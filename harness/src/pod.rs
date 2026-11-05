use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail, ensure};
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use serde_json::json;

use crate::budget;
use crate::clock::Clock;
use crate::cloud::{
    Candidate, Cloud, CloudError, Clouds, CreatePodRequest, GpuRequest, Mounts, NO_VAST_KEY, Offer, PersistentMount,
    Pod, PodStatus, SshEndpoint, find_candidates,
};
use crate::events::{Events, Kind};
use crate::ledger::{
    ADOPT_WINDOW, Ledger, LedgerStore, PodEntry, PodState, VastAccount, check_accounts, check_vast_accounts,
};
use crate::session::{Remote, TRANSFER_LIMIT, Target, Watch, start_job, wait_job};
use crate::spec::{IssueSpec, Provider};
use crate::vast::{VastApi, VastOffer};
use crate::vast_box::{
    VAST_FIRST_READ, base64, vast_create_request, vast_entrypoint, wait_vast_watchdog, watchdog_gone,
};

pub const ENTRYPOINT: &str = include_str!("../pod/entrypoint.sh");
/// sshd, the watchdog's curl, procps (pkill, free) for the pod scripts, and patch for patch_pkg.sh, whatever
/// the issue lists: slim images ship none of them.
const REQUIRED_APT: [&str; 4] = ["openssh-server", "curl", "procps", "patch"];

#[derive(Debug)]
pub struct Interrupted;

impl fmt::Display for Interrupted {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("interrupted")
    }
}

impl std::error::Error for Interrupted {}

pub struct Ctx<'a> {
    pub repo: &'a Path,
    /// RunPod's client always, Vast's when a Vast key is configured.
    pub cloud: &'a dyn Clouds,
    pub remote: &'a dyn Remote,
    pub clock: &'a dyn Clock,
    pub events: &'a Events<'a>,
    pub interrupted: &'a AtomicBool,
    /// The RunPod key's organization from `Cloud::account`, read once per command.
    pub account: Option<&'a str>,
}

impl Ctx<'_> {
    pub fn ledger(&self, issue: &str) -> LedgerStore {
        let vast = match self.cloud.vast() {
            Some(_) => VastAccount::of(self.cloud.vast_account()),
            None => VastAccount::Absent,
        };
        LedgerStore::new(self.repo, issue)
            .with_account(self.account)
            .with_vast_account(vast)
    }

    /// The client for a pod on `provider`.
    pub fn cloud_on(&self, provider: Provider) -> Result<&dyn Cloud> {
        self.cloud
            .on(provider)
            .with_context(|| format!("this pod is on {provider}, but {NO_VAST_KEY}"))
    }

    pub fn check_interrupted(&self) -> Result<()> {
        if self.interrupted.load(Ordering::SeqCst) {
            return Err(Interrupted.into());
        }
        Ok(())
    }

    pub fn target(&self, issue: &str, pod_name: &str, endpoint: &SshEndpoint) -> Target {
        Target {
            host: endpoint.host.clone(),
            port: endpoint.port,
            known_hosts: self
                .repo
                .join("results")
                .join(issue)
                .join(".ssh")
                .join(format!("known_hosts-{pod_name}")),
        }
    }

    pub fn warn(&self, issue: &str, detail: String) {
        self.events.emit(Some(issue), Kind::Warn, json!({ "detail": detail }));
    }
}

#[derive(Debug)]
pub struct LivePod {
    pub entry: PodEntry,
    pub target: Target,
}

/// Who ends the pod when `up` fails after the pod became usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AfterFailure {
    /// `harness up`: the pod stays for debugging until `harness down` or its deadline.
    KeepPod,
    /// `harness run`: its teardown ends the pod right after.
    CallerEnds,
}

/// Context on an error after which the pod still bills: which pod, until when, and how to stop it now. `main`
/// reads it back to put the same facts into the error event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StillLive {
    pub issue: String,
    pub pod: String,
    pub id: Option<String>,
    pub deadline: Timestamp,
}

impl StillLive {
    pub fn of(issue: &str, entry: &PodEntry) -> Self {
        Self {
            issue: issue.to_string(),
            pod: entry.name.clone(),
            id: entry.id.clone(),
            deadline: entry.deadline,
        }
    }

    /// The deadline on the laptop's clock, since the person reading it acts on local time.
    pub fn deadline_local(&self) -> String {
        self.deadline
            .to_zoned(TimeZone::system())
            .strftime("%Y-%m-%d %H:%M:%S %Z")
            .to_string()
    }

    pub fn next_step(&self) -> String {
        format!("harness down {}", self.issue)
    }
}

impl fmt::Display for StillLive {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "pod {} (id {}) is still up and billing until its watchdog ends it at {}; end it now with `{}`",
            self.pod,
            self.id.as_deref().unwrap_or("unknown"),
            self.deadline_local(),
            self.next_step()
        )
    }
}

/// The harness destroyed a pod whose watchdog was gone, so the pod no longer bills and no error after it should say
/// it does.
#[derive(Debug)]
pub struct WatchdogGone {
    pub pod: String,
    pub seen: String,
}

impl fmt::Display for WatchdogGone {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "pod {}'s watchdog is gone, so nothing on it would have ended it at its deadline; it was destroyed. The \
             check printed: {:?}",
            self.pod, self.seen
        )
    }
}

impl std::error::Error for WatchdogGone {}

/// Records a command's error as an event, with the live pod's facts when the error carries them, so a monitor
/// tailing events.jsonl sees every failure without reading stderr.
pub fn report_error(events: &Events, issue: Option<&str>, error: &anyhow::Error) {
    let mut fields = json!({ "detail": format!("{error:#}") });
    if let Some(live) = error.downcast_ref::<StillLive>() {
        fields["pod"] = json!(live.pod);
        fields["id"] = json!(live.id);
        fields["deadline"] = json!(live.deadline);
        fields["deadline_local"] = json!(live.deadline_local());
        fields["next"] = json!(live.next_step());
    }
    events.emit(issue, Kind::Error, fields);
}

/// The key's RunPod organization, checked against every ledger before anything reads a 404 as gone. Commands that
/// only end pods (`down`, `reap`) warn and go on when the check fails or cannot run: a terminate never sent costs
/// more than one sent with another account's key, which RunPod answers with a harmless 404, and the ledger write
/// after it still refuses to book that account's pod as gone.
pub fn key_account(
    cloud: &dyn Clouds,
    repo: &Path,
    events: &Events,
    issue: Option<&str>,
    ends_pods: bool,
) -> Result<Option<String>> {
    let read = cloud.runpod().account();
    let check = |account: Option<&str>| check_accounts(repo, account);
    settle_account(read, "the key's RunPod organization", check, events, issue, ends_pods)
}

/// As `key_account`, for the Vast key against every ledger's Vast pods.
pub fn vast_key_account(
    vast: &dyn VastApi,
    repo: &Path,
    events: &Events,
    issue: Option<&str>,
    ends_pods: bool,
) -> Result<Option<String>> {
    let check = |account: Option<&str>| check_vast_accounts(repo, &VastAccount::of(account));
    settle_account(
        vast.account(),
        "the Vast key's account",
        check,
        events,
        issue,
        ends_pods,
    )
}

fn settle_account(
    read: Result<Option<String>, CloudError>,
    what: &str,
    check: impl Fn(Option<&str>) -> Result<()>,
    events: &Events,
    issue: Option<&str>,
    ends_pods: bool,
) -> Result<Option<String>> {
    let warn = |e: anyhow::Error| {
        let detail = format!("{e:#}; ending pods anyway, and no ledger write under this key books one as gone");
        events.emit(issue, Kind::Warn, json!({ "detail": detail }));
    };
    let account = match read {
        Ok(account) => account,
        Err(e) => {
            let e = anyhow!(e).context(format!("reading {what}"));
            if !ends_pods {
                return Err(e);
            }
            warn(e);
            None
        }
    };
    if let Err(e) = check(account.as_deref()) {
        if !ends_pods {
            return Err(e);
        }
        warn(e);
    }
    Ok(account)
}

pub fn cents(usd: f64) -> f64 {
    let rounded = (usd * 100.0).round() / 100.0;
    // An empty float sum is -0.0, which would print as "-0.0".
    if rounded == 0.0 { 0.0 } else { rounded }
}

pub fn pod_name(issue: &str) -> Result<String> {
    let mut bytes = [0u8; 3];
    File::open("/dev/urandom")
        .and_then(|mut random| random.read_exact(&mut bytes))
        .context("reading /dev/urandom")?;
    Ok(format!(
        "sm120-{issue}-{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2]
    ))
}

/// True only for names `pod_name(issue)` makes, so issue `sglang-44291` never claims `sglang-44291-stage2`'s pods.
pub fn is_pod_of(issue: &str, name: &str) -> bool {
    name.strip_prefix("sm120-")
        .and_then(|rest| rest.strip_prefix(issue))
        .and_then(|rest| rest.strip_prefix('-'))
        .is_some_and(|suffix| suffix.len() == 6 && suffix.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
}

/// A pod of this issue that the ledger does not track was made by another checkout or before a git revert of the
/// ledger. Its reservation lives in a ledger this process cannot see, so reserving beside it would spend the
/// budget twice.
fn refuse_untracked(issue: &str, ledger: &Ledger, provider: Provider, pods: &[Pod]) -> Result<()> {
    let untracked = pods.iter().find(|pod| {
        pod.status != PodStatus::Terminated
            && is_pod_of(issue, &pod.name)
            && !ledger.tracks_on(provider, &pod.id, &pod.name)
    });
    if let Some(pod) = untracked {
        bail!(
            "{provider} lists pod {} (id {}, {:?}) for issue {issue}, but issues/{issue}/harness-ledger.json does \
             not track it; most likely another checkout made it or the ledger was reverted. Nothing was reserved. \
             Its watchdog ends it at its deadline; restore the ledger that holds it, or wait for that, before `up`",
            pod.name,
            pod.id,
            pod.status
        );
    }
    Ok(())
}

/// The watchdog's deadline epoch comes from the laptop clock, and a pod keeps billing for as long as that clock
/// is ahead of the real one.
const MAX_CLOCK_SKEW: SignedDuration = SignedDuration::from_secs(120);

fn refuse_clock_skew(ctx: &Ctx, issue: &str, provider: Provider) -> Result<()> {
    let Some(skew) = ctx.cloud_on(provider)?.clock_skew() else {
        ctx.warn(
            issue,
            format!("{provider} sent no Date header, so the laptop clock went unchecked"),
        );
        return Ok(());
    };
    ensure!(
        skew.abs() <= MAX_CLOCK_SKEW,
        "this laptop's clock is {} s off {provider}'s, and the pod's watchdog deadline is set from it; fix the clock \
         (System Settings > Date & Time, or `sudo sntp -sS time.apple.com`) before `up`. Nothing was reserved",
        -skew.as_secs()
    );
    Ok(())
}

/// Vast runs its own sshd, which a reinstall of openssh-server could disturb, so a Vast box goes without it.
pub(crate) fn apt_packages(spec: &IssueSpec, provider: Provider) -> String {
    let mut apt: Vec<&str> = REQUIRED_APT
        .into_iter()
        .filter(|package| provider == Provider::Runpod || *package != "openssh-server")
        .collect();
    for package in &spec.pod.apt {
        if !apt.contains(&package.as_str()) {
            apt.push(package);
        }
    }
    apt.join(" ")
}

pub fn create_request(spec: &IssueSpec, entry: &PodEntry, data_centers: &[String]) -> CreatePodRequest {
    let mut env = BTreeMap::from([
        (
            "HARNESS_DEADLINE_EPOCH".to_string(),
            entry.deadline.as_second().to_string(),
        ),
        ("HARNESS_APT".to_string(), apt_packages(spec, Provider::Runpod)),
        ("HF_HOME".to_string(), "/workspace/hf".to_string()),
    ]);
    if spec.pod.needs_hf_token {
        // A RunPod secret reference: the token itself never appears in the request, the ledger, or the events.
        env.insert("HF_TOKEN".into(), "{{ RUNPOD_SECRET_hf_token }}".into());
    }
    CreatePodRequest {
        name: entry.name.clone(),
        image: spec.pod.image.clone(),
        entrypoint: vec!["/bin/bash".into(), "-c".into(), ENTRYPOINT.into()],
        env,
        ports: vec!["22/tcp".into()],
        disk: spec.pod.container_disk_gb,
        mounts: Mounts {
            persistent: PersistentMount {
                size: spec.pod.volume_gb,
                path: "/workspace".into(),
            },
        },
        cloud: entry.cloud,
        data_center_ids: data_centers.to_vec(),
        gpu: GpuRequest {
            id: entry.gpu.clone(),
            count: entry.count,
            min_cuda_version: spec.pod.min_cuda.clone(),
        },
        start_ssh: true,
    }
}

/// What `nvidia-smi topo -m` prints between two GPUs that share no NVLink; NVLink pairs show NV1, NV2, ...
const PCIE_LINKS: [&str; 6] = ["X", "PIX", "PXB", "PHB", "NODE", "SYS"];

/// nvidia-smi underlines the header with ANSI escapes even when stdout is not a terminal.
fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            if chars.next() == Some('[') {
                for c in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&c) {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn is_device(cell: &str, kind: &str) -> bool {
    cell.strip_prefix(kind)
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// Fails closed: the matrix must name exactly GPU0..GPU{count-1} and every GPU-to-GPU cell must be a PCIe link,
/// so empty or garbled output (a driver error, a wedged GPU) refuses the box instead of passing it. NIC rows and
/// columns and the legend are ignored.
pub fn check_pcie_topology(topo: &str, count: u32) -> Result<()> {
    let text = strip_ansi(topo);
    let rows: Vec<Vec<&str>> = text
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .filter(|cells| cells.first().is_some_and(|first| is_device(first, "GPU")))
        .collect();
    // The header starts with GPU0 like a row does, but its second cell names a device or a column, not a link.
    let is_header = |cells: &Vec<&str>| {
        cells
            .get(1)
            .is_none_or(|cell| !PCIE_LINKS.contains(cell) && !cell.starts_with("NV"))
    };
    let Some(header) = rows.iter().find(|cells| is_header(cells)) else {
        bail!("{TOPOLOGY} printed no GPU matrix, so the PCIe-only check cannot pass:\n{topo}");
    };
    let columns: Vec<&str> = header
        .iter()
        .copied()
        .take_while(|cell| is_device(cell, "GPU") || is_device(cell, "NIC"))
        .collect();
    let expected: Vec<String> = (0..count).map(|i| format!("GPU{i}")).collect();
    let gpu_columns: Vec<(usize, &str)> = columns
        .iter()
        .enumerate()
        .filter(|(_, cell)| is_device(cell, "GPU"))
        .map(|(i, cell)| (i, *cell))
        .collect();
    let gpu_rows: Vec<&Vec<&str>> = rows.iter().filter(|cells| !is_header(cells)).collect();
    let column_names: Vec<&str> = gpu_columns.iter().map(|(_, cell)| *cell).collect();
    let row_names: Vec<&str> = gpu_rows.iter().map(|cells| cells[0]).collect();
    ensure!(
        column_names == expected && row_names == expected,
        "{TOPOLOGY} does not show exactly the {count} GPUs this pod rents, so the PCIe-only check cannot pass:\n{topo}"
    );
    for row in gpu_rows {
        for &(column, peer) in &gpu_columns {
            let cell = row.get(column + 1).copied().unwrap_or("");
            ensure!(
                !cell.starts_with("NV"),
                "{TOPOLOGY} shows NVLink ({} to {peer}: {cell}), and this repro needs a PCIe-only box:\n{topo}",
                row[0]
            );
            ensure!(
                PCIE_LINKS.contains(&cell),
                "{TOPOLOGY} shows {cell:?} between {} and {peer}, which is not a PCIe link:\n{topo}",
                row[0]
            );
        }
    }
    Ok(())
}

/// Fails closed like the topology gate: a reply with no readable GPU line refuses the box.
pub fn check_gpus_idle(reply: &str) -> Result<()> {
    let mut seen = 0;
    for line in reply.lines().filter(|line| !line.trim().is_empty()) {
        let reading = line
            .split_once(',')
            .and_then(|(index, used)| Some((index.trim(), used.trim().parse::<u64>().ok()?)));
        let Some((index, used)) = reading else {
            bail!("{GPU_MEMORY} printed {line:?}, not a GPU index and a MiB count:\n{reply}");
        };
        ensure!(
            used < IDLE_MIB,
            "GPU {index} has {used} MiB in use before anything of ours ran, so a process outside this container \
             still holds it:\n{reply}"
        );
        seen += 1;
    }
    ensure!(seen > 0, "{GPU_MEMORY} printed no GPUs:\n{reply}");
    Ok(())
}

/// Fails closed like the topology gate: the matrix must name exactly GPU0..GPU{count-1}, and every pair must read
/// OK. Any other reading (NS, CNS, GNS, TNS, U) means no peer access between those two GPUs, where NCCL with P2P
/// on has hung PCIe boxes before.
pub fn check_p2p(matrix: &str, count: u32) -> Result<()> {
    let text = strip_ansi(matrix);
    let rows: Vec<Vec<&str>> = text
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .filter(|cells| cells.first().is_some_and(|first| is_device(first, "GPU")))
        .collect();
    // The header names GPUs only; a row follows its GPU's name with readings.
    let is_header = |cells: &Vec<&str>| cells.iter().all(|cell| is_device(cell, "GPU"));
    let expected: Vec<String> = (0..count).map(|i| format!("GPU{i}")).collect();
    let header: Option<&Vec<&str>> = rows.iter().find(|cells| is_header(cells));
    let body: Vec<&Vec<&str>> = rows.iter().filter(|cells| !is_header(cells)).collect();
    let row_names: Vec<&str> = body.iter().map(|cells| cells[0]).collect();
    ensure!(
        header.is_some_and(|header| *header == expected) && row_names == expected,
        "{P2P} does not show exactly the {count} GPUs this pod rents, so the P2P check cannot pass:\n{matrix}"
    );
    for row in body {
        for (column, peer) in expected.iter().enumerate() {
            let cell = row.get(column + 1).copied().unwrap_or("");
            let wanted = if row[0] == peer { "X" } else { "OK" };
            ensure!(
                cell == wanted,
                "{P2P} shows {cell:?} between {} and {peer}, and this repro needs P2P on every pair:\n{matrix}",
                row[0]
            );
        }
    }
    Ok(())
}

/// Read only on a pod whose issue sets `require_p2p`.
pub const P2P: &str = "timeout 60 nvidia-smi topo -p2p r";

/// A GPU that a process outside the container still holds passes the topology gate, then fails the first variant's
/// idle wait minutes later, after setup has paid for the weights. nvidia-smi lists no process for it, since the
/// holder lives in another PID namespace.
pub const GPU_MEMORY: &str = "timeout 60 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits";
/// wait_gpu_idle.sh's bar: a fresh container reads far below it, and a leftover model far above.
const IDLE_MIB: u64 = 2048;
/// The coreutils `timeout` ends nvidia-smi on a wedged GPU well inside the ssh command limit, so the gate fails
/// with a message instead of a killed ssh.
pub const TOPOLOGY: &str = "timeout 60 nvidia-smi topo -m";
pub const POLL: Duration = Duration::from_secs(10);
/// The pull before a delete bills by the minute; results left on a pod cost less than a slow link that keeps it
/// alive.
pub const PULL_BEFORE_DELETE: Duration = Duration::from_secs(5 * 60);
/// Image pulls for serving images run to tens of GB.
const READY_TIMEOUT: SignedDuration = SignedDuration::from_mins(30);
/// RunPod publishes 22/tcp a little after the container starts; a host without public IPs never does, and the
/// harness needs direct SSH for scp and tar (the proxy carries a shell only). It counts from the container start,
/// not from RUNNING: RunPod reports RUNNING while an uncached image still pulls, which can take longer than this.
const NO_DIRECT_SSH: SignedDuration = SignedDuration::from_mins(3);
const NAME_LOOKUPS: usize = 3;
const GONE_CHECKS: usize = 6;

pub fn up(ctx: &Ctx, spec: &IssueSpec, run_setup: bool, after_failure: AfterFailure) -> Result<LivePod> {
    up_with(ctx, spec, None, run_setup, after_failure)
}

/// As `up`, from offers a search just found, so `watch` spends one search, not two, on the offer it rents, and the
/// offer has less time to be taken.
pub fn up_from(
    ctx: &Ctx,
    spec: &IssueSpec,
    candidates: Vec<Candidate>,
    run_setup: bool,
    after_failure: AfterFailure,
) -> Result<LivePod> {
    up_with(ctx, spec, Some(candidates), run_setup, after_failure)
}

fn up_with(
    ctx: &Ctx,
    spec: &IssueSpec,
    found: Option<Vec<Candidate>>,
    run_setup: bool,
    after_failure: AfterFailure,
) -> Result<LivePod> {
    let issue = spec.issue.as_str();
    let ledger = ctx.ledger(issue).load()?;
    if let Some(active) = ledger.active() {
        bail!(
            "issue {issue} already has pod {} ({:?}); run `harness down {issue}` first",
            active.name,
            active.state
        );
    }
    let remaining = ledger.remaining_usd()?;
    ctx.remote.preflight()?;
    let candidates = match found {
        Some(found) => found,
        None => find_candidates(ctx.cloud, &spec.pod)?,
    };
    if candidates.is_empty() {
        return Err(NothingInStock {
            issue: issue.to_string(),
        }
        .into());
    }
    budget::emit_quote(ctx, spec, &candidates, Some(remaining))?;
    let on_vast = candidates.iter().any(|c| c.provider() == Provider::Vast);
    let providers: Vec<Provider> = Provider::ALL
        .into_iter()
        .filter(|p| *p == Provider::Runpod || on_vast)
        .collect();
    for &provider in &providers {
        refuse_untracked(issue, &ledger, provider, &ctx.cloud_on(provider)?.list_pods()?)?;
    }
    for &provider in &providers {
        refuse_clock_skew(ctx, issue, provider)?;
    }
    let Created {
        entry,
        pod,
        cloud,
        cap_usd_hr,
    } = create_first_available(ctx, spec, &candidates)?;
    let limits = Limits {
        avoid: &spec.pod.avoid_data_centers,
        cap_usd_hr,
    };
    let ready = first_read(cloud, &entry, pod)
        .and_then(|pod| wait_ready(ctx, cloud, issue, &entry, pod, &limits))
        .and_then(|target| gate_and_copy(ctx, spec, &entry, &target).map(|()| target));
    let target = match ready {
        Ok(target) => target,
        Err(e) => {
            // The pod is not usable yet, so nothing is lost by ending it now.
            ctx.warn(
                issue,
                format!("up failed before the pod was usable: {e:#}; terminating {}", entry.name),
            );
            return match end_box(ctx, issue, &entry) {
                Ok(()) => Err(e),
                Err(stop) => {
                    let e = e.context(format!("then terminating it failed: {stop:#}"));
                    Err(match after_failure {
                        AfterFailure::KeepPod => e.context(StillLive::of(issue, &entry)),
                        AfterFailure::CallerEnds => e,
                    })
                }
            };
        }
    };
    ctx.events.emit(
        Some(issue),
        Kind::Ready,
        json!({"pod": entry.name, "id": entry.id, "host": target.host, "port": target.port, "deadline": entry.deadline}),
    );
    match setup(ctx, spec, &entry, &target, run_setup) {
        Ok(()) => Ok(LivePod { entry, target }),
        Err(e) if after_failure == AfterFailure::KeepPod && e.downcast_ref::<WatchdogGone>().is_none() => {
            Err(e.context(StillLive::of(issue, &entry)))
        }
        Err(e) => Err(e),
    }
}

/// `up` found no choice in stock. `watch` keeps looking after it; every other command stops.
#[derive(Debug)]
pub struct NothingInStock {
    pub issue: String,
}

impl fmt::Display for NothingInStock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "nothing in the preference list of issues/{}/harness.toml is in stock",
            self.issue
        )
    }
}

impl std::error::Error for NothingInStock {}

/// A pod `up` created and the ledger tracks, with the client of the provider it is on.
struct Created<'c> {
    entry: PodEntry,
    pod: Pod,
    cloud: &'c dyn Cloud,
    cap_usd_hr: Option<f64>,
}

/// A candidate with its provider's client, found before the reservation so that a missing client cannot come
/// between a reservation and its create.
enum Bound<'c, 'o> {
    Runpod {
        cloud: &'c dyn Cloud,
        offer: &'o Offer,
    },
    Vast {
        vast: &'c dyn VastApi,
        offer: &'o VastOffer,
    },
}

impl<'c> Bound<'c, '_> {
    fn cloud(&self) -> &'c dyn Cloud {
        match *self {
            Self::Runpod { cloud, .. } => cloud,
            Self::Vast { vast, .. } => vast,
        }
    }

    /// Vast's reply names only the new id, so its pod is the neutral view before Vast reports anything about it;
    /// `first_read` replaces it.
    fn create(&self, spec: &IssueSpec, entry: &PodEntry, vast_entrypoint: &str) -> Result<Pod, CloudError> {
        match *self {
            Self::Runpod { cloud, offer } => cloud.create_pod(&create_request(spec, entry, &offer.data_centers)),
            Self::Vast { vast, offer } => {
                let id = vast.create_instance(offer.offer_id, &vast_create_request(spec, entry, vast_entrypoint))?;
                Ok(Pod {
                    id,
                    name: entry.name.clone(),
                    status: PodStatus::Provisioning,
                    cost: None,
                    ssh: None,
                    runtime: None,
                    data_center_id: None,
                })
            }
        }
    }
}

impl Candidate {
    fn bind<'c, 'o>(&'o self, clouds: &'c dyn Clouds) -> Result<Bound<'c, 'o>> {
        Ok(match self {
            Self::Runpod(offer) => Bound::Runpod {
                cloud: clouds.runpod(),
                offer,
            },
            Self::Vast { offer, .. } => Bound::Vast {
                vast: clouds
                    .vast()
                    .with_context(|| format!("{}: {NO_VAST_KEY}", Provider::Vast))?,
                offer,
            },
        })
    }

    /// The size of the entrypoint a Vast box gets in its env goes with it, since the live box is where a limit on
    /// that size would show.
    fn reserve_fields(
        &self,
        name: &str,
        entry: &PodEntry,
        spec: &IssueSpec,
        vast_entrypoint: &str,
    ) -> serde_json::Value {
        let mut fields = serde_json::Map::from_iter([("pod".to_string(), json!(name))]);
        fields.extend(self.fields());
        fields.insert("reserved_usd".into(), json!(cents(entry.reserved_usd)));
        fields.insert("deadline".into(), json!(entry.deadline));
        fields.extend(self.reservation_fields(&spec.pod));
        if let Self::Vast { .. } = self {
            fields.insert("entrypoint_b64_bytes".into(), json!(vast_entrypoint.len()));
        }
        fields.into()
    }
}

/// Everything a create needs that could fail is ready before the reservation, so a reservation is always followed by
/// a create.
fn create_first_available<'c>(ctx: &Ctx<'c>, spec: &IssueSpec, candidates: &[Candidate]) -> Result<Created<'c>> {
    let issue = spec.issue.as_str();
    let store = ctx.ledger(issue);
    let vast_entrypoint = if candidates.iter().any(|c| c.provider() == Provider::Vast) {
        base64(vast_entrypoint()?.as_bytes())
    } else {
        String::new()
    };
    let mut rejections = Vec::new();
    for candidate in candidates {
        ctx.check_interrupted()?;
        let bound = candidate.bind(ctx.cloud)?;
        let cloud = bound.cloud();
        let name = pod_name(issue)?;
        let now = ctx.clock.now();
        let download_gb = spec.pod.vast_download_gb();
        let entry =
            store.update(|ledger| ledger.reserve_candidate(&name, candidate, spec.planned_hours, download_gb, now))?;
        ctx.events.emit(
            Some(issue),
            Kind::Reserve,
            candidate.reserve_fields(&name, &entry, spec, &vast_entrypoint),
        );
        match bound.create(spec, &entry, &vast_entrypoint) {
            Ok(pod) => {
                let entry = track(cloud, &store, &name, &pod.id)?;
                ctx.events
                    .emit(Some(issue), Kind::Create, json!({"pod": name, "id": pod.id}));
                return Ok(Created {
                    entry,
                    pod,
                    cloud,
                    cap_usd_hr: candidate.cap_usd_hr(),
                });
            }
            Err(e) if e.create_outcome_unknown() => {
                let (entry, pod) = adopt_by_name(ctx, cloud, &store, &entry, e)?;
                return Ok(Created {
                    entry,
                    pod,
                    cloud,
                    cap_usd_hr: candidate.cap_usd_hr(),
                });
            }
            Err(e) => {
                store.update(|ledger| ledger.entry_mut(&name)?.mark_create_failed())?;
                if !candidate.refusal_moves_on(&e) {
                    return Err(e).context(format!("creating {name}"));
                }
                ctx.warn(issue, format!("{} was refused: {e}", candidate.describe()));
                rejections.push(e.to_string());
            }
        }
    }
    bail!("every in-stock choice was refused: {}", rejections.join(" | "))
}

/// RunPod's create reply is a full pod. Vast's names only the id, so its first read comes from the API, and a read
/// that fails ends `up` the way a failed wait does.
fn first_read(cloud: &dyn Cloud, entry: &PodEntry, pod: Pod) -> Result<Pod> {
    match entry.provider {
        Provider::Runpod => Ok(pod),
        Provider::Vast => Ok(cloud.get_pod(&pod.id)?.unwrap_or(pod)),
    }
}

/// Records a new pod's id. A pod the ledger cannot record is deleted at once: nothing would show it billing, and
/// `run`'s teardown could not find it to end it.
fn track(cloud: &dyn Cloud, store: &LedgerStore, name: &str, id: &str) -> Result<PodEntry> {
    let recorded = store.update(|ledger| {
        let entry = ledger.entry_mut(name)?;
        entry.mark_live(id)?;
        Ok(entry.clone())
    });
    let Err(e) = recorded else {
        return recorded;
    };
    let outcome = match cloud.delete_pod(id) {
        Ok(()) => "deleted it".to_string(),
        Err(delete) => format!("could not delete it ({delete}); its watchdog ends it at its deadline"),
    };
    Err(e.context(format!(
        "recording pod {name} ({id}) in {} failed, so the harness {outcome}",
        store.path().display()
    )))
}

fn adopt_by_name(
    ctx: &Ctx,
    cloud: &dyn Cloud,
    store: &LedgerStore,
    entry: &PodEntry,
    error: CloudError,
) -> Result<(PodEntry, Pod)> {
    let issue = store.issue();
    ctx.warn(
        issue,
        format!(
            "create of {} may have gone through ({error}); looking it up by name",
            entry.name
        ),
    );
    let mut last_error = None;
    for attempt in 0.. {
        // Vast can take minutes to list a new instance, so on Vast the lookup runs to the end of the adopt window.
        let more = match entry.provider {
            Provider::Runpod => attempt < NAME_LOOKUPS,
            Provider::Vast => attempt == 0 || entry.may_be_unlisted(ctx.clock.now()),
        };
        if !more {
            break;
        }
        if attempt > 0 {
            ctx.check_interrupted()?;
            ctx.clock.sleep(POLL);
        }
        match cloud.list_pods() {
            Ok(pods) => {
                if let Some(pod) = pods.into_iter().find(|pod| pod.name == entry.name) {
                    let entry = track(cloud, store, &entry.name, &pod.id)?;
                    ctx.events.emit(
                        Some(issue),
                        Kind::Create,
                        json!({"pod": entry.name, "id": pod.id, "adopted": true}),
                    );
                    return Ok((entry, pod));
                }
            }
            Err(e) => last_error = Some(e),
        }
    }
    let lookup = last_error.map_or_else(String::new, |e| format!(" (last lookup: {e})"));
    if entry.provider == Provider::Vast {
        store.update(|ledger| {
            let entry = ledger.entry_mut(&entry.name)?;
            entry.mark_create_failed()?;
            entry.abandoned = true;
            Ok(())
        })?;
        return Err(anyhow!(error)).context(format!(
            "creating {} failed and no instance with that label was listed within {} min{lookup}; it is booked as a \
             failed create and marked abandoned, so the reaper destroys it if Vast lists it later",
            entry.name,
            ADOPT_WINDOW.as_mins()
        ));
    }
    Err(anyhow!(error)).context(format!(
        "creating {} failed and no pod by that name is listed{lookup}; the reservation stays until the reaper adopts \
         or releases it",
        entry.name
    ))
}

/// Where a new pod may land and what it may bill, checked on every read while it starts.
struct Limits<'s> {
    avoid: &'s [String],
    cap_usd_hr: Option<f64>,
}

/// The deadline was sized for the quoted rate, so a pod that bills above it, or whose rate cannot be read, could
/// outspend its reservation before the watchdog fires. A Vast choice's `max_usd_hr` holds without the margin.
fn check_rate(ctx: &Ctx, issue: &str, entry: &PodEntry, pod: &Pod, cap: Option<f64>, warned: &mut bool) -> Result<()> {
    let provider = entry.provider;
    let reserved_rate = entry.reserved_rate_usd_hr();
    if pod.cost.is_none() && !*warned {
        *warned = true;
        ctx.warn(
            issue,
            format!("{provider}'s reply for pod {} has no cost field", entry.name),
        );
    }
    if let (Some(cost), Some(cap)) = (pod.cost, cap) {
        ensure!(
            cost <= cap,
            "pod {} bills ${cost:.3}/hr, above the ${cap:.2}/hr max_usd_hr its choice allows",
            entry.name
        );
    }
    match pod.cost {
        Some(cost) if cost > reserved_rate => bail!(
            "pod {} bills ${cost:.3}/hr, above the ${reserved_rate:.3}/hr its deadline was sized for",
            entry.name
        ),
        // RunPod reports 0 only for a pod that is not running, so a running pod at 0 means the rate is unknown.
        Some(cost) if cost <= 0.0 && pod.status == PodStatus::Running => bail!(
            "pod {} is RUNNING but {provider} reports its cost as ${cost}/hr, so its rate cannot be checked against \
             the ${reserved_rate:.3}/hr its deadline was sized for",
            entry.name
        ),
        None if pod.status == PodStatus::Running => bail!(
            "pod {} is RUNNING but {provider} reports no cost, so its rate cannot be checked against the \
             ${reserved_rate:.3}/hr its deadline was sized for",
            entry.name
        ),
        _ => Ok(()),
    }
}

/// The v2 schema calls `dataCenterIds` preferred data centers, so the placement is checked too: a pod in an avoided
/// data center, where setup may never finish, ends before setup at a cost of minutes instead of the whole budget.
fn check_placement(entry: &PodEntry, pod: &Pod, avoid: &[String]) -> Result<()> {
    match &pod.data_center_id {
        Some(dc) if avoid.contains(dc) => bail!(
            "RunPod placed pod {} in data center {dc}, which pod.avoid_data_centers excludes",
            entry.name
        ),
        _ => Ok(()),
    }
}

fn wait_ready(
    ctx: &Ctx,
    cloud: &dyn Cloud,
    issue: &str,
    entry: &PodEntry,
    mut pod: Pod,
    limits: &Limits,
) -> Result<Target> {
    let avoid = limits.avoid;
    let provider = entry.provider;
    let started = ctx.clock.now();
    let mut warned_no_cost = false;
    let mut warned_no_container = false;
    let mut proxy_only_since = None;
    let mut placed = false;
    loop {
        check_rate(ctx, issue, entry, &pod, limits.cap_usd_hr, &mut warned_no_cost)?;
        check_placement(entry, &pod, avoid)?;
        placed |= pod.data_center_id.is_some();
        let now = ctx.clock.now();
        match pod.status {
            PodStatus::Exited | PodStatus::Error | PodStatus::Terminated => {
                bail!("pod {} went {:?} while starting", entry.name, pod.status)
            }
            PodStatus::Running => match pod.ssh_direct() {
                Some(endpoint) => {
                    proxy_only_since = None;
                    let target = ctx.target(issue, &entry.name, endpoint);
                    if ctx.remote.reachable(&target) {
                        if !avoid.is_empty() && !placed {
                            ctx.warn(
                                issue,
                                format!(
                                    "RunPod reported no data center for pod {} on any read, so \
                                     pod.avoid_data_centers went unchecked after placement",
                                    entry.name
                                ),
                            );
                        }
                        return Ok(target);
                    }
                }
                None => match pod.container() {
                    Some(container) => {
                        let since = *proxy_only_since.get_or_insert(now);
                        // Vast's proxy is a fallback the endpoint already takes, so there neither way is open.
                        let missing = match provider {
                            Provider::Runpod => "no direct SSH endpoint (proxy-only host)",
                            Provider::Vast => "no SSH endpoint (no mapped 22/tcp and no SSH proxy)",
                        };
                        ensure!(
                            now.duration_since(since) < NO_DIRECT_SSH,
                            "pod {}: {provider} gave {missing} within {} min of its container starting; runtime \
                             ports: {}",
                            entry.name,
                            NO_DIRECT_SSH.as_mins(),
                            container.ports
                        );
                    }
                    None if !warned_no_container => {
                        warned_no_container = true;
                        ctx.warn(
                            issue,
                            format!(
                                "pod {} is RUNNING but has no container yet (image still pulling); waiting up to {} \
                                 min from create",
                                entry.name,
                                READY_TIMEOUT.as_mins()
                            ),
                        );
                    }
                    None => {}
                },
            },
            PodStatus::Provisioning | PodStatus::Starting | PodStatus::Unknown => {}
        }
        ctx.check_interrupted()?;
        ensure!(
            now < entry.deadline,
            "pod {} reached its deadline {} while starting",
            entry.name,
            entry.deadline
        );
        ensure!(
            now.duration_since(started) < READY_TIMEOUT,
            "pod {} did not answer on SSH within {} min",
            entry.name,
            READY_TIMEOUT.as_mins()
        );
        ctx.clock.sleep(POLL);
        let id = &pod.id;
        pod = match cloud.get_pod(id)? {
            Some(read) => read,
            // A new Vast instance can take a moment to show; ending it on the first empty read would destroy a box
            // that may be about to appear.
            None if provider == Provider::Vast && ctx.clock.now().duration_since(started) < VAST_FIRST_READ => pod,
            None => bail!("pod {id} disappeared while starting"),
        };
    }
}

/// The start of the line entrypoint.sh's watchdog logs as it arms, after the timestamp; its pid follows.
pub(crate) fn watchdog_armed_prefix(deadline: Timestamp) -> String {
    format!("watchdog armed for epoch {} in pid ", deadline.as_second())
}

/// Prints WATCHDOG_ARMED only while the watchdog that logged the armed line is the container's main process:
/// PID 1 itself, or the only child of a docker-init or tini PID 1. Both inits exit when that child exits and the
/// kernel then kills the whole container, so nothing can kill the watchdog and leave the pod running. RunPod
/// starts containers under docker-init (seen on a live pod, 2026-10-16). sshd also ends up a child of PID 1 once it
/// daemonizes, so the watchdog's own command line has to be the entrypoint too.
fn watchdog_check_script(deadline: Timestamp, log: &str, proc: &str) -> String {
    let armed = watchdog_armed_prefix(deadline);
    format!(
        r#"log={log}; proc={proc}
pid=$(sed -nE 's/^[^ ]* {armed}([0-9]+)$/\1/p' "$log" 2>/dev/null | tail -n 1)
main=no
if [[ $pid == 1 ]]; then
    main=yes
elif [[ -n $pid && -r $proc/$pid/status ]]; then
    ppid=$(awk '$1 == "PPid:" {{print $2}}' "$proc/$pid/status")
    init=$(tr '\0' ' ' <"$proc/1/cmdline" 2>/dev/null)
    own=$(tr '\0' ' ' <"$proc/$pid/cmdline" 2>/dev/null)
    init_re='^(/sbin/docker-init|[^ ]*/tini(-static)?) (-[^ ]+ )*-- /bin/bash -c #!/usr/bin/env bash'
    if [[ $ppid == 1 && $init =~ $init_re && $own == "/bin/bash -c #!/usr/bin/env bash"* ]]; then
        main=yes
    fi
fi
if [[ $main == yes ]]; then
    echo WATCHDOG_ARMED
else
    echo WATCHDOG_MISSING
    tail -n 5 "$log" 2>/dev/null || true
    echo '--- pid 1:'
    tr '\0' ' ' <"$proc/1/cmdline" 2>/dev/null
    echo
    ps -eo pid,ppid,user,args --forest 2>/dev/null | head -n 30 | cut -c1-160 || true
fi"#
    )
}

pub(crate) const WATCHDOG_LOG: &str = "/workspace/harness-watchdog.log";

/// The watchdog is what ends the pod once the laptop is gone. An image ENTRYPOINT or an API change that skips
/// entrypoint.sh leaves a pod that only `down` or a later reaper pass can end, so such a pod is ended before use.
/// The script also sources the container env the entrypoint wrote, which fails the same way.
fn check_watchdog(ctx: &Ctx, entry: &PodEntry, target: &Target) -> Result<()> {
    if entry.provider == Provider::Vast {
        return wait_vast_watchdog(ctx, entry, target);
    }
    let script = watchdog_check_script(entry.deadline, WATCHDOG_LOG, "/proc");
    let reply = ctx.remote.run(target, &script).context("checking the pod's watchdog")?;
    ensure!(
        reply.lines().any(|line| line == "WATCHDOG_ARMED"),
        "pod {} shows no watchdog armed as the container's main process for its deadline {}, so nothing on the \
         pod would end it; the watchdog log ends: {:?}",
        entry.name,
        entry.deadline,
        reply.trim()
    );
    Ok(())
}

/// Every box's P2P matrix is kept, in a file of its own beside env.txt (env_report.sh's): a passing one is what a
/// benchmark on the box rests on, and a failing one is all that is left of the box.
fn gate_p2p(ctx: &Ctx, spec: &IssueSpec, entry: &PodEntry, target: &Target) -> Result<()> {
    let matrix = ctx.remote.run(target, P2P)?;
    let dir = ctx.repo.join("results").join(&spec.issue);
    let saved = fs::create_dir_all(&dir).and_then(|()| {
        fs::write(
            dir.join(format!("p2p-{}.txt", entry.name)),
            format!("## P2P on pod {} ({P2P})\n{matrix}", entry.name),
        )
    });
    if let Err(write) = saved {
        ctx.warn(&spec.issue, format!("saving the P2P matrix failed: {write}"));
    }
    check_p2p(&matrix, spec.pod.count)
}

fn gate_and_copy(ctx: &Ctx, spec: &IssueSpec, entry: &PodEntry, target: &Target) -> Result<()> {
    check_watchdog(ctx, entry, target)?;
    if spec.pod.require_pcie() {
        check_pcie_topology(&ctx.remote.run(target, TOPOLOGY)?, spec.pod.count)?;
    }
    if spec.pod.require_p2p {
        gate_p2p(ctx, spec, entry, target)?;
    }
    check_gpus_idle(&ctx.remote.run(target, GPU_MEMORY)?)?;
    let issue_dir = format!("issues/{}", spec.issue);
    ctx.remote.upload(target, ctx.repo, &["harness/pod", &issue_dir])
}

fn setup(ctx: &Ctx, spec: &IssueSpec, entry: &PodEntry, target: &Target, run_setup: bool) -> Result<()> {
    let issue = spec.issue.as_str();
    let mut command = format!("bash harness/pod/env_report.sh > results/{issue}/env.txt 2>&1");
    let setup_sh = run_setup && ctx.repo.join("issues").join(issue).join("setup.sh").is_file();
    if setup_sh {
        command.push_str(&format!(" && bash issues/{issue}/setup.sh"));
    }
    start_job(ctx.remote, target, issue, "setup", &command)?;
    let watch = Watch {
        on_line: &mut |_| {},
        every: None,
    };
    let code = wait_job(ctx, issue, entry, target, "setup", "/dev/null", watch)?;
    if let Err(e) = pull(ctx, issue, target, TRANSFER_LIMIT) {
        ctx.warn(issue, format!("pulling setup output failed: {e:#}"));
    }
    ensure!(code == 0, "setup exited {code}; see results/{issue}/jobs/setup.log");
    ctx.events.emit(
        Some(issue),
        Kind::Setup,
        json!({"pod": entry.name, "setup_sh": setup_sh}),
    );
    Ok(())
}

/// Pulls all of results/<issue>.
pub fn pull(ctx: &Ctx, issue: &str, target: &Target, limit: Duration) -> Result<()> {
    pull_paths(ctx, issue, target, &[format!("results/{issue}")], limit)
}

pub fn pull_paths(ctx: &Ctx, issue: &str, target: &Target, paths: &[String], limit: Duration) -> Result<()> {
    let refs: Vec<&str> = paths.iter().map(String::as_str).collect();
    ctx.remote.download(target, &refs, ctx.repo, limit)?;
    ctx.events.emit(Some(issue), Kind::Pull, json!({ "paths": paths }));
    Ok(())
}

fn live_entry(ctx: &Ctx, issue: &str) -> Result<Option<PodEntry>> {
    let ledger = ctx.ledger(issue).load()?;
    Ok(ledger.pods.into_iter().find(|p| p.state == PodState::Live))
}

fn target_for(ctx: &Ctx, issue: &str, entry: &PodEntry) -> Result<Target> {
    let id = entry.id.as_deref().context("live pod without an id")?;
    let pod = ctx
        .cloud_on(entry.provider)?
        .get_pod(id)?
        .with_context(|| format!("pod {id} is no longer listed"))?;
    let endpoint = pod
        .ssh_direct()
        .with_context(|| format!("pod {id} has no direct SSH endpoint (status {:?})", pod.status))?;
    Ok(ctx.target(issue, &entry.name, endpoint))
}

pub fn live_pod(ctx: &Ctx, issue: &str) -> Result<LivePod> {
    let entry =
        live_entry(ctx, issue)?.with_context(|| format!("issue {issue} has no live pod; run `harness up {issue}`"))?;
    let target = target_for(ctx, issue, &entry)?;
    Ok(LivePod { entry, target })
}

pub fn down(ctx: &Ctx, issue: &str) -> Result<()> {
    let Some(entry) = live_entry(ctx, issue)? else {
        // A reservation whose create outcome is unknown may still have made a pod, so it is not "down".
        let ledger = ctx.ledger(issue).load()?;
        if let Some(reserved) = ledger.active() {
            bail!(
                "pod {} is reserved without an id; `harness reap` adopts and ends it once {} lists it, or releases \
                 it {} min after {}",
                reserved.name,
                reserved.provider,
                ADOPT_WINDOW.as_mins(),
                reserved.reserved_at
            );
        }
        return Ok(());
    };
    let pulled = target_for(ctx, issue, &entry).and_then(|target| pull(ctx, issue, &target, PULL_BEFORE_DELETE));
    if let Err(e) = pulled {
        ctx.warn(issue, format!("pull before terminate failed: {e:#}"));
    }
    terminate(ctx, issue, &entry).map_err(|e| e.context(StillLive::of(issue, &entry)))
}

/// RunPod's watchdog is the container's main process, so it cannot die and leave the pod up. Vast's runs beside the
/// box's own init, so a box in use is checked again, and one whose watchdog is gone is destroyed.
pub fn check_watchdog_alive(ctx: &Ctx, issue: &str, entry: &PodEntry, target: &Target) -> Result<()> {
    if entry.provider != Provider::Vast {
        return Ok(());
    }
    let Some(seen) = watchdog_gone(ctx, issue, entry, target) else {
        return Ok(());
    };
    end_box(ctx, issue, entry).map_err(|e| {
        e.context(format!(
            "pod {}'s watchdog is gone ({seen:?}), and destroying the box failed",
            entry.name
        ))
    })?;
    Err(WatchdogGone {
        pod: entry.name.clone(),
        seen,
    }
    .into())
}

/// How often a box nobody will use is destroyed before the harness gives up and leaves it to the reaper.
const END_ATTEMPTS: usize = 3;

/// A box nobody will use any more bills until it is gone, so its destroy is tried more than once.
pub fn end_box(ctx: &Ctx, issue: &str, entry: &PodEntry) -> Result<()> {
    let mut attempt = 1;
    loop {
        match terminate(ctx, issue, entry) {
            Ok(()) => return Ok(()),
            Err(e) if attempt >= END_ATTEMPTS => return Err(e),
            Err(e) => ctx.warn(
                issue,
                format!("destroying pod {} failed, trying again: {e:#}", entry.name),
            ),
        }
        attempt += 1;
        ctx.clock.sleep(POLL);
    }
}

/// Deletes the pod (on Vast, destroys it: never a stop, which keeps billing the disk), waits until its provider no
/// longer lists it, then books the estimate and any billing record.
pub fn terminate(ctx: &Ctx, issue: &str, entry: &PodEntry) -> Result<()> {
    end_pod(ctx, issue, entry, false)
}

/// The reaper's delete of a pod past its deadline: as `terminate`, with the estimate capped at the reservation.
pub fn terminate_overdue(ctx: &Ctx, issue: &str, entry: &PodEntry) -> Result<()> {
    end_pod(ctx, issue, entry, true)
}

fn end_pod(ctx: &Ctx, issue: &str, entry: &PodEntry, overdue: bool) -> Result<()> {
    let provider = entry.provider;
    let id = entry
        .id
        .as_deref()
        .with_context(|| format!("pod {} has no id to terminate", entry.name))?;
    let cloud = ctx.cloud_on(provider)?;
    // A Vast instance no read has shown yet may still appear, so after its destroy, absence would prove nothing. The
    // read only decides what the ledger may book: a failed one must never hold back the destroy itself.
    let unseen = provider == Provider::Vast
        && entry.may_be_unlisted(ctx.clock.now())
        && !matches!(cloud.get_pod(id), Ok(Some(_)));
    let destroyed = cloud.delete_pod(id).map_err(anyhow::Error::from).and_then(|()| {
        ensure!(
            !unseen,
            "Vast did not show instance {id} before its destroy, so its destroy cannot be confirmed; it stays live in \
             the ledger, marked abandoned: the reaper destroys it once Vast lists it, or books it once Vast has had {} \
             min to show it",
            ADOPT_WINDOW.as_mins()
        );
        confirm_gone(ctx, cloud, id)
    });
    if let Err(e) = destroyed {
        if provider == Provider::Vast {
            abandon(ctx, issue, &entry.name);
        }
        return Err(e);
    }
    let at = ctx.clock.now();
    let entry = ctx.ledger(issue).update(|ledger| {
        let entry = ledger.entry_mut(&entry.name)?;
        if entry.state == PodState::Live {
            entry.mark_terminated(at)?;
            if overdue {
                entry.cap_estimate_at_reservation();
            }
        }
        Ok(entry.clone())
    })?;
    let entry = record_billing(ctx, cloud, issue, entry);
    ctx.events.emit(
        Some(issue),
        Kind::Terminate,
        json!({"pod": entry.name, "id": id, "cost_usd": entry.cost_usd.map(cents), "cost_source": entry.cost_source}),
    );
    Ok(())
}

/// Marks a Vast pod the harness tried and failed to end, so that the reaper destroys it as soon as Vast lists it
/// instead of at its deadline. A failed mark only warns: the error that led here says more.
fn abandon(ctx: &Ctx, issue: &str, name: &str) {
    let marked = ctx.ledger(issue).update(|ledger| {
        let entry = ledger.entry_mut(name)?;
        if entry.state != PodState::Terminated {
            entry.abandoned = true;
        }
        Ok(())
    });
    if let Err(e) = marked {
        ctx.warn(issue, format!("marking pod {name} abandoned failed: {e:#}"));
    }
}

/// The reaper's path for an instance Vast listed under a failed create's label: nothing uses it, so it is destroyed
/// at once and booked from its reservation on.
pub fn destroy_late_instance(ctx: &Ctx, issue: &str, name: &str, id: &str) -> Result<()> {
    let cloud = ctx.cloud_on(Provider::Vast)?;
    cloud.delete_pod(id)?;
    confirm_gone(ctx, cloud, id)?;
    let at = ctx.clock.now();
    let entry = ctx.ledger(issue).update(|ledger| {
        let entry = ledger.entry_mut(name)?;
        if entry.state == PodState::CreateFailed {
            entry.book_late_instance(id, at)?;
        }
        Ok(entry.clone())
    })?;
    let entry = record_billing(ctx, cloud, issue, entry);
    ctx.events.emit(
        Some(issue),
        Kind::Terminate,
        json!({"pod": name, "id": id, "detail": "listed after its create failed",
               "cost_usd": entry.cost_usd.map(cents), "cost_source": entry.cost_source}),
    );
    Ok(())
}

fn still_billing(pod: Option<Pod>) -> Option<PodStatus> {
    pod.map(|pod| pod.status)
        .filter(|status| *status != PodStatus::Terminated)
}

/// RunPod answers 404, and Vast an empty reply, both for a deleted pod and for one this key cannot see, so a GET
/// that says gone counts only when the full pod list agrees.
fn confirm_gone(ctx: &Ctx, cloud: &dyn Cloud, id: &str) -> Result<()> {
    let mut seen = String::new();
    for check in 0..GONE_CHECKS {
        if check > 0 {
            ctx.clock.sleep(POLL);
        }
        if let Some(status) = still_billing(cloud.get_pod(id)?) {
            seen = format!("GET shows it {status:?}");
            continue;
        }
        let pods = cloud.list_pods()?;
        match still_billing(pods.into_iter().find(|pod| pod.id == id)) {
            Some(status) => seen = format!("GET finds no pod, but the pod list shows it {status:?}"),
            None => return Ok(()),
        }
    }
    bail!("pod {id} may still bill after delete ({seen}); it stays live in the ledger")
}

/// The reaper's path for a live pod missing from a complete pod list. A GET must agree, so a list that briefly
/// drops a pod never books one that still bills.
pub fn mark_gone(ctx: &Ctx, issue: &str, name: &str) -> Result<()> {
    let ledger = ctx.ledger(issue).load()?;
    let Some(entry) = ledger.pods.iter().find(|entry| entry.name == name) else {
        bail!("ledger for {issue} has no pod named {name}");
    };
    let id = entry.id.as_deref().context("live pod without an id")?;
    let provider = entry.provider;
    let cloud = ctx.cloud_on(provider)?;
    if let Some(status) = still_billing(cloud.get_pod(id)?) {
        bail!(
            "pod {name} ({id}) is missing from the pod list but GET shows it {status:?}; it stays live in the ledger"
        );
    }
    let now = ctx.clock.now();
    let entry = ctx.ledger(issue).update(|ledger| {
        let entry = ledger.entry_mut(name)?;
        if entry.state == PodState::Live {
            // A watchdog ends the pod at its deadline; billing corrects an earlier or later end.
            let end = now.min(entry.deadline);
            entry.mark_terminated(end)?;
        }
        Ok(entry.clone())
    })?;
    let entry = record_billing(ctx, cloud, issue, entry);
    ctx.events.emit(
        Some(issue),
        Kind::Terminate,
        json!({"pod": entry.name, "id": entry.id, "detail": format!("no longer listed by {provider}"),
               "cost_usd": entry.cost_usd.map(cents), "cost_source": entry.cost_source}),
    );
    Ok(())
}

fn record_billing(ctx: &Ctx, cloud: &dyn Cloud, issue: &str, entry: PodEntry) -> PodEntry {
    let now = ctx.clock.now();
    let Some(id) = entry.id.clone().filter(|_| entry.needs_billing(now)) else {
        return entry;
    };
    let billed = match cloud.pod_billing_usd(&id) {
        Ok(Some(billed)) => billed,
        Ok(None) => return entry,
        Err(e) => {
            ctx.warn(issue, format!("billing lookup for {id} failed: {e}"));
            return entry;
        }
    };
    let updated = ctx.ledger(issue).update(|ledger| {
        let found = ledger.entry_mut(&entry.name)?;
        found.apply_billing(billed, now);
        Ok(found.clone())
    });
    match updated {
        Ok(updated) => updated,
        Err(e) => {
            ctx.warn(issue, format!("recording billing for {id} failed: {e:#}"));
            entry
        }
    }
}

/// Upgrades estimates to billing records where the pod's provider has them; `budget show` calls it every time.
pub fn reconcile_billing(ctx: &Ctx, issue: &str) -> Result<()> {
    let now = ctx.clock.now();
    let ledger = ctx.ledger(issue).load()?;
    for entry in ledger.pods.iter().filter(|entry| entry.needs_billing(now)) {
        let provider = entry.provider;
        match ctx.cloud.on(provider) {
            Some(cloud) => {
                record_billing(ctx, cloud, issue, entry.clone());
            }
            None => ctx.warn(
                issue,
                format!(
                    "pod {} keeps its estimate: its billing is on {provider}, and {NO_VAST_KEY}",
                    entry.name
                ),
            ),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PCIE: &str = include_str!("../tests/fixtures/topo_pcie.txt");

    fn topology_error(topo: &str, count: u32) -> String {
        format!("{:#}", check_pcie_topology(topo, count).unwrap_err())
    }

    #[test]
    fn a_pcie_matrix_with_nics_ansi_and_a_legend_passes() {
        assert!(PCIE.contains("\u{1b}[4m") && PCIE.contains("NIC3") && PCIE.contains("NV#"));
        check_pcie_topology(PCIE, 2).unwrap();
        let one = "\t\u{1b}[4mGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\u{1b}[0m\nGPU0\t X \t0-63\t0\t\tN/A\n";
        check_pcie_topology(one, 1).unwrap();
        let every_link = "\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\nGPU0\t X \tPIX\tPXB\tPHB\t0\n\
                          GPU1\tPIX\t X \tNODE\tSYS\t0\nGPU2\tPXB\tNODE\t X \tSYS\t0\nGPU3\tPHB\tSYS\tSYS\t X \t0\n";
        check_pcie_topology(every_link, 4).unwrap();
    }

    #[test]
    fn nvlink_fails_the_gate() {
        let nvlink = "\tGPU0\tGPU1\tCPU Affinity\nGPU0\t X \tNV18\t0-63\nGPU1\tNV18\t X \t0-63\n";
        assert!(topology_error(nvlink, 2).contains("shows NVLink (GPU0 to GPU1: NV18)"));
    }

    #[test]
    fn output_that_proves_nothing_fails_the_gate() {
        for garbled in [
            "",
            "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n",
            "GPU0\t X \tNODE\nNIC0\tNODE\t X\n",
            "\tGPU0\tGPU1\nGPU0\t X \tSYS\n",
            "\tGPU0\tGPU1\nGPU0\t X \tSYS\nGPU1\t??\t X \n",
            "\tGPU0\tGPU1\nGPU0\t X\nGPU1\tSYS\t X \n",
        ] {
            assert!(check_pcie_topology(garbled, 2).is_err(), "{garbled:?} passed");
        }
        assert!(topology_error(PCIE, 1).contains("exactly the 1 GPUs"));
        assert!(topology_error(PCIE, 4).contains("exactly the 4 GPUs"));
    }

    #[test]
    fn idle_gpus_pass_the_memory_gate() {
        check_gpus_idle("0, 0\n1, 2047\n").unwrap();
    }

    #[test]
    fn a_gpu_held_from_outside_fails_the_memory_gate() {
        let err = format!("{:#}", check_gpus_idle("0, 3\n1, 90892\n2, 3\n").unwrap_err());
        assert!(err.contains("GPU 1 has 90892 MiB in use"), "{err}");
    }

    #[test]
    fn output_that_proves_nothing_fails_the_memory_gate() {
        for garbled in [
            "",
            "\n",
            "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n",
            "0, [N/A]\n",
            "0, 0\n1\n",
        ] {
            assert!(check_gpus_idle(garbled).is_err(), "{garbled:?} passed");
        }
    }

    #[test]
    fn the_watchdog_check_looks_for_the_line_the_entrypoint_logs() {
        assert!(ENTRYPOINT.contains("log \"watchdog armed for epoch $deadline in pid $$\""));
        let deadline: Timestamp = "2026-10-16T20:03:00Z".parse().unwrap();
        assert_eq!(
            watchdog_armed_prefix(deadline),
            "watchdog armed for epoch 1792180980 in pid "
        );
    }

    /// Runs the real check script in local bash against a fake /proc and watchdog log.
    fn watchdog_check(log_lines: &[&str], procs: &[(u32, u32, &str)]) -> String {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("sm120-watchdog-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for (pid, ppid, cmdline) in procs {
            let p = dir.join("proc").join(pid.to_string());
            std::fs::create_dir_all(&p).unwrap();
            std::fs::write(p.join("status"), format!("Name:\tx\nPid:\t{pid}\nPPid:\t{ppid}\n")).unwrap();
            std::fs::write(p.join("cmdline"), cmdline.replace(' ', "\0") + "\0").unwrap();
        }
        let log = dir.join("watchdog.log");
        std::fs::write(&log, log_lines.join("\n") + "\n").unwrap();
        let deadline: Timestamp = "2026-10-16T20:03:00Z".parse().unwrap();
        let script = watchdog_check_script(deadline, log.to_str().unwrap(), dir.join("proc").to_str().unwrap());
        let out = std::process::Command::new("bash")
            .arg("-c")
            .arg(script)
            .output()
            .unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .next()
            .unwrap_or("")
            .to_string()
    }

    #[test]
    fn the_watchdog_counts_as_armed_only_as_the_containers_main_process() {
        const ENTRY: &str = "/bin/bash -c #!/usr/bin/env bash\n# Pod entrypoint";
        let docker_init = format!("/sbin/docker-init -- {ENTRY}");
        let armed = |pid: u32| format!("2026-10-16T19:48:00Z watchdog armed for epoch 1792180980 in pid {pid}");
        let other_epoch = "2026-10-16T19:48:00Z watchdog armed for epoch 1792180000 in pid 1";
        // What RunPod runs: docker-init as PID 1, the entrypoint as its child, sshd reparented to PID 1.
        let runpod = [
            (1, 0, docker_init.as_str()),
            (497, 1, ENTRY),
            (2294, 1, "sshd: /usr/sbin/sshd"),
        ];
        assert_eq!(watchdog_check(&[&armed(497)], &runpod), "WATCHDOG_ARMED");
        assert_eq!(watchdog_check(&[&armed(1)], &[(1, 0, ENTRY)]), "WATCHDOG_ARMED");
        let tini = format!("/usr/bin/tini -s -g -- {ENTRY}");
        assert_eq!(
            watchdog_check(&[&armed(497)], &[(1, 0, tini.as_str()), (497, 1, ENTRY)]),
            "WATCHDOG_ARMED"
        );
        for (lines, procs) in [
            // a deadline that is not the ledger's
            (vec![other_epoch], &runpod[..]),
            // the watchdog has died since it logged
            (vec![&armed(498)], &runpod[..]),
            // a pid that is not the entrypoint, though PID 1 is its parent
            (vec![&armed(2294)], &runpod[..]),
            // the newest line wins over an older armed line
            (vec![&armed(1), &armed(2294)], &runpod[..]),
        ] {
            assert_eq!(watchdog_check(&lines, procs), "WATCHDOG_MISSING", "{lines:?}");
        }
        // an init that outlives its child, or an entrypoint that is not the init's direct child
        let supervisor = [(1, 0, "/usr/bin/supervisord -n"), (497, 1, ENTRY)];
        assert_eq!(watchdog_check(&[&armed(497)], &supervisor), "WATCHDOG_MISSING");
        let nested = [(1, 0, docker_init.as_str()), (300, 1, ENTRY), (497, 300, ENTRY)];
        assert_eq!(watchdog_check(&[&armed(497)], &nested), "WATCHDOG_MISSING");
        assert_eq!(watchdog_check(&[], &runpod), "WATCHDOG_MISSING");
    }

    /// `nvidia-smi topo -p2p r` with one pair set to `pair` and the header underlined as nvidia-smi does on a tty.
    fn p2p_matrix(count: usize, pair: Option<(usize, usize, &str)>) -> String {
        let names: Vec<String> = (0..count).map(|i| format!("GPU{i}")).collect();
        let mut out = format!(" \t\u{1b}[4m{}\t\u{1b}[0m\n", names.join("\t"));
        for row in 0..count {
            let cells: Vec<&str> = (0..count)
                .map(|col| match pair {
                    _ if row == col => "X",
                    Some((a, b, cell)) if (row, col) == (a, b) || (row, col) == (b, a) => cell,
                    _ => "OK",
                })
                .collect();
            out.push_str(&format!(" GPU{row}\t{}\t\n", cells.join("\t")));
        }
        out + "\nLegend:\n\n  X    = Self\n  OK   = Status Ok\n  CNS  = Chipset not supported\n  NS   = Not supported\n"
    }

    #[test]
    fn a_matrix_where_every_pair_reads_ok_passes_the_p2p_gate() {
        check_p2p(&p2p_matrix(8, None), 8).unwrap();
        check_p2p(&p2p_matrix(2, None), 2).unwrap();
        check_p2p(&p2p_matrix(1, None), 1).unwrap();
    }

    #[test]
    fn any_pair_without_p2p_fails_the_gate() {
        for cell in ["NS", "CNS", "GNS", "TNS", "U"] {
            let err = format!("{:#}", check_p2p(&p2p_matrix(8, Some((2, 7, cell))), 8).unwrap_err());
            assert!(
                err.contains(&format!("shows \"{cell}\" between GPU2 and GPU7")),
                "{err}"
            );
            assert!(err.contains("Legend:"), "the error carries the matrix: {err}");
        }
    }

    #[test]
    fn a_p2p_reply_that_proves_nothing_fails_the_gate() {
        for garbled in [
            String::new(),
            "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n".into(),
            p2p_matrix(4, None),
            p2p_matrix(8, None).replace(" GPU7\tOK", " GPU7\t"),
            p2p_matrix(8, None).replacen("\tX\t", "\tOK\t", 1),
        ] {
            assert!(check_p2p(&garbled, 8).is_err(), "{garbled:?} passed");
        }
    }

    #[test]
    fn pod_names_are_unique_and_carry_the_issue() {
        let a = pod_name("vllm-47990").unwrap();
        let b = pod_name("vllm-47990").unwrap();
        assert_ne!(a, b);
        let suffix = a.strip_prefix("sm120-vllm-47990-").unwrap();
        assert!(suffix.len() == 6 && suffix.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert!(is_pod_of("vllm-47990", &a));
    }

    #[test]
    fn a_pod_belongs_to_one_issue_only() {
        assert!(is_pod_of("sglang-44291", "sm120-sglang-44291-0a9f3c"));
        for name in [
            "sm120-sglang-44291-stage2-0a9f3c",
            "sm120-sglang-44291-0A9F3C",
            "sm120-sglang-44291-0a9f3",
            "sm120-sglang-44291-0a9f3c1",
            "sm120-sglang-4429-10a9f3c",
            "sm120-sglang-44291-0a9f3g",
            "sglang-44291-0a9f3c",
        ] {
            assert!(!is_pod_of("sglang-44291", name), "{name}");
        }
    }
}
