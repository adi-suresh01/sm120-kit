use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail, ensure};
use jiff::SignedDuration;
use serde_json::json;

use crate::clock::Clock;
use crate::cloud::{
    Cloud, CloudError, CreatePodRequest, GpuRequest, Mounts, PersistentMount, Pod, PodStatus, SshEndpoint, find_offers,
};
use crate::events::{Events, Kind};
use crate::ledger::{ADOPT_WINDOW, Ledger, LedgerStore, Offer, PodEntry, PodState, plan_reservation};
use crate::session::{Remote, TRANSFER_LIMIT, Target, poll_job, start_job};
use crate::spec::IssueSpec;

pub const ENTRYPOINT: &str = include_str!("../pod/entrypoint.sh");
/// sshd, the watchdog's curl, and procps (pkill, free) for the pod scripts, whatever the issue lists.
const REQUIRED_APT: [&str; 3] = ["openssh-server", "curl", "procps"];

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
    pub cloud: &'a dyn Cloud,
    pub remote: &'a dyn Remote,
    pub clock: &'a dyn Clock,
    pub events: &'a Events<'a>,
    pub interrupted: &'a AtomicBool,
    /// From `Cloud::account`, read once per command.
    pub account: Option<&'a str>,
}

impl Ctx<'_> {
    pub fn ledger(&self, issue: &str) -> LedgerStore {
        LedgerStore::new(self.repo, issue).with_account(self.account)
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

pub struct LivePod {
    pub entry: PodEntry,
    pub target: Target,
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
fn refuse_untracked(issue: &str, ledger: &Ledger, pods: &[Pod]) -> Result<()> {
    let untracked = pods.iter().find(|pod| {
        pod.status != PodStatus::Terminated && is_pod_of(issue, &pod.name) && !ledger.tracks(&pod.id, &pod.name)
    });
    if let Some(pod) = untracked {
        bail!(
            "RunPod lists pod {} (id {}, {:?}) for issue {issue}, but issues/{issue}/harness-ledger.json does not \
             track it; most likely another checkout made it or the ledger was reverted. Nothing was reserved. Its \
             watchdog ends it at its deadline; restore the ledger that holds it, or wait for that, before `up`",
            pod.name,
            pod.id,
            pod.status
        );
    }
    Ok(())
}

pub fn create_request(spec: &IssueSpec, entry: &PodEntry) -> CreatePodRequest {
    let mut apt: Vec<&str> = REQUIRED_APT.to_vec();
    for package in &spec.pod.apt {
        if !apt.contains(&package.as_str()) {
            apt.push(package);
        }
    }
    let mut env = BTreeMap::from([
        (
            "HARNESS_DEADLINE_EPOCH".to_string(),
            entry.deadline.as_second().to_string(),
        ),
        ("HARNESS_APT".to_string(), apt.join(" ")),
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

/// The coreutils `timeout` ends nvidia-smi on a wedged GPU well inside the ssh command limit, so the gate fails
/// with a message instead of a killed ssh.
pub const TOPOLOGY: &str = "timeout 60 nvidia-smi topo -m";
pub const POLL: Duration = Duration::from_secs(10);
/// The pull before a delete bills by the minute; results left on a pod cost less than a slow link that keeps it
/// alive.
pub const PULL_BEFORE_DELETE: Duration = Duration::from_secs(5 * 60);
/// Image pulls for serving images run to tens of GB.
const READY_TIMEOUT: SignedDuration = SignedDuration::from_mins(30);
const NAME_LOOKUPS: usize = 3;
const GONE_CHECKS: usize = 6;
/// About five minutes of failed polls, at POLL apart, before a job counts as lost.
const POLL_FAILURES: u32 = 30;

pub fn quote(ctx: &Ctx, spec: &IssueSpec) -> Result<()> {
    let offers = find_offers(ctx.cloud, &spec.pod)?;
    let Some(first) = offers.first() else {
        bail!(
            "nothing in the preference list of issues/{}/harness.toml is in stock",
            spec.issue
        );
    };
    let remaining = ctx.ledger(&spec.issue).load()?.remaining_usd()?;
    let plan = plan_reservation(ctx.clock.now(), spec.planned_hours, remaining, first.rate_usd_hr)?;
    let fallbacks: Vec<String> = offers[1..]
        .iter()
        .map(|o| format!("{} {} ${:.2}/hr", o.gpu, o.cloud.as_str(), o.rate_usd_hr))
        .collect();
    ctx.events.emit(
        Some(&spec.issue),
        Kind::Quote,
        json!({
            "gpu": first.gpu, "cloud": first.cloud, "count": first.count, "rate_usd_hr": first.rate_usd_hr,
            "deadline": plan.deadline, "worst_case_usd": cents(plan.reserved_usd),
            "remaining_usd": cents(remaining), "fallbacks": fallbacks,
        }),
    );
    Ok(())
}

pub fn up(ctx: &Ctx, spec: &IssueSpec, run_setup: bool) -> Result<LivePod> {
    let issue = spec.issue.as_str();
    let ledger = ctx.ledger(issue).load()?;
    if let Some(active) = ledger.active() {
        bail!(
            "issue {issue} already has pod {} ({:?}); run `harness down {issue}` first",
            active.name,
            active.state
        );
    }
    ctx.remote.preflight()?;
    let offers = find_offers(ctx.cloud, &spec.pod)?;
    ensure!(
        !offers.is_empty(),
        "nothing in the preference list of issues/{issue}/harness.toml is in stock"
    );
    refuse_untracked(issue, &ledger, &ctx.cloud.list_pods()?)?;
    let (entry, pod) = create_first_available(ctx, spec, &offers)?;
    let ready =
        wait_ready(ctx, issue, &entry, pod).and_then(|target| gate_and_copy(ctx, spec, &target).map(|()| target));
    let target = match ready {
        Ok(target) => target,
        Err(e) => {
            // The pod is not usable yet, so nothing is lost by ending it now.
            ctx.warn(
                issue,
                format!("up failed before the pod was usable: {e:#}; terminating {}", entry.name),
            );
            if let Err(stop) = terminate(ctx, issue, &entry) {
                ctx.events.emit(
                    Some(issue),
                    Kind::Error,
                    json!({"pod": entry.name, "detail": format!("terminate failed: {stop:#}"), "watchdog_at": entry.deadline}),
                );
            }
            return Err(e);
        }
    };
    ctx.events.emit(
        Some(issue),
        Kind::Ready,
        json!({"pod": entry.name, "id": entry.id, "host": target.host, "port": target.port, "deadline": entry.deadline}),
    );
    setup(ctx, spec, &entry, &target, run_setup)?;
    Ok(LivePod { entry, target })
}

fn create_first_available(ctx: &Ctx, spec: &IssueSpec, offers: &[Offer]) -> Result<(PodEntry, Pod)> {
    let issue = spec.issue.as_str();
    let store = ctx.ledger(issue);
    let mut rejections = Vec::new();
    for offer in offers {
        ctx.check_interrupted()?;
        let name = pod_name(issue)?;
        let now = ctx.clock.now();
        let entry = store.update(|ledger| ledger.reserve(&name, offer, spec.planned_hours, now))?;
        ctx.events.emit(
            Some(issue),
            Kind::Reserve,
            json!({
                "pod": name, "gpu": offer.gpu, "cloud": offer.cloud, "count": offer.count,
                "rate_usd_hr": offer.rate_usd_hr, "reserved_usd": cents(entry.reserved_usd), "deadline": entry.deadline,
            }),
        );
        match ctx.cloud.create_pod(&create_request(spec, &entry)) {
            Ok(pod) => {
                let entry = mark_live(&store, &name, &pod.id)?;
                ctx.events
                    .emit(Some(issue), Kind::Create, json!({"pod": name, "id": pod.id}));
                return Ok((entry, pod));
            }
            Err(e) if e.create_outcome_unknown() => return adopt_by_name(ctx, &store, &entry, e),
            Err(e) => {
                store.update(|ledger| ledger.entry_mut(&name)?.mark_create_failed())?;
                // RunPod answers 400 both for a broken request and for no capacity on this GPU; the next choice
                // tells the two apart.
                if e.status() != Some(400) {
                    return Err(e).context(format!("creating {name}"));
                }
                ctx.warn(
                    issue,
                    format!("{} on {} was refused: {e}", offer.gpu, offer.cloud.as_str()),
                );
                rejections.push(e.to_string());
            }
        }
    }
    bail!("every in-stock choice was refused: {}", rejections.join(" | "))
}

fn mark_live(store: &LedgerStore, name: &str, id: &str) -> Result<PodEntry> {
    store.update(|ledger| {
        let entry = ledger.entry_mut(name)?;
        entry.mark_live(id)?;
        Ok(entry.clone())
    })
}

fn adopt_by_name(ctx: &Ctx, store: &LedgerStore, entry: &PodEntry, error: CloudError) -> Result<(PodEntry, Pod)> {
    let issue = store.issue();
    ctx.warn(
        issue,
        format!(
            "create of {} may have gone through ({error}); looking it up by name",
            entry.name
        ),
    );
    let mut last_error = None;
    for attempt in 0..NAME_LOOKUPS {
        if attempt > 0 {
            ctx.clock.sleep(POLL);
        }
        match ctx.cloud.list_pods() {
            Ok(pods) => {
                if let Some(pod) = pods.into_iter().find(|pod| pod.name == entry.name) {
                    let entry = mark_live(store, &entry.name, &pod.id)?;
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
    Err(anyhow!(error)).context(format!(
        "creating {} failed and no pod by that name is listed{lookup}; the reservation stays until the reaper adopts \
         or releases it",
        entry.name
    ))
}

fn wait_ready(ctx: &Ctx, issue: &str, entry: &PodEntry, mut pod: Pod) -> Result<Target> {
    let started = ctx.clock.now();
    let reserved_rate = entry.reserved_rate_usd_hr();
    loop {
        if let Some(cost) = pod.cost
            && cost > reserved_rate
        {
            bail!(
                "pod {} bills ${cost:.3}/hr, above the ${reserved_rate:.3}/hr its deadline was sized for",
                entry.name
            );
        }
        match pod.status {
            PodStatus::Exited | PodStatus::Error | PodStatus::Terminated => {
                bail!("pod {} went {:?} while starting", entry.name, pod.status)
            }
            PodStatus::Running => {
                if let Some(endpoint) = pod.ssh_direct() {
                    let target = ctx.target(issue, &entry.name, endpoint);
                    if ctx.remote.reachable(&target) {
                        return Ok(target);
                    }
                }
            }
            PodStatus::Provisioning | PodStatus::Starting | PodStatus::Unknown => {}
        }
        ctx.check_interrupted()?;
        let now = ctx.clock.now();
        if now.duration_since(started) >= READY_TIMEOUT || now >= entry.deadline {
            bail!(
                "pod {} did not answer on SSH within {} min",
                entry.name,
                READY_TIMEOUT.as_mins()
            );
        }
        ctx.clock.sleep(POLL);
        let id = &pod.id;
        pod = ctx
            .cloud
            .get_pod(id)?
            .with_context(|| format!("pod {id} disappeared while starting"))?;
    }
}

fn gate_and_copy(ctx: &Ctx, spec: &IssueSpec, target: &Target) -> Result<()> {
    if spec.pod.require_pcie() {
        check_pcie_topology(&ctx.remote.run(target, TOPOLOGY)?, spec.pod.count)?;
    }
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
    let code = wait_job(ctx, issue, entry, target, "setup", "/dev/null", &mut |_| {})?;
    if let Err(e) = pull(ctx, issue, target, TRANSFER_LIMIT) {
        ctx.warn(issue, format!("pulling setup output failed: {e:#}"));
    }
    ensure!(
        code == 0,
        "setup exited {code}; see results/{issue}/jobs/setup.log. Pod {} stays up for debugging until {} or \
         `harness down {issue}`",
        entry.name,
        entry.deadline
    );
    ctx.events.emit(
        Some(issue),
        Kind::Setup,
        json!({"pod": entry.name, "setup_sh": setup_sh}),
    );
    Ok(())
}

/// Polls a job started with `start_job` until it exits, handing every complete new line of `tail_path` to
/// `on_line`. Ends early on Ctrl-C, at the pod deadline, or after POLL_FAILURES polls in a row fail.
pub fn wait_job(
    ctx: &Ctx,
    issue: &str,
    entry: &PodEntry,
    target: &Target,
    job: &str,
    tail_path: &str,
    on_line: &mut dyn FnMut(&str),
) -> Result<i32> {
    let mut offset = 0;
    let mut failures = 0;
    loop {
        ctx.check_interrupted()?;
        ensure!(
            ctx.clock.now() < entry.deadline,
            "pod {} reached its deadline while {job} ran",
            entry.name
        );
        match poll_job(ctx.remote, target, issue, job, tail_path, offset) {
            Ok(status) => {
                failures = 0;
                status.lines.iter().for_each(|line| on_line(line));
                offset = status.offset;
                if let Some(code) = status.exit {
                    return Ok(code);
                }
            }
            Err(e) => {
                failures += 1;
                if failures >= POLL_FAILURES {
                    return Err(e).context(format!("lost contact with the pod while {job} ran"));
                }
                ctx.warn(issue, format!("polling {job} failed ({failures} in a row): {e:#}"));
            }
        }
        ctx.clock.sleep(POLL);
    }
}

pub fn pull(ctx: &Ctx, issue: &str, target: &Target, limit: Duration) -> Result<()> {
    let path = format!("results/{issue}");
    ctx.remote.download(target, &[&path], ctx.repo, limit)?;
    ctx.events.emit(Some(issue), Kind::Pull, json!({ "path": path }));
    Ok(())
}

fn live_entry(ctx: &Ctx, issue: &str) -> Result<Option<PodEntry>> {
    Ok(ctx
        .ledger(issue)
        .load()?
        .pods
        .into_iter()
        .find(|p| p.state == PodState::Live))
}

fn target_for(ctx: &Ctx, issue: &str, entry: &PodEntry) -> Result<Target> {
    let id = entry.id.as_deref().context("live pod without an id")?;
    let pod = ctx
        .cloud
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
        if let Some(reserved) = ctx.ledger(issue).load()?.active() {
            bail!(
                "pod {} is reserved without an id; `harness reap` adopts and ends it once RunPod lists it, or \
                 releases it {} min after {}",
                reserved.name,
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
    terminate(ctx, issue, &entry)
}

/// Deletes the pod, waits until RunPod no longer lists it, then books the estimate and any billing record.
pub fn terminate(ctx: &Ctx, issue: &str, entry: &PodEntry) -> Result<()> {
    let id = entry
        .id
        .as_deref()
        .with_context(|| format!("pod {} has no id to terminate", entry.name))?;
    ctx.cloud.delete_pod(id)?;
    confirm_gone(ctx, id)?;
    let at = ctx.clock.now();
    let entry = ctx.ledger(issue).update(|ledger| {
        let entry = ledger.entry_mut(&entry.name)?;
        if entry.state == PodState::Live {
            entry.mark_terminated(at)?;
        }
        Ok(entry.clone())
    })?;
    let entry = record_billing(ctx, issue, entry);
    ctx.events.emit(
        Some(issue),
        Kind::Terminate,
        json!({"pod": entry.name, "id": id, "cost_usd": entry.cost_usd.map(cents), "cost_source": entry.cost_source}),
    );
    Ok(())
}

fn still_billing(pod: Option<Pod>) -> Option<PodStatus> {
    pod.map(|pod| pod.status)
        .filter(|status| *status != PodStatus::Terminated)
}

/// RunPod answers 404 both for a deleted pod and for one this key cannot see, so a GET that says gone counts only
/// when the full pod list agrees.
fn confirm_gone(ctx: &Ctx, id: &str) -> Result<()> {
    let mut seen = String::new();
    for check in 0..GONE_CHECKS {
        if check > 0 {
            ctx.clock.sleep(POLL);
        }
        if let Some(status) = still_billing(ctx.cloud.get_pod(id)?) {
            seen = format!("GET shows it {status:?}");
            continue;
        }
        let pods = ctx.cloud.list_pods()?;
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
    if let Some(status) = still_billing(ctx.cloud.get_pod(id)?) {
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
    let entry = record_billing(ctx, issue, entry);
    ctx.events.emit(
        Some(issue),
        Kind::Terminate,
        json!({"pod": entry.name, "id": entry.id, "detail": "no longer listed by RunPod",
               "cost_usd": entry.cost_usd.map(cents), "cost_source": entry.cost_source}),
    );
    Ok(())
}

fn record_billing(ctx: &Ctx, issue: &str, entry: PodEntry) -> PodEntry {
    let now = ctx.clock.now();
    let Some(id) = entry.id.clone().filter(|_| entry.needs_billing(now)) else {
        return entry;
    };
    let billed = match ctx.cloud.pod_billing_usd(&id) {
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

/// Upgrades estimates to billing records where RunPod has them; `budget show` calls it every time.
pub fn reconcile_billing(ctx: &Ctx, issue: &str) -> Result<()> {
    let now = ctx.clock.now();
    for entry in ctx.ledger(issue).load()?.pods {
        if entry.needs_billing(now) {
            record_billing(ctx, issue, entry);
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
