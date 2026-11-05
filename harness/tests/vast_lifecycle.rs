//! `up`, `down`, `reap`, and `quote` for an issue on Vast, against a fake Vast beside the fake RunPod: the same
//! ledger and safety rules as RunPod, with every call about a Vast pod going to Vast.

mod support;

use std::cell::RefCell;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use anyhow::Result;
use jiff::SignedDuration;
use sm120_harness::budget;
use sm120_harness::clock::Clock;
use sm120_harness::cloud::{
    CatalogGpu, Cloud, CloudError, Clouds, CreatePodRequest, Pod, PodRuntime, PodSsh, PodStatus, Providers, SshEndpoint,
};
use sm120_harness::events::Events;
use sm120_harness::ledger::{Budget, CostSource, Ledger, LedgerStore, PodState, VastPlacement};
use sm120_harness::pod::{self, AfterFailure, Ctx};
use sm120_harness::reap;
use sm120_harness::session::{Remote, Target};
use sm120_harness::spec::{CloudTier, IssueSpec};
use sm120_harness::vast::{OfferQuery, VastApi, VastCreateRequest, VastOffer};
use sm120_harness::watch::{self, Outcome};
use support::fake_cloud::FakeCloud;
use support::fake_remote::FakeRemote;
use support::{FakeClock, TempDir, repo_with_issue};

const ISSUE: &str = "vllm-59203-tp2";
const SPEC: &str = r#"
issue = "vllm-59203-tp2"
planned_hours = 1.0
[pod]
image = "vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
count = 2
min_cuda = "13.0"
container_disk_gb = 40
volume_gb = 80
require_p2p = true
gpus = [
  { provider = "vast", id = "RTX PRO 6000 Max-Q", max_usd_hr = 3.0 },
  { id = "NVIDIA RTX PRO 6000 Blackwell Server Edition", cloud = "SECURE" },
]
[server]
package = "vllm"
model = "m"
launch = ["vllm", "serve", "m"]
port = 8000
fail_pattern = "illegal memory access"
boot_timeout_s = 900
[variants.baseline]
flags = []
"#;

#[derive(Debug, Clone, Copy)]
enum Create {
    Made,
    /// Vast's 200 with `success: false`, as a taken offer gets.
    Taken,
    MadeButReplyLost,
}

struct VastState {
    offers: Vec<VastOffer>,
    creates: VecDeque<Create>,
    requests: Vec<(u64, VastCreateRequest)>,
    instances: Vec<(Pod, u64)>,
    searches: usize,
    /// Searches that find nothing before the offers show up, as while a watch waits.
    empty_searches: usize,
    /// Reads that do not see an instance yet, as right after a create.
    hidden_reads: usize,
    deleted: Vec<String>,
    billing: Option<f64>,
    next_id: u64,
}

struct FakeVast(RefCell<VastState>);

fn offer(offer_id: u64, machine_id: u64, rate_usd_hr: f64) -> VastOffer {
    VastOffer {
        offer_id,
        machine_id,
        host_id: 80000 + machine_id,
        gpu: "RTX PRO 6000 Max-Q".into(),
        count: 2,
        rate_usd_hr,
        reliability2: Some(0.99),
        reliability: None,
        disk_space: 500.0,
        inet_down: 2000.0,
        internet_down_cost_per_tb: Some(3.0),
        rentable: Some(true),
        rented: Some(false),
        cuda_max_good: Some(13.2),
        geolocation: Some("Texas, US".into()),
        hosting_type: Some(0),
    }
}

impl Default for FakeVast {
    fn default() -> Self {
        Self(RefCell::new(VastState {
            offers: vec![offer(9100002, 70002, 2.80), offer(9100001, 70001, 2.46)],
            creates: VecDeque::new(),
            requests: Vec::new(),
            instances: Vec::new(),
            searches: 0,
            empty_searches: 0,
            hidden_reads: 0,
            deleted: Vec::new(),
            billing: None,
            next_id: 31000000,
        }))
    }
}

fn unsupported() -> CloudError {
    CloudError::Protocol("not a Vast call".into()).vast()
}

impl FakeVast {
    fn make(&self, request: &VastCreateRequest, offer_id: u64) -> String {
        let mut state = self.0.borrow_mut();
        state.next_id += 1;
        let rate = state
            .offers
            .iter()
            .find(|o| o.offer_id == offer_id)
            .unwrap()
            .rate_usd_hr;
        let pod = Pod {
            id: state.next_id.to_string(),
            name: request.label.clone(),
            status: PodStatus::Provisioning,
            cost: Some(rate),
            ssh: None,
            runtime: None,
            data_center_id: None,
        };
        let id = pod.id.clone();
        state.instances.push((pod, offer_id));
        id
    }

    fn billing_ids(&self) -> Vec<String> {
        self.0
            .borrow()
            .instances
            .iter()
            .map(|(pod, _)| pod.id.clone())
            .collect()
    }
}

impl Cloud for FakeVast {
    fn gpu_catalog(&self, _: CloudTier, _: u32, _: &str) -> Result<Vec<CatalogGpu>, CloudError> {
        Err(unsupported())
    }

    fn create_pod(&self, _: &CreatePodRequest) -> Result<Pod, CloudError> {
        Err(unsupported())
    }

    fn list_pods(&self) -> Result<Vec<Pod>, CloudError> {
        Ok(self.0.borrow().instances.iter().map(|(pod, _)| pod.clone()).collect())
    }

    /// A new instance loads on its first read and runs with SSH from the second on.
    fn get_pod(&self, id: &str) -> Result<Option<Pod>, CloudError> {
        let mut state = self.0.borrow_mut();
        if state.hidden_reads > 0 {
            state.hidden_reads -= 1;
            return Ok(None);
        }
        let Some((pod, _)) = state.instances.iter_mut().find(|(pod, _)| pod.id == id) else {
            return Ok(None);
        };
        pod.status = match pod.status {
            PodStatus::Provisioning => PodStatus::Starting,
            _ => PodStatus::Running,
        };
        if pod.status == PodStatus::Running {
            pod.ssh = Some(PodSsh {
                direct: Some(SshEndpoint {
                    host: "203.0.113.20".into(),
                    port: 41022,
                }),
            });
            pod.runtime = Some(PodRuntime {
                uptime: Some(60.0),
                ports: serde_json::Value::Null,
            });
        }
        Ok(Some(pod.clone()))
    }

    fn delete_pod(&self, id: &str) -> Result<(), CloudError> {
        let mut state = self.0.borrow_mut();
        state.deleted.push(id.to_string());
        state.instances.retain(|(pod, _)| pod.id != id);
        Ok(())
    }

    fn pod_billing_usd(&self, _: &str) -> Result<Option<f64>, CloudError> {
        Ok(self.0.borrow().billing)
    }

    fn account(&self) -> Result<Option<String>, CloudError> {
        Ok(Some("vast:424242".into()))
    }

    fn clock_skew(&self) -> Option<SignedDuration> {
        Some(SignedDuration::ZERO)
    }
}

impl VastApi for FakeVast {
    fn offers(&self, query: &OfferQuery) -> Result<Vec<VastOffer>, CloudError> {
        let mut state = self.0.borrow_mut();
        state.searches += 1;
        if state.empty_searches > 0 {
            state.empty_searches -= 1;
            return Ok(Vec::new());
        }
        let matching = state
            .offers
            .iter()
            .filter(|o| o.gpu == query.gpu && o.count == query.count);
        Ok(matching.cloned().collect())
    }

    fn create_instance(&self, offer_id: u64, request: &VastCreateRequest) -> Result<String, CloudError> {
        let outcome = {
            let mut state = self.0.borrow_mut();
            state.requests.push((offer_id, request.clone()));
            state.creates.pop_front().unwrap_or(Create::Made)
        };
        match outcome {
            Create::Made => Ok(self.make(request, offer_id)),
            Create::Taken => Err(CloudError::Http {
                status: 200,
                detail: "create refused: no_such_ask".into(),
                retry_after_s: None,
            }
            .vast()),
            Create::MadeButReplyLost => {
                self.make(request, offer_id);
                Err(CloudError::Network("connection reset by peer".into()).vast())
            }
        }
    }
}

/// FakeRemote, plus the P2P matrix each box prints in turn; the last one repeats.
struct Box2 {
    inner: FakeRemote,
    p2p: RefCell<VecDeque<String>>,
    /// What the Vast watchdog check prints, when not FakeRemote's plain WATCHDOG_ARMED.
    watchdog: RefCell<Option<String>>,
}

fn p2p_without(cell: &str) -> String {
    P2P_OK
        .replace("GPU0\tX\tOK", &format!("GPU0\tX\t{cell}"))
        .replace("GPU1\tOK\tX", &format!("GPU1\t{cell}\tX"))
}

const P2P_OK: &str =
    " \tGPU0\tGPU1\t\n GPU0\tX\tOK\t\n GPU1\tOK\tX\t\n\nLegend:\n\n  X    = Self\n  OK   = Status Ok\n";

impl Remote for Box2 {
    fn preflight(&self) -> Result<()> {
        self.inner.preflight()
    }
    fn reachable(&self, target: &Target) -> bool {
        self.inner.reachable(target)
    }
    fn run(&self, target: &Target, script: &str) -> Result<String> {
        if let Some(reply) = self.watchdog.borrow().as_ref()
            && script.contains("watchdog armed for epoch")
        {
            return Ok(reply.clone());
        }
        if script.contains("nvidia-smi topo -p2p r") {
            self.inner.scripts.borrow_mut().push(script.to_string());
            let mut matrices = self.p2p.borrow_mut();
            let matrix = if matrices.len() > 1 {
                matrices.pop_front()
            } else {
                matrices.front().cloned()
            };
            return Ok(matrix.unwrap_or_default());
        }
        self.inner.run(target, script)
    }
    fn run_streaming(&self, target: &Target, script: &str) -> Result<()> {
        self.inner.run_streaming(target, script)
    }
    fn write_file(&self, target: &Target, path: &str, content: &str) -> Result<()> {
        self.inner.write_file(target, path, content)
    }
    fn upload(&self, target: &Target, root: &Path, paths: &[&str]) -> Result<()> {
        self.inner.upload(target, root, paths)
    }
    fn download(&self, target: &Target, paths: &[&str], root: &Path, limit: Duration) -> Result<()> {
        self.inner.download(target, paths, root, limit)
    }
}

struct World {
    repo: TempDir,
    clock: FakeClock,
    runpod: FakeCloud,
    vast: FakeVast,
    remote: Box2,
    interrupted: AtomicBool,
}

impl World {
    fn new() -> Self {
        let repo = repo_with_issue(ISSUE, &[("harness.toml", SPEC)]);
        let clock = FakeClock::at("2026-11-04T17:00:00Z");
        LedgerStore::new(repo.path(), ISSUE)
            .update(|ledger| {
                ledger.budget = Some(Budget {
                    usd: 10.0,
                    approved_at: clock.now(),
                    note: "test".into(),
                });
                Ok(())
            })
            .unwrap();
        Self {
            repo,
            clock,
            runpod: FakeCloud::default(),
            vast: FakeVast::default(),
            remote: Box2 {
                inner: FakeRemote::default(),
                p2p: RefCell::new(VecDeque::from([P2P_OK.to_string()])),
                watchdog: RefCell::new(None),
            },
            interrupted: AtomicBool::new(false),
        }
    }

    fn spec(&self) -> IssueSpec {
        IssueSpec::load(self.repo.path(), ISSUE).unwrap()
    }

    fn with<T>(&self, f: impl FnOnce(&Ctx) -> T) -> T {
        let clouds = Providers {
            runpod: &self.runpod,
            vast: Some(&self.vast),
            vast_account: Some("vast:424242".into()),
        };
        self.with_clouds(&clouds, f)
    }

    fn with_clouds<T>(&self, clouds: &dyn Clouds, f: impl FnOnce(&Ctx) -> T) -> T {
        let events = Events::new(&self.repo.path().join("results"), &self.clock, false);
        let ctx = Ctx {
            repo: self.repo.path(),
            cloud: clouds,
            remote: &self.remote,
            clock: &self.clock,
            events: &events,
            interrupted: &self.interrupted,
            account: Some("org-sm120"),
        };
        f(&ctx)
    }

    fn up(&self) -> Result<pod::LivePod> {
        let spec = self.spec();
        self.with(|ctx| pod::up(ctx, &spec, false, AfterFailure::KeepPod))
    }

    fn ledger(&self) -> Ledger {
        LedgerStore::new(self.repo.path(), ISSUE).load().unwrap()
    }

    fn events(&self) -> String {
        fs::read_to_string(self.repo.path().join("results").join(ISSUE).join("events.jsonl")).unwrap_or_default()
    }

    fn scripts_with(&self, needle: &str) -> usize {
        self.remote
            .inner
            .scripts
            .borrow()
            .iter()
            .filter(|s| s.contains(needle))
            .count()
    }
}

#[test]
fn up_on_vast_rents_the_cheapest_offer_and_runs_every_gate() {
    let w = World::new();
    let live = w.up().unwrap();
    let ledger = w.ledger();
    let entry = &ledger.pods[0];
    assert_eq!((entry.state, entry.id.as_deref()), (PodState::Live, Some("31000001")));
    assert_eq!(
        (entry.gpu.as_str(), entry.count, entry.rate_usd_hr),
        ("RTX PRO 6000 Max-Q", 2, 2.46)
    );
    assert_eq!(
        ledger.vast.pods[&entry.name],
        VastPlacement {
            offer: 9100001,
            machine: 70001,
            host: 150001
        }
    );
    assert_eq!(ledger.vast.account.as_deref(), Some("vast:424242"));
    let (offer_id, request) = w.vast.0.borrow().requests[0].clone();
    assert_eq!(offer_id, 9100001);
    assert_eq!(request.label, entry.name);
    assert_eq!(
        request.env["HARNESS_DEADLINE_EPOCH"],
        entry.deadline.as_second().to_string()
    );
    assert_eq!(request.disk, 120);
    assert_eq!(live.target.port, 41022);
    assert_eq!(w.scripts_with("nvidia-smi topo -p2p r"), 1);
    assert_eq!(w.scripts_with("nvidia-smi topo -m"), 1);
    assert!(w.runpod.0.borrow().requests.is_empty(), "nothing went to RunPod");
    let events = w.events();
    assert!(events.contains("\"provider\":\"vast\",\"machine\":70001"), "{events}");
}

#[test]
fn a_taken_offer_falls_through_to_the_next_machine_and_never_retries_it() {
    let w = World::new();
    w.vast.0.borrow_mut().creates = VecDeque::from([Create::Taken]);
    w.up().unwrap();
    let ledger = w.ledger();
    let states: Vec<PodState> = ledger.pods.iter().map(|p| p.state).collect();
    assert_eq!(states, [PodState::CreateFailed, PodState::Live]);
    let tried: Vec<u64> = w.vast.0.borrow().requests.iter().map(|(id, _)| *id).collect();
    assert_eq!(tried, [9100001, 9100002]);
    assert_eq!(ledger.vast.pods[&ledger.pods[1].name].machine, 70002);
    assert!(w.events().contains("was refused"), "{}", w.events());
}

#[test]
fn a_lost_create_reply_is_adopted_by_its_label() {
    let w = World::new();
    w.vast.0.borrow_mut().creates = VecDeque::from([Create::MadeButReplyLost]);
    w.up().unwrap();
    let ledger = w.ledger();
    assert_eq!(ledger.pods.len(), 1);
    assert_eq!(
        (ledger.pods[0].state, ledger.pods[0].id.as_deref()),
        (PodState::Live, Some("31000001"))
    );
    assert!(w.events().contains("\"adopted\":true"));
}

#[test]
fn a_box_that_cannot_destroy_itself_is_ended_at_once() {
    for reply in [
        "SELF_DESTRUCT_MISSING\n2026-11-04T17:01:00Z VAST SELF-DESTRUCT MISSING: curl is not installed\n",
        "WATCHDOG_ARMED\n2026-11-04T17:01:00Z vast key reads instance 31000001: http 403\n",
    ] {
        let w = World::new();
        *w.remote.watchdog.borrow_mut() = Some(reply.into());
        let err = format!("{:#}", w.up().unwrap_err());
        assert!(
            err.contains("could not destroy itself") || err.contains("cannot destroy itself"),
            "{err}"
        );
        assert_eq!(w.vast.0.borrow().deleted, ["31000001"], "{reply}");
        assert_eq!(w.ledger().pods[0].state, PodState::Terminated);
        assert_eq!(
            w.clock.slept.borrow().len(),
            1,
            "no wait for a box that said it cannot end itself"
        );
    }
}

#[test]
fn a_probe_that_fails_short_of_a_rejected_key_only_warns() {
    let w = World::new();
    *w.remote.watchdog.borrow_mut() =
        Some("WATCHDOG_ARMED\n2026-11-04T17:01:00Z vast key reads instance 31000001: http none\n".into());
    w.up().unwrap();
    assert!(w.events().contains("did not read its instance"), "{}", w.events());
}

#[test]
fn a_new_instance_vast_does_not_show_yet_is_waited_for_not_destroyed() {
    let w = World::new();
    w.vast.0.borrow_mut().hidden_reads = 4;
    w.up().unwrap();
    assert!(w.vast.0.borrow().deleted.is_empty());
    assert_eq!(w.ledger().pods[0].state, PodState::Live);
}

#[test]
fn an_instance_vast_never_shows_is_destroyed_and_booked_only_after_the_adopt_window() {
    let w = World::new();
    w.vast.0.borrow_mut().hidden_reads = usize::MAX;
    let err = format!("{:#}", w.up().unwrap_err());
    assert!(err.contains("pod 31000001 disappeared while starting"), "{err}");
    assert!(err.contains("its destroy cannot be confirmed"), "{err}");
    assert_eq!(w.vast.0.borrow().deleted, ["31000001"]);
    // An instance no read has shown may still appear, so the reservation stays held until the reaper books it.
    assert_eq!(w.ledger().pods[0].state, PodState::Live);
    w.with(reap::reap).unwrap();
    assert_eq!(w.ledger().pods[0].state, PodState::Live);
    w.clock.advance(SignedDuration::from_mins(9));
    w.with(reap::reap).unwrap();
    assert_eq!(w.ledger().pods[0].state, PodState::Terminated);
}

#[test]
fn a_box_without_p2p_on_every_pair_is_destroyed_with_its_matrix_kept() {
    let w = World::new();
    *w.remote.p2p.borrow_mut() = VecDeque::from([p2p_without("CNS")]);
    let err = format!("{:#}", w.up().unwrap_err());
    assert!(err.contains("shows \"CNS\" between GPU0 and GPU1"), "{err}");
    assert_eq!(w.vast.0.borrow().deleted, ["31000001"]);
    assert!(w.vast.billing_ids().is_empty());
    let entry = &w.ledger().pods[0];
    assert_eq!(
        (entry.state, entry.cost_source),
        (PodState::Terminated, Some(CostSource::Estimate))
    );
    let saved = fs::read_to_string(w.repo.path().join("results").join(ISSUE).join("env.txt")).unwrap();
    assert!(saved.contains("GPU0\tX\tCNS") && saved.contains(&entry.name), "{saved}");
}

#[test]
fn down_destroys_on_vast_and_books_its_charges() {
    let w = World::new();
    w.up().unwrap();
    w.vast.0.borrow_mut().billing = Some(0.12);
    w.clock.advance(SignedDuration::from_mins(2));
    w.with(|ctx| pod::down(ctx, ISSUE)).unwrap();
    assert_eq!(w.vast.0.borrow().deleted, ["31000001"]);
    let entry = &w.ledger().pods[0];
    assert_eq!(entry.state, PodState::Terminated);
    assert_eq!(
        (entry.cost_usd, entry.cost_source),
        (Some(0.12), Some(CostSource::Billing))
    );
}

#[test]
fn the_reaper_ends_an_overdue_vast_box_and_nothing_on_runpod() {
    let w = World::new();
    w.up().unwrap();
    let deadline = w.ledger().pods[0].deadline;
    w.clock
        .advance(deadline.duration_since(w.clock.now()) + SignedDuration::from_mins(3));
    w.with(reap::reap).unwrap();
    assert_eq!(w.vast.0.borrow().deleted, ["31000001"]);
    let entry = &w.ledger().pods[0];
    assert_eq!(entry.state, PodState::Terminated);
    assert!(entry.cost_usd.unwrap() <= entry.reserved_usd + 1e-9);
    assert!(w.runpod.billing_pods().is_empty());
}

#[test]
fn without_a_vast_client_its_pods_are_left_alone_and_a_vast_issue_says_why() {
    let w = World::new();
    w.up().unwrap();
    let deadline = w.ledger().pods[0].deadline;
    w.clock
        .advance(deadline.duration_since(w.clock.now()) + SignedDuration::from_mins(3));
    // RunPod's client alone, as on a laptop with no Vast key.
    w.with_clouds(&w.runpod, reap::reap).unwrap();
    assert!(w.vast.0.borrow().deleted.is_empty());
    assert_eq!(
        w.ledger().pods[0].state,
        PodState::Live,
        "never booked as gone from RunPod's list"
    );
    assert!(w.events().contains("went unsettled"), "{}", w.events());
    let spec = w.spec();
    let err = format!(
        "{:#}",
        w.with_clouds(&w.runpod, |ctx| budget::quote(ctx, &spec)).unwrap_err()
    );
    assert!(err.contains("no Vast key is usable"), "{err}");
}

#[test]
fn quote_names_the_machine_its_price_and_the_download_rate() {
    let w = World::new();
    let spec = w.spec();
    w.with(|ctx| budget::quote(ctx, &spec)).unwrap();
    let events = w.events();
    let quote = events.lines().find(|line| line.contains("\"kind\":\"quote\"")).unwrap();
    assert!(quote.contains("\"provider\":\"vast\""), "{quote}");
    assert!(
        quote.contains("\"machine\":70001") && quote.contains("\"rate_usd_hr\":2.46"),
        "{quote}"
    );
    assert!(quote.contains("\"download_usd_tb\":3.0"), "{quote}");
    assert!(quote.contains("on Vast machine 70002 $2.80/hr"), "{quote}");
    // RunPod's fake catalog prices the 2x Server Edition at 2 x $2.49.
    assert!(
        quote.contains("NVIDIA RTX PRO 6000 Blackwell Server Edition SECURE $4.98/hr"),
        "{quote}"
    );
}

const EVERY: Duration = Duration::from_secs(240);
const DAY: Duration = Duration::from_secs(86400);

impl World {
    fn watch(&self, every: Duration, limit: Duration) -> Result<Outcome> {
        let spec = self.spec();
        self.with(|ctx| watch::watch(ctx, spec, every, limit))
    }

    fn machines(&self) -> Vec<(PodState, u64)> {
        let ledger = self.ledger();
        ledger
            .pods
            .iter()
            .map(|p| (p.state, ledger.vast.pods[&p.name].machine))
            .collect()
    }
}

#[test]
fn a_watch_waits_for_an_offer_skips_a_machine_that_failed_and_rents_the_next() {
    let w = World::new();
    w.vast.0.borrow_mut().empty_searches = 1;
    *w.remote.p2p.borrow_mut() = VecDeque::from([p2p_without("NS"), P2P_OK.to_string()]);
    let outcome = w.watch(EVERY, DAY).unwrap();
    assert!(matches!(outcome, Outcome::Rented(_)), "{outcome:?}");
    assert_eq!(w.machines(), [(PodState::Terminated, 70001), (PodState::Live, 70002)]);
    assert_eq!(w.vast.0.borrow().deleted, ["31000001"]);
    let events = w.events();
    assert!(events.contains("\"mode\":\"will rent\""), "{events}");
    assert!(events.contains("avoiding Vast machine [70001]"), "{events}");
}

#[test]
fn a_watch_without_a_budget_reports_the_offer_and_rents_nothing() {
    let w = World::new();
    LedgerStore::new(w.repo.path(), ISSUE)
        .update(|ledger| {
            ledger.budget = None;
            Ok(())
        })
        .unwrap();
    assert!(matches!(w.watch(EVERY, DAY).unwrap(), Outcome::Reported));
    assert!(w.vast.0.borrow().requests.is_empty());
    assert!(w.ledger().pods.is_empty());
    let events = w.events();
    assert!(events.contains("\"mode\":\"report only\""), "{events}");
    assert!(
        events.contains("\"kind\":\"quote\"") && events.contains("\"machine\":70001"),
        "{events}"
    );
}

#[test]
fn a_budget_short_of_the_planned_hours_at_the_cheapest_cap_only_reports() {
    let w = World::new();
    // 3.00 x 1.10 x 1 h = 3.30 is needed.
    LedgerStore::new(w.repo.path(), ISSUE)
        .update(|ledger| {
            ledger.budget.as_mut().unwrap().usd = 3.2;
            Ok(())
        })
        .unwrap();
    assert!(matches!(w.watch(EVERY, DAY).unwrap(), Outcome::Reported));
    assert!(w.vast.0.borrow().requests.is_empty());
}

#[test]
fn a_watch_that_finds_nothing_ends_when_its_time_runs_out() {
    let w = World::new();
    w.vast.0.borrow_mut().empty_searches = usize::MAX;
    // The RunPod fallback is a choice too, and would be rented while in stock.
    w.runpod.0.borrow_mut().secure.clear();
    let outcome = w.watch(EVERY, Duration::from_secs(600)).unwrap();
    assert!(matches!(outcome, Outcome::TimedOut), "{outcome:?}");
    // Searches at 0, 4, 8, and 12 minutes; the last one is past the 10 minutes.
    assert_eq!(w.vast.0.borrow().searches, 4);
    assert!(w.ledger().pods.is_empty());
}

#[test]
fn a_watch_stops_renting_after_three_boxes_fail_their_checks() {
    let w = World::new();
    w.vast.0.borrow_mut().offers = (0..5)
        .map(|i| offer(9100010 + i, 70010 + i, 2.0 + 0.1 * i as f64))
        .collect();
    *w.remote.p2p.borrow_mut() = VecDeque::from([p2p_without("CNS")]);
    let err = format!("{:#}", w.watch(EVERY, DAY).unwrap_err());
    assert!(err.contains("3 boxes failed a check in this watch"), "{err}");
    assert_eq!(
        w.machines(),
        [
            (PodState::Terminated, 70010),
            (PodState::Terminated, 70011),
            (PodState::Terminated, 70012)
        ]
    );
    assert!(w.vast.billing_ids().is_empty());
}

#[test]
fn a_watch_stops_when_every_offer_is_refused_three_rounds_in_a_row() {
    let w = World::new();
    w.runpod.0.borrow_mut().secure.clear();
    w.vast.0.borrow_mut().creates = (0..20).map(|_| Create::Taken).collect();
    let err = format!("{:#}", w.watch(EVERY, DAY).unwrap_err());
    assert!(err.contains("refused at the create 3 rounds in a row"), "{err}");
    assert!(err.contains("no_such_ask"), "{err}");
    // Two offers a round, each tried once.
    assert_eq!(w.vast.0.borrow().requests.len(), 6);
    assert!(w.ledger().pods.iter().all(|p| p.state == PodState::CreateFailed));
}

#[test]
fn ctrl_c_ends_a_watch_between_searches() {
    let w = World::new();
    w.vast.0.borrow_mut().empty_searches = usize::MAX;
    w.interrupted.store(true, std::sync::atomic::Ordering::SeqCst);
    let err = w.watch(EVERY, DAY).unwrap_err();
    assert!(err.downcast_ref::<pod::Interrupted>().is_some(), "{err:#}");
    assert!(w.ledger().pods.is_empty());
}
