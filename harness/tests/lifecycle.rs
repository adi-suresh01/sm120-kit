mod support;

use std::fs;
use std::panic::{self, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::atomic::AtomicBool;

use jiff::SignedDuration;
use sm120_harness::cloud::PodStatus;
use sm120_harness::events::Events;
use sm120_harness::exec;
use sm120_harness::ledger::{Budget, CostSource, Ledger, LedgerStore, PodState};
use sm120_harness::pod::{self, Ctx};
use sm120_harness::reap;
use sm120_harness::spec::IssueSpec;
use support::fake_cloud::{Create, FakeCloud};
use support::fake_remote::FakeRemote;
use support::{FakeClock, TempDir, repo_with_issue};

const ISSUE: &str = "vllm-47990";
const SPEC: &str = r#"
issue = "vllm-47990"
planned_hours = 1.0
[pod]
image = "vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
count = 1
min_cuda = "13.0"
container_disk_gb = 40
volume_gb = 80
gpus = [
  { id = "NVIDIA RTX PRO 6000 Blackwell Server Edition", cloud = "SECURE" },
  { id = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", cloud = "SECURE" },
]
[server]
package = "vllm"
model = "nvidia/Qwen3.6-35B-A3B-NVFP4"
launch = ["vllm", "serve", "nvidia/Qwen3.6-35B-A3B-NVFP4"]
port = 8000
fail_pattern = "illegal memory access"
boot_timeout_s = 900
[variants.baseline]
flags = ["--moe-backend", "flashinfer_b12x"]
"#;

struct World {
    repo: TempDir,
    clock: FakeClock,
    cloud: FakeCloud,
    remote: FakeRemote,
    interrupted: Rc<AtomicBool>,
}

impl Default for World {
    fn default() -> Self {
        let repo = repo_with_issue(ISSUE, &[("setup.sh", "echo setup\n")]);
        LedgerStore::new(repo.path(), ISSUE)
            .update(|ledger| {
                ledger.budget = Some(Budget {
                    usd: 20.0,
                    approved_at: "2026-10-16T16:00:00Z".parse().unwrap(),
                    note: "approved in chat".into(),
                });
                Ok(())
            })
            .unwrap();
        Self {
            repo,
            clock: FakeClock::at("2026-10-16T17:00:00Z"),
            cloud: FakeCloud::default(),
            remote: FakeRemote::default(),
            interrupted: Rc::new(AtomicBool::new(false)),
        }
    }
}

impl World {
    fn spec(&self) -> IssueSpec {
        IssueSpec::parse(SPEC, ISSUE, |_| true).unwrap()
    }

    fn ledger(&self) -> Ledger {
        LedgerStore::new(self.repo.path(), ISSUE).load().unwrap()
    }

    fn with<T>(&self, f: impl FnOnce(&Ctx) -> T) -> T {
        let events = Events::new(&self.repo.path().join("results"), &self.clock, false);
        let ctx = Ctx {
            repo: self.repo.path(),
            cloud: &self.cloud,
            remote: &self.remote,
            clock: &self.clock,
            events: &events,
            interrupted: &self.interrupted,
        };
        f(&ctx)
    }

    fn events(&self) -> String {
        fs::read_to_string(self.repo.path().join("results").join(ISSUE).join("events.jsonl")).unwrap_or_default()
    }

    fn states(&self) -> Vec<PodState> {
        self.ledger().pods.iter().map(|pod| pod.state).collect()
    }

    /// The safety invariant: every pod that bills is in a ledger, with the same deadline its watchdog holds.
    fn assert_every_billing_pod_is_tracked(&self) {
        let ledgers: Vec<Ledger> = LedgerStore::all(self.repo.path())
            .unwrap()
            .iter()
            .map(|store| store.load().unwrap())
            .collect();
        for (id, name, env) in self.cloud.billing_pods() {
            let entry = ledgers
                .iter()
                .flat_map(|ledger| &ledger.pods)
                .find(|entry| match entry.state {
                    PodState::Live => entry.id.as_deref() == Some(id.as_str()),
                    PodState::Reserved => entry.name == name,
                    PodState::Terminated | PodState::CreateFailed => false,
                })
                .unwrap_or_else(|| panic!("pod {name} ({id}) bills but no ledger tracks it"));
            assert_eq!(
                env["HARNESS_DEADLINE_EPOCH"],
                entry.deadline.as_second().to_string(),
                "the watchdog deadline of {name} differs from the ledger"
            );
        }
    }

    fn up(&self) -> anyhow::Result<pod::LivePod> {
        let spec = self.spec();
        self.with(|ctx| pod::up(ctx, &spec, true))
    }
}

fn err_text<T>(result: anyhow::Result<T>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn up_reserves_creates_waits_copies_and_sets_up() {
    let w = World::default();
    let live = w.up().unwrap();
    assert_eq!(w.states(), [PodState::Live]);
    let entry = &w.ledger().pods[0];
    assert_eq!(entry.gpu, "NVIDIA RTX PRO 6000 Blackwell Server Edition");
    assert_eq!(entry.deadline, "2026-10-16T18:00:00Z".parse().unwrap());
    assert_eq!(live.target.port, 40122);
    assert_eq!(*w.remote.uploads.borrow(), [vec!["harness/pod", "issues/vllm-47990"]]);
    let setup = w.remote.jobs.borrow()[0].clone();
    assert!(setup.contains("env_report.sh > results/vllm-47990/env.txt 2>&1 && bash issues/vllm-47990/setup.sh"));
    let events = w.events();
    for kind in ["reserve", "create", "ready", "setup", "pull"] {
        assert!(
            events.contains(&format!("\"kind\":\"{kind}\"")),
            "no {kind} event in {events}"
        );
    }
    w.assert_every_billing_pod_is_tracked();
}

#[test]
fn the_create_request_carries_the_deadline_and_never_a_token() {
    let w = World::default();
    w.up().unwrap();
    let (_, _, env) = w.cloud.billing_pods().remove(0);
    let deadline: jiff::Timestamp = "2026-10-16T18:00:00Z".parse().unwrap();
    assert_eq!(env["HARNESS_DEADLINE_EPOCH"], deadline.as_second().to_string());
    assert_eq!(env["HARNESS_APT"], "openssh-server curl procps");
    assert!(!env.contains_key("HF_TOKEN"));
    let w = World::default();
    let spec = IssueSpec::parse(
        &SPEC.replace("volume_gb = 80", "volume_gb = 80\nneeds_hf_token = true"),
        ISSUE,
        |_| true,
    )
    .unwrap();
    w.with(|ctx| pod::up(ctx, &spec, false)).unwrap();
    let (_, _, env) = w.cloud.billing_pods().remove(0);
    assert_eq!(env["HF_TOKEN"], "{{ RUNPOD_SECRET_hf_token }}");
}

#[test]
fn quote_prices_the_first_choice_without_reserving() {
    let w = World::default();
    let spec = w.spec();
    w.with(|ctx| pod::quote(ctx, &spec)).unwrap();
    let events = w.events();
    assert!(
        events.contains("\"gpu\":\"NVIDIA RTX PRO 6000 Blackwell Server Edition\""),
        "{events}"
    );
    assert!(
        events.contains("\"deadline\":\"2026-10-16T18:00:00Z\",\"worst_case_usd\":2.74"),
        "{events}"
    );
    assert!(w.ledger().pods.is_empty());
}

#[test]
fn ssh_agent_trouble_stops_up_before_any_money_moves() {
    let w = World::default();
    w.remote.preflight_ok.set(false);
    assert!(err_text(w.up()).contains("ssh-agent"));
    assert!(w.ledger().pods.is_empty());
    assert_eq!(w.cloud.0.borrow().create_calls, 0);
}

#[test]
fn out_of_stock_reserves_nothing() {
    let w = World::default();
    w.cloud.0.borrow_mut().secure.clear();
    assert!(err_text(w.up()).contains("is in stock"));
    assert!(w.ledger().pods.is_empty());
}

#[test]
fn a_refused_create_falls_through_to_the_next_choice() {
    let w = World::default();
    w.cloud.script(&[Create::Refused(400), Create::Made]);
    w.up().unwrap();
    assert_eq!(w.states(), [PodState::CreateFailed, PodState::Live]);
    assert_eq!(
        w.ledger().pods[1].gpu,
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    );
    w.assert_every_billing_pod_is_tracked();
}

#[test]
fn an_auth_failure_on_create_stops_at_once() {
    for status in [401, 403, 422] {
        let w = World::default();
        w.cloud.script(&[Create::Refused(status)]);
        assert!(err_text(w.up()).contains(&format!("fake {status}")));
        assert_eq!(w.states(), [PodState::CreateFailed]);
        assert_eq!(w.cloud.0.borrow().create_calls, 1);
    }
}

#[test]
fn a_lost_reply_is_adopted_by_name_without_a_second_create() {
    let w = World::default();
    w.cloud.script(&[Create::MadeButReplyLost]);
    w.up().unwrap();
    assert_eq!(w.cloud.0.borrow().create_calls, 1);
    assert_eq!(w.cloud.billing_pods().len(), 1);
    assert_eq!(w.states(), [PodState::Live]);
    assert!(w.events().contains("\"adopted\":true"));
    w.assert_every_billing_pod_is_tracked();
}

#[test]
fn a_pod_stuck_booting_is_terminated() {
    let w = World::default();
    w.cloud.0.borrow_mut().boot = vec![PodStatus::Starting];
    assert!(err_text(w.up()).contains("did not answer on SSH within 30 min"));
    assert_eq!(w.states(), [PodState::Terminated]);
    assert!(w.cloud.billing_pods().is_empty());
}

#[test]
fn a_pod_whose_ssh_never_answers_is_terminated() {
    let w = World::default();
    w.remote.reachable.set(false);
    assert!(err_text(w.up()).contains("did not answer on SSH"));
    assert_eq!(w.states(), [PodState::Terminated]);
    assert!(w.cloud.billing_pods().is_empty());
}

#[test]
fn a_pod_that_exits_while_starting_is_terminated() {
    for status in [PodStatus::Exited, PodStatus::Error] {
        let w = World::default();
        w.cloud.0.borrow_mut().boot = vec![PodStatus::Starting, status];
        assert!(err_text(w.up()).contains("while starting"));
        assert_eq!(w.states(), [PodState::Terminated]);
    }
}

#[test]
fn a_pod_billing_above_the_reserved_rate_is_terminated_at_once() {
    let w = World::default();
    w.cloud.0.borrow_mut().cost = 2.75;
    assert!(err_text(w.up()).contains("above the $2.739/hr"));
    assert_eq!(w.states(), [PodState::Terminated]);
    assert!(w.cloud.billing_pods().is_empty());
    let w = World::default();
    w.cloud.0.borrow_mut().cost = 2.70;
    w.up().unwrap();
}

#[test]
fn ctrl_c_while_booting_terminates_the_pod() {
    let w = World::default();
    w.cloud.0.borrow_mut().interrupt_on_create = Some(Rc::clone(&w.interrupted));
    w.cloud.0.borrow_mut().boot = vec![PodStatus::Starting];
    assert_eq!(err_text(w.up()), "interrupted");
    assert_eq!(w.states(), [PodState::Terminated]);
    assert!(w.cloud.billing_pods().is_empty());
}

#[test]
fn nvlink_fails_the_topology_gate_and_terminates() {
    let w = World::default();
    *w.remote.topo.borrow_mut() = "\tGPU0\tGPU1\nGPU0\t X \tNV18\nGPU1\tNV18\t X \n".into();
    let spec = IssueSpec::parse(&SPEC.replace("count = 1", "count = 2"), ISSUE, |_| true).unwrap();
    let err = w.with(|ctx| pod::up(ctx, &spec, true));
    assert!(err_text(err).contains("shows NVLink"));
    assert_eq!(w.states(), [PodState::Terminated]);
}

#[test]
fn an_api_outage_after_create_keeps_the_pod_in_the_ledger() {
    let w = World::default();
    {
        let mut state = w.cloud.0.borrow_mut();
        state.get_error = Some(401);
        state.delete_error = Some(401);
    }
    assert!(w.up().is_err());
    assert_eq!(w.states(), [PodState::Live]);
    assert_eq!(w.cloud.billing_pods().len(), 1);
    w.assert_every_billing_pod_is_tracked();
    assert!(w.events().contains("terminate failed"));
}

#[test]
fn a_failed_setup_leaves_the_pod_up_for_debugging() {
    let w = World::default();
    w.remote.exits.borrow_mut().push_back(3);
    assert!(err_text(w.up()).contains("setup exited 3"));
    assert_eq!(w.states(), [PodState::Live]);
    w.assert_every_billing_pod_is_tracked();
}

#[test]
fn a_second_up_is_refused_while_a_pod_is_live() {
    let w = World::default();
    w.up().unwrap();
    assert!(err_text(w.up()).contains("already has pod"));
    assert_eq!(w.cloud.0.borrow().create_calls, 1);
}

#[test]
fn down_pulls_terminates_and_books_the_estimate() {
    let w = World::default();
    w.up().unwrap();
    w.clock.advance(SignedDuration::from_mins(30));
    let pulls = w.remote.downloads.get();
    w.with(|ctx| pod::down(ctx, ISSUE)).unwrap();
    assert_eq!(w.remote.downloads.get(), pulls + 1);
    let entry = &w.ledger().pods[0];
    assert_eq!(entry.state, PodState::Terminated);
    assert_eq!(entry.cost_source, Some(CostSource::Estimate));
    assert!(entry.cost_usd.unwrap() > 1.24 && entry.cost_usd.unwrap() < 1.4);
    assert!(w.cloud.billing_pods().is_empty());
}

#[test]
fn down_uses_billing_when_it_is_not_below_the_estimate() {
    let w = World::default();
    w.up().unwrap();
    w.cloud.0.borrow_mut().billing = Some(5.0);
    w.with(|ctx| pod::down(ctx, ISSUE)).unwrap();
    let entry = &w.ledger().pods[0];
    assert_eq!(
        (entry.cost_usd, entry.cost_source),
        (Some(5.0), Some(CostSource::Billing))
    );
}

#[test]
fn reconcile_takes_a_lower_bill_only_once_it_has_settled() {
    let w = World::default();
    w.up().unwrap();
    w.clock.advance(SignedDuration::from_mins(60));
    w.with(|ctx| pod::down(ctx, ISSUE)).unwrap();
    w.cloud.0.borrow_mut().billing = Some(2.0);
    w.with(|ctx| pod::reconcile_billing(ctx, ISSUE)).unwrap();
    assert_eq!(w.ledger().pods[0].cost_source, Some(CostSource::Estimate));
    w.clock.advance(SignedDuration::from_hours(25));
    w.with(|ctx| pod::reconcile_billing(ctx, ISSUE)).unwrap();
    let entry = &w.ledger().pods[0];
    assert_eq!(
        (entry.cost_usd, entry.cost_source),
        (Some(2.0), Some(CostSource::Billing))
    );
}

#[test]
fn a_delete_that_does_not_take_keeps_the_pod_live_in_the_ledger() {
    let w = World::default();
    w.up().unwrap();
    w.cloud.0.borrow_mut().delete_ignored = true;
    assert!(err_text(w.with(|ctx| pod::down(ctx, ISSUE))).contains("may still bill"));
    assert_eq!(w.states(), [PodState::Live]);
    w.assert_every_billing_pod_is_tracked();
}

fn run_reaper(w: &World) {
    w.with(reap::reap).unwrap();
}

#[test]
fn a_5xx_or_lost_request_leaves_the_reservation_for_the_reaper() {
    for outcome in [Create::Refused(502), Create::RequestLost] {
        let w = World::default();
        w.cloud.script(&[outcome]);
        assert!(err_text(w.up()).contains("reservation stays"));
        assert_eq!(w.cloud.0.borrow().create_calls, 1);
        assert_eq!(w.states(), [PodState::Reserved]);
        assert!(err_text(w.with(|ctx| pod::down(ctx, ISSUE))).contains("reserved without an id"));
        run_reaper(&w);
        assert_eq!(w.states(), [PodState::Reserved]);
        w.clock.advance(SignedDuration::from_mins(11));
        run_reaper(&w);
        assert_eq!(w.states(), [PodState::CreateFailed]);
        assert!(w.ledger().remaining_usd().unwrap() > 19.99);
    }
}

#[test]
fn a_pod_listed_late_is_adopted_by_the_reaper_and_downed() {
    let w = World::default();
    w.cloud.script(&[Create::MadeButListedLater]);
    assert!(w.up().is_err());
    w.assert_every_billing_pod_is_tracked();
    w.cloud.reveal_unlisted();
    run_reaper(&w);
    assert_eq!(w.states(), [PodState::Live]);
    w.with(|ctx| pod::down(ctx, ISSUE)).unwrap();
    assert_eq!(w.states(), [PodState::Terminated]);
    assert!(w.cloud.billing_pods().is_empty());
}

#[test]
fn killed_between_reserve_and_create() {
    let w = World::default();
    w.cloud.script(&[Create::KilledBeforeSend]);
    assert!(panic::catch_unwind(AssertUnwindSafe(|| w.up())).is_err());
    assert_eq!(w.states(), [PodState::Reserved]);
    assert!(w.cloud.billing_pods().is_empty());
    w.clock.advance(SignedDuration::from_mins(11));
    run_reaper(&w);
    assert_eq!(w.states(), [PodState::CreateFailed]);
}

#[test]
fn killed_between_create_and_the_ledger_update() {
    let w = World::default();
    w.cloud.script(&[Create::KilledAfterMade]);
    assert!(panic::catch_unwind(AssertUnwindSafe(|| w.up())).is_err());
    assert_eq!(w.states(), [PodState::Reserved]);
    w.assert_every_billing_pod_is_tracked();
    run_reaper(&w);
    assert_eq!(w.states(), [PodState::Live]);
    w.clock.advance(SignedDuration::from_mins(63));
    run_reaper(&w);
    assert_eq!(w.states(), [PodState::Terminated]);
    assert!(w.cloud.billing_pods().is_empty());
}

#[test]
fn the_reaper_deletes_a_pod_past_deadline_plus_grace() {
    let w = World::default();
    w.up().unwrap();
    w.clock.advance(SignedDuration::from_mins(61));
    run_reaper(&w);
    assert_eq!(w.states(), [PodState::Live]);
    w.clock.advance(SignedDuration::from_mins(2));
    run_reaper(&w);
    assert_eq!(w.states(), [PodState::Terminated]);
    assert!(w.cloud.billing_pods().is_empty());
}

#[test]
fn the_reaper_books_a_vanished_pod_up_to_its_deadline() {
    let w = World::default();
    w.up().unwrap();
    let id = w.ledger().pods[0].id.clone().unwrap();
    w.cloud.remove_outside_harness(&id);
    w.clock.advance(SignedDuration::from_hours(5));
    run_reaper(&w);
    let entry = &w.ledger().pods[0];
    assert_eq!(entry.state, PodState::Terminated);
    assert_eq!(entry.terminated_at, Some(entry.deadline));
    assert!((entry.cost_usd.unwrap() - 2.49).abs() < 1e-9);
}

#[test]
fn the_reaper_never_deletes_an_untracked_pod() {
    let w = World::default();
    w.up().unwrap();
    let ledger = LedgerStore::new(w.repo.path(), ISSUE);
    ledger
        .update(|l| {
            l.pods.clear();
            Ok(())
        })
        .unwrap();
    w.clock.advance(SignedDuration::from_hours(5));
    run_reaper(&w);
    assert_eq!(w.cloud.billing_pods().len(), 1);
    let general = fs::read_to_string(w.repo.path().join("results/harness-events.jsonl")).unwrap();
    assert!(general.contains("no ledger tracks"), "{general}");
}

fn run(w: &World, variants: &[&str]) -> anyhow::Result<()> {
    let spec = w.spec();
    let names: Vec<String> = variants.iter().map(|name| name.to_string()).collect();
    w.with(|ctx| exec::run(ctx, &spec, &names))
}

fn assert_down(w: &World) {
    assert_eq!(w.states().last(), Some(&PodState::Terminated));
    assert!(w.cloud.billing_pods().is_empty());
    w.assert_every_billing_pod_is_tracked();
}

#[test]
fn run_goes_up_runs_every_variant_and_comes_down() {
    let w = World::default();
    run(&w, &[]).unwrap();
    assert_down(&w);
    let files = w.remote.files.borrow();
    assert_eq!(files[0].0, "results/vllm-47990/variants/baseline.sh");
    assert!(
        files[0]
            .1
            .contains("SERVER_CMD=('vllm' 'serve' 'nvidia/Qwen3.6-35B-A3B-NVFP4' '--moe-backend'")
    );
    let job = w.remote.jobs.borrow()[1].clone();
    assert!(job.contains(
        "bash harness/pod/serve_variant.sh issues/vllm-47990 results/vllm-47990/variants/baseline.sh results/vllm-47990"
    ));
    let events = w.events();
    assert!(events.contains("\"kind\":\"variant_start\",\"variant\":\"baseline\""));
    assert!(events.contains("\"kind\":\"verdict\",\"variant\":\"baseline\",\"verdict\":\"OK\""));
}

#[test]
fn run_comes_down_when_setup_or_a_variant_fails() {
    for exits in [[2, 0], [0, 2]] {
        let w = World::default();
        w.remote.exits.borrow_mut().extend(exits);
        assert!(err_text(run(&w, &[])).contains("exit"));
        assert_down(&w);
    }
}

#[test]
fn run_comes_down_when_a_variant_reports_no_verdict() {
    let w = World::default();
    w.remote.verdict.borrow_mut().clear();
    assert!(err_text(run(&w, &[])).contains("0 verdict lines"));
    assert_down(&w);
}

#[test]
fn run_comes_down_on_ctrl_c() {
    let w = World::default();
    *w.remote.interrupt_on_job.borrow_mut() = Some(Rc::clone(&w.interrupted));
    assert_eq!(err_text(run(&w, &[])), "interrupted");
    assert_down(&w);
}

#[test]
fn run_comes_down_after_a_panic() {
    let w = World::default();
    w.remote.panic_on_write.set(true);
    assert!(panic::catch_unwind(AssertUnwindSafe(|| run(&w, &[]))).is_err());
    assert_down(&w);
}

#[test]
fn run_comes_down_on_an_unknown_variant() {
    let w = World::default();
    assert!(err_text(run(&w, &["nope"])).contains("no variant \"nope\""));
    assert_down(&w);
}

#[test]
fn run_after_a_lost_create_leaves_only_a_tracked_reservation() {
    let w = World::default();
    w.cloud.script(&[Create::MadeButListedLater]);
    assert!(run(&w, &[]).is_err());
    assert_eq!(w.states(), [PodState::Reserved]);
    w.assert_every_billing_pod_is_tracked();
}
