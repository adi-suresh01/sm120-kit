//! `up`, `down`, `reap`, and `quote` for an issue on Vast, against a fake Vast beside the fake RunPod: the same
//! ledger and safety rules as RunPod, with every call about a Vast pod going to Vast.

#[path = "support/fake_vast.rs"]
mod fake_vast;
mod support;

use std::collections::VecDeque;
use std::fs;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use anyhow::Result;
use fake_vast::{Create, FakeVast, P2P_OK, VastBox, WatchClock, offer, p2p_without};
use jiff::SignedDuration;
use sm120_harness::clock::Clock;
use sm120_harness::cloud::{Clouds, Providers};
use sm120_harness::events::Events;
use sm120_harness::ledger::{Budget, CostSource, Ledger, LedgerStore, PodState, VastPlacement};
use sm120_harness::pod::{self, AfterFailure, Ctx};
use sm120_harness::reap;
use sm120_harness::spec::IssueSpec;
use sm120_harness::watch::{self, Outcome};
use sm120_harness::{budget, exec};
use support::fake_cloud::FakeCloud;
use support::{TempDir, repo_with_issue};

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

struct World {
    repo: TempDir,
    clock: Rc<WatchClock>,
    runpod: FakeCloud,
    vast: FakeVast,
    remote: VastBox,
    interrupted: Rc<AtomicBool>,
}

impl World {
    fn new() -> Self {
        Self::with_spec(SPEC)
    }

    fn with_spec(spec: &str) -> Self {
        let repo = repo_with_issue(ISSUE, &[("harness.toml", spec)]);
        let clock = Rc::new(WatchClock::at("2026-11-04T17:00:00Z"));
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
            remote: VastBox::new(Rc::clone(&clock)),
            clock,
            runpod: FakeCloud::default(),
            vast: FakeVast::default(),
            interrupted: Rc::new(AtomicBool::new(false)),
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
        let events = Events::new(&self.repo.path().join("results"), &*self.clock, false);
        let ctx = Ctx {
            repo: self.repo.path(),
            cloud: clouds,
            remote: &self.remote,
            clock: &*self.clock,
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
        entry.placement,
        Some(VastPlacement {
            offer: 9100001,
            machine: 70001,
            host: 150001
        })
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
    let b64 = &request.env["HARNESS_ENTRYPOINT_B64"];
    assert!(
        events.contains(&format!("\"entrypoint_b64_bytes\":{}", b64.len())),
        "{events}"
    );
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
    assert_eq!(ledger.pods[1].placement.unwrap().machine, 70002);
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
        "WATCHDOG_ARMED\n2026-11-04T17:01:00Z vast key reads instance {id}: http 403\nBOX_CLOCK {now}\n",
    ] {
        let w = World::new();
        w.remote.arm_check_prints(reply);
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
fn a_box_whose_key_did_not_read_back_this_instance_is_ended() {
    let read = |code: &str| format!("2026-11-04T17:01:00Z vast key reads instance {{id}}: http {code}\n");
    let ok = |id: &str| format!("2026-11-04T17:01:00Z vast probe ok id={id}\n");
    for lines in [
        // The read failed short of a refused key.
        read("none"),
        // A 200 that names no instance, as Vast answers a key that does not own it.
        read("200"),
        // A reply about another instance.
        read("200") + &ok("310000012"),
        // An earlier start read it back; the latest did not.
        read("200") + &ok("{id}") + &read("none"),
        // No read logged at all.
        String::new(),
    ] {
        let w = World::new();
        w.remote
            .arm_check_prints(&format!("WATCHDOG_ARMED\n{lines}BOX_CLOCK {{now}}\n"));
        let err = format!("{:#}", w.up().unwrap_err());
        assert!(err.contains("did not read back instance 31000001"), "{lines}: {err}");
        assert_eq!(w.vast.0.borrow().deleted, ["31000001"], "{lines}");
        assert_eq!(w.ledger().pods[0].state, PodState::Terminated, "{lines}");
    }
}

#[test]
fn a_box_whose_clock_is_off_or_unread_is_ended() {
    let unread = fake_vast::ARMED.replace("BOX_CLOCK {now}\n", "");
    let garbled = fake_vast::ARMED.replace("BOX_CLOCK {now}", "BOX_CLOCK soon");
    for (offset, reply, refusal) in [
        (0, unread.as_str(), Some("printed no clock reading")),
        (0, garbled.as_str(), Some("printed no clock reading")),
        (-121, fake_vast::ARMED, Some("the box's clock is -121 s off")),
        (600, fake_vast::ARMED, Some("the box's clock is 600 s off")),
        (-120, fake_vast::ARMED, None),
        (120, fake_vast::ARMED, None),
    ] {
        let w = World::new();
        w.remote.arm_check_prints(reply);
        w.remote.clock_offset_s.set(offset);
        match refusal {
            Some(refusal) => {
                let err = format!("{:#}", w.up().unwrap_err());
                assert!(err.contains(refusal), "{offset}: {err}");
                assert_eq!(w.vast.0.borrow().deleted, ["31000001"], "{offset}");
            }
            None => {
                w.up().unwrap();
            }
        }
    }
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
            .map(|p| (p.state, p.placement.unwrap().machine))
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
    // 3.00 x 1.10 x 1 h = 3.30, plus 120 GB at $20 per TB = 2.40, is needed.
    LedgerStore::new(w.repo.path(), ISSUE)
        .update(|ledger| {
            ledger.budget.as_mut().unwrap().usd = 5.69;
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

#[test]
fn a_destroy_vast_refuses_leaves_the_box_billing_in_the_ledger() {
    let w = World::new();
    w.up().unwrap();
    w.vast.0.borrow_mut().delete_error = Some(500);
    let err = format!("{:#}", w.with(|ctx| pod::down(ctx, ISSUE)).unwrap_err());
    assert!(
        err.contains("fake destroy failure") && err.contains("still up"),
        "{err}"
    );
    assert_eq!(w.ledger().pods[0].state, PodState::Live);
    assert_eq!(w.vast.billing_ids(), ["31000001"]);
}

#[test]
fn a_failed_vast_list_still_lets_runpods_pods_settle() {
    let w = World::new();
    // No Vast offer, so `up` rents the RunPod fallback.
    w.vast.0.borrow_mut().offers.clear();
    w.up().unwrap();
    let id = w.ledger().pods[0].id.clone().unwrap();
    w.runpod.remove_outside_harness(&id);
    w.vast.0.borrow_mut().list_error = Some(502);
    let err = format!("{:#}", w.with(reap::reap).unwrap_err());
    assert!(err.contains("fake list failure"), "{err}");
    assert_eq!(w.ledger().pods[0].state, PodState::Terminated);
}

#[test]
fn a_create_refused_for_the_key_or_a_rate_limit_stops_instead_of_moving_on() {
    for status in [401, 403, 429] {
        let w = World::new();
        w.vast.0.borrow_mut().creates = VecDeque::from([Create::Refused(status)]);
        let err = format!("{:#}", w.up().unwrap_err());
        assert!(err.contains(&format!("returned {status}")), "{status}: {err}");
        assert_eq!(w.vast.0.borrow().requests.len(), 1, "{status}");
        let states: Vec<PodState> = w.ledger().pods.iter().map(|p| p.state).collect();
        assert_eq!(states, [PodState::CreateFailed], "{status}");
        assert!(w.runpod.0.borrow().requests.is_empty(), "{status}");
    }
}

#[test]
fn a_destroyed_instance_still_listed_for_a_while_is_waited_out_then_booked() {
    let w = World::new();
    w.up().unwrap();
    w.vast.0.borrow_mut().linger_reads = 3;
    let slept_before = w.clock.slept.borrow().len();
    w.with(|ctx| pod::down(ctx, ISSUE)).unwrap();
    assert_eq!(w.ledger().pods[0].state, PodState::Terminated);
    assert_eq!(w.clock.slept.borrow().len() - slept_before, 3);
    assert!(w.vast.billing_ids().is_empty());
}

#[test]
fn a_read_that_fails_never_holds_back_the_destroy() {
    let w = World::new();
    w.vast.0.borrow_mut().get_fails = true;
    let err = format!("{:#}", w.up().unwrap_err());
    assert!(err.contains("its destroy cannot be confirmed"), "{err}");
    assert_eq!(w.vast.0.borrow().deleted, ["31000001"]);
    assert!(w.vast.billing_ids().is_empty());
    // Destroyed, but no read agreed, so the ledger still holds it.
    assert_eq!(w.ledger().pods[0].state, PodState::Live);
}

#[test]
fn a_watch_whose_reads_fail_destroys_the_box_before_it_stops() {
    let w = World::new();
    w.vast.0.borrow_mut().get_fails = true;
    let err = format!("{:#}", w.watch(EVERY, DAY).unwrap_err());
    assert!(err.contains("its destroy cannot be confirmed"), "{err}");
    assert_eq!(w.vast.0.borrow().deleted.first().map(String::as_str), Some("31000001"));
    assert!(w.vast.billing_ids().is_empty());
}

#[test]
fn a_create_vast_lists_minutes_late_is_still_adopted_by_its_label() {
    let w = World::new();
    {
        let mut vast = w.vast.0.borrow_mut();
        vast.creates = VecDeque::from([Create::MadeButListedLater]);
        vast.reveal_after_lists = Some(18);
    }
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
fn a_create_vast_never_lists_is_given_up_then_destroyed_when_it_shows() {
    let w = World::new();
    w.vast.0.borrow_mut().creates = VecDeque::from([Create::MadeButListedLater]);
    let reserved_at = w.clock.now();
    let err = format!("{:#}", w.up().unwrap_err());
    assert!(err.contains("marked abandoned"), "{err}");
    assert!(w.clock.now().duration_since(reserved_at) >= SignedDuration::from_mins(10));
    let entry = w.ledger().pods[0].clone();
    assert_eq!((entry.state, entry.abandoned), (PodState::CreateFailed, true));
    // Vast lists it at last: the next reaper pass destroys it at once and books it.
    w.vast.reveal_unlisted();
    w.with(reap::reap).unwrap();
    assert_eq!(w.vast.0.borrow().deleted, ["31000001"]);
    assert!(w.vast.billing_ids().is_empty());
    let entry = &w.ledger().pods[0];
    assert_eq!(
        (entry.state, entry.id.as_deref(), entry.abandoned),
        (PodState::Terminated, Some("31000001"), false)
    );
    assert_eq!(entry.cost_source, Some(CostSource::Estimate));
}

#[test]
fn a_box_vast_lists_only_after_its_destroy_is_destroyed_on_sight() {
    let w = World::new();
    w.vast.0.borrow_mut().creates = VecDeque::from([Create::MadeListedLater]);
    let err = format!("{:#}", w.up().unwrap_err());
    assert!(err.contains("its destroy cannot be confirmed"), "{err}");
    let entry = w.ledger().pods[0].clone();
    assert_eq!((entry.state, entry.abandoned), (PodState::Live, true));
    assert_eq!(w.vast.billing_ids(), ["31000001"]);
    w.vast.reveal_unlisted();
    // Long before its deadline, which is when the reaper would end a box it had not given up on.
    assert!(w.clock.now() < entry.deadline);
    w.with(reap::reap).unwrap();
    assert!(w.vast.billing_ids().is_empty());
    assert_eq!(w.ledger().pods[0].state, PodState::Terminated);
}

const WATCHDOG_GONE: &str = "WATCHDOG_MISSING\n2026-11-04T17:01:00Z watchdog armed for epoch 1 in pid 812\n";

#[test]
fn a_box_whose_watchdog_died_is_ended_before_the_next_variant() {
    let w = World::new();
    let live = w.up().unwrap();
    w.remote.arm_check_prints(WATCHDOG_GONE);
    let spec = w.spec();
    let names = vec!["baseline".to_string()];
    let err = format!("{:#}", w.with(|ctx| exec::exec(ctx, &spec, &live, &names)).unwrap_err());
    assert!(err.contains("watchdog is gone") && !err.contains("still up"), "{err}");
    assert!(
        w.remote
            .inner
            .jobs
            .borrow()
            .iter()
            .all(|job| !job.contains("serve_variant.sh"))
    );
    assert_eq!(w.vast.0.borrow().deleted, ["31000001"]);
    assert_eq!(w.ledger().pods[0].state, PodState::Terminated);
}

#[test]
fn a_failed_poll_on_vast_checks_the_watchdog_again() {
    let w = World::new();
    let live = w.up().unwrap();
    // Armed before the variant, gone by the time a poll fails.
    *w.remote.watchdog.borrow_mut() = VecDeque::from([fake_vast::ARMED.to_string(), WATCHDOG_GONE.to_string()]);
    w.remote.inner.die_after(0);
    let spec = w.spec();
    let names = vec!["baseline".to_string()];
    let err = format!("{:#}", w.with(|ctx| exec::exec(ctx, &spec, &live, &names)).unwrap_err());
    assert!(err.contains("watchdog is gone"), "{err}");
    assert_eq!(w.vast.0.borrow().deleted, ["31000001"]);
    assert_eq!(w.ledger().pods[0].state, PodState::Terminated);
}

#[test]
fn a_watchdog_check_that_cannot_run_ends_nothing() {
    let w = World::new();
    let live = w.up().unwrap();
    w.remote.arm_check_prints("");
    let spec = w.spec();
    let names = vec!["baseline".to_string()];
    w.with(|ctx| exec::exec(ctx, &spec, &live, &names)).unwrap();
    assert!(w.vast.0.borrow().deleted.is_empty());
    assert!(w.events().contains("went unchecked"), "{}", w.events());
}

#[test]
fn a_box_billing_above_max_usd_hr_is_ended_even_inside_the_margin() {
    for (cost, ended) in [(3.05, true), (3.0, false)] {
        let w = World::new();
        {
            let mut vast = w.vast.0.borrow_mut();
            // $2.80/hr reserves $3.08/hr with the margin, but the choice allows $3.00/hr at most.
            vast.offers = vec![offer(9100002, 70002, 2.80)];
            vast.instance_cost = Some(cost);
        }
        let result = w.up();
        if ended {
            let err = format!("{:#}", result.unwrap_err());
            assert!(err.contains("above the $3.00/hr max_usd_hr"), "{err}");
            assert_eq!(w.vast.0.borrow().deleted, ["31000001"]);
        } else {
            result.unwrap();
        }
    }
}

#[test]
fn a_vast_reservation_holds_the_downloads_on_top_of_the_hours() {
    let w = World::new();
    w.up().unwrap();
    let entry = &w.ledger().pods[0];
    // $2.46 x 1.10 x 1 h, plus 120 GB (the whole disk, with no expected_download_gb) at $3 per TB.
    assert!(
        (entry.reserved_usd - (2.706 + 0.36)).abs() < 1e-9,
        "{}",
        entry.reserved_usd
    );
    let events = w.events();
    let reserve = events
        .lines()
        .find(|line| line.contains("\"kind\":\"reserve\""))
        .unwrap();
    assert!(
        reserve.contains("\"expected_download_gb\":120,\"download_reserved_usd\":0.36"),
        "{reserve}"
    );
}

#[test]
fn quote_holds_the_downloads_that_expected_download_gb_names() {
    let w = World::with_spec(&SPEC.replace("volume_gb = 80", "volume_gb = 80\nexpected_download_gb = 40"));
    let spec = w.spec();
    w.with(|ctx| budget::quote(ctx, &spec)).unwrap();
    let events = w.events();
    let quote = events.lines().find(|line| line.contains("\"kind\":\"quote\"")).unwrap();
    // $2.71 for the hour and 40 GB at $3 per TB.
    assert!(quote.contains("\"worst_case_usd\":2.83"), "{quote}");
    assert!(
        quote.contains("\"expected_download_gb\":40,\"download_reserved_usd\":0.12"),
        "{quote}"
    );
}
