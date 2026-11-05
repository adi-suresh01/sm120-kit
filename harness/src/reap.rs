use anyhow::Result;
use jiff::{SignedDuration, Timestamp};
use serde_json::json;

use crate::cloud::{NO_VAST_KEY, Pod, PodStatus};
use crate::events::Kind;
use crate::ledger::{ADOPT_WINDOW, Ledger, LedgerStore, PodEntry, PodState};
use crate::pod::{self, Ctx};
use crate::spec::Provider;

/// The pod's own watchdog fires at the deadline; the reaper steps in only after it had time to.
pub const GRACE: SignedDuration = SignedDuration::from_mins(2);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Terminate {
        issue: String,
        name: String,
        id: String,
        /// What RunPod listed last: EXITED means the watchdog only managed a stop.
        status: PodStatus,
    },
    MarkGone {
        issue: String,
        name: String,
    },
    Adopt {
        issue: String,
        name: String,
        id: String,
    },
    Release {
        issue: String,
        name: String,
    },
    /// Destroy at once, whatever the deadline: the harness gave up on this pod, or Vast listed it after its create
    /// failed, so nothing uses it.
    Destroy {
        issue: String,
        name: String,
        id: String,
    },
    WarnUntracked {
        name: String,
        id: String,
    },
}

fn listed<'a>(pods: &'a [Pod], id: &str) -> Option<&'a Pod> {
    pods.iter()
        .find(|pod| pod.id == id && pod.status != PodStatus::Terminated)
}

fn live_action(
    provider: Provider,
    issue: &str,
    entry: &PodEntry,
    id: &str,
    pods: &[Pod],
    now: Timestamp,
) -> Option<Action> {
    let (issue, name, id) = (issue.to_string(), entry.name.clone(), id.to_string());
    // A deadline too close to the end of time to add the grace to is not one the watchdog will ever reach.
    let overdue = entry.deadline.checked_add(GRACE).is_ok_and(|end| now > end);
    match listed(pods, &id) {
        // Vast can leave a new instance out of its list for a while; another command's `up` may have just made it.
        None if provider == Provider::Vast && entry.may_be_unlisted(now) => None,
        None => Some(Action::MarkGone { issue, name }),
        Some(_) if entry.abandoned => Some(Action::Destroy { issue, name, id }),
        Some(pod) if overdue => Some(Action::Terminate {
            issue,
            name,
            id,
            status: pod.status,
        }),
        Some(_) => None,
    }
}

/// Pure: what to do about every RunPod ledger entry and every listed RunPod pod, given the time.
pub fn decide(ledgers: &[Ledger], pods: &[Pod], now: Timestamp) -> Vec<Action> {
    decide_on(Provider::Runpod, ledgers, pods, now)
}

/// As `decide`, for one provider: its entries against its own pod list. Entries on another provider are left
/// alone, so a list from one provider never releases, books, or ends a pod on the other.
pub fn decide_on(provider: Provider, ledgers: &[Ledger], pods: &[Pod], now: Timestamp) -> Vec<Action> {
    let mut actions = Vec::new();
    for ledger in ledgers {
        let issue = ledger.issue.as_str();
        let ours = ledger.pods.iter().filter(|entry| entry.provider == provider);
        for entry in ours {
            match (entry.state, entry.id.as_deref()) {
                (PodState::Live, Some(id)) => actions.extend(live_action(provider, issue, entry, id, pods, now)),
                (PodState::Reserved, _) => match pods.iter().find(|pod| pod.name == entry.name) {
                    Some(pod) => {
                        actions.push(Action::Adopt {
                            issue: issue.to_string(),
                            name: entry.name.clone(),
                            id: pod.id.clone(),
                        });
                        actions.extend(live_action(provider, issue, entry, &pod.id, pods, now));
                    }
                    None if now.duration_since(entry.reserved_at) > ADOPT_WINDOW => actions.push(Action::Release {
                        issue: issue.to_string(),
                        name: entry.name.clone(),
                    }),
                    None => {}
                },
                (PodState::CreateFailed, _) if provider == Provider::Vast => {
                    let late = pods
                        .iter()
                        .find(|pod| pod.name == entry.name && pod.status != PodStatus::Terminated);
                    if let Some(pod) = late {
                        actions.push(Action::Destroy {
                            issue: issue.to_string(),
                            name: entry.name.clone(),
                            id: pod.id.clone(),
                        });
                    }
                }
                (PodState::Live, None) | (PodState::Terminated | PodState::CreateFailed, _) => {}
            }
        }
    }
    let tracked = |pod: &Pod| {
        ledgers
            .iter()
            .any(|ledger| ledger.tracks_on(provider, &pod.id, &pod.name))
    };
    for pod in pods {
        if pod.name.starts_with("sm120-") && pod.status != PodStatus::Terminated && !tracked(pod) {
            actions.push(Action::WarnUntracked {
                name: pod.name.clone(),
                id: pod.id.clone(),
            });
        }
    }
    actions
}

/// Runs first in every command. A ledger that fails to load is reported and skipped, so one bad file cannot stop
/// the others from being settled. Each provider with a client is listed and settled on its own, so one provider's
/// outage leaves the other's pods settled; the first list that failed is the error.
pub fn reap(ctx: &Ctx) -> Result<()> {
    let mut ledgers = Vec::new();
    for store in LedgerStore::all(ctx.repo)? {
        match store.load() {
            Ok(ledger) => ledgers.push(ledger),
            Err(e) => ctx.events.emit(
                Some(store.issue()),
                Kind::Error,
                json!({"detail": format!("reaper skipped this ledger: {e:#}")}),
            ),
        }
    }
    let mut failed = None;
    for provider in Provider::ALL {
        let Some(cloud) = ctx.cloud.on(provider) else {
            warn_unsettled(ctx, provider, &ledgers);
            continue;
        };
        let pods = match cloud.list_pods() {
            Ok(pods) => pods,
            Err(e) => {
                failed.get_or_insert(e);
                continue;
            }
        };
        for action in decide_on(provider, &ledgers, &pods, ctx.clock.now()) {
            if let Err(e) = apply(ctx, provider, &action) {
                ctx.events.emit(
                    action_issue(ctx, &action),
                    Kind::Error,
                    json!({"detail": format!("reaper action {action:?} failed: {e:#}")}),
                );
            }
        }
    }
    match failed {
        Some(e) => Err(e.into()),
        None => Ok(()),
    }
}

/// A ledger can hold a Vast pod while this command has no Vast key; nothing settles it until one is configured.
fn warn_unsettled(ctx: &Ctx, provider: Provider, ledgers: &[Ledger]) {
    for ledger in ledgers {
        let open = ledger
            .pods
            .iter()
            .filter(|entry| matches!(entry.state, PodState::Live | PodState::Reserved) && entry.provider == provider);
        for entry in open {
            ctx.events.emit(
                Some(&ledger.issue),
                Kind::Warn,
                json!({"pod": entry.name, "detail": format!(
                    "{provider} pod {} ({:?}) went unsettled: {NO_VAST_KEY}", entry.name, entry.state
                )}),
            );
        }
    }
}

/// Where the action's events go: its ledger's issue, or for an untracked pod the issue its name points at, so
/// a monitor tailing one issue's events sees everything about that issue's pods.
fn action_issue<'a>(ctx: &Ctx, action: &'a Action) -> Option<&'a str> {
    match action {
        Action::Terminate { issue, .. }
        | Action::MarkGone { issue, .. }
        | Action::Adopt { issue, .. }
        | Action::Release { issue, .. }
        | Action::Destroy { issue, .. } => Some(issue),
        Action::WarnUntracked { name, .. } => name
            .strip_prefix("sm120-")
            .and_then(|rest| rest.rsplit_once('-'))
            .map(|(issue, _)| issue)
            .filter(|issue| pod::is_pod_of(issue, name) && ctx.repo.join("issues").join(issue).is_dir()),
    }
}

/// Each change re-checks the entry under the ledger lock, since another harness process may have moved it on.
fn apply(ctx: &Ctx, provider: Provider, action: &Action) -> Result<()> {
    match action {
        Action::Terminate {
            issue,
            name,
            id,
            status,
        } => {
            let ledger = ctx.ledger(issue).load()?;
            let entry = ledger.pods.iter().find(|e| &e.name == name);
            if let Some(entry) = entry.filter(|e| e.state == PodState::Live && e.id.as_deref() == Some(id.as_str())) {
                ctx.events.emit(
                    Some(issue),
                    Kind::Reap,
                    json!({"pod": name, "id": id, "action": "terminate", "deadline": entry.deadline,
                           "last_status": status}),
                );
                pod::terminate_overdue(ctx, issue, entry)?;
            }
        }
        Action::MarkGone { issue, name } => {
            pod::mark_gone(ctx, issue, name)?;
            ctx.events
                .emit(Some(issue), Kind::Reap, json!({"pod": name, "action": "mark_gone"}));
        }
        Action::Adopt { issue, name, id } => {
            ctx.ledger(issue).update(|ledger| {
                let entry = ledger.entry_mut(name)?;
                if entry.state == PodState::Reserved {
                    entry.mark_live(id)?;
                }
                Ok(())
            })?;
            ctx.events.emit(
                Some(issue),
                Kind::Reap,
                json!({"pod": name, "id": id, "action": "adopt"}),
            );
        }
        Action::Release { issue, name } => {
            ctx.ledger(issue).update(|ledger| {
                let entry = ledger.entry_mut(name)?;
                if entry.state == PodState::Reserved {
                    entry.mark_create_failed()?;
                }
                Ok(())
            })?;
            ctx.events
                .emit(Some(issue), Kind::Reap, json!({"pod": name, "action": "release"}));
        }
        Action::Destroy { issue, name, id } => {
            let ledger = ctx.ledger(issue).load()?;
            let Some(entry) = ledger.pods.iter().find(|e| &e.name == name) else {
                return Ok(());
            };
            let event = json!({"pod": name, "id": id, "action": "destroy", "state": entry.state});
            match entry.state {
                PodState::Live if entry.id.as_deref() == Some(id.as_str()) => {
                    ctx.events.emit(Some(issue), Kind::Reap, event);
                    pod::terminate(ctx, issue, entry)?;
                }
                PodState::CreateFailed => {
                    ctx.events.emit(Some(issue), Kind::Reap, event);
                    pod::destroy_late_instance(ctx, issue, name, id)?;
                }
                PodState::Live | PodState::Reserved | PodState::Terminated => {}
            }
        }
        Action::WarnUntracked { name, id } => {
            let detail = match provider {
                Provider::Runpod => "sm120 pod that no ledger tracks; not terminated, check it by hand".to_string(),
                other => format!("sm120 pod on {other} that no ledger tracks; not terminated, check it by hand"),
            };
            ctx.events.emit(
                action_issue(ctx, action),
                Kind::Warn,
                json!({"pod": name, "id": id, "detail": detail}),
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud::Offer;
    use crate::ledger::Budget;
    use crate::spec::CloudTier;

    fn at(s: &str) -> Timestamp {
        s.parse().unwrap()
    }

    fn pod(id: &str, name: &str, status: PodStatus) -> Pod {
        Pod {
            id: id.into(),
            name: name.into(),
            status,
            cost: Some(2.49),
            ssh: None,
            runtime: None,
            data_center_id: None,
        }
    }

    /// One reservation made at 17:00 with a 1 h deadline, moved to `state`.
    fn ledger_with(state: PodState) -> Ledger {
        let mut ledger = Ledger::new("vllm-47990");
        ledger.budget = Some(Budget {
            usd: 20.0,
            approved_at: at("2026-10-16T16:00:00Z"),
            note: "chat".into(),
        });
        let offer = Offer {
            gpu: "g".into(),
            cloud: CloudTier::Secure,
            count: 1,
            rate_usd_hr: 2.0,
            data_centers: Vec::new(),
            unlisted_avoided: Vec::new(),
        };
        ledger
            .reserve("sm120-vllm-47990-aaaaaa", &offer, 1.0, at("2026-10-16T17:00:00Z"))
            .unwrap();
        let entry = &mut ledger.pods[0];
        match state {
            PodState::Reserved => {}
            PodState::Live => entry.mark_live("p1").unwrap(),
            PodState::CreateFailed => entry.mark_create_failed().unwrap(),
            PodState::Terminated => {
                entry.mark_live("p1").unwrap();
                entry.mark_terminated(at("2026-10-16T17:30:00Z")).unwrap();
            }
        }
        ledger
    }

    const NAME: &str = "sm120-vllm-47990-aaaaaa";

    fn terminate(status: PodStatus) -> Action {
        Action::Terminate {
            issue: "vllm-47990".into(),
            name: NAME.into(),
            id: "p1".into(),
            status,
        }
    }

    #[test]
    fn live_pod_before_deadline_plus_grace_is_left_alone() {
        let ledgers = [ledger_with(PodState::Live)];
        let pods = [pod("p1", NAME, PodStatus::Running)];
        assert!(decide(&ledgers, &pods, at("2026-10-16T18:01:59Z")).is_empty());
        assert_eq!(
            decide(&ledgers, &pods, at("2026-10-16T18:02:01Z")),
            [terminate(PodStatus::Running)]
        );
    }

    #[test]
    fn a_stopped_pod_past_its_deadline_is_still_deleted() {
        let pods = [pod("p1", NAME, PodStatus::Exited)];
        let actions = decide(&[ledger_with(PodState::Live)], &pods, at("2026-10-16T19:00:00Z"));
        assert_eq!(actions, [terminate(PodStatus::Exited)]);
    }

    #[test]
    fn live_pod_missing_from_the_api_is_marked_gone() {
        let gone = Action::MarkGone {
            issue: "vllm-47990".into(),
            name: NAME.into(),
        };
        let ledgers = [ledger_with(PodState::Live)];
        let listed_terminated = [pod("p1", NAME, PodStatus::Terminated)];
        for pods in [&[][..], &listed_terminated[..]] {
            assert_eq!(decide(&ledgers, pods, at("2026-10-16T17:10:00Z")), vec![gone.clone()]);
        }
    }

    #[test]
    fn reservation_is_adopted_by_exact_name() {
        let pods = [
            pod("p9", "sm120-vllm-47990-aaaaab", PodStatus::Running),
            pod("p1", NAME, PodStatus::Starting),
        ];
        let actions = decide(&[ledger_with(PodState::Reserved)], &pods, at("2026-10-16T17:01:00Z"));
        let adopt = Action::Adopt {
            issue: "vllm-47990".into(),
            name: NAME.into(),
            id: "p1".into(),
        };
        let stray = Action::WarnUntracked {
            name: "sm120-vllm-47990-aaaaab".into(),
            id: "p9".into(),
        };
        assert_eq!(actions, [adopt, stray]);
    }

    #[test]
    fn an_adopted_pod_past_its_deadline_is_terminated_in_the_same_pass() {
        let pods = [pod("p1", NAME, PodStatus::Running)];
        let actions = decide(&[ledger_with(PodState::Reserved)], &pods, at("2026-10-16T18:30:00Z"));
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[1], terminate(PodStatus::Running));
    }

    #[test]
    fn reservation_without_a_pod_is_released_after_ten_minutes() {
        let ledgers = [ledger_with(PodState::Reserved)];
        assert!(decide(&ledgers, &[], at("2026-10-16T17:09:59Z")).is_empty());
        let release = Action::Release {
            issue: "vllm-47990".into(),
            name: NAME.into(),
        };
        assert_eq!(decide(&ledgers, &[], at("2026-10-16T17:10:01Z")), [release]);
    }

    #[test]
    fn untracked_sm120_pods_warn_and_other_pods_are_ignored() {
        let pods = [
            pod("x1", "sm120-sglang-1-ffffff", PodStatus::Running),
            pod("x2", "comfyui", PodStatus::Running),
            pod("x3", "sm120-old-000000", PodStatus::Terminated),
        ];
        let actions = decide(
            &[ledger_with(PodState::CreateFailed)],
            &pods,
            at("2026-10-16T17:30:00Z"),
        );
        let warn = Action::WarnUntracked {
            name: "sm120-sglang-1-ffffff".into(),
            id: "x1".into(),
        };
        assert_eq!(actions, [warn]);
    }

    const VAST_ISSUE: &str = "sglang-33422-tp8";
    const VAST_NAME: &str = "sm120-sglang-33422-tp8-cccccc";

    /// A Vast reservation made at 17:00 with a 1 h deadline, moved to `state` with instance id 31000001.
    fn vast_ledger_with(state: PodState) -> Ledger {
        let mut ledger = Ledger::new(VAST_ISSUE);
        ledger.budget = Some(Budget {
            usd: 50.0,
            approved_at: at("2026-10-16T16:00:00Z"),
            note: "chat".into(),
        });
        let offer = crate::vast::sample_offer(16.0);
        ledger
            .reserve_vast(VAST_NAME, &offer, 1.0, 0, at("2026-10-16T17:00:00Z"))
            .unwrap();
        if state == PodState::Live {
            ledger.pods[0].mark_live("31000001").unwrap();
        }
        ledger
    }

    #[test]
    fn a_vast_pod_is_settled_only_against_vasts_own_list() {
        let ledgers = [ledger_with(PodState::Live), vast_ledger_with(PodState::Live)];
        let runpod = [pod("p1", NAME, PodStatus::Running)];
        // RunPod's list never names the Vast pod, and that is no evidence it is gone.
        assert!(decide(&ledgers, &runpod, at("2026-10-16T17:30:00Z")).is_empty());
        let gone = Action::MarkGone {
            issue: VAST_ISSUE.into(),
            name: VAST_NAME.into(),
        };
        assert_eq!(
            decide_on(Provider::Vast, &ledgers, &[], at("2026-10-16T17:30:00Z")),
            [gone]
        );
        let listed = [pod("31000001", VAST_NAME, PodStatus::Running)];
        assert!(decide_on(Provider::Vast, &ledgers, &listed, at("2026-10-16T17:30:00Z")).is_empty());
        let overdue = Action::Terminate {
            issue: VAST_ISSUE.into(),
            name: VAST_NAME.into(),
            id: "31000001".into(),
            status: PodStatus::Exited,
        };
        let stopped = [pod("31000001", VAST_NAME, PodStatus::Exited)];
        assert_eq!(
            decide_on(Provider::Vast, &ledgers, &stopped, at("2026-10-16T18:02:01Z")),
            [overdue]
        );
    }

    #[test]
    fn a_new_vast_pod_missing_from_the_list_is_not_booked_gone_inside_the_adopt_window() {
        let ledgers = [vast_ledger_with(PodState::Live)];
        assert!(decide_on(Provider::Vast, &ledgers, &[], at("2026-10-16T17:09:59Z")).is_empty());
        let gone = Action::MarkGone {
            issue: VAST_ISSUE.into(),
            name: VAST_NAME.into(),
        };
        assert_eq!(
            decide_on(Provider::Vast, &ledgers, &[], at("2026-10-16T17:10:01Z")),
            [gone]
        );
    }

    #[test]
    fn a_vast_reservation_is_adopted_or_released_by_vast_alone() {
        let ledgers = [vast_ledger_with(PodState::Reserved)];
        assert!(decide(&ledgers, &[], at("2026-10-16T17:30:00Z")).is_empty());
        let release = Action::Release {
            issue: VAST_ISSUE.into(),
            name: VAST_NAME.into(),
        };
        assert_eq!(
            decide_on(Provider::Vast, &ledgers, &[], at("2026-10-16T17:30:00Z")),
            [release]
        );
        // A RunPod pod that happens to carry the name is not the Vast reservation's.
        let same_name = [pod("p7", VAST_NAME, PodStatus::Running)];
        let stray = Action::WarnUntracked {
            name: VAST_NAME.into(),
            id: "p7".into(),
        };
        assert_eq!(decide(&ledgers, &same_name, at("2026-10-16T17:05:00Z")), [stray]);
        let adopt = Action::Adopt {
            issue: VAST_ISSUE.into(),
            name: VAST_NAME.into(),
            id: "31000001".into(),
        };
        let listed = [pod("31000001", VAST_NAME, PodStatus::Starting)];
        assert_eq!(
            decide_on(Provider::Vast, &ledgers, &listed, at("2026-10-16T17:05:00Z")),
            [adopt]
        );
    }

    #[test]
    fn a_vast_pod_the_harness_gave_up_on_is_destroyed_as_soon_as_it_is_listed() {
        let mut ledger = vast_ledger_with(PodState::Live);
        ledger.pods[0].abandoned = true;
        let ledgers = [ledger];
        let destroy = Action::Destroy {
            issue: VAST_ISSUE.into(),
            name: VAST_NAME.into(),
            id: "31000001".into(),
        };
        let listed = [pod("31000001", VAST_NAME, PodStatus::Running)];
        assert_eq!(
            decide_on(Provider::Vast, &ledgers, &listed, at("2026-10-16T17:05:00Z")),
            std::slice::from_ref(&destroy)
        );
        // Unlisted, it is booked gone once the adopt window has passed, as any Vast pod is.
        assert!(decide_on(Provider::Vast, &ledgers, &[], at("2026-10-16T17:05:00Z")).is_empty());
        let mut failed = vast_ledger_with(PodState::Reserved);
        failed.pods[0].mark_create_failed().unwrap();
        let failed = [failed];
        assert!(decide_on(Provider::Vast, &failed, &[], at("2026-10-16T17:30:00Z")).is_empty());
        assert_eq!(
            decide_on(Provider::Vast, &failed, &listed, at("2026-10-16T17:30:00Z")),
            [destroy]
        );
        let stopped = [pod("31000001", VAST_NAME, PodStatus::Terminated)];
        assert!(decide_on(Provider::Vast, &failed, &stopped, at("2026-10-16T17:30:00Z")).is_empty());
    }

    #[test]
    fn a_runpod_pod_listed_under_a_failed_create_only_warns() {
        let pods = [pod("p1", NAME, PodStatus::Running)];
        let actions = decide(
            &[ledger_with(PodState::CreateFailed)],
            &pods,
            at("2026-10-16T17:30:00Z"),
        );
        let warn = Action::WarnUntracked {
            name: NAME.into(),
            id: "p1".into(),
        };
        assert_eq!(actions, [warn]);
    }

    #[test]
    fn an_id_tracked_on_one_provider_does_not_track_a_pod_on_the_other() {
        let mut runpod = ledger_with(PodState::Live);
        runpod.pods[0].id = Some("31000001".into());
        let vast_listed = [
            pod("31000001", "sm120-sglang-1-ffffff", PodStatus::Running),
            pod("31000009", "", PodStatus::Running),
        ];
        let warn = Action::WarnUntracked {
            name: "sm120-sglang-1-ffffff".into(),
            id: "31000001".into(),
        };
        assert_eq!(
            decide_on(Provider::Vast, &[runpod], &vast_listed, at("2026-10-16T17:30:00Z")),
            [warn]
        );
    }

    #[test]
    fn a_terminated_entry_whose_pod_is_still_listed_is_flagged_not_deleted() {
        let pods = [pod("p1", NAME, PodStatus::Running)];
        let actions = decide(&[ledger_with(PodState::Terminated)], &pods, at("2026-10-16T19:00:00Z"));
        let warn = Action::WarnUntracked {
            name: NAME.into(),
            id: "p1".into(),
        };
        assert_eq!(actions, [warn]);
    }
}
