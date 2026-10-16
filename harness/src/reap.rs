use anyhow::Result;
use jiff::{SignedDuration, Timestamp};
use serde_json::json;

use crate::cloud::{Pod, PodStatus};
use crate::events::Kind;
use crate::ledger::{ADOPT_WINDOW, Ledger, LedgerStore, PodEntry, PodState};
use crate::pod::{self, Ctx};

/// The pod's own watchdog fires at the deadline; the reaper steps in only after it had time to.
pub const GRACE: SignedDuration = SignedDuration::from_mins(2);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Terminate { issue: String, name: String, id: String },
    MarkGone { issue: String, name: String },
    Adopt { issue: String, name: String, id: String },
    Release { issue: String, name: String },
    WarnUntracked { name: String, id: String },
}

fn listed<'a>(pods: &'a [Pod], id: &str) -> Option<&'a Pod> {
    pods.iter()
        .find(|pod| pod.id == id && pod.status != PodStatus::Terminated)
}

fn live_action(issue: &str, entry: &PodEntry, id: &str, pods: &[Pod], now: Timestamp) -> Option<Action> {
    let (issue, name, id) = (issue.to_string(), entry.name.clone(), id.to_string());
    match listed(pods, &id) {
        None => Some(Action::MarkGone { issue, name }),
        Some(_) if now > entry.deadline + GRACE => Some(Action::Terminate { issue, name, id }),
        Some(_) => None,
    }
}

/// Pure: what to do about every ledger entry and every listed pod, given the time.
pub fn decide(ledgers: &[Ledger], pods: &[Pod], now: Timestamp) -> Vec<Action> {
    let mut actions = Vec::new();
    for ledger in ledgers {
        let issue = ledger.issue.as_str();
        for entry in &ledger.pods {
            match (entry.state, entry.id.as_deref()) {
                (PodState::Live, Some(id)) => actions.extend(live_action(issue, entry, id, pods, now)),
                (PodState::Reserved, _) => match pods.iter().find(|pod| pod.name == entry.name) {
                    Some(pod) => {
                        actions.push(Action::Adopt {
                            issue: issue.to_string(),
                            name: entry.name.clone(),
                            id: pod.id.clone(),
                        });
                        actions.extend(live_action(issue, entry, &pod.id, pods, now));
                    }
                    None if now.duration_since(entry.reserved_at) > ADOPT_WINDOW => actions.push(Action::Release {
                        issue: issue.to_string(),
                        name: entry.name.clone(),
                    }),
                    None => {}
                },
                (PodState::Live, None) | (PodState::Terminated | PodState::CreateFailed, _) => {}
            }
        }
    }
    let tracked = |pod: &Pod| ledgers.iter().any(|ledger| ledger.tracks(&pod.id, &pod.name));
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
/// the others from being settled.
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
    let pods = ctx.cloud.list_pods()?;
    for action in decide(&ledgers, &pods, ctx.clock.now()) {
        if let Err(e) = apply(ctx, &action) {
            ctx.events.emit(
                action_issue(ctx, &action),
                Kind::Error,
                json!({"detail": format!("reaper action {action:?} failed: {e:#}")}),
            );
        }
    }
    Ok(())
}

/// Where the action's events go: its ledger's issue, or for an untracked pod the issue its name points at, so
/// a monitor tailing one issue's events sees everything about that issue's pods.
fn action_issue<'a>(ctx: &Ctx, action: &'a Action) -> Option<&'a str> {
    match action {
        Action::Terminate { issue, .. }
        | Action::MarkGone { issue, .. }
        | Action::Adopt { issue, .. }
        | Action::Release { issue, .. } => Some(issue),
        Action::WarnUntracked { name, .. } => name
            .strip_prefix("sm120-")
            .and_then(|rest| rest.rsplit_once('-'))
            .map(|(issue, _)| issue)
            .filter(|issue| pod::is_pod_of(issue, name) && ctx.repo.join("issues").join(issue).is_dir()),
    }
}

/// Each change re-checks the entry under the ledger lock, since another harness process may have moved it on.
fn apply(ctx: &Ctx, action: &Action) -> Result<()> {
    match action {
        Action::Terminate { issue, name, id } => {
            let ledger = ctx.ledger(issue).load()?;
            let entry = ledger.pods.iter().find(|e| &e.name == name);
            if let Some(entry) = entry.filter(|e| e.state == PodState::Live && e.id.as_deref() == Some(id.as_str())) {
                ctx.events.emit(
                    Some(issue),
                    Kind::Reap,
                    json!({"pod": name, "id": id, "action": "terminate", "deadline": entry.deadline}),
                );
                pod::terminate(ctx, issue, entry)?;
            }
        }
        Action::MarkGone { issue, name } => {
            pod::mark_gone(ctx, issue, name)?;
            ctx.events.emit(Some(issue), Kind::Reap, json!({"pod": name, "action": "mark_gone"}));
        }
        Action::Adopt { issue, name, id } => {
            ctx.ledger(issue).update(|ledger| {
                let entry = ledger.entry_mut(name)?;
                if entry.state == PodState::Reserved {
                    entry.mark_live(id)?;
                }
                Ok(())
            })?;
            ctx.events.emit(Some(issue), Kind::Reap, json!({"pod": name, "id": id, "action": "adopt"}));
        }
        Action::Release { issue, name } => {
            ctx.ledger(issue).update(|ledger| {
                let entry = ledger.entry_mut(name)?;
                if entry.state == PodState::Reserved {
                    entry.mark_create_failed()?;
                }
                Ok(())
            })?;
            ctx.events.emit(Some(issue), Kind::Reap, json!({"pod": name, "action": "release"}));
        }
        Action::WarnUntracked { name, id } => ctx.events.emit(
            action_issue(ctx, action),
            Kind::Warn,
            json!({"pod": name, "id": id, "detail": "sm120 pod that no ledger tracks; not terminated, check it by hand"}),
        ),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::{Budget, Offer};
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

    fn terminate() -> Action {
        Action::Terminate {
            issue: "vllm-47990".into(),
            name: NAME.into(),
            id: "p1".into(),
        }
    }

    #[test]
    fn live_pod_before_deadline_plus_grace_is_left_alone() {
        let ledgers = [ledger_with(PodState::Live)];
        let pods = [pod("p1", NAME, PodStatus::Running)];
        assert!(decide(&ledgers, &pods, at("2026-10-16T18:01:59Z")).is_empty());
        assert_eq!(decide(&ledgers, &pods, at("2026-10-16T18:02:01Z")), [terminate()]);
    }

    #[test]
    fn a_stopped_pod_past_its_deadline_is_still_deleted() {
        let pods = [pod("p1", NAME, PodStatus::Exited)];
        let actions = decide(&[ledger_with(PodState::Live)], &pods, at("2026-10-16T19:00:00Z"));
        assert_eq!(actions, [terminate()]);
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
        assert_eq!(actions[1], terminate());
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
