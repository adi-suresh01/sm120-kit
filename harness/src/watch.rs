//! `harness watch`: searches every few minutes for an offer the issue's choices accept, for boxes that come free
//! rarely and go within minutes. It rents only for an issue whose budget is already set (`harness budget set` is the
//! approval) and holds `planned_hours` at the cheapest cap; otherwise it reports the first offer and exits.

use std::collections::HashSet;
use std::time::Duration;

use anyhow::{Result, bail};
use jiff::SignedDuration;
use serde_json::json;

use crate::budget;
use crate::cloud::{CloudError, find_candidates};
use crate::events::Kind;
use crate::ledger::{MARGIN, PodState};
use crate::pod::{self, AfterFailure, Ctx, Interrupted, LivePod, NothingInStock, POLL};
use crate::spec::{IssueSpec, PodSpec, Provider};

/// A box that fails a check costs its minutes before the check; past this many in one watch, something about the
/// choices is wrong, and renting on would only spend the budget on more of the same.
pub const MAX_FAILED_BOXES: usize = 3;
/// A free box is rare enough that someone else taking every offer first, round after round, is not the likely
/// story; a create every offer refuses (a bad image, a body Vast rejects) is, and each round adds ledger entries.
pub const MAX_REFUSED_ROUNDS: usize = 3;
/// Vast's search has a daily quota, which a tighter schedule spends for little: a free box stays free for minutes.
pub const MIN_EVERY: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Rent,
    ReportOnly,
}

impl Mode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Rent => "will rent",
            Self::ReportOnly => "report only",
        }
    }
}

/// The lowest `max_usd_hr` among the Vast choices: the least an offer worth renting can cost.
pub fn cheapest_cap(pod: &PodSpec) -> Option<f64> {
    pod.vast_choices().map(|choice| choice.max_usd_hr).reduce(f64::min)
}

/// Rent only with a budget that covers the planned hours, at the margin a reservation holds, for a box at the
/// cheapest cap. RunPod-only choices have no cap, and their own reservation decides what the budget buys.
pub fn mode(remaining: Option<f64>, pod: &PodSpec, planned_hours: f64) -> Mode {
    let Some(remaining) = remaining.filter(|usd| *usd > 0.0) else {
        return Mode::ReportOnly;
    };
    match cheapest_cap(pod) {
        Some(cap) if remaining < cap * MARGIN * planned_hours => Mode::ReportOnly,
        _ => Mode::Rent,
    }
}

#[derive(Debug)]
pub enum Outcome {
    Rented(Box<LivePod>),
    Reported,
    TimedOut,
}

pub fn watch(ctx: &Ctx, mut spec: IssueSpec, every: Duration, limit: Duration) -> Result<Outcome> {
    let issue = spec.issue.clone();
    let remaining = ctx.ledger(&issue).load()?.remaining_usd().ok();
    let mode = mode(remaining, &spec.pod, spec.planned_hours);
    ctx.events.emit(
        Some(&issue),
        Kind::Watch,
        json!({"mode": mode.as_str(), "every_s": every.as_secs(), "for_s": limit.as_secs(),
               "remaining_usd": remaining.map(pod::cents), "cheapest_cap_usd_hr": cheapest_cap(&spec.pod)}),
    );
    let started = ctx.clock.now();
    let limit = SignedDuration::try_from(limit)?;
    let mut failed_boxes = 0;
    let mut refused_rounds = 0;
    loop {
        ctx.check_interrupted()?;
        let found = match find_candidates(ctx.cloud, &spec.pod) {
            Ok(found) => found,
            Err(e) if e.downcast_ref::<CloudError>().is_some() => {
                ctx.warn(&issue, format!("search failed, watching on: {e:#}"));
                Vec::new()
            }
            Err(e) => return Err(e),
        };
        if !found.is_empty() {
            match mode {
                Mode::ReportOnly => {
                    budget::emit_quote(ctx, &spec, &found, None)?;
                    return Ok(Outcome::Reported);
                }
                Mode::Rent => match rent(ctx, &mut spec)? {
                    Rental::Live(live) => return Ok(Outcome::Rented(live)),
                    Rental::Gone => refused_rounds = 0,
                    Rental::Refused(error) => {
                        refused_rounds += 1;
                        if refused_rounds >= MAX_REFUSED_ROUNDS {
                            return Err(error.context(format!(
                                "every offer was refused at the create {refused_rounds} rounds in a row, which a \
                                 request every offer refuses explains better than offers taken first"
                            )));
                        }
                    }
                    Rental::BoxFailed => {
                        refused_rounds = 0;
                        failed_boxes += 1;
                        if failed_boxes >= MAX_FAILED_BOXES {
                            bail!(
                                "{failed_boxes} boxes failed a check in this watch and were destroyed; stopping \
                                 before more of the budget goes the same way (see results/{issue}/events.jsonl)"
                            );
                        }
                    }
                },
            }
        }
        if ctx.clock.now().duration_since(started) >= limit {
            return Ok(Outcome::TimedOut);
        }
        nap(ctx, every)?;
    }
}

/// Sleeps in steps, so Ctrl-C ends the watch within one.
fn nap(ctx: &Ctx, every: Duration) -> Result<()> {
    let mut left = every;
    while !left.is_zero() {
        ctx.check_interrupted()?;
        let step = left.min(POLL);
        ctx.clock.sleep(step);
        left -= step;
    }
    Ok(())
}

enum Rental {
    Live(Box<LivePod>),
    /// None was left by the time `up` searched, or the search failed.
    Gone,
    /// Every offer was refused at the create, as an offer someone takes first is.
    Refused(anyhow::Error),
    /// A box failed a check and `up` destroyed it; its machine is avoided for the rest of the watch.
    BoxFailed,
}

/// `up` exactly as `harness up` runs it. What its error means for the watch is read from the ledger, which says
/// whether anything still bills: a new entry that is live or reserved ends the watch with the error, as `up` would.
fn rent(ctx: &Ctx, spec: &mut IssueSpec) -> Result<Rental> {
    let issue = spec.issue.clone();
    let before: HashSet<String> = ctx.ledger(&issue).load()?.pods.into_iter().map(|p| p.name).collect();
    let error = match pod::up(ctx, spec, true, AfterFailure::KeepPod) {
        Ok(live) => return Ok(Rental::Live(Box::new(live))),
        Err(e) if e.downcast_ref::<Interrupted>().is_some() => return Err(e),
        Err(e) => e,
    };
    let ledger = ctx.ledger(&issue).load()?;
    let new: Vec<_> = ledger.pods.iter().filter(|p| !before.contains(&p.name)).collect();
    if new
        .iter()
        .any(|p| matches!(p.state, PodState::Live | PodState::Reserved))
    {
        return Err(error);
    }
    let ended: Vec<_> = new.iter().filter(|p| p.state == PodState::Terminated).collect();
    if !ended.is_empty() {
        // RunPod picks the host, so avoiding it is not possible, and the next box could fail the same way.
        if ended.iter().any(|p| p.provider == Provider::Runpod) {
            return Err(error);
        }
        let machines: Vec<u64> = ended
            .iter()
            .filter_map(|p| p.placement)
            .map(|placement| placement.machine)
            .collect();
        spec.pod.avoid_machines.extend(&machines);
        ctx.warn(
            &issue,
            format!(
                "a box failed a check and was destroyed; avoiding Vast machine {machines:?} for this watch: {error:#}"
            ),
        );
        return Ok(Rental::BoxFailed);
    }
    if !new.is_empty() {
        ctx.warn(
            &issue,
            format!("every offer was refused at the create, watching on: {error:#}"),
        );
        return Ok(Rental::Refused(error));
    }
    if error.downcast_ref::<NothingInStock>().is_some() {
        return Ok(Rental::Gone);
    }
    if error.downcast_ref::<CloudError>().is_some() {
        ctx.warn(&issue, format!("up failed before reserving, watching on: {error:#}"));
        return Ok(Rental::Gone);
    }
    Err(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pod(choices: &str) -> PodSpec {
        let text = format!(
            r#"
issue = "x-1"
planned_hours = 2.0
[pod]
image = "i@sha256:{}"
count = 8
min_cuda = "13.0"
container_disk_gb = 40
volume_gb = 80
gpus = [{choices}]
[server]
package = "vllm"
model = "m"
launch = ["vllm"]
port = 8000
fail_pattern = "x"
boot_timeout_s = 60
"#,
            "a".repeat(64)
        );
        IssueSpec::parse(&text, "x-1", |_| true).unwrap().pod
    }

    const TWO_CAPS: &str = r#"{ provider = "vast", id = "RTX PRO 6000 Max-Q", max_usd_hr = 17.0 },
        { provider = "vast", id = "RTX PRO 6000 WS", max_usd_hr = 12.0 }"#;

    #[test]
    fn it_rents_only_with_a_budget_that_holds_the_planned_hours_at_the_cheapest_cap() {
        let pod = pod(TWO_CAPS);
        assert_eq!(cheapest_cap(&pod), Some(12.0));
        // 12 x 1.10 x 2 h = 26.40
        assert_eq!(mode(None, &pod, 2.0), Mode::ReportOnly);
        assert_eq!(mode(Some(26.39), &pod, 2.0), Mode::ReportOnly);
        assert_eq!(mode(Some(26.41), &pod, 2.0), Mode::Rent);
        assert_eq!(mode(Some(0.0), &pod, 2.0), Mode::ReportOnly);
        assert_eq!(mode(Some(-3.0), &pod, 2.0), Mode::ReportOnly);
    }

    #[test]
    fn runpod_only_choices_rent_with_any_budget_left() {
        let pod = pod(r#"{ id = "NVIDIA RTX PRO 6000 Blackwell Server Edition", cloud = "SECURE" }"#);
        assert_eq!(cheapest_cap(&pod), None);
        assert_eq!(mode(Some(1.0), &pod, 2.0), Mode::Rent);
        assert_eq!(mode(None, &pod, 2.0), Mode::ReportOnly);
    }
}
