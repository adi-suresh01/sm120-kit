use anyhow::{Result, bail, ensure};
use serde_json::json;

use crate::cloud::{Offer, find_offers};
use crate::events::Kind;
use crate::ledger::{Budget, plan_reservation};
use crate::pod::{Ctx, cents, reconcile_billing};
use crate::spec::{IssueSpec, is_slug};

/// Records Aditya's approval. A budget cut below what is already spent and reserved does not shorten a live
/// pod's deadline: the watchdog already holds it, so the warning says how to end the pod sooner.
pub fn set(ctx: &Ctx, issue: &str, usd: f64, note: &str) -> Result<()> {
    ensure!(
        is_slug(issue),
        "issue {issue:?} must be lowercase letters, digits, and dashes"
    );
    ensure!(
        ctx.repo.join("issues").join(issue).is_dir(),
        "issues/{issue} does not exist"
    );
    ensure!(
        usd.is_finite() && usd > 0.0,
        "the budget must be a positive dollar amount"
    );
    let approved_at = ctx.clock.now();
    let ledger = ctx.ledger(issue).update(|ledger| {
        ledger.budget = Some(Budget {
            usd,
            approved_at,
            note: note.to_string(),
        });
        Ok(ledger.clone())
    })?;
    let committed = ledger.spent_usd() + ledger.reserved_usd();
    if let Some(active) = ledger.active()
        && usd < committed
    {
        ctx.warn(
            issue,
            format!(
                "the new budget ${usd:.2} is below the ${committed:.2} already spent and reserved; pod {} keeps its \
                 deadline {} and can spend up to its reservation. End it sooner with `harness down {issue}`",
                active.name, active.deadline
            ),
        );
    }
    show(ctx, issue)
}

pub fn show(ctx: &Ctx, issue: &str) -> Result<()> {
    reconcile_billing(ctx, issue)?;
    let ledger = ctx.ledger(issue).load()?;
    let pods: Vec<_> = ledger
        .pods
        .iter()
        .map(|p| {
            json!({"pod": p.name, "state": p.state, "gpu": p.gpu, "rate_usd_hr": p.rate_usd_hr,
                   "deadline": p.deadline, "cost_usd": p.cost_usd.map(cents), "cost_source": p.cost_source})
        })
        .collect();
    ctx.events.emit(
        Some(issue),
        Kind::Budget,
        json!({
            "budget_usd": ledger.budget.as_ref().map(|b| b.usd), "note": ledger.budget.as_ref().map(|b| &b.note),
            "spent_usd": cents(ledger.spent_usd()), "reserved_usd": cents(ledger.reserved_usd()),
            "remaining_usd": ledger.remaining_usd().ok().map(cents), "pods": pods,
        }),
    );
    Ok(())
}

/// Works before any budget exists, since its figures are what Aditya approves the budget from.
pub fn quote(ctx: &Ctx, spec: &IssueSpec) -> Result<()> {
    let offers = find_offers(ctx.cloud, &spec.pod)?;
    if offers.is_empty() {
        bail!(
            "nothing in the preference list of issues/{}/harness.toml is in stock",
            spec.issue
        );
    }
    let remaining = ctx.ledger(&spec.issue).load()?.remaining_usd().ok();
    emit_quote(ctx, spec, &offers, remaining)
}

/// What a reservation for the first offer would hold. With no budget, the planned hours alone set the deadline,
/// so the worst case is rate x margin x planned hours and what remains is unknown.
pub fn emit_quote(ctx: &Ctx, spec: &IssueSpec, offers: &[Offer], remaining: Option<f64>) -> Result<()> {
    let Some(first) = offers.first() else {
        bail!("no offer to quote for {}", spec.issue);
    };
    let plan = plan_reservation(
        ctx.clock.now(),
        spec.planned_hours,
        remaining.unwrap_or(f64::INFINITY),
        first.rate_usd_hr,
    )?;
    let fallbacks: Vec<String> = offers[1..]
        .iter()
        .map(|o| {
            let mut line = format!("{} {} ${:.2}/hr", o.gpu, o.cloud.as_str(), o.rate_usd_hr);
            if !o.data_centers.is_empty() {
                line.push_str(&format!(" in {}", o.data_centers.join(", ")));
            }
            line
        })
        .collect();
    let mut fields = json!({
        "gpu": first.gpu, "cloud": first.cloud, "count": first.count, "rate_usd_hr": first.rate_usd_hr,
        "deadline": plan.deadline, "worst_case_usd": cents(plan.reserved_usd),
        "remaining_usd": remaining.map(cents), "fallbacks": fallbacks,
    });
    if !first.data_centers.is_empty() {
        fields["data_centers"] = json!(first.data_centers);
    }
    ctx.events.emit(Some(&spec.issue), Kind::Quote, fields);
    if !first.unlisted_avoided.is_empty() {
        ctx.warn(
            &spec.issue,
            format!(
                "pod.avoid_data_centers lists {}, which the catalog names for no data center of {}x {}: either it \
                 has no stock there now, or RunPod uses no such id and the list does not avoid what was meant",
                first.unlisted_avoided.join(", "),
                first.count,
                first.gpu
            ),
        );
    }
    Ok(())
}
