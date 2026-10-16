use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use serde_json::json;
use sm120_harness::clock::SystemClock;
use sm120_harness::cloud::{API_BASE, Cloud, RunpodRest};
use sm120_harness::events::{Events, Kind};
use sm120_harness::ledger::{Budget, check_accounts};
use sm120_harness::pod::{self, Ctx, cents};
use sm120_harness::session::{Remote, SshRemote, TRANSFER_LIMIT};
use sm120_harness::spec::{IssueSpec, is_slug};
use sm120_harness::{exec, reap};

/// Rents a RunPod pod for an issue, runs server variants on it, pulls the results, and terminates it, within the
/// GPU budget recorded for the issue.
#[derive(Parser)]
#[command(name = "harness")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Record or show an issue's approved GPU budget
    Budget {
        #[command(subcommand)]
        action: BudgetAction,
    },
    /// Price the first in-stock GPU choice and the deadline `up` would set
    Quote { issue: String },
    /// Reserve budget, create the pod, wait for SSH, copy the kit, run setup.sh
    Up {
        issue: String,
        /// Skip issues/<issue>/setup.sh (env_report.sh still runs)
        #[arg(long)]
        no_setup: bool,
    },
    /// Run variants on the live pod, pulling results after each one
    Exec {
        issue: String,
        #[arg(required = true)]
        variants: Vec<String>,
    },
    /// Pull results/<issue> from the live pod
    Pull { issue: String },
    /// Pull results, terminate the pod, and book its cost
    Down { issue: String },
    /// up, exec (every variant when none are named), then down, which runs on every way out
    Run { issue: String, variants: Vec<String> },
    /// Settle every ledger against RunPod; every other command does this first
    Reap,
    /// Run a command on the live pod, in the kit directory with the container environment loaded
    Ssh {
        issue: String,
        #[arg(last = true, required = true)]
        command: Vec<String>,
    },
}

#[derive(Subcommand)]
enum BudgetAction {
    /// Record Aditya's approval for this issue's GPU spend
    Set {
        issue: String,
        usd: f64,
        #[arg(long)]
        note: String,
    },
    /// Print spent, reserved, and remaining, after reconciling with RunPod billing
    Show { issue: String },
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let _ = writeln!(io::stderr().lock(), "harness: {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// The repo root is the nearest directory up from here that holds both issues/ and harness/pod/.
fn find_repo() -> Result<PathBuf> {
    let here = std::env::current_dir()?;
    here.ancestors()
        .find(|dir| dir.join("issues").is_dir() && dir.join("harness/pod/entrypoint.sh").is_file())
        .map(Path::to_path_buf)
        .context("run harness from inside the sm120-kit repo")
}

fn load(repo: &Path, issue: &str) -> Result<IssueSpec> {
    ensure!(
        is_slug(issue),
        "issue {issue:?} must be lowercase letters, digits, and dashes"
    );
    IssueSpec::load(repo, issue)
}

fn run(cli: Cli) -> Result<()> {
    let repo = find_repo()?;
    let api_key = std::env::var("RUNPOD_API_KEY")
        .ok()
        .filter(|key| !key.is_empty())
        .context("RUNPOD_API_KEY is not set")?;
    let interrupted = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&interrupted);
    // Only a flag: every loop stops at a safe point, and `run` still reaches `down`.
    ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst)).context("installing the signal handler")?;
    let clock = SystemClock;
    let cloud = RunpodRest::new(API_BASE, &api_key, &clock);
    let events = Events::new(&repo.join("results"), &clock, true);
    let account = cloud.account().context("reading the key's RunPod organization")?;
    check_accounts(&repo, account.as_deref())?;
    let ctx = Ctx {
        repo: &repo,
        cloud: &cloud,
        remote: &SshRemote,
        clock: &clock,
        events: &events,
        interrupted: &interrupted,
        account: account.as_deref(),
    };
    let reaped = reap::reap(&ctx);
    if let Command::Reap = cli.command {
        return reaped;
    }
    if let Err(e) = reaped {
        events.emit(None, Kind::Warn, json!({"detail": format!("reaper failed: {e:#}")}));
    }
    match cli.command {
        Command::Budget {
            action: BudgetAction::Set { issue, usd, note },
        } => budget_set(&ctx, &issue, usd, &note),
        Command::Budget {
            action: BudgetAction::Show { issue },
        } => budget_show(&ctx, &issue),
        Command::Quote { issue } => pod::quote(&ctx, &load(&repo, &issue)?),
        Command::Up { issue, no_setup } => pod::up(&ctx, &load(&repo, &issue)?, !no_setup).map(drop),
        Command::Exec { issue, variants } => {
            let spec = load(&repo, &issue)?;
            exec::exec(&ctx, &spec, &pod::live_pod(&ctx, &issue)?, &variants)
        }
        Command::Pull { issue } => pod::pull(&ctx, &issue, &pod::live_pod(&ctx, &issue)?.target, TRANSFER_LIMIT),
        Command::Down { issue } => pod::down(&ctx, &issue),
        Command::Run { issue, variants } => exec::run(&ctx, &load(&repo, &issue)?, &variants),
        Command::Reap => Ok(()),
        Command::Ssh { issue, command } => {
            SshRemote.run_streaming(&pod::live_pod(&ctx, &issue)?.target, &command.join(" "))
        }
    }
}

fn budget_set(ctx: &Ctx, issue: &str, usd: f64, note: &str) -> Result<()> {
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
    ctx.ledger(issue).update(|ledger| {
        ledger.budget = Some(Budget {
            usd,
            approved_at,
            note: note.to_string(),
        });
        Ok(())
    })?;
    budget_show(ctx, issue)
}

fn budget_show(ctx: &Ctx, issue: &str) -> Result<()> {
    pod::reconcile_billing(ctx, issue)?;
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

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    #[test]
    fn cli_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn ssh_takes_everything_after_the_double_dash() {
        let cli = Cli::try_parse_from(["harness", "ssh", "vllm-47990", "--", "nvidia-smi", "-L"]).unwrap();
        match cli.command {
            Command::Ssh { issue, command } => assert_eq!(
                (issue.as_str(), command.join(" ")),
                ("vllm-47990", "nvidia-smi -L".into())
            ),
            _ => panic!("parsed as another command"),
        }
    }

    #[test]
    fn exec_needs_at_least_one_variant_and_run_does_not() {
        assert!(Cli::try_parse_from(["harness", "exec", "vllm-47990"]).is_err());
        assert!(Cli::try_parse_from(["harness", "run", "vllm-47990"]).is_ok());
    }
}
