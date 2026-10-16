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
use sm120_harness::ledger::{Budget, LedgerStore, check_accounts};
use sm120_harness::pod::{self, AfterFailure, Ctx, cents};
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

impl Command {
    fn issue(&self) -> Option<&str> {
        match self {
            Self::Budget {
                action: BudgetAction::Set { issue, .. } | BudgetAction::Show { issue },
            }
            | Self::Quote { issue }
            | Self::Up { issue, .. }
            | Self::Exec { issue, .. }
            | Self::Pull { issue }
            | Self::Down { issue }
            | Self::Run { issue, .. }
            | Self::Ssh { issue, .. } => Some(issue),
            Self::Reap => None,
        }
    }
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

/// Printed from the signal thread at once: teardown can take minutes, and a person who sees nothing presses
/// Ctrl-C again. Only `run` ends the pod here; the other commands stop at the next safe point.
fn interrupt_notice(repo: &Path, issue: Option<&str>, ends_pod: bool) -> String {
    let pod = issue.and_then(|issue| {
        let ledger = LedgerStore::new(repo, issue).load().ok()?;
        ledger.active().map(|entry| entry.name.clone())
    });
    match (ends_pod, pod) {
        (true, Some(pod)) => format!("harness: interrupted; tearing down pod {pod}, do not interrupt again"),
        (true, None) => "harness: interrupted; tearing down, do not interrupt again".into(),
        (false, _) => "harness: interrupted; stopping at the next safe point".into(),
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
    let issue = cli.command.issue().map(str::to_string);
    let interrupted = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&interrupted);
    let (handler_repo, handler_issue) = (repo.clone(), issue.clone());
    let ends_pod = matches!(cli.command, Command::Run { .. });
    // Only a flag: every loop stops at a safe point, and `run` still reaches `down`.
    ctrlc::set_handler(move || {
        flag.store(true, Ordering::SeqCst);
        let notice = interrupt_notice(&handler_repo, handler_issue.as_deref(), ends_pod);
        let _ = writeln!(io::stderr().lock(), "{notice}");
    })
    .context("installing the signal handler")?;
    let clock = SystemClock;
    let cloud = RunpodRest::new(API_BASE, &api_key, &clock);
    let events = Events::new(&repo.join("results"), &clock, true);
    let result = dispatch(&repo, &cloud, &clock, &events, &interrupted, cli.command);
    if let Err(e) = &result {
        pod::report_error(&events, issue.as_deref(), e);
    }
    result
}

fn dispatch(
    repo: &Path,
    cloud: &RunpodRest,
    clock: &SystemClock,
    events: &Events,
    interrupted: &AtomicBool,
    command: Command,
) -> Result<()> {
    let account = cloud.account().context("reading the key's RunPod organization")?;
    check_accounts(repo, account.as_deref())?;
    let ctx = Ctx {
        repo,
        cloud,
        remote: &SshRemote,
        clock,
        events,
        interrupted,
        account: account.as_deref(),
    };
    let reaped = reap::reap(&ctx);
    if let Command::Reap = command {
        return reaped;
    }
    if let Err(e) = reaped {
        events.emit(
            command.issue(),
            Kind::Warn,
            json!({"detail": format!("reaper failed: {e:#}")}),
        );
    }
    match command {
        Command::Budget {
            action: BudgetAction::Set { issue, usd, note },
        } => budget_set(&ctx, &issue, usd, &note),
        Command::Budget {
            action: BudgetAction::Show { issue },
        } => budget_show(&ctx, &issue),
        Command::Quote { issue } => pod::quote(&ctx, &load(repo, &issue)?),
        Command::Up { issue, no_setup } => {
            pod::up(&ctx, &load(repo, &issue)?, !no_setup, AfterFailure::KeepPod).map(drop)
        }
        Command::Exec { issue, variants } => {
            let spec = load(repo, &issue)?;
            exec::exec(&ctx, &spec, &pod::live_pod(&ctx, &issue)?, &variants)
        }
        Command::Pull { issue } => pod::pull(&ctx, &issue, &pod::live_pod(&ctx, &issue)?.target, TRANSFER_LIMIT),
        Command::Down { issue } => pod::down(&ctx, &issue),
        Command::Run { issue, variants } => exec::run(&ctx, &load(repo, &issue)?, &variants),
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
    fn ctrl_c_under_run_names_the_pod_it_tears_down() {
        let repo = std::env::temp_dir().join(format!("sm120-notice-{}", std::process::id()));
        std::fs::create_dir_all(repo.join("issues/vllm-47990")).unwrap();
        let ledger = r#"{"issue": "vllm-47990", "budget": null, "pods": [{"name": "sm120-vllm-47990-7f3a2c",
            "id": "p1", "state": "live", "gpu": "g", "count": 1, "cloud": "SECURE", "rate_usd_hr": 2.0,
            "reserved_usd": 2.2, "reserved_at": "2026-10-16T17:00:00Z", "deadline": "2026-10-16T18:00:00Z",
            "terminated_at": null, "cost_usd": null, "cost_source": null}]}"#;
        std::fs::write(repo.join("issues/vllm-47990/harness-ledger.json"), ledger).unwrap();
        assert_eq!(
            interrupt_notice(&repo, Some("vllm-47990"), true),
            "harness: interrupted; tearing down pod sm120-vllm-47990-7f3a2c, do not interrupt again"
        );
        assert!(interrupt_notice(&repo, Some("vllm-47990"), false).ends_with("stopping at the next safe point"));
        assert!(interrupt_notice(&repo, Some("other"), true).ends_with("tearing down, do not interrupt again"));
        std::fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn every_command_but_reap_names_its_issue() {
        for args in [
            &["harness", "budget", "show", "x-1"][..],
            &["harness", "budget", "set", "x-1", "5", "--note", "n"],
            &["harness", "quote", "x-1"],
            &["harness", "up", "x-1"],
            &["harness", "exec", "x-1", "a"],
            &["harness", "pull", "x-1"],
            &["harness", "down", "x-1"],
            &["harness", "run", "x-1"],
            &["harness", "ssh", "x-1", "--", "ls"],
        ] {
            assert_eq!(
                Cli::try_parse_from(args).unwrap().command.issue(),
                Some("x-1"),
                "{args:?}"
            );
        }
        assert_eq!(Cli::try_parse_from(["harness", "reap"]).unwrap().command.issue(), None);
    }

    #[test]
    fn exec_needs_at_least_one_variant_and_run_does_not() {
        assert!(Cli::try_parse_from(["harness", "exec", "vllm-47990"]).is_err());
        assert!(Cli::try_parse_from(["harness", "run", "vllm-47990"]).is_ok());
    }
}
