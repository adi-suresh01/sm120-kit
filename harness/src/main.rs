use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use jiff::SignedDuration;
use serde_json::json;
use sm120_harness::clock::SystemClock;
use sm120_harness::cloud::{API_BASE, Providers, RunpodRest};
use sm120_harness::events::{Events, Kind};
use sm120_harness::ledger::LedgerStore;
use sm120_harness::pod::{self, AfterFailure, Ctx};
use sm120_harness::session::{SshRemote, TRANSFER_LIMIT};
use sm120_harness::spec::{IssueSpec, is_slug};
use sm120_harness::vast::{self, VastApi, VastRest};
use sm120_harness::watch::{self, Outcome};
use sm120_harness::{budget, exec, reap};

/// Rents a RunPod pod or a Vast box for an issue, runs server variants on it, pulls the results, and terminates it,
/// within the GPU budget recorded for the issue.
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
    /// Settle every ledger against RunPod and Vast; every other command does this first
    Reap,
    /// Search for an offer every few minutes; rent it as `up` would when the budget is set, else report it
    Watch {
        issue: String,
        /// Time between searches, such as 4m or 90s
        #[arg(long, default_value = "4m", value_parser = duration)]
        every: Duration,
        /// How long to keep searching; exit status 3 when it runs out
        #[arg(long = "for", default_value = "24h", value_parser = duration)]
        limit: Duration,
    },
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
    /// Print spent, reserved, and remaining, after reconciling with RunPod and Vast billing
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
            | Self::Watch { issue, .. }
            | Self::Ssh { issue, .. } => Some(issue),
            Self::Reap => None,
        }
    }
}

/// `4m`, `90s`, `24h`, or `1h 30m`; zero would search without pause.
fn duration(text: &str) -> Result<Duration, String> {
    let parsed: SignedDuration = text
        .parse()
        .map_err(|e| format!("{text:?} is not a duration such as 4m: {e}"))?;
    Duration::try_from(parsed)
        .ok()
        .filter(|d| !d.is_zero())
        .ok_or_else(|| format!("{text:?} must be longer than zero"))
}

/// What `watch` exits with when `--for` runs out without an offer.
const WATCH_TIMED_OUT: u8 = 3;

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => code,
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

fn run(cli: Cli) -> Result<ExitCode> {
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
    let events = Events::new(&repo.join("results"), &clock, true);
    let runpod = RunpodRest::new(API_BASE, &api_key, &clock);
    let vast_key = match vast::configured_key() {
        Ok(key) => key.filter(|_| vast_in_use(&repo, issue.as_deref())),
        Err(e) => {
            let detail = format!("Vast is set aside for this command, and its pods are left alone: {e:#}");
            events.emit(issue.as_deref(), Kind::Warn, json!({ "detail": detail }));
            None
        }
    };
    let vast = vast_key.map(|key| VastRest::new(vast::API_BASE, &key, &clock));
    let result = dispatch(
        &repo,
        &runpod,
        vast.as_ref(),
        &clock,
        &events,
        &interrupted,
        cli.command,
    );
    if let Err(e) = &result {
        pod::report_error(&events, issue.as_deref(), e);
    }
    result
}

/// Whether this command may touch Vast: some ledger holds a Vast pod, or the command's issue has a Vast choice. Any
/// other command never calls Vast, so a hung Vast API cannot hold up a RunPod `down` while its pod bills. A ledger
/// that does not load could hold a Vast pod, so it counts as one.
fn vast_in_use(repo: &Path, issue: Option<&str>) -> bool {
    let ledgers = LedgerStore::all(repo).map(|stores| {
        stores
            .iter()
            .any(|store| store.load().map_or(true, |ledger| !ledger.vast.pods.is_empty()))
    });
    let spec = issue
        .and_then(|issue| IssueSpec::load(repo, issue).ok())
        .is_some_and(|spec| spec.pod.vast_choices().next().is_some());
    ledgers.unwrap_or(true) || spec
}

/// The Vast client, once its key's account checks out against every ledger. A command that only ends pods keeps it
/// either way, as with RunPod; any other command sets it aside on a failed check, so a Vast outage or a stray key
/// never stops a RunPod-only command, and nothing on Vast is touched.
fn usable_vast<'v>(
    vast: Option<&'v VastRest<'v>>,
    repo: &Path,
    events: &Events,
    issue: Option<&str>,
    ends_pods: bool,
) -> Option<(&'v VastRest<'v>, Option<String>)> {
    let vast = vast?;
    match pod::vast_key_account(vast, repo, events, issue, ends_pods) {
        Ok(account) => Some((vast, account)),
        Err(e) => {
            let detail = format!("Vast is set aside for this command, and its pods are left alone: {e:#}");
            events.emit(issue, Kind::Warn, json!({ "detail": detail }));
            None
        }
    }
}

fn dispatch(
    repo: &Path,
    runpod: &RunpodRest,
    vast: Option<&VastRest>,
    clock: &SystemClock,
    events: &Events,
    interrupted: &AtomicBool,
    command: Command,
) -> Result<ExitCode> {
    let ends_pods = matches!(command, Command::Down { .. } | Command::Reap);
    let account = pod::key_account(runpod, repo, events, command.issue(), ends_pods)?;
    let (vast, vast_account) = usable_vast(vast, repo, events, command.issue(), ends_pods).unzip();
    let clouds = Providers {
        runpod,
        vast: vast.map(|vast| vast as &dyn VastApi),
        vast_account: vast_account.flatten(),
    };
    let remote = SshRemote::from_env();
    let ctx = Ctx {
        repo,
        cloud: &clouds,
        remote: &remote,
        clock,
        events,
        interrupted,
        account: account.as_deref(),
    };
    let reaped = reap::reap(&ctx);
    if let Command::Reap = command {
        return reaped.map(|()| ExitCode::SUCCESS);
    }
    if let Err(e) = reaped {
        events.emit(
            command.issue(),
            Kind::Warn,
            json!({"detail": format!("reaper failed: {e:#}")}),
        );
    }
    if let Command::Watch { issue, every, limit } = command {
        return match watch::watch(&ctx, load(repo, &issue)?, every, limit)? {
            Outcome::Rented(_) | Outcome::Reported => Ok(ExitCode::SUCCESS),
            Outcome::TimedOut => Ok(ExitCode::from(WATCH_TIMED_OUT)),
        };
    }
    let done = match command {
        Command::Budget {
            action: BudgetAction::Set { issue, usd, note },
        } => budget::set(&ctx, &issue, usd, &note),
        Command::Budget {
            action: BudgetAction::Show { issue },
        } => budget::show(&ctx, &issue),
        Command::Quote { issue } => budget::quote(&ctx, &load(repo, &issue)?),
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
        Command::Reap | Command::Watch { .. } => Ok(()),
        Command::Ssh { issue, command } => ctx
            .remote
            .run_streaming(&pod::live_pod(&ctx, &issue)?.target, &command.join(" ")),
    };
    done.map(|()| ExitCode::SUCCESS)
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
    fn vast_is_touched_only_where_a_ledger_or_the_issue_uses_it() {
        let repo = std::env::temp_dir().join(format!("sm120-vast-in-use-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&repo);
        let issue = |name: &str| repo.join("issues").join(name);
        std::fs::create_dir_all(issue("rp-1")).unwrap();
        let runpod_ledger = r#"{"issue": "rp-1", "budget": null, "pods": []}"#;
        std::fs::write(issue("rp-1").join("harness-ledger.json"), runpod_ledger).unwrap();
        assert!(!vast_in_use(&repo, Some("rp-1")) && !vast_in_use(&repo, None));
        let real = Path::new(env!("CARGO_MANIFEST_DIR")).join("../issues/harness-test-vast/harness.toml");
        std::fs::create_dir_all(issue("harness-test-vast")).unwrap();
        std::fs::copy(real, issue("harness-test-vast").join("harness.toml")).unwrap();
        assert!(vast_in_use(&repo, Some("harness-test-vast")) && !vast_in_use(&repo, Some("rp-1")));
        std::fs::write(issue("rp-1").join("harness-ledger.json"), "{not json").unwrap();
        assert!(
            vast_in_use(&repo, None),
            "a ledger that does not load could hold a Vast pod"
        );
        std::fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn watch_reads_its_schedule_and_names_its_issue() {
        let parse = |args: &[&str]| match Cli::try_parse_from(args).map(|cli| cli.command) {
            Ok(Command::Watch { issue, every, limit }) => Ok((issue, every, limit)),
            Ok(_) => panic!("parsed as another command"),
            Err(e) => Err(e.to_string()),
        };
        let (issue, every, limit) = parse(&["harness", "watch", "x-1"]).unwrap();
        assert_eq!(
            (issue.as_str(), every, limit),
            ("x-1", Duration::from_secs(240), Duration::from_secs(86400))
        );
        let (_, every, limit) = parse(&["harness", "watch", "x-1", "--every", "90s", "--for", "1h 30m"]).unwrap();
        assert_eq!((every, limit), (Duration::from_secs(90), Duration::from_secs(5400)));
        for bad in ["0s", "-4m", "soon"] {
            assert!(parse(&["harness", "watch", "x-1", "--every", bad]).is_err(), "{bad}");
        }
        let cli = Cli::try_parse_from(["harness", "watch", "x-1"]).unwrap();
        assert_eq!(cli.command.issue(), Some("x-1"));
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
