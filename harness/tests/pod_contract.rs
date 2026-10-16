//! Both halves of `harness exec` against each other, with no pod: a harness.toml loaded and rendered by the real
//! Rust code, each variant run by the real serve_variant.sh on this machine (fake server and packages, shims for
//! the GPU and process tools), and its progress.jsonl tailed by the real job poll and line handler. A variant-file
//! field, a phase line, or a verdict line that one side changes without the other fails here.

mod support;

use std::fs;
use std::net::TcpListener;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::thread;
use std::time::Duration;

use anyhow::{Result, bail, ensure};
use jiff::{SignedDuration, Timestamp};
use serde_json::{Value, json};
use sm120_harness::clock::Clock;
use sm120_harness::events::Events;
use sm120_harness::exec::{self, render_variant};
use sm120_harness::ledger::{PodEntry, PodState};
use sm120_harness::pod::{Ctx, LivePod};
use sm120_harness::session::{Remote, Target};
use sm120_harness::spec::{CloudTier, IssueSpec};
use support::fake_cloud::FakeCloud;
use support::{TempDir, modern_bash, path_with_modern_bash};

const ISSUE: &str = "contract-1";
/// A server flag and an env value with every character the variant file has to carry through bash intact.
const NOTE: &str = "say \"hi\" to $HOME `id`\nback\\slash";
const ODD: &str = "it's $HOME `id` \"q\"\nline2";

/// The pod as a directory on this machine. Scripts run in it with the fakes first on PATH, the way SshRemote runs
/// them in the kit directory.
struct LocalPod {
    kit: PathBuf,
    env: Vec<(&'static str, String)>,
}

impl Remote for LocalPod {
    fn preflight(&self) -> Result<()> {
        Ok(())
    }

    fn reachable(&self, _: &Target) -> bool {
        true
    }

    fn run(&self, _: &Target, script: &str) -> Result<String> {
        let out = Command::new(modern_bash())
            .arg("-c")
            .arg(format!("set -euo pipefail\n{script}"))
            .current_dir(&self.kit)
            .env_remove("RUNPOD_API_KEY")
            .env_remove("HF_TOKEN")
            .envs(self.env.iter().map(|(key, value)| (key, value)))
            .output()?;
        ensure!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        Ok(String::from_utf8(out.stdout)?)
    }

    fn run_streaming(&self, _: &Target, _: &str) -> Result<()> {
        bail!("exec never streams")
    }

    fn write_file(&self, _: &Target, path: &str, content: &str) -> Result<()> {
        let path = self.kit.join(path);
        fs::create_dir_all(path.parent().unwrap())?;
        Ok(fs::write(path, content)?)
    }

    fn upload(&self, _: &Target, local_root: &Path, paths: &[&str]) -> Result<()> {
        for path in paths {
            copy_tree(&local_root.join(path), &self.kit.join(path));
        }
        Ok(())
    }

    /// The results stay in the kit, where the test reads them.
    fn download(&self, _: &Target, _: &[&str], _: &Path, _: Duration) -> Result<()> {
        Ok(())
    }
}

/// Real time, with the 10 s poll interval cut to 100 ms.
struct Wall;

impl Clock for Wall {
    fn now(&self) -> Timestamp {
        Timestamp::now()
    }

    fn sleep(&self, _: Duration) {
        thread::sleep(Duration::from_millis(100));
    }
}

fn copy_tree(from: &Path, to: &Path) {
    let _ = fs::remove_dir_all(to);
    fs::create_dir_all(to.parent().unwrap()).unwrap();
    assert!(
        Command::new("cp")
            .arg("-R")
            .arg(from)
            .arg(to)
            .status()
            .unwrap()
            .success()
    );
}

fn command_path(tool: &str) -> Option<String> {
    let out = Command::new("sh")
        .args(["-c", &format!("command -v {tool}")])
        .output()
        .unwrap();
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn shim(bin: &Path, name: &str, body: &str) {
    let path = bin.join(name);
    fs::write(&path, format!("#!/usr/bin/env bash\n{body}")).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
}

/// nvidia-smi reports idle GPUs; pkill only logs the cleanup's `-f` kill, which on this machine could match
/// unrelated processes; setsid and timeout stand in where macOS has none. The setsid stand-in starts a new session
/// too, since serve_and_smoke.sh ends POST_SMOKE by its process group.
fn write_shims(bin: &Path) {
    fs::create_dir_all(bin).unwrap();
    shim(bin, "nvidia-smi", "echo 0\n");
    shim(bin, "pkill", "printf '%s\\n' \"$*\" >>\"$WORK/pkill.log\"\n");
    if command_path("setsid").is_none() {
        shim(
            bin,
            "setsid",
            "exec python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \"$@\"\n",
        );
    }
    if command_path("timeout").is_none() {
        shim(
            bin,
            "timeout",
            "while [[ $1 == -* ]]; do\n    [[ $1 == -k || $1 == -s ]] && shift\n    shift\ndone\nshift\nexec \"$@\"\n",
        );
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// A string patch on the served package, a table patch on a second package at another strip level, a second
/// variant that patches only the served package, so the second package must come back unpatched for it, and a
/// third whose server crashes at boot, which is a verdict and not a harness error.
fn harness_toml(port: u16, server: &Path) -> String {
    let toml_string = |s: &str| Value::String(s.into()).to_string();
    format!(
        r#"
issue = "{ISSUE}"
planned_hours = 1.0

[pod]
image = "lmsysorg/sglang:dev-cu13@sha256:0a1ba33aa4e032b8c5bcebdc69c45664074c032074add4f7db01629ec1fcc381"
count = 1
min_cuda = "13.0"
container_disk_gb = 10
volume_gb = 10
gpus = [{{ id = "NVIDIA RTX PRO 6000 Blackwell Server Edition", cloud = "SECURE" }}]

[server]
package = "sglang"
model = "fake/model"
launch = ["python3", {server}, "ok"]
base_flags = ["--note", {note}]
port = {port}
fail_pattern = "illegal memory access"
boot_timeout_s = 60

[variants.patched]
flags = ["--speculative-algorithm", "NEXTN"]
patches = ["demo", {{ name = "infer", package = "fakeinfer", strip = 2 }}]
env = {{ CONTRACT_ODD = {odd}, CONTRACT_FIRST = "1" }}
smoke_concurrency = 2

[variants.patched.quality]
gsm8k = 5
bench = [[1, 8], [16, 64]]

[variants.plain]
flags = ["--tp-size", "1"]
patches = ["demo"]
env = {{ CONTRACT_ODD = "plain" }}

[variants.crash]
flags = ["--fake-mode", "fatal"]
"#,
        server = toml_string(&server.display().to_string()),
        note = toml_string(NOTE),
        odd = toml_string(ODD),
    )
}

const INFER_DIFF: &str = "--- a/pkgs/fakeinfer/core.py\n+++ b/pkgs/fakeinfer/core.py\n@@ -1 +1 @@\n\
                          -KERNEL = \"upstream\"\n+KERNEL = \"patched\"\n";

fn json_lines(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
        .lines()
        .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}")))
        .collect()
}

/// One short line per event, so a whole run compares as one list.
fn trace(event: &Value) -> String {
    let field = |name: &str| event[name].as_str().unwrap_or("?").to_string();
    match field("kind").as_str() {
        "variant_start" => format!("start {}", field("variant")),
        "phase" => format!("{} {}", field("variant"), field("phase")),
        "verdict" => format!("verdict {} {}", field("variant"), field("verdict")),
        "pull" => "pull".into(),
        other => format!("{other}: {event}"),
    }
}

#[test]
fn exec_runs_rendered_variants_through_serve_variant_and_reads_every_line() {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let work = TempDir::new("contract");
    let (repo, kit, bin) = (
        work.path().join("repo"),
        work.path().join("kit"),
        work.path().join("bin"),
    );
    let issue_dir = repo.join("issues").join(ISSUE);
    fs::create_dir_all(issue_dir.join("patches")).unwrap();
    let port = free_port();
    let server = here.join("tests/pod/fake_server.py");
    fs::write(issue_dir.join("harness.toml"), harness_toml(port, &server)).unwrap();
    fs::copy(here.join("tests/pod/demo.diff"), issue_dir.join("patches/demo.diff")).unwrap();
    fs::write(issue_dir.join("patches/infer.diff"), INFER_DIFF).unwrap();
    // What `up` copies: the pod scripts. `exec` sends the issue folder itself.
    copy_tree(&here.join("pod"), &kit.join("harness/pod"));
    copy_tree(&here.join("tests/pod/fake_site"), &work.path().join("site"));
    write_shims(&bin);
    let record = work.path().join("server.jsonl");
    let remote = LocalPod {
        kit: kit.clone(),
        env: vec![
            ("PATH", format!("{}:{}", bin.display(), path_with_modern_bash())),
            ("PYTHONPATH", work.path().join("site").display().to_string()),
            ("PKG_SNAPSHOT_DIR", work.path().join("snap").display().to_string()),
            ("FAKE_SERVER_RECORD", record.display().to_string()),
            ("WORK", work.path().display().to_string()),
            ("BOOT_POLL_S", "0.2".into()),
            ("POST_POLL_S", "0.2".into()),
            ("STOP_POLL_S", "0.1".into()),
        ],
    };

    let spec = IssueSpec::load(&repo, ISSUE).unwrap();
    let (clock, cloud, interrupted) = (Wall, FakeCloud::default(), AtomicBool::new(false));
    let events = Events::new(&repo.join("results"), &clock, false);
    let ctx = Ctx {
        repo: &repo,
        cloud: &cloud,
        remote: &remote,
        clock: &clock,
        events: &events,
        interrupted: &interrupted,
        account: None,
    };
    let now = clock.now();
    let live = LivePod {
        entry: PodEntry {
            name: format!("sm120-{ISSUE}-0a0b0c"),
            id: Some("localpod".into()),
            state: PodState::Live,
            gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition".into(),
            count: 1,
            cloud: CloudTier::Secure,
            rate_usd_hr: 1.0,
            reserved_usd: 1.1,
            reserved_at: now,
            deadline: now + SignedDuration::from_hours(1),
            terminated_at: None,
            cost_usd: None,
            cost_source: None,
        },
        target: Target {
            host: "127.0.0.1".into(),
            port: 22,
            known_hosts: work.path().join("known_hosts"),
        },
    };
    let names = ["patched".to_string(), "plain".to_string(), "crash".to_string()];
    let result = exec::exec(&ctx, &spec, &live, &names);
    let results = kit.join("results").join(ISSUE);
    if let Err(e) = result {
        let logs: Vec<String> = names
            .iter()
            .map(|name| fs::read_to_string(results.join(format!("jobs/variant-{name}.log"))).unwrap_or_default())
            .collect();
        panic!("exec failed: {e:#}\n{}", logs.join("\n---\n"));
    }

    // The file serve_variant.sh sourced is the one render_variant made.
    for name in &names {
        let sent = fs::read_to_string(results.join("variants").join(format!("{name}.sh"))).unwrap();
        assert_eq!(sent, render_variant(&spec, spec.variant(name).unwrap()).unwrap());
    }

    let all = json_lines(&repo.join("results").join(ISSUE).join("events.jsonl"));
    let got: Vec<String> = all.iter().map(trace).collect();
    let patched = [
        "restore",
        "patch",
        "patch",
        "idle_wait",
        "launch",
        "boot_ok",
        "smoke_ok",
        "quality_start",
        "quality_done",
        "stop",
    ];
    let plain = ["restore", "patch", "idle_wait", "launch", "boot_ok", "smoke_ok", "stop"];
    let mut want = vec!["start patched".to_string()];
    want.extend(patched.iter().map(|phase| format!("patched {phase}")));
    want.extend(["verdict patched OK".into(), "pull".into(), "start plain".into()]);
    want.extend(plain.iter().map(|phase| format!("plain {phase}")));
    want.extend(["verdict plain OK".into(), "pull".into(), "start crash".into()]);
    want.extend(["restore", "idle_wait", "launch", "boot_fail", "stop"].map(|phase| format!("crash {phase}")));
    want.extend(["verdict crash BOOT_FAIL".into(), "pull".into()]);
    assert_eq!(got, want);

    let details = |variant: &str, phase: &str| -> Vec<Value> {
        all.iter()
            .filter(|e| e["kind"] == "phase" && e["variant"] == variant && e["phase"] == phase)
            .map(|e| e["detail"].clone())
            .collect()
    };
    assert!(all.iter().filter(|e| e["kind"] == "phase").all(|e| e["t"].is_i64()));
    // Both packages come back before every variant, whatever it patches.
    for variant in ["patched", "plain", "crash"] {
        assert_eq!(details(variant, "restore"), [json!("sglang fakeinfer")], "{variant}");
    }
    assert_eq!(
        details("patched", "patch"),
        [json!("demo -> sglang (-p1)"), json!("infer -> fakeinfer (-p2)")]
    );
    assert_eq!(details("plain", "patch"), [json!("demo -> sglang (-p1)")]);
    let launch = |flags: &[&str]| {
        let mut argv = vec!["python3", server.to_str().unwrap(), "ok", "--note", NOTE];
        argv.extend(flags);
        let port = port.to_string();
        argv.extend(["--host", "127.0.0.1", "--port", &port]);
        argv.iter().map(|arg| arg.to_string()).collect::<Vec<String>>()
    };
    let (patched_argv, plain_argv) = (
        launch(&["--speculative-algorithm", "NEXTN"]),
        launch(&["--tp-size", "1"]),
    );
    assert_eq!(details("patched", "launch"), [json!(patched_argv.join(" "))]);
    assert_eq!(details("plain", "launch"), [json!(plain_argv.join(" "))]);

    // Each verdict event carries the verdict object unchanged, and it is the run's verdict.json.
    for name in &names {
        let mut verdicts: Vec<Value> = all
            .iter()
            .filter(|e| e["kind"] == "verdict" && e["variant"] == name.as_str())
            .cloned()
            .collect();
        assert_eq!(verdicts.len(), 1, "{name}");
        let event = verdicts[0].as_object_mut().unwrap();
        for key in ["ts", "issue", "kind"] {
            event.remove(key);
        }
        let written = json_lines(&results.join(name).join("verdict.json"));
        assert_eq!(written, [Value::Object(event.clone())], "{name}");
    }
    let verdict = |name: &str| json_lines(&results.join(name).join("verdict.json")).remove(0);
    assert_eq!(verdict("patched")["gsm8k"], json!(0.97));
    assert_eq!(verdict("patched")["tok_s"], json!({"1": 100.0, "16": 1600.0}));
    assert_eq!(verdict("plain")["gsm8k"], Value::Null);
    assert!(
        verdict("crash")["fault"]
            .as_str()
            .unwrap()
            .contains("illegal memory access")
    );
    assert_eq!(json_lines(&results.join("verdicts.jsonl")).len(), 3);

    // What the server itself was started with: the exact argv, the variant's env and no other's, and the packages
    // as that variant patched them.
    let started = json_lines(&record);
    assert_eq!(started.len(), 3, "{started:?}");
    assert_eq!(started[0]["argv"], json!(patched_argv[1..]));
    assert_eq!(started[1]["argv"], json!(plain_argv[1..]));
    assert_eq!(started[0]["env"], json!({"CONTRACT_ODD": ODD, "CONTRACT_FIRST": "1"}));
    assert_eq!(started[1]["env"], json!({"CONTRACT_ODD": "plain"}));
    assert_eq!(
        (&started[0]["sglang"], &started[0]["fakeinfer"]),
        (&json!("patched"), &json!("patched"))
    );
    assert_eq!(
        (&started[1]["sglang"], &started[1]["fakeinfer"]),
        (&json!("patched"), &json!("upstream"))
    );
    assert_eq!(
        (&started[2]["sglang"], &started[2]["fakeinfer"], &started[2]["env"]),
        (&json!("upstream"), &json!("upstream"), &json!({}))
    );
}
