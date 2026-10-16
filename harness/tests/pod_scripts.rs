//! Runs the plain-bash tests under tests/pod/ so `cargo test` covers the pod side too.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// The pod images run bash 5, and the scripts rely on 4.4 behavior such as empty arrays under `set -u`. macOS
/// still ships 3.2 as /bin/bash.
fn modern_bash() -> &'static Path {
    static FOUND: OnceLock<PathBuf> = OnceLock::new();
    FOUND.get_or_init(|| {
        for candidate in ["bash", "/opt/homebrew/bin/bash", "/usr/local/bin/bash"] {
            let Ok(out) = Command::new(candidate)
                .args(["-c", "echo ${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"])
                .output()
            else {
                continue;
            };
            let version = String::from_utf8_lossy(&out.stdout);
            let mut parts = version.split_whitespace().map(|part| part.parse::<u32>().unwrap_or(0));
            if (parts.next().unwrap_or(0), parts.next().unwrap_or(0)) >= (4, 4) {
                return PathBuf::from(candidate);
            }
        }
        panic!("these tests need bash >= 4.4 (brew install bash)");
    })
}

fn run_bash_test(name: &str) {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pod").join(name);
    let bash = modern_bash();
    let mut command = Command::new(bash);
    // Scripts call `bash` by name, so the modern one has to come first on PATH.
    if let Some(dir) = bash.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        let path = std::env::var("PATH").unwrap_or_default();
        command.env("PATH", format!("{}:{path}", dir.display()));
    }
    let out = command.arg(&script).output().unwrap();
    assert!(
        out.status.success(),
        "{name} failed\n--- stdout\n{}\n--- stderr\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn entrypoint_watchdog() {
    run_bash_test("entrypoint_test.sh");
}

#[test]
fn pkg_snapshot() {
    run_bash_test("pkg_snapshot_test.sh");
}

#[test]
fn verdict_replays_stage_2() {
    run_bash_test("verdict_replay_test.sh");
}

#[test]
fn serve_variant_against_a_fake_server() {
    run_bash_test("serve_variant_test.sh");
}
