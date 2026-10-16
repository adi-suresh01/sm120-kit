//! Runs the plain-bash tests under tests/pod/ so `cargo test` covers the pod side too.

mod support;

use std::path::Path;
use std::process::Command;

use support::{modern_bash, path_with_modern_bash};

fn run_bash_test(name: &str) {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pod").join(name);
    let out = Command::new(modern_bash())
        .env("PATH", path_with_modern_bash())
        .arg(&script)
        .output()
        .unwrap();
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
fn patch_pkg() {
    run_bash_test("patch_pkg_test.sh");
}

#[test]
fn verdict_replays_stage_2() {
    run_bash_test("verdict_replay_test.sh");
}

#[test]
fn verdict_survives_unexpected_files() {
    run_bash_test("verdict_errors_test.sh");
}

#[test]
fn nvidia_smi_calls_are_bounded() {
    run_bash_test("gpu_tools_test.sh");
}

#[test]
fn memory_guard_on_fake_cgroups() {
    run_bash_test("mem_guard_test.sh");
}

#[test]
fn serve_variant_against_a_fake_server() {
    run_bash_test("serve_variant_test.sh");
}
