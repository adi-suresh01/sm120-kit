//! Runs tests/pod/entrypoint_vast_test.sh on the entrypoint and onstart loader exactly as `up` sends them to Vast.

mod support;

use std::fs;
use std::path::Path;
use std::process::Command;

use sm120_harness::pod::{VAST_ONSTART, vast_entrypoint};
use support::{TempDir, modern_bash, path_with_modern_bash};

#[test]
fn vast_entrypoint_and_loader() {
    let dir = TempDir::new("vast-entrypoint");
    let (entrypoint, onstart) = (dir.path().join("entrypoint.sh"), dir.path().join("onstart.sh"));
    fs::write(&entrypoint, vast_entrypoint().unwrap()).unwrap();
    fs::write(&onstart, VAST_ONSTART).unwrap();
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pod/entrypoint_vast_test.sh");
    let out = Command::new(modern_bash())
        .env("PATH", path_with_modern_bash())
        .env("VAST_ENTRYPOINT_FILE", &entrypoint)
        .env("VAST_ONSTART_FILE", &onstart)
        .arg(&script)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "--- stdout\n{}\n--- stderr\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
