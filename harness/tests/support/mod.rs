// Every integration test crate compiles this module and each uses a different part of it.
#![allow(dead_code)]

use std::cell::{Cell, RefCell};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use jiff::{SignedDuration, Timestamp};
use sm120_harness::clock::Clock;

pub mod fake_cloud;
pub mod fake_remote;

pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let n = NEXT.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!("sm120-harness-{tag}-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A repo layout with `issues/<issue>/` and the given files under it.
pub fn repo_with_issue(issue: &str, files: &[(&str, &str)]) -> TempDir {
    let repo = TempDir::new(issue);
    let dir = repo.path().join("issues").join(issue);
    fs::create_dir_all(&dir).unwrap();
    for (name, body) in files {
        let path = dir.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, body).unwrap();
    }
    repo
}

pub struct FakeClock {
    now: Cell<Timestamp>,
    pub slept: RefCell<Vec<Duration>>,
}

impl FakeClock {
    pub fn at(rfc3339: &str) -> Self {
        Self {
            now: Cell::new(rfc3339.parse().unwrap()),
            slept: RefCell::new(Vec::new()),
        }
    }

    pub fn advance(&self, by: SignedDuration) {
        self.now.set(self.now.get() + by);
    }
}

impl Clock for FakeClock {
    fn now(&self) -> Timestamp {
        self.now.get()
    }

    fn sleep(&self, duration: Duration) {
        self.slept.borrow_mut().push(duration);
        self.advance(SignedDuration::try_from(duration).unwrap());
    }
}
