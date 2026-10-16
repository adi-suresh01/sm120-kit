use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Result, bail};
use sm120_harness::session::{POLL_SENTINEL, Remote, Target};

/// A pod that answers the harness's scripts the way the pod-side bash would: each job reports RUNNING once, then
/// exits with the next code from `exits` (0 when empty) and, for a variant job that exits 0, one verdict line.
pub struct FakeRemote {
    pub preflight_ok: Cell<bool>,
    pub reachable: Cell<bool>,
    pub topo: RefCell<String>,
    /// Every script `run` was given, in order.
    pub scripts: RefCell<Vec<String>>,
    pub jobs: RefCell<Vec<String>>,
    pub exits: RefCell<VecDeque<i32>>,
    exit: Cell<i32>,
    polls_before_exit: Cell<u32>,
    pub verdict: RefCell<String>,
    pub uploads: RefCell<Vec<Vec<String>>>,
    pub files: RefCell<Vec<(String, String)>>,
    pub downloads: RefCell<Vec<(Vec<String>, Duration)>>,
    pub download_fails: Cell<bool>,
    /// The next job is killed without writing an exit code.
    pub job_dies: Cell<bool>,
    pub interrupt_on_job: RefCell<Option<Rc<AtomicBool>>>,
    pub panic_on_write: Cell<bool>,
}

impl Default for FakeRemote {
    fn default() -> Self {
        Self {
            preflight_ok: Cell::new(true),
            reachable: Cell::new(true),
            topo: RefCell::new(include_str!("../fixtures/topo_pcie.txt").to_string()),
            scripts: RefCell::new(Vec::new()),
            jobs: RefCell::new(Vec::new()),
            exits: RefCell::new(VecDeque::new()),
            exit: Cell::new(0),
            polls_before_exit: Cell::new(1),
            verdict: RefCell::new(r#"{"variant":"baseline","verdict":"OK","boot_s":180}"#.into()),
            uploads: RefCell::new(Vec::new()),
            files: RefCell::new(Vec::new()),
            downloads: RefCell::new(Vec::new()),
            download_fails: Cell::new(false),
            job_dies: Cell::new(false),
            interrupt_on_job: RefCell::new(None),
            panic_on_write: Cell::new(false),
        }
    }
}

impl FakeRemote {
    fn poll(&self, script: &str) -> String {
        // What a login shell might print before the poll script runs.
        let mut reply = format!("Welcome to Ubuntu\n{POLL_SENTINEL}\n");
        let left = self.polls_before_exit.get();
        if left > 0 {
            self.polls_before_exit.set(left - 1);
            reply.push_str("RUNNING\n");
            return reply;
        }
        if self.job_dies.get() {
            reply.push_str("DEAD\n");
            return reply;
        }
        reply.push_str(&format!("EXIT {}\n", self.exit.get()));
        if script.contains("verdict.json") && self.exit.get() == 0 {
            reply.push_str(&self.verdict.borrow());
            reply.push('\n');
        }
        reply
    }

    pub fn download_count(&self) -> usize {
        self.downloads.borrow().len()
    }
}

impl Remote for FakeRemote {
    fn preflight(&self) -> Result<()> {
        if !self.preflight_ok.get() {
            bail!("ssh-agent holds no key");
        }
        Ok(())
    }

    fn reachable(&self, _: &Target) -> bool {
        self.reachable.get()
    }

    fn run(&self, _: &Target, script: &str) -> Result<String> {
        self.scripts.borrow_mut().push(script.to_string());
        if script.contains("nvidia-smi topo -m") {
            return Ok(self.topo.borrow().clone());
        }
        if script.contains("setsid nohup") {
            self.jobs.borrow_mut().push(script.to_string());
            self.polls_before_exit.set(1);
            self.exit.set(self.exits.borrow_mut().pop_front().unwrap_or(0));
            if let Some(flag) = self.interrupt_on_job.borrow().as_ref() {
                flag.store(true, Ordering::SeqCst);
            }
            return Ok("STARTED\n".into());
        }
        if script.contains(".exit ]]") {
            return Ok(self.poll(script));
        }
        Ok(String::new())
    }

    fn run_streaming(&self, _: &Target, _: &str) -> Result<()> {
        Ok(())
    }

    fn write_file(&self, _: &Target, path: &str, content: &str) -> Result<()> {
        if self.panic_on_write.get() {
            panic!("harness bug while writing {path}");
        }
        self.files.borrow_mut().push((path.to_string(), content.to_string()));
        Ok(())
    }

    fn upload(&self, _: &Target, _: &Path, paths: &[&str]) -> Result<()> {
        self.uploads
            .borrow_mut()
            .push(paths.iter().map(|p| p.to_string()).collect());
        Ok(())
    }

    fn download(&self, _: &Target, paths: &[&str], _: &Path, limit: Duration) -> Result<()> {
        self.downloads
            .borrow_mut()
            .push((paths.iter().map(|p| p.to_string()).collect(), limit));
        if self.download_fails.get() {
            bail!("pulling {paths:?} ran past its {limit:?} limit and was killed");
        }
        Ok(())
    }
}
