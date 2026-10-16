use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::fs;
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
    polls: Cell<u32>,
    /// progress.jsonl lines a variant job writes before its verdict.
    pub phases: RefCell<Vec<String>>,
    pub verdict: RefCell<String>,
    pub uploads: RefCell<Vec<Vec<String>>>,
    pub files: RefCell<Vec<(String, String)>>,
    pub downloads: RefCell<Vec<(Vec<String>, Duration)>>,
    pub download_fails: Cell<bool>,
    /// Each download that succeeds writes `pull <n>` to server.log under its first path, as a pull of a live variant
    /// brings back its log so far.
    pub downloads_write: Cell<bool>,
    /// The container is gone once this many polls were answered: later polls and downloads fail, as on a pod RunPod
    /// stopped.
    dies_after_polls: Cell<u32>,
    /// RUNNING replies a variant job gives before it exits.
    pub variant_polls: Cell<u32>,
    /// The next job is killed without writing an exit code.
    pub job_dies: Cell<bool>,
    /// Variant jobs never exit, like a server that hangs past every timeout.
    pub variant_hangs: Cell<bool>,
    pub interrupt_on_job: RefCell<Option<Rc<AtomicBool>>>,
    /// Sets the flag once this many polls have been answered, as a Ctrl-C mid-job would.
    pub interrupt_after_polls: RefCell<Option<(u32, Rc<AtomicBool>)>>,
    pub panic_on_write: Cell<bool>,
    /// Whether the pod's watchdog log shows it armed, as the entrypoint writes at boot.
    pub watchdog_armed: Cell<bool>,
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
            polls: Cell::new(0),
            phases: RefCell::new(vec![
                r#"{"phase":"launch","t":1760000000,"detail":"vllm serve"}"#.into(),
            ]),
            verdict: RefCell::new(r#"{"variant":"baseline","verdict":"OK","boot_s":180}"#.into()),
            uploads: RefCell::new(Vec::new()),
            files: RefCell::new(Vec::new()),
            downloads: RefCell::new(Vec::new()),
            download_fails: Cell::new(false),
            downloads_write: Cell::new(false),
            dies_after_polls: Cell::new(u32::MAX),
            variant_polls: Cell::new(1),
            job_dies: Cell::new(false),
            variant_hangs: Cell::new(false),
            interrupt_on_job: RefCell::new(None),
            interrupt_after_polls: RefCell::new(None),
            panic_on_write: Cell::new(false),
            watchdog_armed: Cell::new(true),
        }
    }
}

impl FakeRemote {
    fn poll(&self, script: &str) -> String {
        let polls = self.polls.get() + 1;
        self.polls.set(polls);
        // A wait loop that lost its deadline check would spin here; fail instead of hanging the test run.
        assert!(polls < 10_000, "polled {polls} times without the harness giving up");
        if let Some((after, flag)) = self.interrupt_after_polls.borrow().as_ref()
            && polls >= *after
        {
            flag.store(true, Ordering::SeqCst);
        }
        // What a login shell might print before the poll script runs.
        let mut reply = format!("Welcome to Ubuntu\n{POLL_SENTINEL}\n");
        let variant = script.contains("progress.jsonl");
        let left = self.polls_before_exit.get();
        if left > 0 || (variant && self.variant_hangs.get()) {
            self.polls_before_exit.set(left.saturating_sub(1));
            reply.push_str("RUNNING\n");
            return reply;
        }
        if self.job_dies.get() {
            reply.push_str("DEAD\n");
            return reply;
        }
        reply.push_str(&format!("EXIT {}\n", self.exit.get()));
        if variant && self.exit.get() == 0 {
            for line in self.phases.borrow().iter() {
                reply.push_str(line);
                reply.push('\n');
            }
            reply.push_str(&self.verdict.borrow());
            reply.push('\n');
        }
        reply
    }

    pub fn download_count(&self) -> usize {
        self.downloads.borrow().len()
    }

    /// The container goes away after `more` further polls are answered.
    pub fn die_after(&self, more: u32) {
        self.dies_after_polls.set(self.polls.get() + more);
    }

    fn dead(&self) -> bool {
        self.polls.get() >= self.dies_after_polls.get()
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
        if script.contains("watchdog armed for epoch") {
            let reply = if self.watchdog_armed.get() {
                "WATCHDOG_ARMED\n"
            } else {
                "WATCHDOG_MISSING\n"
            };
            return Ok(reply.into());
        }
        if script.contains("nvidia-smi topo -m") {
            return Ok(self.topo.borrow().clone());
        }
        if script.contains("setsid nohup") {
            self.jobs.borrow_mut().push(script.to_string());
            let running = if script.contains("serve_variant.sh") {
                self.variant_polls.get()
            } else {
                1
            };
            self.polls_before_exit.set(running);
            self.exit.set(self.exits.borrow_mut().pop_front().unwrap_or(0));
            if let Some(flag) = self.interrupt_on_job.borrow().as_ref() {
                flag.store(true, Ordering::SeqCst);
            }
            return Ok("STARTED\n".into());
        }
        if script.contains(".exit ]]") {
            if self.dead() {
                bail!("ssh: connect to host 203.0.113.7 port 40122: Connection refused");
            }
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

    fn download(&self, _: &Target, paths: &[&str], local_root: &Path, limit: Duration) -> Result<()> {
        self.downloads
            .borrow_mut()
            .push((paths.iter().map(|p| p.to_string()).collect(), limit));
        if self.dead() {
            bail!("pulling {paths:?} failed (ssh: connect to host 203.0.113.7 port 40122: Connection refused)");
        }
        if self.download_fails.get() {
            bail!("pulling {paths:?} ran past its {limit:?} limit and was killed");
        }
        if self.downloads_write.get() {
            let dir = local_root.join(paths[0]);
            fs::create_dir_all(&dir)?;
            fs::write(dir.join("server.log"), format!("pull {}", self.download_count()))?;
        }
        Ok(())
    }
}
