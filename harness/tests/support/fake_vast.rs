//! A fake Vast beside the fake RunPod, and a fake box that answers the Vast checks. Included with `#[path]` by the
//! test crates that use it, next to `mod support;`.
#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ops::Deref;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use jiff::{SignedDuration, Timestamp};
use sm120_harness::clock::Clock;
use sm120_harness::cloud::{
    CatalogGpu, Cloud, CloudError, CreatePodRequest, Pod, PodRuntime, PodSsh, PodStatus, SshEndpoint,
};
use sm120_harness::session::{Remote, Target};
use sm120_harness::spec::CloudTier;
use sm120_harness::vast::{OfferQuery, VastApi, VastCreateRequest, VastOffer};

use crate::support::FakeClock;
use crate::support::fake_remote::FakeRemote;

/// A 2x Max-Q offer that passes every default limit; tests change the fields they are about.
pub fn offer(offer_id: u64, machine_id: u64, rate_usd_hr: f64) -> VastOffer {
    VastOffer {
        offer_id,
        machine_id,
        host_id: 80000 + machine_id,
        gpu: "RTX PRO 6000 Max-Q".into(),
        count: 2,
        rate_usd_hr,
        reliability2: Some(0.99),
        reliability: None,
        disk_space: 500.0,
        inet_down: 2000.0,
        internet_down_cost_per_tb: Some(3.0),
        rentable: Some(true),
        rented: Some(false),
        cuda_max_good: Some(13.2),
        geolocation: Some("Texas, US".into()),
        hosting_type: Some(0),
    }
}

/// What the next create does.
#[derive(Debug, Clone, Copy)]
pub enum Create {
    Made,
    /// Vast's 200 with `success: false`, as a taken offer gets.
    Taken,
    MadeButReplyLost,
    /// The reply is lost, and no read shows the instance until `reveal_unlisted`.
    MadeButListedLater,
    /// The reply names the instance, and no read shows it until `reveal_unlisted`.
    MadeListedLater,
    Refused(u16),
}

pub struct VastState {
    pub offers: Vec<VastOffer>,
    pub creates: VecDeque<Create>,
    pub requests: Vec<(u64, VastCreateRequest)>,
    pub instances: Vec<(Pod, u64)>,
    pub unlisted: Vec<(Pod, u64)>,
    /// The unlisted instances show from this many list calls on.
    pub reveal_after_lists: Option<usize>,
    pub searches: usize,
    /// Searches that find nothing before the offers show up, as while a watch waits.
    pub empty_searches: usize,
    /// Every search fails with this status and detail.
    pub search_error: Option<(u16, &'static str)>,
    /// Reads that do not see an instance yet, as right after a create.
    pub hidden_reads: usize,
    /// GET fails as on a reply Vast garbled; the list still answers.
    pub get_fails: bool,
    pub list_error: Option<u16>,
    pub delete_error: Option<u16>,
    /// Reads, GET or list, that still show a destroyed instance, as Vast does while it tears one down.
    pub linger_reads: usize,
    lingering: Vec<Pod>,
    pub deleted: Vec<String>,
    /// What a running instance bills per hour, when not its offer's rate.
    pub instance_cost: Option<f64>,
    pub billing: Option<f64>,
    pub list_calls: usize,
    next_id: u64,
}

pub struct FakeVast(pub RefCell<VastState>);

impl Default for FakeVast {
    fn default() -> Self {
        Self(RefCell::new(VastState {
            offers: vec![offer(9100002, 70002, 2.80), offer(9100001, 70001, 2.46)],
            creates: VecDeque::new(),
            requests: Vec::new(),
            instances: Vec::new(),
            unlisted: Vec::new(),
            reveal_after_lists: None,
            searches: 0,
            empty_searches: 0,
            search_error: None,
            hidden_reads: 0,
            get_fails: false,
            list_error: None,
            delete_error: None,
            linger_reads: 0,
            lingering: Vec::new(),
            deleted: Vec::new(),
            instance_cost: None,
            billing: None,
            list_calls: 0,
            next_id: 31000000,
        }))
    }
}

fn http(status: u16, detail: &str) -> CloudError {
    CloudError::Http {
        status,
        detail: detail.into(),
        retry_after_s: None,
    }
    .vast()
}

fn unsupported() -> CloudError {
    CloudError::Protocol("not a Vast call".into()).vast()
}

impl FakeVast {
    fn make(&self, request: &VastCreateRequest, offer_id: u64, listed: bool) -> String {
        let mut state = self.0.borrow_mut();
        state.next_id += 1;
        let rate = state
            .offers
            .iter()
            .find(|o| o.offer_id == offer_id)
            .unwrap()
            .rate_usd_hr;
        let pod = Pod {
            id: state.next_id.to_string(),
            name: request.label.clone(),
            status: PodStatus::Provisioning,
            cost: Some(state.instance_cost.unwrap_or(rate)),
            ssh: None,
            runtime: None,
            data_center_id: None,
        };
        let id = pod.id.clone();
        if listed {
            state.instances.push((pod, offer_id));
        } else {
            state.unlisted.push((pod, offer_id));
        }
        id
    }

    /// Instances that exist and bill, listed or not.
    pub fn billing_ids(&self) -> Vec<String> {
        let state = self.0.borrow();
        state
            .instances
            .iter()
            .chain(&state.unlisted)
            .map(|(pod, _)| pod.id.clone())
            .collect()
    }

    pub fn reveal_unlisted(&self) {
        let mut state = self.0.borrow_mut();
        let unlisted: Vec<(Pod, u64)> = state.unlisted.drain(..).collect();
        state.instances.extend(unlisted);
    }

    /// One read's worth of the instances still shown after their destroy.
    fn lingering(&self) -> Vec<Pod> {
        let mut state = self.0.borrow_mut();
        if state.lingering.is_empty() {
            return Vec::new();
        }
        if state.linger_reads == 0 {
            state.lingering.clear();
            return Vec::new();
        }
        state.linger_reads -= 1;
        state.lingering.clone()
    }
}

impl Cloud for FakeVast {
    fn gpu_catalog(&self, _: CloudTier, _: u32, _: &str) -> Result<Vec<CatalogGpu>, CloudError> {
        Err(unsupported())
    }

    fn create_pod(&self, _: &CreatePodRequest) -> Result<Pod, CloudError> {
        Err(unsupported())
    }

    fn list_pods(&self) -> Result<Vec<Pod>, CloudError> {
        let lingering = self.lingering();
        let due = {
            let mut state = self.0.borrow_mut();
            let due = state.reveal_after_lists.map(|left| left.saturating_sub(1));
            state.reveal_after_lists = due;
            due == Some(0)
        };
        if due {
            self.reveal_unlisted();
        }
        let mut state = self.0.borrow_mut();
        state.list_calls += 1;
        if let Some(status) = state.list_error {
            return Err(http(status, "fake list failure"));
        }
        let mut pods: Vec<Pod> = state.instances.iter().map(|(pod, _)| pod.clone()).collect();
        pods.extend(lingering);
        Ok(pods)
    }

    /// A new instance loads on its first read and runs with SSH from the second on.
    fn get_pod(&self, id: &str) -> Result<Option<Pod>, CloudError> {
        if let Some(pod) = self.lingering().into_iter().find(|pod| pod.id == id) {
            return Ok(Some(pod));
        }
        let mut state = self.0.borrow_mut();
        if state.get_fails {
            return Err(CloudError::Decode("missing field `instances`".into()).vast());
        }
        if state.hidden_reads > 0 {
            state.hidden_reads -= 1;
            return Ok(None);
        }
        let Some((pod, _)) = state.instances.iter_mut().find(|(pod, _)| pod.id == id) else {
            return Ok(None);
        };
        pod.status = match pod.status {
            PodStatus::Provisioning => PodStatus::Starting,
            _ => PodStatus::Running,
        };
        if pod.status == PodStatus::Running {
            pod.ssh = Some(PodSsh {
                direct: Some(SshEndpoint {
                    host: "203.0.113.20".into(),
                    port: 41022,
                }),
            });
            pod.runtime = Some(PodRuntime {
                uptime: Some(60.0),
                ports: serde_json::Value::Null,
            });
        }
        Ok(Some(pod.clone()))
    }

    fn delete_pod(&self, id: &str) -> Result<(), CloudError> {
        let mut state = self.0.borrow_mut();
        if let Some(status) = state.delete_error {
            return Err(http(status, "fake destroy failure"));
        }
        state.deleted.push(id.to_string());
        let gone: Vec<Pod> = state
            .instances
            .iter()
            .filter(|(pod, _)| pod.id == id)
            .map(|(pod, _)| Pod {
                status: PodStatus::Exited,
                ..pod.clone()
            })
            .collect();
        if state.linger_reads > 0 {
            state.lingering.extend(gone);
        }
        // Vast answers 404 for an instance it does not list yet, so the destroy leaves an unlisted one billing.
        state.instances.retain(|(pod, _)| pod.id != id);
        Ok(())
    }

    fn pod_billing_usd(&self, _: &str) -> Result<Option<f64>, CloudError> {
        Ok(self.0.borrow().billing)
    }

    fn account(&self) -> Result<Option<String>, CloudError> {
        Ok(Some("vast:424242".into()))
    }

    fn clock_skew(&self) -> Option<SignedDuration> {
        Some(SignedDuration::ZERO)
    }
}

impl VastApi for FakeVast {
    fn offers(&self, query: &OfferQuery) -> Result<Vec<VastOffer>, CloudError> {
        let mut state = self.0.borrow_mut();
        state.searches += 1;
        if let Some((status, detail)) = state.search_error {
            return Err(http(status, detail));
        }
        if state.empty_searches > 0 {
            state.empty_searches -= 1;
            return Ok(Vec::new());
        }
        let matching = state
            .offers
            .iter()
            .filter(|o| o.gpu == query.gpu && o.count == query.count);
        Ok(matching.cloned().collect())
    }

    fn create_instance(&self, offer_id: u64, request: &VastCreateRequest) -> Result<String, CloudError> {
        let outcome = {
            let mut state = self.0.borrow_mut();
            state.requests.push((offer_id, request.clone()));
            state.creates.pop_front().unwrap_or(Create::Made)
        };
        match outcome {
            Create::Made => Ok(self.make(request, offer_id, true)),
            Create::Taken => Err(CloudError::Refused("create: no_such_ask".into()).vast()),
            Create::MadeButReplyLost => {
                self.make(request, offer_id, true);
                Err(CloudError::Network("connection reset by peer".into()).vast())
            }
            Create::MadeButListedLater => {
                self.make(request, offer_id, false);
                Err(CloudError::Network("timed out".into()).vast())
            }
            Create::MadeListedLater => Ok(self.make(request, offer_id, false)),
            Create::Refused(status) => Err(http(status, "fake create refusal")),
        }
    }
}

/// FakeClock that can raise a flag on its next sleep, as a Ctrl-C between two steps would.
pub struct WatchClock {
    pub inner: FakeClock,
    pub interrupt_on_sleep: RefCell<Option<Rc<AtomicBool>>>,
}

impl WatchClock {
    pub fn at(rfc3339: &str) -> Self {
        Self {
            inner: FakeClock::at(rfc3339),
            interrupt_on_sleep: RefCell::new(None),
        }
    }
}

impl Deref for WatchClock {
    type Target = FakeClock;

    fn deref(&self) -> &FakeClock {
        &self.inner
    }
}

impl Clock for WatchClock {
    fn now(&self) -> Timestamp {
        self.inner.now()
    }

    fn sleep(&self, duration: Duration) {
        self.inner.sleep(duration);
        if let Some(flag) = self.interrupt_on_sleep.borrow_mut().take() {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

pub const P2P_OK: &str =
    " \tGPU0\tGPU1\t\n GPU0\tX\tOK\t\n GPU1\tOK\tX\t\n\nLegend:\n\n  X    = Self\n  OK   = Status Ok\n";

pub fn p2p_without(cell: &str) -> String {
    P2P_OK
        .replace("GPU0\tX\tOK", &format!("GPU0\tX\t{cell}"))
        .replace("GPU1\tOK\tX", &format!("GPU1\t{cell}\tX"))
}

/// A Vast box: FakeRemote, plus the P2P matrix each box prints in turn (the last one repeats) and what the Vast
/// watchdog check prints when not FakeRemote's plain WATCHDOG_ARMED.
pub struct VastBox {
    pub inner: FakeRemote,
    pub p2p: RefCell<VecDeque<String>>,
    pub watchdog: RefCell<Option<String>>,
}

impl Default for VastBox {
    fn default() -> Self {
        Self {
            inner: FakeRemote::default(),
            p2p: RefCell::new(VecDeque::from([P2P_OK.to_string()])),
            watchdog: RefCell::new(None),
        }
    }
}

impl Remote for VastBox {
    fn preflight(&self) -> Result<()> {
        self.inner.preflight()
    }
    fn reachable(&self, target: &Target) -> bool {
        self.inner.reachable(target)
    }
    fn run(&self, target: &Target, script: &str) -> Result<String> {
        if let Some(reply) = self.watchdog.borrow().as_ref()
            && script.contains("watchdog armed for epoch")
        {
            self.inner.scripts.borrow_mut().push(script.to_string());
            return Ok(reply.clone());
        }
        if script.contains("nvidia-smi topo -p2p r") {
            self.inner.scripts.borrow_mut().push(script.to_string());
            let mut matrices = self.p2p.borrow_mut();
            let matrix = if matrices.len() > 1 {
                matrices.pop_front()
            } else {
                matrices.front().cloned()
            };
            return Ok(matrix.unwrap_or_default());
        }
        self.inner.run(target, script)
    }
    fn run_streaming(&self, target: &Target, script: &str) -> Result<()> {
        self.inner.run_streaming(target, script)
    }
    fn write_file(&self, target: &Target, path: &str, content: &str) -> Result<()> {
        self.inner.write_file(target, path, content)
    }
    fn upload(&self, target: &Target, root: &Path, paths: &[&str]) -> Result<()> {
        self.inner.upload(target, root, paths)
    }
    fn download(&self, target: &Target, paths: &[&str], root: &Path, limit: Duration) -> Result<()> {
        self.inner.download(target, paths, root, limit)
    }
}
