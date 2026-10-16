use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};

use sm120_harness::cloud::{CatalogGpu, Cloud, CloudError, CreatePodRequest, Pod, PodSsh, PodStatus, SshEndpoint};
use sm120_harness::spec::CloudTier;

/// What the next create call does. Several leave a pod behind while reporting failure, which is exactly what the
/// harness must survive.
#[derive(Debug, Clone, Copy)]
pub enum Create {
    Made,
    Refused(u16),
    MadeButReplyLost,
    MadeBut5xx,
    RequestLost,
    MadeButListedLater,
    KilledBeforeSend,
    KilledAfterMade,
}

pub struct FakePod {
    pub pod: Pod,
    pub env: BTreeMap<String, String>,
    boot: VecDeque<PodStatus>,
}

pub struct State {
    pub secure: Vec<CatalogGpu>,
    pub community: Vec<CatalogGpu>,
    pub creates: VecDeque<Create>,
    /// Statuses a new pod reports on successive reads; the last one repeats.
    pub boot: Vec<PodStatus>,
    pub cost: Option<f64>,
    /// A running pod never gets `ssh.direct`, as on a host without public IPs.
    pub proxy_only: bool,
    pub pods: Vec<FakePod>,
    pub unlisted: Vec<FakePod>,
    pub get_error: Option<u16>,
    /// GET answers 404 for every pod, as RunPod does for a pod the key cannot see.
    pub get_not_found: bool,
    /// The pod list leaves out every pod while GET still finds them.
    pub list_omits_all: bool,
    pub delete_error: Option<u16>,
    pub delete_ignored: bool,
    pub account: Option<String>,
    pub billing: Option<f64>,
    pub interrupt_on_create: Option<Rc<AtomicBool>>,
    /// Runs right after a pod is made, before the reply: the moment a local failure hurts most.
    pub after_create: Option<Box<dyn Fn()>>,
    pub create_calls: usize,
    next_id: u32,
}

pub struct FakeCloud(pub RefCell<State>);

fn http(status: u16) -> CloudError {
    CloudError::Http {
        status,
        detail: format!("fake {status}"),
        retry_after_s: None,
    }
}

fn catalog(json: &str) -> Vec<CatalogGpu> {
    let page: serde_json::Value = serde_json::from_str(json).unwrap();
    serde_json::from_value(page["gpus"].clone()).unwrap()
}

impl Default for FakeCloud {
    fn default() -> Self {
        Self(RefCell::new(State {
            secure: catalog(include_str!("../fixtures/catalog_secure_x1.json")),
            community: catalog(include_str!("../fixtures/catalog_community_x1.json")),
            creates: VecDeque::new(),
            boot: vec![PodStatus::Starting, PodStatus::Running],
            cost: Some(2.49),
            proxy_only: false,
            pods: Vec::new(),
            unlisted: Vec::new(),
            get_error: None,
            get_not_found: false,
            list_omits_all: false,
            delete_error: None,
            delete_ignored: false,
            account: Some("org-sm120".into()),
            billing: None,
            interrupt_on_create: None,
            after_create: None,
            create_calls: 0,
            next_id: 0,
        }))
    }
}

impl FakeCloud {
    pub fn script(&self, creates: &[Create]) {
        self.0.borrow_mut().creates = creates.iter().copied().collect();
    }

    /// Pods that exist and would bill, listed or not.
    pub fn billing_pods(&self) -> Vec<(String, String, BTreeMap<String, String>)> {
        let state = self.0.borrow();
        state
            .pods
            .iter()
            .chain(&state.unlisted)
            .filter(|fake| fake.pod.status != PodStatus::Terminated)
            .map(|fake| (fake.pod.id.clone(), fake.pod.name.clone(), fake.env.clone()))
            .collect()
    }

    pub fn reveal_unlisted(&self) {
        let mut state = self.0.borrow_mut();
        let unlisted: Vec<FakePod> = state.unlisted.drain(..).collect();
        state.pods.extend(unlisted);
    }

    pub fn remove_outside_harness(&self, id: &str) {
        self.0.borrow_mut().pods.retain(|fake| fake.pod.id != id);
    }

    fn make(&self, request: &CreatePodRequest, listed: bool) -> Pod {
        let mut state = self.0.borrow_mut();
        state.next_id += 1;
        let pod = Pod {
            id: format!("fakepod{:04}", state.next_id),
            name: request.name.clone(),
            status: PodStatus::Provisioning,
            cost: state.cost,
            ssh: Some(PodSsh { direct: None }),
        };
        let fake = FakePod {
            pod: pod.clone(),
            env: request.env.clone(),
            boot: state.boot.iter().copied().collect(),
        };
        if listed {
            state.pods.push(fake);
        } else {
            state.unlisted.push(fake);
        }
        if let Some(hook) = &state.after_create {
            hook();
        }
        pod
    }
}

impl Cloud for FakeCloud {
    fn gpu_catalog(&self, cloud: CloudTier, _: u32, _: &str) -> Result<Vec<CatalogGpu>, CloudError> {
        let state = self.0.borrow();
        Ok(match cloud {
            CloudTier::Secure => state.secure.clone(),
            CloudTier::Community => state.community.clone(),
        })
    }

    fn create_pod(&self, request: &CreatePodRequest) -> Result<Pod, CloudError> {
        let outcome = {
            let mut state = self.0.borrow_mut();
            state.create_calls += 1;
            if let Some(flag) = &state.interrupt_on_create {
                flag.store(true, Ordering::SeqCst);
            }
            state.creates.pop_front().unwrap_or(Create::Made)
        };
        match outcome {
            Create::Made => Ok(self.make(request, true)),
            Create::Refused(status) => Err(http(status)),
            Create::MadeButReplyLost => {
                self.make(request, true);
                Err(CloudError::Network("connection reset by peer".into()))
            }
            Create::MadeBut5xx => {
                self.make(request, true);
                Err(http(502))
            }
            Create::RequestLost => Err(CloudError::Network("connection refused".into())),
            Create::MadeButListedLater => {
                self.make(request, false);
                Err(CloudError::Network("timed out".into()))
            }
            Create::KilledBeforeSend => panic!("harness killed before the create request left"),
            Create::KilledAfterMade => {
                self.make(request, true);
                panic!("harness killed after the pod was made")
            }
        }
    }

    fn list_pods(&self) -> Result<Vec<Pod>, CloudError> {
        let state = self.0.borrow();
        if state.list_omits_all {
            return Ok(Vec::new());
        }
        Ok(state.pods.iter().map(|fake| fake.pod.clone()).collect())
    }

    fn get_pod(&self, id: &str) -> Result<Option<Pod>, CloudError> {
        let mut state = self.0.borrow_mut();
        if let Some(status) = state.get_error {
            return Err(http(status));
        }
        if state.get_not_found {
            return Ok(None);
        }
        let proxy_only = state.proxy_only;
        let Some(fake) = state.pods.iter_mut().find(|fake| fake.pod.id == id) else {
            return Ok(None);
        };
        if fake.boot.len() > 1 {
            fake.boot.pop_front();
        }
        fake.pod.status = fake.boot[0];
        if fake.pod.status == PodStatus::Running && !proxy_only {
            fake.pod.ssh = Some(PodSsh {
                direct: Some(SshEndpoint {
                    host: "203.0.113.7".into(),
                    port: 40122,
                }),
            });
        }
        Ok(Some(fake.pod.clone()))
    }

    fn delete_pod(&self, id: &str) -> Result<(), CloudError> {
        let mut state = self.0.borrow_mut();
        if let Some(status) = state.delete_error {
            return Err(http(status));
        }
        if !state.delete_ignored {
            state.pods.retain(|fake| fake.pod.id != id);
        }
        Ok(())
    }

    fn pod_billing_usd(&self, _: &str) -> Result<Option<f64>, CloudError> {
        Ok(self.0.borrow().billing)
    }

    fn account(&self) -> Result<Option<String>, CloudError> {
        Ok(self.0.borrow().account.clone())
    }
}
