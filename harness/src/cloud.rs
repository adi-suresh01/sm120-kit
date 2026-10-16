use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use ureq::http::Response;
use ureq::{Agent, Body};

use crate::clock::Clock;
use crate::ledger::Offer;
use crate::spec::{CloudTier, GpuChoice, PodSpec};

const BACKOFF_S: [u64; 5] = [1, 2, 4, 8, 16];
/// A Retry-After for an hourly window would stall `down`; past this the pod watchdog is the better bet.
const MAX_RETRY_AFTER_S: u64 = 60;

#[derive(Debug)]
pub enum CloudError {
    /// The request may or may not have reached RunPod.
    Network(String),
    Http {
        status: u16,
        detail: String,
        retry_after_s: Option<u64>,
    },
    Decode(String),
}

impl CloudError {
    pub fn retryable(&self) -> bool {
        match self {
            Self::Network(_) => true,
            Self::Http { status, .. } => *status == 429 || *status >= 500,
            Self::Decode(_) => false,
        }
    }

    pub fn rate_limited(&self) -> bool {
        matches!(self, Self::Http { status: 429, .. })
    }

    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Http { status, .. } => Some(*status),
            Self::Network(_) | Self::Decode(_) => None,
        }
    }

    /// After a create fails this way RunPod may still have made the pod, so the caller must look it up by name
    /// instead of creating again.
    pub fn create_outcome_unknown(&self) -> bool {
        match self {
            Self::Network(_) | Self::Decode(_) => true,
            Self::Http { status, .. } => *status >= 500,
        }
    }
}

impl fmt::Display for CloudError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Network(e) => write!(f, "RunPod API unreachable: {e}"),
            Self::Http { status, detail, .. } => write!(f, "RunPod API returned {status}: {detail}"),
            Self::Decode(e) => write!(f, "RunPod API response did not decode: {e}"),
        }
    }
}

impl std::error::Error for CloudError {}

pub fn with_retry<T>(
    sleep: &dyn Fn(Duration),
    retry_on: impl Fn(&CloudError) -> bool,
    mut call: impl FnMut() -> Result<T, CloudError>,
) -> Result<T, CloudError> {
    let mut backoff = BACKOFF_S.iter();
    loop {
        match call() {
            Err(e) if retry_on(&e) => {
                let Some(&base) = backoff.next() else { return Err(e) };
                let wait = match &e {
                    CloudError::Http {
                        retry_after_s: Some(after),
                        ..
                    } => (*after).clamp(base, MAX_RETRY_AFTER_S),
                    _ => base,
                };
                sleep(Duration::from_secs(wait));
            }
            result => return result,
        }
    }
}

pub trait Cloud {
    fn gpu_catalog(&self, cloud: CloudTier, count: u32, min_cuda: &str) -> Result<Vec<CatalogGpu>, CloudError>;
    fn create_pod(&self, request: &CreatePodRequest) -> Result<Pod, CloudError>;
    /// Every page: an error rather than a partial list, since callers read absence as "gone".
    fn list_pods(&self) -> Result<Vec<Pod>, CloudError>;
    /// `None` on 404, which RunPod also answers for a pod this key cannot see: never proof on its own.
    fn get_pod(&self, id: &str) -> Result<Option<Pod>, CloudError>;
    /// Also succeeds on 404, for the same reason not proof that the pod is gone.
    fn delete_pod(&self, id: &str) -> Result<(), CloudError>;
    fn pod_billing_usd(&self, id: &str) -> Result<Option<f64>, CloudError>;
    /// The RunPod organization the key belongs to, or `None` when the key cannot tell (no organization, or no
    /// membership scope).
    fn account(&self) -> Result<Option<String>, CloudError>;
}

#[derive(Debug, Clone, Deserialize)]
pub struct PerCloud<T> {
    pub secure: Option<T>,
    pub community: Option<T>,
}

impl<T: Copy> PerCloud<T> {
    pub fn get(&self, cloud: CloudTier) -> Option<T> {
        match cloud {
            CloudTier::Secure => self.secure,
            CloudTier::Community => self.community,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Availability {
    None,
    Low,
    Medium,
    High,
    #[serde(other)]
    Unknown,
}

impl Availability {
    fn in_stock(self) -> bool {
        matches!(self, Self::Low | Self::Medium | Self::High)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogGpu {
    pub id: String,
    pub price: PerCloud<f64>,
    pub max_count: PerCloud<u32>,
    #[serde(default)]
    pub availability: Option<Availability>,
    #[serde(default)]
    pub data_centers: Vec<DataCenterStock>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataCenterStock {
    pub id: String,
    pub availability: Availability,
}

pub fn in_stock(gpu: &CatalogGpu, cloud: CloudTier) -> bool {
    match cloud {
        // Only the per-data-center list follows the requested GPU count; the top-level level can say LOW while
        // no data center has that many free on one machine.
        CloudTier::Secure => gpu.data_centers.iter().any(|dc| dc.availability.in_stock()),
        // Community hosts sit outside RunPod's data centers, so the catalog never lists any for them.
        CloudTier::Community => gpu.availability.is_some_and(Availability::in_stock),
    }
}

pub fn offer_for(choice: &GpuChoice, count: u32, catalog: &[CatalogGpu]) -> Option<Offer> {
    let gpu = catalog.iter().find(|gpu| gpu.id == choice.id)?;
    let price = gpu.price.get(choice.cloud).filter(|price| *price > 0.0)?;
    let max_count = gpu.max_count.get(choice.cloud)?;
    (max_count >= count && in_stock(gpu, choice.cloud)).then(|| Offer {
        gpu: choice.id.clone(),
        cloud: choice.cloud,
        count,
        rate_usd_hr: price * f64::from(count),
    })
}

/// Every in-stock choice, in preference order. One catalog read per cloud tier that the preferences use.
pub fn find_offers(api: &dyn Cloud, pod: &PodSpec) -> Result<Vec<Offer>, CloudError> {
    let catalog_for = |tier: CloudTier| -> Result<Vec<CatalogGpu>, CloudError> {
        if pod.gpus.iter().any(|choice| choice.cloud == tier) {
            api.gpu_catalog(tier, pod.count, &pod.min_cuda)
        } else {
            Ok(Vec::new())
        }
    };
    let secure = catalog_for(CloudTier::Secure)?;
    let community = catalog_for(CloudTier::Community)?;
    Ok(pod
        .gpus
        .iter()
        .filter_map(|choice| {
            let catalog = match choice.cloud {
                CloudTier::Secure => &secure,
                CloudTier::Community => &community,
            };
            offer_for(choice, pod.count, catalog)
        })
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PodStatus {
    Provisioning,
    Starting,
    Running,
    Exited,
    Error,
    Terminated,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Pod {
    pub id: String,
    pub name: String,
    pub status: PodStatus,
    #[serde(default)]
    pub cost: Option<f64>,
    #[serde(default)]
    pub ssh: Option<PodSsh>,
}

impl Pod {
    pub fn ssh_direct(&self) -> Option<&SshEndpoint> {
        self.ssh.as_ref()?.direct.as_ref()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PodSsh {
    #[serde(default)]
    pub direct: Option<SshEndpoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SshEndpoint {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreatePodRequest {
    pub name: String,
    pub image: String,
    pub entrypoint: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub ports: Vec<String>,
    pub disk: u32,
    pub mounts: Mounts,
    pub cloud: CloudTier,
    pub gpu: GpuRequest,
    pub start_ssh: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Mounts {
    pub persistent: PersistentMount,
}

#[derive(Debug, Clone, Serialize)]
pub struct PersistentMount {
    pub size: u32,
    pub path: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuRequest {
    pub id: String,
    pub count: u32,
    pub min_cuda_version: String,
}

pub const API_BASE: &str = "https://api.runpod.io";

#[derive(Deserialize)]
struct CatalogPage {
    gpus: Vec<CatalogGpu>,
}

#[derive(Deserialize)]
struct PodPage {
    pods: Vec<Pod>,
    #[serde(default)]
    pagination: Option<Pagination>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Pagination {
    has_next_page: bool,
    next_cursor: Option<String>,
}

#[derive(Deserialize)]
struct BillingPage {
    records: Vec<serde_json::Value>,
    metadata: BillingMetadata,
}

#[derive(Deserialize)]
struct BillingMetadata {
    totals: BillingTotals,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BillingTotals {
    total_amount: f64,
}

#[derive(Deserialize)]
struct Organization {
    id: String,
}

#[derive(Deserialize)]
struct Problem {
    title: Option<String>,
    detail: Option<String>,
    #[serde(default)]
    errors: Vec<String>,
}

/// RunPod REST v2 over blocking HTTP. Deliberately not `Debug`: it holds the API key.
pub struct RunpodRest<'a> {
    agent: Agent,
    base: String,
    authorization: String,
    clock: &'a dyn Clock,
}

impl<'a> RunpodRest<'a> {
    pub fn new(base: &str, api_key: &str, clock: &'a dyn Clock) -> Self {
        let config = Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(60)))
            .build();
        Self {
            agent: Agent::new_with_config(config),
            base: base.trim_end_matches('/').to_string(),
            authorization: format!("Bearer {api_key}"),
            clock,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn get_json<T: DeserializeOwned>(&self, path: &str, query: &[(&str, &str)]) -> Result<T, CloudError> {
        with_retry(&|d| self.clock.sleep(d), CloudError::retryable, || {
            let mut request = self
                .agent
                .get(self.url(path))
                .header("Authorization", &self.authorization);
            for (key, value) in query {
                request = request.query(*key, *value);
            }
            read_json(request.call().map_err(network)?)
        })
    }
}

fn network(e: ureq::Error) -> CloudError {
    CloudError::Network(e.to_string())
}

fn ensure_success(response: &mut Response<Body>) -> Result<(), CloudError> {
    let status = response.status().as_u16();
    if (200..300).contains(&status) {
        return Ok(());
    }
    let retry_after_s = response
        .headers()
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse().ok());
    let body = response.body_mut().read_to_string().unwrap_or_default();
    Err(CloudError::Http {
        status,
        detail: problem_detail(&body),
        retry_after_s,
    })
}

fn read_json<T: DeserializeOwned>(mut response: Response<Body>) -> Result<T, CloudError> {
    ensure_success(&mut response)?;
    response
        .body_mut()
        .read_json()
        .map_err(|e| CloudError::Decode(e.to_string()))
}

fn problem_detail(body: &str) -> String {
    match serde_json::from_str::<Problem>(body) {
        Ok(problem) => {
            let mut detail = [problem.title, problem.detail]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(": ");
            for error in problem.errors {
                detail.push_str("; ");
                detail.push_str(&error);
            }
            detail
        }
        Err(_) => body.chars().take(300).collect(),
    }
}

impl Cloud for RunpodRest<'_> {
    fn gpu_catalog(&self, cloud: CloudTier, count: u32, min_cuda: &str) -> Result<Vec<CatalogGpu>, CloudError> {
        let count = count.to_string();
        let query = [
            ("include", "AVAILABILITY"),
            ("product", "POD"),
            ("count", count.as_str()),
            ("cloud", cloud.as_str()),
            ("minCudaVersion", min_cuda),
        ];
        Ok(self.get_json::<CatalogPage>("/v2/catalog/gpus", &query)?.gpus)
    }

    /// Only a 429 is retried: RunPod rejects it before doing anything, while any other failure may have made a pod.
    fn create_pod(&self, request: &CreatePodRequest) -> Result<Pod, CloudError> {
        with_retry(&|d| self.clock.sleep(d), CloudError::rate_limited, || {
            let response = self
                .agent
                .post(self.url("/v2/pods"))
                .header("Authorization", &self.authorization)
                .send_json(request)
                .map_err(network)?;
            read_json(response)
        })
    }

    fn list_pods(&self) -> Result<Vec<Pod>, CloudError> {
        let mut pods = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let query: Vec<(&str, &str)> = cursor.iter().map(|c| ("cursor", c.as_str())).collect();
            let page: PodPage = self.get_json("/v2/pods", &query)?;
            pods.extend(page.pods);
            match page.pagination {
                Some(Pagination {
                    has_next_page: true,
                    next_cursor: Some(next),
                }) => cursor = Some(next),
                _ => return Ok(pods),
            }
        }
    }

    fn get_pod(&self, id: &str) -> Result<Option<Pod>, CloudError> {
        match self.get_json(&format!("/v2/pods/{id}"), &[]) {
            Ok(pod) => Ok(Some(pod)),
            Err(e) if e.status() == Some(404) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn delete_pod(&self, id: &str) -> Result<(), CloudError> {
        let result = with_retry(&|d| self.clock.sleep(d), CloudError::retryable, || {
            let mut response = self
                .agent
                .delete(self.url(&format!("/v2/pods/{id}")))
                .header("Authorization", &self.authorization)
                .call()
                .map_err(network)?;
            ensure_success(&mut response)
        });
        match result {
            Err(e) if e.status() == Some(404) => Ok(()),
            other => other,
        }
    }

    fn pod_billing_usd(&self, id: &str) -> Result<Option<f64>, CloudError> {
        let page: BillingPage = self.get_json("/v2/billing/pods", &[("podId", id), ("bucketSize", "day")])?;
        Ok((!page.records.is_empty()).then_some(page.metadata.totals.total_amount))
    }

    /// The OpenAPI spec answers 404 for a caller outside any organization and 403 for a key without
    /// `membership:read`; both mean the account cannot be fingerprinted, not that something failed.
    fn account(&self) -> Result<Option<String>, CloudError> {
        match self.get_json::<Organization>("/v2/organization", &[]) {
            Ok(org) => Ok(Some(org.id)),
            Err(e) if matches!(e.status(), Some(401 | 403 | 404)) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    fn http(status: u16, retry_after_s: Option<u64>) -> CloudError {
        CloudError::Http {
            status,
            detail: String::new(),
            retry_after_s,
        }
    }

    fn run_with_failures(failures: Vec<CloudError>) -> (Result<u32, CloudError>, Vec<u64>) {
        let slept = RefCell::new(Vec::new());
        let failures = RefCell::new(failures);
        let mut calls = 0;
        let result = with_retry(&|d| slept.borrow_mut().push(d.as_secs()), CloudError::retryable, || {
            calls += 1;
            match failures.borrow_mut().pop() {
                Some(e) => Err(e),
                None => Ok(calls),
            }
        });
        (result, slept.into_inner())
    }

    #[test]
    fn retries_network_429_and_5xx_with_backoff() {
        let (result, slept) =
            run_with_failures(vec![http(503, None), http(429, None), CloudError::Network("x".into())]);
        assert_eq!(result.unwrap(), 4);
        assert_eq!(slept, [1, 2, 4]);
    }

    #[test]
    fn gives_up_after_five_retries() {
        let (result, slept) = run_with_failures((0..6).map(|_| http(502, None)).collect());
        assert_eq!(result.unwrap_err().status(), Some(502));
        assert_eq!(slept, [1, 2, 4, 8, 16]);
    }

    #[test]
    fn never_retries_auth_or_other_client_errors() {
        for status in [400, 401, 403, 404, 422] {
            let (result, slept) = run_with_failures(vec![http(status, None)]);
            assert_eq!(result.unwrap_err().status(), Some(status));
            assert!(slept.is_empty(), "{status} was retried");
        }
    }

    #[test]
    fn honors_retry_after_within_a_minute() {
        let (_, slept) = run_with_failures(vec![http(429, Some(3600)), http(429, Some(12))]);
        assert_eq!(slept, [12, 60]);
    }

    #[test]
    fn create_outcome_is_unknown_for_network_5xx_and_garbled_replies() {
        assert!(CloudError::Network("reset".into()).create_outcome_unknown());
        assert!(CloudError::Decode("eof".into()).create_outcome_unknown());
        assert!(http(502, None).create_outcome_unknown());
        assert!(!http(400, None).create_outcome_unknown());
        assert!(!http(429, None).create_outcome_unknown());
    }

    const SECURE_X1: &str = include_str!("../tests/fixtures/catalog_secure_x1.json");
    const COMMUNITY_X1: &str = include_str!("../tests/fixtures/catalog_community_x1.json");
    const SECURE_X2: &str = include_str!("../tests/fixtures/catalog_secure_x2.json");
    const SERVER: &str = "NVIDIA RTX PRO 6000 Blackwell Server Edition";
    const WORKSTATION: &str = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition";

    fn offer(catalog: &str, id: &str, cloud: CloudTier, count: u32) -> Option<Offer> {
        let page: serde_json::Value = serde_json::from_str(catalog).unwrap();
        let gpus: Vec<CatalogGpu> = serde_json::from_value(page["gpus"].clone()).unwrap();
        let choice = GpuChoice { id: id.into(), cloud };
        offer_for(&choice, count, &gpus)
    }

    #[test]
    fn prices_an_in_stock_gpu_per_pod() {
        let one = offer(SECURE_X1, SERVER, CloudTier::Secure, 1).unwrap();
        assert_eq!((one.rate_usd_hr, one.count, one.cloud), (2.49, 1, CloudTier::Secure));
        let two = offer(SECURE_X2, SERVER, CloudTier::Secure, 2).unwrap();
        assert!((two.rate_usd_hr - 4.98).abs() < 1e-9);
    }

    #[test]
    fn secure_stock_comes_from_data_centers_not_the_top_level() {
        // The recorded x2 catalog says LOW for the 5090 but lists no data center with two free.
        assert!(offer(SECURE_X2, "NVIDIA GeForce RTX 5090", CloudTier::Secure, 2).is_none());
        assert!(offer(SECURE_X1, "NVIDIA GeForce RTX 5090", CloudTier::Secure, 1).is_some());
    }

    #[test]
    fn community_stock_comes_from_the_top_level() {
        assert_eq!(
            offer(COMMUNITY_X1, WORKSTATION, CloudTier::Community, 1)
                .unwrap()
                .rate_usd_hr,
            1.69
        );
        assert!(offer(COMMUNITY_X1, SERVER, CloudTier::Community, 1).is_none());
    }

    #[test]
    fn skips_gpus_without_stock_price_or_capacity() {
        assert!(offer(SECURE_X1, "NVIDIA RTX 6000 Ada Generation", CloudTier::Secure, 1).is_none());
        // Max-Q lists a secure price but a secure max count of 0: RunPod does not offer it there.
        let maxq = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition";
        assert!(offer(SECURE_X1, maxq, CloudTier::Secure, 1).is_none());
        assert!(offer(SECURE_X1, WORKSTATION, CloudTier::Secure, 9).is_none());
        assert!(offer(SECURE_X1, "NVIDIA H100 NVL", CloudTier::Secure, 1).is_none());
    }
}
