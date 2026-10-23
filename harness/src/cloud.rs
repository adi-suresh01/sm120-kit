use std::cell::Cell;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::time::Duration;

use jiff::fmt::rfc2822::DateTimeParser;
use jiff::{SignedDuration, Timestamp};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use ureq::http::{HeaderMap, Response};
use ureq::{Agent, Body};

use crate::clock::Clock;
use crate::ledger::ADOPT_WINDOW;
use crate::spec::{CloudTier, GpuChoice, PodSpec};

const BACKOFF_S: [u64; 5] = [1, 2, 4, 8, 16];
/// A request that has not finished in this long is treated as lost; for a create, the name lookup takes over.
const HTTP_TIMEOUT_S: u64 = 60;
/// A wait the server asks for is capped: an hourly window would stall `down`, and past this the pod watchdog is
/// the better bet.
pub const MAX_RETRY_AFTER_S: u64 = 60;
/// Lower for creates: their whole retry run has to end inside the adopt window, or a reaper in another harness
/// process could release the reservation of a create that is still in flight.
pub const MAX_CREATE_RETRY_AFTER_S: u64 = 30;
/// Every create attempt running to the timeout, plus every capped wait between them.
pub const CREATE_WORST_CASE_S: u64 =
    (BACKOFF_S.len() as u64 + 1) * HTTP_TIMEOUT_S + BACKOFF_S.len() as u64 * MAX_CREATE_RETRY_AFTER_S;
const _: () = assert!(CREATE_WORST_CASE_S < ADOPT_WINDOW.as_secs() as u64);
/// RunPod returns up to 1000 pods a page, so 100 pages is far past any real account; more means a loop.
const MAX_POD_PAGES: usize = 100;

#[derive(Debug)]
pub enum CloudError {
    /// The request may or may not have reached RunPod.
    Network(String),
    Http {
        status: u16,
        detail: String,
        retry_after_s: Option<u64>,
    },
    /// The body arrived whole but is not the JSON the spec promises.
    Decode(String),
    /// Replies that each decode but together break the protocol, such as a pagination cursor that repeats.
    Protocol(String),
}

impl CloudError {
    pub fn retryable(&self) -> bool {
        match self {
            Self::Network(_) => true,
            Self::Http { status, .. } => *status == 429 || *status >= 500,
            Self::Decode(_) | Self::Protocol(_) => false,
        }
    }

    pub fn rate_limited(&self) -> bool {
        matches!(self, Self::Http { status: 429, .. })
    }

    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Http { status, .. } => Some(*status),
            Self::Network(_) | Self::Decode(_) | Self::Protocol(_) => None,
        }
    }

    /// After a create fails this way RunPod may still have made the pod, so the caller must look it up by name
    /// instead of creating again.
    pub fn create_outcome_unknown(&self) -> bool {
        match self {
            Self::Network(_) | Self::Decode(_) | Self::Protocol(_) => true,
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
            Self::Protocol(e) => write!(f, "RunPod API broke its protocol: {e}"),
        }
    }
}

impl std::error::Error for CloudError {}

/// Retries with backoff while `retry_on` holds. A wait the server asked for replaces the backoff step, within
/// `[step, max_wait_s]`.
pub fn with_retry<T>(
    sleep: &dyn Fn(Duration),
    retry_on: impl Fn(&CloudError) -> bool,
    max_wait_s: u64,
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
                    } => (*after).clamp(base, max_wait_s.max(base)),
                    _ => base,
                };
                sleep(Duration::from_secs(wait));
            }
            result => return result,
        }
    }
}

/// How long the server asked a rate-limited client to wait: `Retry-After`, else `RateLimit-Reset`, else the
/// longest reset among the exhausted windows of the structured `RateLimit` header the OpenAPI spec documents
/// (`"minute";r=0;t=12, "hour";r=2800;t=1812`).
pub fn server_wait(retry_after: Option<&str>, reset: Option<&str>, rate_limit: Option<&str>) -> Option<u64> {
    let seconds = |value: &str| value.trim().parse::<u64>().ok();
    retry_after
        .and_then(seconds)
        .or_else(|| reset.and_then(seconds))
        .or_else(|| {
            rate_limit?
                .split(',')
                .filter_map(|member| {
                    let params: BTreeMap<&str, &str> = member
                        .split(';')
                        .skip(1)
                        .filter_map(|param| param.trim().split_once('='))
                        .collect();
                    (params.get("r").and_then(|r| seconds(r)) == Some(0))
                        .then(|| params.get("t").and_then(|t| seconds(t)))
                        .flatten()
                })
                .max()
        })
}

pub trait Cloud {
    fn gpu_catalog(&self, cloud: CloudTier, count: u32, min_cuda: &str) -> Result<Vec<CatalogGpu>, CloudError>;
    fn create_pod(&self, request: &CreatePodRequest) -> Result<Pod, CloudError>;
    /// Every page: an error rather than a partial list, since callers read absence as "gone".
    fn list_pods(&self) -> Result<Vec<Pod>, CloudError>;
    /// `None` on 404, which RunPod also answers for a pod this key cannot see: never proof on its own.
    fn get_pod(&self, id: &str) -> Result<Option<Pod>, CloudError>;
    /// As `get_pod`, in one attempt without retries, for a check inside a loop that has to stay prompt.
    fn get_pod_once(&self, id: &str) -> Result<Option<Pod>, CloudError> {
        self.get_pod(id)
    }
    /// Also succeeds on 404, for the same reason not proof that the pod is gone.
    fn delete_pod(&self, id: &str) -> Result<(), CloudError>;
    fn pod_billing_usd(&self, id: &str) -> Result<Option<f64>, CloudError>;
    /// The RunPod organization the key belongs to, or `None` when the key cannot tell (no organization, or no
    /// membership scope).
    fn account(&self) -> Result<Option<String>, CloudError>;
    /// RunPod's clock minus the laptop's, from the `Date` header of the last reply, if any reply had one.
    fn clock_skew(&self) -> Option<SignedDuration>;
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

/// A GPU choice with a live price: what a reservation is made for.
#[derive(Debug, Clone, PartialEq)]
pub struct Offer {
    pub gpu: String,
    pub cloud: CloudTier,
    pub count: u32,
    pub rate_usd_hr: f64,
    /// The in-stock data centers outside the avoid list, in catalog order; empty when nothing is avoided.
    pub data_centers: Vec<String>,
    /// Avoided ids the catalog names for no data center of this GPU. The list filtered nothing for them, which a
    /// data center without stock and an id RunPod does not use both explain, so `quote` and `up` say so.
    pub unlisted_avoided: Vec<String>,
}

/// With nothing to avoid, the create sends no list and RunPod places the pod anywhere it has stock.
fn allowed_data_centers(gpu: &CatalogGpu, avoid: &[String]) -> Vec<String> {
    if avoid.is_empty() {
        return Vec::new();
    }
    gpu.data_centers
        .iter()
        .filter(|dc| dc.availability.in_stock() && !avoid.contains(&dc.id))
        .map(|dc| dc.id.clone())
        .collect()
}

pub fn offer_for(choice: &GpuChoice, count: u32, avoid: &[String], catalog: &[CatalogGpu]) -> Option<Offer> {
    let gpu = catalog.iter().find(|gpu| gpu.id == choice.id)?;
    let price = gpu.price.get(choice.cloud).filter(|price| *price > 0.0)?;
    let max_count = gpu.max_count.get(choice.cloud)?;
    let data_centers = allowed_data_centers(gpu, avoid);
    // Stock that sits only in avoided data centers is no stock for this issue.
    let placeable = avoid.is_empty() || !data_centers.is_empty();
    (max_count >= count && in_stock(gpu, choice.cloud) && placeable).then(|| Offer {
        gpu: choice.id.clone(),
        cloud: choice.cloud,
        count,
        rate_usd_hr: price * f64::from(count),
        data_centers,
        unlisted_avoided: avoid
            .iter()
            .filter(|id| gpu.data_centers.iter().all(|dc| dc.id != **id))
            .cloned()
            .collect(),
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
            offer_for(choice, pod.count, &pod.avoid_data_centers, catalog)
        })
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[serde(rename_all = "camelCase")]
pub struct Pod {
    pub id: String,
    pub name: String,
    pub status: PodStatus,
    #[serde(default)]
    pub cost: Option<f64>,
    #[serde(default)]
    pub ssh: Option<PodSsh>,
    #[serde(default)]
    pub runtime: Option<PodRuntime>,
    /// Null until the scheduler places the pod, as in the reply to a create.
    #[serde(default)]
    pub data_center_id: Option<String>,
}

impl Pod {
    pub fn ssh_direct(&self) -> Option<&SshEndpoint> {
        self.ssh.as_ref()?.direct.as_ref()
    }

    /// RunPod reports RUNNING while an uncached image still pulls, before any container exists; only a runtime
    /// with an uptime shows one.
    pub fn container(&self) -> Option<&PodRuntime> {
        self.runtime.as_ref().filter(|runtime| runtime.uptime.is_some())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PodRuntime {
    /// A float, so a fractional uptime cannot fail the decode of the whole pod and end it.
    #[serde(default)]
    pub uptime: Option<f64>,
    /// Kept as RunPod sent it: only error messages read it, so a change in its shape must not fail the decode.
    #[serde(default)]
    pub ports: serde_json::Value,
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
    /// The v2 schema calls these "preferred data centers for placement", so `up` also checks where the pod landed.
    /// Left out when empty, so an issue that avoids nothing sends the body it always did.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub data_center_ids: Vec<String>,
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
    skew: Cell<Option<SignedDuration>>,
}

impl<'a> RunpodRest<'a> {
    pub fn new(base: &str, api_key: &str, clock: &'a dyn Clock) -> Self {
        let config = Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(HTTP_TIMEOUT_S)))
            .build();
        Self {
            agent: Agent::new_with_config(config),
            base: base.trim_end_matches('/').to_string(),
            authorization: format!("Bearer {api_key}"),
            clock,
            skew: Cell::new(None),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    /// Notes the server's clock from every reply, error replies included.
    fn observe(&self, response: Result<Response<Body>, ureq::Error>) -> Result<Response<Body>, CloudError> {
        let response = response.map_err(|e| CloudError::Network(e.to_string()))?;
        let date = header(response.headers(), "date").and_then(http_date);
        if let Some(server) = date {
            self.skew.set(Some(server.duration_since(self.clock.now())));
        }
        Ok(response)
    }

    fn get_json<T: DeserializeOwned>(&self, path: &str, query: &[(&str, &str)]) -> Result<T, CloudError> {
        with_retry(
            &|d| self.clock.sleep(d),
            CloudError::retryable,
            MAX_RETRY_AFTER_S,
            || self.get_json_once(path, query),
        )
    }

    fn get_json_once<T: DeserializeOwned>(&self, path: &str, query: &[(&str, &str)]) -> Result<T, CloudError> {
        let mut request = self
            .agent
            .get(self.url(path))
            .header("Authorization", &self.authorization);
        for (key, value) in query {
            request = request.query(*key, *value);
        }
        read_json(self.observe(request.call())?)
    }
}

fn found_or_gone(result: Result<Pod, CloudError>) -> Result<Option<Pod>, CloudError> {
    match result {
        Ok(pod) => Ok(Some(pod)),
        Err(e) if e.status() == Some(404) => Ok(None),
        Err(e) => Err(e),
    }
}

/// HTTP dates (`Fri, 16 Oct 2026 17:03:00 GMT`) are the RFC 2822 form with a GMT zone.
fn http_date(value: &str) -> Option<Timestamp> {
    DateTimeParser::new().parse_timestamp(value.trim()).ok()
}

fn header<'h>(headers: &'h HeaderMap, name: &str) -> Option<&'h str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

fn ensure_success(response: &mut Response<Body>) -> Result<(), CloudError> {
    let status = response.status().as_u16();
    if (200..300).contains(&status) {
        return Ok(());
    }
    let headers = response.headers();
    let retry_after_s = server_wait(
        header(headers, "retry-after"),
        header(headers, "ratelimit-reset"),
        header(headers, "ratelimit"),
    );
    let body = response.body_mut().read_to_string().unwrap_or_default();
    Err(CloudError::Http {
        status,
        detail: problem_detail(&body),
        retry_after_s,
    })
}

/// A body cut off by a timeout or a reset is a network failure, which a GET may retry; only a body that arrived
/// whole and is not the expected JSON is a decode failure.
fn read_json<T: DeserializeOwned>(mut response: Response<Body>) -> Result<T, CloudError> {
    ensure_success(&mut response)?;
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|e| CloudError::Network(format!("reading the reply: {e}")))?;
    serde_json::from_str(&text).map_err(|e| CloudError::Decode(e.to_string()))
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
        let sleep = |d| self.clock.sleep(d);
        with_retry(&sleep, CloudError::rate_limited, MAX_CREATE_RETRY_AFTER_S, || {
            let response = self
                .agent
                .post(self.url("/v2/pods"))
                .header("Authorization", &self.authorization)
                .send_json(request);
            read_json(self.observe(response)?)
        })
    }

    fn list_pods(&self) -> Result<Vec<Pod>, CloudError> {
        let mut pods = Vec::new();
        let mut cursor: Option<String> = None;
        let mut seen = HashSet::new();
        for _ in 0..MAX_POD_PAGES {
            let query: Vec<(&str, &str)> = cursor.iter().map(|c| ("cursor", c.as_str())).collect();
            let page: PodPage = self.get_json("/v2/pods", &query)?;
            pods.extend(page.pods);
            match page.pagination {
                Some(Pagination {
                    has_next_page: true,
                    next_cursor: Some(next),
                }) => {
                    if !seen.insert(next.clone()) {
                        return Err(CloudError::Protocol(format!(
                            "pod list cursor {next:?} came back twice"
                        )));
                    }
                    cursor = Some(next);
                }
                Some(Pagination {
                    has_next_page: true,
                    next_cursor: None,
                }) => {
                    return Err(CloudError::Protocol(
                        "pod list says more pages follow but gives no cursor".into(),
                    ));
                }
                _ => return Ok(pods),
            }
        }
        Err(CloudError::Protocol(format!(
            "pod list still had more after {MAX_POD_PAGES} pages"
        )))
    }

    fn get_pod(&self, id: &str) -> Result<Option<Pod>, CloudError> {
        found_or_gone(self.get_json(&format!("/v2/pods/{id}"), &[]))
    }

    fn get_pod_once(&self, id: &str) -> Result<Option<Pod>, CloudError> {
        found_or_gone(self.get_json_once(&format!("/v2/pods/{id}"), &[]))
    }

    fn delete_pod(&self, id: &str) -> Result<(), CloudError> {
        let sleep = |d| self.clock.sleep(d);
        let result = with_retry(&sleep, CloudError::retryable, MAX_RETRY_AFTER_S, || {
            let request = self
                .agent
                .delete(self.url(&format!("/v2/pods/{id}")))
                .header("Authorization", &self.authorization);
            ensure_success(&mut self.observe(request.call())?)
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

    /// A caller outside any organization gets 404 per the OpenAPI spec, and since 2026-10-22 also 400 "Organizations
    /// are not enabled for this user"; a key without `membership:read` gets 403, and a rejected key 401. Each means the
    /// account cannot be fingerprinted, not that something failed.
    fn account(&self) -> Result<Option<String>, CloudError> {
        match self.get_json::<Organization>("/v2/organization", &[]) {
            Ok(org) => Ok(Some(org.id)),
            Err(e) if matches!(e.status(), Some(400 | 401 | 403 | 404)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn clock_skew(&self) -> Option<SignedDuration> {
        self.skew.get()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use serde_json::json;

    use super::*;

    fn http(status: u16, retry_after_s: Option<u64>) -> CloudError {
        CloudError::Http {
            status,
            detail: String::new(),
            retry_after_s,
        }
    }

    fn run_with_failures(failures: Vec<CloudError>) -> (Result<u32, CloudError>, Vec<u64>) {
        run_capped(failures, CloudError::retryable, MAX_RETRY_AFTER_S)
    }

    fn run_capped(
        failures: Vec<CloudError>,
        retry_on: fn(&CloudError) -> bool,
        max_wait_s: u64,
    ) -> (Result<u32, CloudError>, Vec<u64>) {
        let slept = RefCell::new(Vec::new());
        let failures = RefCell::new(failures);
        let mut calls = 0;
        let result = with_retry(&|d| slept.borrow_mut().push(d.as_secs()), retry_on, max_wait_s, || {
            calls += 1;
            match failures.borrow_mut().pop() {
                Some(e) => Err(e),
                None => Ok(calls),
            }
        });
        (result, slept.into_inner())
    }

    #[test]
    fn create_retries_end_inside_the_adopt_window() {
        // Worst case: every attempt hits the timeout and every 429 asks for an hour.
        let (result, slept) = run_capped(
            (0..6).map(|_| http(429, Some(3600))).collect(),
            CloudError::rate_limited,
            MAX_CREATE_RETRY_AFTER_S,
        );
        assert!(result.is_err());
        let total = slept.iter().sum::<u64>() + 6 * HTTP_TIMEOUT_S;
        assert!(total <= CREATE_WORST_CASE_S, "{total}");
        assert!(CREATE_WORST_CASE_S < ADOPT_WINDOW.as_secs() as u64);
    }

    #[test]
    fn the_server_wait_comes_from_retry_after_then_the_ratelimit_headers() {
        assert_eq!(server_wait(Some("7"), Some("50"), None), Some(7));
        assert_eq!(server_wait(None, Some(" 50 "), None), Some(50));
        let limits = "\"minute\";r=0;t=12, \"hour\";r=2800;t=1812, \"day\";r=0;t=45";
        assert_eq!(server_wait(None, None, Some(limits)), Some(45));
        assert_eq!(server_wait(Some("soon"), None, Some("\"minute\";r=5;t=12")), None);
        assert_eq!(server_wait(None, None, None), None);
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
        assert!(CloudError::Protocol("loop".into()).create_outcome_unknown());
        assert!(http(502, None).create_outcome_unknown());
        assert!(!http(400, None).create_outcome_unknown());
        assert!(!http(429, None).create_outcome_unknown());
    }

    #[test]
    fn a_running_pod_has_a_container_only_once_its_runtime_has_an_uptime() {
        let running: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/pod_running.json")).unwrap();
        let pod: Pod = serde_json::from_value(running.clone()).unwrap();
        let container = pod.container().unwrap();
        assert_eq!(container.uptime, Some(121.0));
        assert_eq!(
            container.ports,
            json!([{"private": 22, "public": 40122, "type": "tcp", "ip": "203.0.113.7"}])
        );
        let with_runtime = |runtime: Option<serde_json::Value>| -> Pod {
            let mut reply = running.clone();
            match runtime {
                Some(runtime) => reply["runtime"] = runtime,
                None => {
                    reply.as_object_mut().unwrap().remove("runtime");
                }
            }
            serde_json::from_value(reply).unwrap()
        };
        let started = with_runtime(Some(json!({"uptime": 3, "ports": null})));
        assert_eq!(started.container().unwrap().ports, serde_json::Value::Null);
        for runtime in [
            None,
            Some(json!(null)),
            Some(json!({})),
            Some(json!({"uptime": null, "ports": null})),
        ] {
            let pod = with_runtime(runtime.clone());
            assert_eq!(pod.status, PodStatus::Running);
            assert!(pod.container().is_none(), "{runtime:?}");
        }
    }

    const SECURE_X1: &str = include_str!("../tests/fixtures/catalog_secure_x1.json");
    const COMMUNITY_X1: &str = include_str!("../tests/fixtures/catalog_community_x1.json");
    const SECURE_X2: &str = include_str!("../tests/fixtures/catalog_secure_x2.json");
    const SERVER: &str = "NVIDIA RTX PRO 6000 Blackwell Server Edition";
    const WORKSTATION: &str = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition";

    fn gpus(catalog: &str) -> Vec<CatalogGpu> {
        let page: serde_json::Value = serde_json::from_str(catalog).unwrap();
        serde_json::from_value(page["gpus"].clone()).unwrap()
    }

    fn offer(catalog: &str, id: &str, cloud: CloudTier, count: u32) -> Option<Offer> {
        let choice = GpuChoice { id: id.into(), cloud };
        offer_for(&choice, count, &[], &gpus(catalog))
    }

    fn secure_avoiding(gpus: &[CatalogGpu], id: &str, avoid: &[&str]) -> Option<Offer> {
        let choice = GpuChoice {
            id: id.into(),
            cloud: CloudTier::Secure,
        };
        let avoid: Vec<String> = avoid.iter().map(|dc| dc.to_string()).collect();
        offer_for(&choice, 1, &avoid, gpus)
    }

    #[test]
    fn stock_only_in_avoided_data_centers_is_no_offer() {
        let gpu: CatalogGpu = serde_json::from_value(json!({
            "id": SERVER, "price": {"secure": 2.49}, "maxCount": {"secure": 8}, "availability": "HIGH",
            "dataCenters": [{"id": "EUR-IS-2", "availability": "HIGH"}, {"id": "US-NC-2", "availability": "NONE"}],
        }))
        .unwrap();
        // US-NC-2 is listed but has none free, so EUR-IS-2 is the only stock.
        let one = [gpu];
        assert!(secure_avoiding(&one, SERVER, &["EUR-IS-2"]).is_none());
        assert_eq!(secure_avoiding(&one, SERVER, &[]).unwrap().rate_usd_hr, 2.49);
        assert!(secure_avoiding(&gpus(SECURE_X1), SERVER, &["CA-MTL-3", "EU-CZ-1"]).is_none());
    }

    #[test]
    fn an_offer_carries_the_in_stock_data_centers_left_after_the_avoid_list() {
        let x1 = gpus(SECURE_X1);
        let offer = secure_avoiding(&x1, SERVER, &["EU-CZ-1"]).unwrap();
        assert_eq!(
            (offer.data_centers, offer.rate_usd_hr),
            (vec!["CA-MTL-3".to_string()], 2.49)
        );
        assert!(offer.unlisted_avoided.is_empty());
        let nothing_avoided = secure_avoiding(&x1, SERVER, &[]).unwrap();
        assert!(nothing_avoided.data_centers.is_empty() && nothing_avoided.unlisted_avoided.is_empty());
        let gpu: CatalogGpu = serde_json::from_value(json!({
            "id": SERVER, "price": {"secure": 2.49}, "maxCount": {"secure": 8}, "availability": "HIGH",
            "dataCenters": [
                {"id": "US-NC-1", "availability": "HIGH"}, {"id": "EUR-IS-2", "availability": "LOW"},
                {"id": "CA-MTL-3", "availability": "MEDIUM"},
            ],
        }))
        .unwrap();
        let offer = secure_avoiding(&[gpu], SERVER, &["EUR-IS-2"]).unwrap();
        assert_eq!(offer.data_centers, ["US-NC-1", "CA-MTL-3"]);
    }

    #[test]
    fn an_offer_names_the_avoided_ids_the_catalog_lists_for_no_data_center() {
        let x1 = gpus(SECURE_X1);
        let offer = secure_avoiding(&x1, SERVER, &["EUR-IS-3", "EU-CZ-1", "US-NC-1"]).unwrap();
        assert_eq!(offer.data_centers, ["CA-MTL-3"]);
        assert_eq!(offer.unlisted_avoided, ["EUR-IS-3", "US-NC-1"]);
        let offer = secure_avoiding(&x1, WORKSTATION, &["EUR-IS-3"]).unwrap();
        assert_eq!(
            (offer.data_centers, offer.unlisted_avoided),
            (
                vec!["EU-CZ-1".to_string(), "EU-RO-1".to_string()],
                vec!["EUR-IS-3".to_string()]
            )
        );
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
