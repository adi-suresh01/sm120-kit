//! Vast.ai's REST API, as the vastai 1.10 CLI calls it (`vastai/api/*.py`). Vast rents whole machines from many
//! hosts, each listed as an offer at the host's own price, so `up` creates from one offer, not from a GPU type.

use std::cell::Cell;
use std::collections::HashSet;
use std::fs;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::time::Duration;

use jiff::SignedDuration;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use ureq::http::Response;
use ureq::{Agent, Body};

use crate::clock::Clock;
use crate::cloud::{
    CatalogGpu, Cloud, CloudError, CreatePodRequest, HTTP_TIMEOUT_S, MAX_CREATE_RETRY_AFTER_S, MAX_POD_PAGES,
    MAX_RETRY_AFTER_S, Pod, PodRuntime, PodSsh, PodStatus, SshEndpoint, ensure_success_with, found_or_gone, header,
    http_date, read_body_json, with_retry,
};
use crate::ledger::VastPlacement;
use crate::spec::{CloudTier, PodSpec, VastChoice};

pub const API_BASE: &str = "https://console.vast.ai";
/// The CLI's own page size for the instance list.
const INSTANCE_PAGE: &str = "25";
const CHARGE_PAGE: &str = "100";
/// How far back the charge lookup reads. `budget show` settles pods booked as estimates until billing names them,
/// and Vast lists charges by day.
const CHARGE_WINDOW: SignedDuration = SignedDuration::from_hours(62 * 24);
/// Vast filters the search, and the harness checks every offer again, so it asks for more than it keeps.
const SEARCH_LIMIT: u32 = 64;
/// A taken offer falls through to the next cheapest; a few cover the race without reserving for the whole market.
const OFFERS_PER_CHOICE: usize = 5;

/// What Vast's search and create do that RunPod's API has no counterpart for. The calls about pods that exist
/// (list, read, destroy, charges, account) are the `Cloud` half.
pub trait VastApi: Cloud {
    /// One search of on-demand offers. An offer that does not decode is left out; a reply where none decodes is an
    /// error, so a change in Vast's format never reads as "nothing on the market".
    fn offers(&self, query: &OfferQuery) -> Result<Vec<VastOffer>, CloudError>;
    /// The new instance's id. Vast also returns the instance's own API key here; it is never kept.
    fn create_instance(&self, offer_id: u64, request: &VastCreateRequest) -> Result<String, CloudError>;
}

/// One Vast offer: a specific machine at the host's price for the whole of it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct VastOffer {
    #[serde(rename = "id")]
    pub offer_id: u64,
    pub machine_id: u64,
    pub host_id: u64,
    #[serde(rename = "gpu_name")]
    pub gpu: String,
    #[serde(rename = "num_gpus")]
    pub count: u32,
    /// Every GPU plus the disk at the size the search priced, per hour.
    #[serde(rename = "dph_total")]
    pub rate_usd_hr: f64,
    #[serde(default)]
    pub reliability2: Option<f64>,
    #[serde(default)]
    pub reliability: Option<f64>,
    pub disk_space: f64,
    pub inet_down: f64,
    /// Downloads bill on top of the hourly rate, and setup downloads the weights on every box.
    #[serde(default)]
    pub internet_down_cost_per_tb: Option<f64>,
    #[serde(default)]
    pub rentable: Option<bool>,
    #[serde(default)]
    pub rented: Option<bool>,
    #[serde(default)]
    pub cuda_max_good: Option<f64>,
    #[serde(default)]
    pub geolocation: Option<String>,
    /// 1 for a datacenter host, 0 for any other.
    #[serde(default)]
    pub hosting_type: Option<i64>,
}

impl VastOffer {
    pub fn reliability(&self) -> Option<f64> {
        self.reliability2.or(self.reliability)
    }

    /// The ledger's `cloud` field is RunPod's tier; for a Vast pod it records Vast's own split, datacenter hosts
    /// (Vast's "secure cloud") against all others.
    pub fn cloud(&self) -> CloudTier {
        if self.hosting_type == Some(1) {
            CloudTier::Secure
        } else {
            CloudTier::Community
        }
    }

    pub fn placement(&self) -> VastPlacement {
        VastPlacement {
            offer: self.offer_id,
            machine: self.machine_id,
            host: self.host_id,
        }
    }

    pub fn describe(&self) -> String {
        let mut line = format!(
            "{} x{} on Vast machine {} ${:.2}/hr",
            self.gpu, self.count, self.machine_id, self.rate_usd_hr
        );
        if let Some(reliability) = self.reliability() {
            line.push_str(&format!(", reliability {reliability:.3}"));
        }
        if let Some(per_tb) = self.internet_down_cost_per_tb {
            line.push_str(&format!(", downloads ${per_tb:.2}/TB"));
        }
        if let Some(place) = self
            .geolocation
            .as_deref()
            .map(str::trim)
            .filter(|place| !place.is_empty())
        {
            line.push_str(&format!(", {place}"));
        }
        line
    }
}

/// What one Vast choice asks of an offer, for the search and for the check of each offer it returns.
#[derive(Debug, Clone, PartialEq)]
pub struct OfferQuery {
    pub gpu: String,
    pub count: u32,
    pub disk_gb: u32,
    pub max_usd_hr: f64,
    pub min_reliability: f64,
    pub min_inet_down_mbps: f64,
    pub max_download_usd_tb: f64,
    pub min_cuda: f64,
}

/// Vast prices downloads per GB in `inet_down_cost` and reports `internet_down_cost_per_tb` as 1024 times that.
const GB_PER_TB: f64 = 1024.0;

impl OfferQuery {
    pub fn new(choice: &VastChoice, pod: &PodSpec) -> Self {
        Self {
            gpu: choice.id.clone(),
            count: pod.count,
            disk_gb: pod.vast_disk_gb(),
            max_usd_hr: choice.max_usd_hr,
            min_reliability: choice.min_reliability,
            min_inet_down_mbps: choice.min_inet_down_mbps,
            max_download_usd_tb: choice.max_download_usd_tb,
            // The spec validates min_cuda as major.minor, which parses as a float.
            min_cuda: pod.min_cuda.parse().unwrap_or(f64::INFINITY),
        }
    }

    /// The body of `POST /bundles/`: the CLI's default filters plus this choice's limits. `allocated_storage` makes
    /// `dph_total` include the disk at the size the create asks for, so the rate reserved is the rate billed.
    pub fn body(&self) -> Value {
        json!({
            "verified": {"eq": true},
            "external": {"eq": false},
            "rentable": {"eq": true},
            "rented": {"eq": false},
            "gpu_name": {"eq": self.gpu},
            "num_gpus": {"eq": self.count},
            "disk_space": {"gte": self.disk_gb},
            "dph_total": {"lte": self.max_usd_hr},
            "reliability": {"gte": self.min_reliability},
            "inet_down": {"gte": self.min_inet_down_mbps},
            "inet_down_cost": {"lte": self.max_download_usd_tb / GB_PER_TB},
            "cuda_max_good": {"gte": self.min_cuda},
            "order": [["dph_total", "asc"]],
            "type": "on-demand",
            "limit": SEARCH_LIMIT,
            "allocated_storage": self.disk_gb,
        })
    }

    /// Checked again here because a filter the server ignores or reads differently would otherwise rent the wrong
    /// box. A field the offer leaves out fails its check.
    pub fn admits(&self, offer: &VastOffer, avoid_machines: &[u64]) -> bool {
        offer.gpu == self.gpu
            && offer.count == self.count
            && offer.rentable == Some(true)
            && offer.rented == Some(false)
            && offer.rate_usd_hr > 0.0
            && offer.rate_usd_hr <= self.max_usd_hr
            && offer.reliability().is_some_and(|r| r >= self.min_reliability)
            && offer.inet_down >= self.min_inet_down_mbps
            && offer
                .internet_down_cost_per_tb
                .is_some_and(|per_tb| per_tb <= self.max_download_usd_tb + 1e-9)
            && offer.disk_space >= f64::from(self.disk_gb)
            && offer.cuda_max_good.is_some_and(|cuda| cuda + 1e-9 >= self.min_cuda)
            && !avoid_machines.contains(&offer.machine_id)
    }
}

/// The offers one Vast choice may rent, cheapest first, then most reliable.
pub fn offers_for(api: &dyn VastApi, choice: &VastChoice, pod: &PodSpec) -> Result<Vec<VastOffer>, CloudError> {
    let query = OfferQuery::new(choice, pod);
    Ok(pick(&query, api.offers(&query)?, &pod.avoid_machines))
}

pub fn pick(query: &OfferQuery, offers: Vec<VastOffer>, avoid_machines: &[u64]) -> Vec<VastOffer> {
    let mut admitted: Vec<VastOffer> = offers
        .into_iter()
        .filter(|offer| query.admits(offer, avoid_machines))
        .collect();
    admitted.sort_by(|a, b| {
        a.rate_usd_hr.total_cmp(&b.rate_usd_hr).then_with(|| {
            let reliability = |offer: &VastOffer| offer.reliability().unwrap_or(0.0);
            reliability(b).total_cmp(&reliability(a))
        })
    });
    admitted.truncate(OFFERS_PER_CHOICE);
    admitted
}

/// The body of `PUT /asks/{offer}/`.
#[derive(Debug, Clone, Serialize)]
pub struct VastCreateRequest {
    pub client_id: String,
    pub image: String,
    pub env: std::collections::BTreeMap<String, String>,
    pub disk: u32,
    pub label: String,
    pub onstart: String,
    pub runtype: String,
    /// A taken offer is refused instead of queued, so `up` moves on to the next one.
    pub cancel_unavail: bool,
}

/// Vast's states for an instance, mapped to the ones the harness acts on. A stopped instance still bills its disk,
/// so a stop requested or done reads as exited even while the container still runs.
pub fn status_of(actual: Option<&str>, intended: Option<&str>, current: Option<&str>) -> PodStatus {
    let norm = |value: Option<&str>| value.map(|v| v.trim().to_ascii_lowercase()).unwrap_or_default();
    let (actual, intended, current) = (norm(actual), norm(intended), norm(current));
    if matches!(actual.as_str(), "offline" | "error") {
        return PodStatus::Error;
    }
    if intended == "stopped" || current == "stopped" {
        return PodStatus::Exited;
    }
    match actual.as_str() {
        "" => PodStatus::Provisioning,
        "created" | "loading" => PodStatus::Starting,
        "running" => PodStatus::Running,
        "exited" | "stopped" => PodStatus::Exited,
        _ => PodStatus::Unknown,
    }
}

/// Vast sends numbers as numbers or, in places, as strings.
fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// An id, which Vast sends as an integer or a string of digits.
fn whole(value: &Value) -> Option<u64> {
    match value {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn text(value: &Value) -> Option<&str> {
    value.as_str().map(str::trim).filter(|s| !s.is_empty())
}

fn port(value: &Value) -> Option<u16> {
    number(value)
        .filter(|n| n.fract() == 0.0 && (1.0..=65535.0).contains(n))
        .map(|n| n as u16)
}

/// The ports map as Vast sends it: an object, or the same object encoded as a string.
fn ports_of(row: &Value) -> Value {
    match &row["ports"] {
        Value::String(encoded) => serde_json::from_str(encoded).unwrap_or(Value::Null),
        other => other.clone(),
    }
}

/// Direct SSH to the host's public address where the instance maps 22/tcp, else Vast's SSH proxy, which forwards
/// to the container's sshd and so carries scp and tar as direct SSH does.
fn ssh_of(row: &Value, ports: &Value) -> Option<SshEndpoint> {
    let direct = text(&row["public_ipaddr"]).zip(port(&ports["22/tcp"][0]["HostPort"]));
    let proxy = text(&row["ssh_host"]).zip(port(&row["ssh_port"]));
    direct.or(proxy).map(|(host, port)| SshEndpoint {
        host: host.to_string(),
        port,
    })
}

/// One instance row in the neutral `Pod` view. Only the id is required, since a row with no id cannot be acted on.
pub fn instance_pod(row: &Value) -> Result<Pod, CloudError> {
    let Some(id) = whole(&row["id"]) else {
        return Err(CloudError::Decode(format!("instance row without an id: {}", row["id"])));
    };
    let id = id.to_string();
    let status = status_of(
        row["actual_status"].as_str(),
        row["intended_status"].as_str(),
        row["cur_state"].as_str(),
    );
    let running = status == PodStatus::Running;
    let ports = ports_of(row);
    let ssh = running.then(|| PodSsh {
        direct: ssh_of(row, &ports),
    });
    // Vast reports the container running only once it exists, unlike RunPod during an image pull.
    let runtime = running.then(|| PodRuntime {
        uptime: Some(number(&row["uptime_mins"]).map_or(0.0, |mins| mins * 60.0)),
        ports,
    });
    Ok(Pod {
        id,
        name: text(&row["label"]).unwrap_or_default().to_string(),
        status,
        cost: number(&row["dph_total"]),
        ssh,
        runtime,
        data_center_id: None,
    })
}

/// Vast's error bodies are `{"success": false, "error": <code>, "msg": <text>}`; the code is kept, since the search
/// quota is told apart by it.
fn vast_detail(body: &str) -> String {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let fields: Vec<&str> = parsed
        .iter()
        .flat_map(|reply| ["error", "msg", "detail"].map(|key| reply[key].as_str()))
        .flatten()
        .collect();
    if fields.is_empty() {
        body.chars().take(300).collect()
    } else {
        fields.join(": ")
    }
}

/// The daily search quota resets only at 00:00 UTC, so a retry spends another search for nothing.
pub fn search_quota_used(error: &CloudError) -> bool {
    error.status() == Some(429) && error.detail().is_some_and(|d| d.contains("search_quota_exceeded"))
}

/// The key the vastai CLI uses: `VAST_API_KEY`, else the file `vastai set api-key` writes. `None` when neither is
/// set; RunPod-only issues never need one.
pub fn configured_key() -> anyhow::Result<Option<String>> {
    if let Some(key) = std::env::var("VAST_API_KEY").ok().filter(|key| !key.trim().is_empty()) {
        return Ok(Some(key.trim().to_string()));
    }
    let Some(home) = std::env::var_os("HOME") else {
        return Ok(None);
    };
    let path = PathBuf::from(home).join(".config/vastai/vast_api_key");
    match fs::read_to_string(&path) {
        Ok(key) => Ok(Some(key.trim().to_string()).filter(|key| !key.is_empty())),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow::Error::new(e).context(format!("reading {}", path.display()))),
    }
}

#[derive(Deserialize)]
struct OfferPage {
    offers: Vec<Value>,
}

/// `instances` is required in both shapes: a reply without it, such as Vast's `{"success": false, ...}`, must fail
/// as garbled, never read as an empty list or an instance gone.
#[derive(Deserialize)]
struct InstancePage {
    instances: Vec<Value>,
    #[serde(default)]
    next_token: Option<String>,
}

#[derive(Deserialize)]
struct OneInstance {
    instances: Value,
}

#[derive(Deserialize)]
struct CreateReply {
    #[serde(default)]
    success: Option<bool>,
    #[serde(default)]
    new_contract: Value,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    msg: Option<String>,
}

#[derive(Deserialize)]
struct ChargePage {
    results: Vec<Value>,
    #[serde(default)]
    next_token: Option<String>,
}

/// Whether a charge row is this instance's. The CLI reads `source` as `instance-<id>` or as the bare id.
fn charge_of(row: &Value, id: &str) -> bool {
    let kind_ok = row["type"]
        .as_str()
        .is_none_or(|kind| kind.eq_ignore_ascii_case("instance"));
    let source = text(&row["source"]).is_some_and(|source| source == id || source.rsplit('-').next() == Some(id));
    let metadata = ["instance_id", "contract_id"]
        .iter()
        .any(|key| whole(&row["metadata"][key]).is_some_and(|n| n.to_string() == id));
    kind_ok && (source || metadata)
}

/// The sum of this instance's charges across every page, or `None` when no row names it.
pub fn charges_usd(pages: &[Vec<Value>], id: &str) -> Option<f64> {
    let amounts: Vec<f64> = pages
        .iter()
        .flatten()
        .filter(|row| charge_of(row, id))
        .filter_map(|row| number(&row["amount"]))
        .collect();
    (!amounts.is_empty()).then(|| amounts.iter().sum())
}

/// Vast REST over blocking HTTP. Deliberately not `Debug`: it holds the API key.
pub struct VastRest<'a> {
    agent: Agent,
    base: String,
    authorization: String,
    clock: &'a dyn Clock,
    skew: Cell<Option<SignedDuration>>,
}

impl<'a> VastRest<'a> {
    /// Redirects are not followed: Vast redirects a path missing its trailing slash, and the HTTP client drops the
    /// key on a redirect, so a followed one reads as a 403. Unfollowed, a wrong path fails with its 3xx instead.
    pub fn new(base: &str, api_key: &str, clock: &'a dyn Clock) -> Self {
        let config = Agent::config_builder()
            .http_status_as_error(false)
            .max_redirects(0)
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

    fn sleep(&self, duration: Duration) {
        self.clock.sleep(duration);
    }

    /// Notes the server's clock from every reply, error replies included, and reads the body as JSON.
    fn read<T: DeserializeOwned>(&self, response: Result<Response<Body>, ureq::Error>) -> Result<T, CloudError> {
        read_body_json(self.observe(response)?)
    }

    /// Notes the server's clock from every reply, error replies included, and fails on a status outside 2xx. A
    /// call whose reply body says nothing the harness acts on stops here.
    fn observe(&self, response: Result<Response<Body>, ureq::Error>) -> Result<Response<Body>, CloudError> {
        let mut response = response.map_err(|e| CloudError::Network(e.to_string()))?;
        if let Some(server) = header(response.headers(), "date").and_then(http_date) {
            self.skew.set(Some(server.duration_since(self.clock.now())));
        }
        ensure_success_with(&mut response, vast_detail)?;
        Ok(response)
    }

    fn get_once<T: DeserializeOwned>(&self, path: &str, query: &[(&str, &str)]) -> Result<T, CloudError> {
        let mut request = self
            .agent
            .get(self.url(path))
            .header("Authorization", &self.authorization);
        for (key, value) in query {
            request = request.query(*key, *value);
        }
        self.read(request.call())
    }

    fn get<T: DeserializeOwned>(&self, path: &str, query: &[(&str, &str)]) -> Result<T, CloudError> {
        with_retry(&|d| self.sleep(d), CloudError::retryable, MAX_RETRY_AFTER_S, || {
            self.get_once(path, query)
        })
        .map_err(CloudError::vast)
    }

    fn instance(&self, id: &str, retry: bool) -> Result<Option<Pod>, CloudError> {
        let path = format!("/api/v0/instances/{id}/");
        let query = [("owner", "me")];
        let reply: Result<OneInstance, CloudError> = if retry {
            self.get(&path, &query)
        } else {
            self.get_once(&path, &query).map_err(CloudError::vast)
        };
        // `instances: null` is what Vast answers for an instance this key does not own: never proof on its own.
        match reply.map(|one| one.instances) {
            Ok(Value::Null) => Ok(None),
            Ok(row) => found_or_gone(instance_pod(&row).map_err(CloudError::vast)),
            Err(e) => found_or_gone(Err(e)),
        }
    }
}

impl Cloud for VastRest<'_> {
    /// Vast has no catalog of GPU types; `VastApi::offers` searches its market instead.
    fn gpu_catalog(&self, _: CloudTier, _: u32, _: &str) -> Result<Vec<CatalogGpu>, CloudError> {
        Err(CloudError::Protocol("Vast has no RunPod GPU catalog; its offers come from VastApi::offers".into()).vast())
    }

    fn create_pod(&self, _: &CreatePodRequest) -> Result<Pod, CloudError> {
        Err(CloudError::Protocol("Vast creates from an offer through VastApi::create_instance".into()).vast())
    }

    /// The v1 list the CLI pages through, every page or an error, since callers read absence as "gone".
    fn list_pods(&self) -> Result<Vec<Pod>, CloudError> {
        let order = json!([{"col": "id", "dir": "asc"}]).to_string();
        let mut pods = Vec::new();
        let mut token: Option<String> = None;
        let mut seen = HashSet::new();
        for _ in 0..MAX_POD_PAGES {
            let mut query = vec![
                ("select_filters", "{}"),
                ("order_by", order.as_str()),
                ("limit", INSTANCE_PAGE),
            ];
            if let Some(token) = &token {
                query.push(("after_token", token.as_str()));
            }
            let page: InstancePage = self.get("/api/v1/instances/", &query)?;
            for row in &page.instances {
                pods.push(instance_pod(row).map_err(CloudError::vast)?);
            }
            match page.next_token.filter(|next| !next.is_empty()) {
                Some(next) if !seen.insert(next.clone()) => {
                    return Err(CloudError::Protocol(format!("instance list token {next:?} came back twice")).vast());
                }
                Some(next) => token = Some(next),
                None => return Ok(pods),
            }
        }
        Err(CloudError::Protocol(format!("instance list still had more after {MAX_POD_PAGES} pages")).vast())
    }

    fn get_pod(&self, id: &str) -> Result<Option<Pod>, CloudError> {
        self.instance(id, true)
    }

    fn get_pod_once(&self, id: &str) -> Result<Option<Pod>, CloudError> {
        self.instance(id, false)
    }

    /// Destroy, never stop: a stopped instance keeps billing its disk. Vast answers a destroy it will not do with a
    /// 200 that says `success: false`, which is no destroy.
    fn delete_pod(&self, id: &str) -> Result<(), CloudError> {
        let result = with_retry(&|d| self.sleep(d), CloudError::retryable, MAX_RETRY_AFTER_S, || {
            let response = self
                .agent
                .delete(self.url(&format!("/api/v0/instances/{id}/")))
                .header("Authorization", &self.authorization)
                .force_send_body()
                .send_json(json!({}));
            let body = self.observe(response)?.body_mut().read_to_string().unwrap_or_default();
            match serde_json::from_str::<Value>(&body) {
                Ok(reply) if reply["success"] == Value::Bool(false) => {
                    Err(CloudError::Refused(format!("destroy: {}", vast_detail(&body))))
                }
                _ => Ok(()),
            }
        })
        .map_err(CloudError::vast);
        match result {
            Err(e) if e.status() == Some(404) => Ok(()),
            other => other,
        }
    }

    fn pod_billing_usd(&self, id: &str) -> Result<Option<f64>, CloudError> {
        let now = self.clock.now();
        let filters = json!({
            "day": {"gte": (now - CHARGE_WINDOW).as_second(), "lte": (now + SignedDuration::from_hours(24)).as_second()},
            "type": {"in": ["instance"]},
        })
        .to_string();
        let mut pages = Vec::new();
        let mut token: Option<String> = None;
        let mut seen = HashSet::new();
        for _ in 0..MAX_POD_PAGES {
            let mut query = vec![
                ("select_filters", filters.as_str()),
                ("latest_first", "false"),
                ("limit", CHARGE_PAGE),
                ("format", "table"),
            ];
            if let Some(token) = &token {
                query.push(("after_token", token.as_str()));
            }
            let page: ChargePage = self.get("/api/v0/charges/", &query)?;
            pages.push(page.results);
            match page.next_token.filter(|next| !next.is_empty()) {
                Some(next) if !seen.insert(next.clone()) => {
                    return Err(CloudError::Protocol(format!("charge list token {next:?} came back twice")).vast());
                }
                Some(next) => token = Some(next),
                None => return Ok(charges_usd(&pages, id)),
            }
        }
        Err(CloudError::Protocol(format!("charge list still had more after {MAX_POD_PAGES} pages")).vast())
    }

    fn account(&self) -> Result<Option<String>, CloudError> {
        let user: Value = self.get("/api/v0/users/current/", &[])?;
        match whole(&user["id"]) {
            Some(id) => Ok(Some(format!("vast:{id}"))),
            None => Err(CloudError::Decode("the current user has no id".into()).vast()),
        }
    }

    fn clock_skew(&self) -> Option<SignedDuration> {
        self.skew.get()
    }
}

impl VastApi for VastRest<'_> {
    fn offers(&self, query: &OfferQuery) -> Result<Vec<VastOffer>, CloudError> {
        let body = query.body();
        let retry_on = |e: &CloudError| e.retryable() && !search_quota_used(e);
        let page: OfferPage = with_retry(&|d| self.sleep(d), retry_on, MAX_RETRY_AFTER_S, || {
            let response = self
                .agent
                .post(self.url("/api/v0/bundles/"))
                .header("Authorization", &self.authorization)
                .send_json(&body);
            self.read(response)
        })
        .map_err(CloudError::vast)?;
        decode_offers(page.offers)
    }

    /// Only a 429 is retried, as for RunPod: any other failure may have made an instance.
    fn create_instance(&self, offer_id: u64, request: &VastCreateRequest) -> Result<String, CloudError> {
        let reply: CreateReply = with_retry(
            &|d| self.sleep(d),
            CloudError::rate_limited,
            MAX_CREATE_RETRY_AFTER_S,
            || {
                let response = self
                    .agent
                    .put(self.url(&format!("/api/v0/asks/{offer_id}/")))
                    .header("Authorization", &self.authorization)
                    .send_json(request);
                self.read(response)
            },
        )
        .map_err(CloudError::vast)?;
        created_id(reply)
    }
}

/// A 200 that says `success: false` is a refusal, as a taken offer gets; one that says success without an id may
/// have made an instance, so it reads as a garbled reply and the caller looks the instance up by name.
fn created_id(reply: CreateReply) -> Result<String, CloudError> {
    if reply.success == Some(false) {
        let detail = [reply.error, reply.msg]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(": ");
        return Err(CloudError::Refused(format!("create: {detail}")).vast());
    }
    match whole(&reply.new_contract) {
        Some(id) => Ok(id.to_string()),
        None => Err(CloudError::Decode("the create reply names no new_contract".into()).vast()),
    }
}

pub fn decode_offers(rows: Vec<Value>) -> Result<Vec<VastOffer>, CloudError> {
    let total = rows.len();
    let mut first_error = None;
    let mut offers = Vec::new();
    for row in rows {
        match serde_json::from_value::<VastOffer>(row) {
            Ok(offer) => offers.push(offer),
            Err(e) => {
                first_error.get_or_insert(e);
            }
        }
    }
    match first_error {
        Some(e) if offers.is_empty() => {
            Err(CloudError::Decode(format!("none of {total} offers decoded, the first: {e}")).vast())
        }
        _ => Ok(offers),
    }
}

/// An 8x Max-Q offer that passes every default limit, for the tests of the modules that reserve from one.
#[cfg(test)]
pub(crate) fn sample_offer(rate_usd_hr: f64) -> VastOffer {
    VastOffer {
        offer_id: 9100005,
        machine_id: 70005,
        host_id: 80005,
        gpu: "RTX PRO 6000 Max-Q".into(),
        count: 8,
        rate_usd_hr,
        reliability2: Some(0.99),
        reliability: None,
        disk_space: 500.0,
        inet_down: 2000.0,
        internet_down_cost_per_tb: Some(4.0),
        rentable: Some(true),
        rented: Some(false),
        cuda_max_good: Some(13.2),
        geolocation: None,
        hosting_type: Some(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scrubbed from `vastai search offers 'gpu_name="RTX PRO 6000 Max-Q"' --raw` on 2026-11-04: ids, machines,
    /// hosts, and addresses are fake, every other field is as Vast sent it.
    const MAXQ: &str = include_str!("../tests/fixtures/vast/offers_maxq.json");
    const INSTANCES: &str = include_str!("../tests/fixtures/vast/instances_page.json");

    fn maxq() -> Vec<VastOffer> {
        decode_offers(serde_json::from_str(MAXQ).unwrap()).unwrap()
    }

    fn query(count: u32, max_usd_hr: f64) -> OfferQuery {
        OfferQuery {
            gpu: "RTX PRO 6000 Max-Q".into(),
            count,
            disk_gb: 80,
            max_usd_hr,
            min_reliability: 0.95,
            min_inet_down_mbps: 500.0,
            max_download_usd_tb: 50.0,
            min_cuda: 13.0,
        }
    }

    #[test]
    fn real_offers_decode_with_the_fields_up_reads() {
        let offers = maxq();
        assert_eq!(offers.len(), 6);
        let first = &offers[0];
        assert_eq!(
            (first.offer_id, first.machine_id, first.host_id, first.count),
            (9100000, 70000, 80000, 1)
        );
        assert_eq!(first.gpu, "RTX PRO 6000 Max-Q");
        assert!((first.rate_usd_hr - 1.2299).abs() < 1e-3);
        assert_eq!(first.reliability(), Some(0.994592));
        assert_eq!((first.rentable, first.rented), (Some(true), Some(false)));
        assert_eq!(first.cuda_max_good, Some(13.2));
        assert_eq!(first.cloud(), CloudTier::Community);
        assert_eq!(offers[2].cloud(), CloudTier::Secure);
        let eight = &offers[5];
        assert_eq!((eight.count, eight.reliability()), (8, Some(0.5999925)));
        assert!(
            eight
                .describe()
                .starts_with("RTX PRO 6000 Max-Q x8 on Vast machine 70005 $21.34/hr")
        );
        assert!(eight.describe().contains("downloads $40.00/TB"), "{}", eight.describe());
    }

    #[test]
    fn an_offer_is_admitted_only_when_every_limit_holds() {
        let picked = pick(&query(1, 2.0), maxq(), &[]);
        let ids: Vec<u64> = picked.iter().map(|o| o.offer_id).collect();
        // 9100003 is cheapest but under the reliability floor; the 2x, 4x, and 8x are the wrong count.
        assert_eq!(ids, [9100000, 9100002]);
        let avoided = pick(&query(1, 2.0), maxq(), &[70000]);
        assert_eq!(avoided.iter().map(|o| o.offer_id).collect::<Vec<_>>(), [9100002]);
        assert!(pick(&query(1, 1.5), maxq(), &[]).iter().all(|o| o.rate_usd_hr <= 1.5));
        // The real 8x box sat at 0.60 reliability and $21.34/hr: neither cap nor floor lets it through.
        assert!(pick(&query(8, 17.0), maxq(), &[]).is_empty());
        let mut lax = query(8, 22.0);
        lax.min_reliability = 0.5;
        assert_eq!(pick(&lax, maxq(), &[]).len(), 1);
        // 9100000 charges $30.62 per TB downloaded, 9100002 $12.
        let mut thrifty = query(1, 2.0);
        thrifty.max_download_usd_tb = 20.0;
        let ids: Vec<u64> = pick(&thrifty, maxq(), &[]).iter().map(|o| o.offer_id).collect();
        assert_eq!(ids, [9100002]);
    }

    #[test]
    fn a_host_whose_driver_is_older_than_the_image_needs_is_no_offer() {
        // The two cheapest 1x offers on 2026-11-04, scrubbed: a GTX 1650 on CUDA 13.2 and a P4000 on 12.2.
        let cheap =
            decode_offers(serde_json::from_str(include_str!("../tests/fixtures/vast/offers_cheap.json")).unwrap())
                .unwrap();
        let query = |gpu: &str| OfferQuery {
            gpu: gpu.into(),
            count: 1,
            disk_gb: 30,
            max_usd_hr: 0.25,
            min_reliability: 0.95,
            min_inet_down_mbps: 500.0,
            max_download_usd_tb: 50.0,
            min_cuda: 12.8,
        };
        assert_eq!(pick(&query("GTX 1650"), cheap.clone(), &[]).len(), 1);
        assert_eq!(cheap[1].cuda_max_good, Some(12.2));
        assert!(pick(&query("Quadro P4000"), cheap, &[]).is_empty());
    }

    #[test]
    fn a_missing_field_fails_its_check() {
        let q = query(1, 2.0);
        let base = maxq().remove(0);
        assert!(q.admits(&base, &[]));
        let variants: [fn(&mut VastOffer); 8] = [
            |o| o.internet_down_cost_per_tb = None,
            |o| o.rented = None,
            |o| o.rentable = Some(false),
            |o| o.cuda_max_good = None,
            |o| (o.reliability2, o.reliability) = (None, None),
            |o| o.disk_space = 79.9,
            |o| o.inet_down = 499.0,
            |o| o.gpu = "RTX PRO 6000 Max-Q ".into(),
        ];
        for change in variants {
            let mut offer = base.clone();
            change(&mut offer);
            assert!(!q.admits(&offer, &[]), "{offer:?}");
        }
        let mut old_driver = base.clone();
        old_driver.cuda_max_good = Some(12.8);
        assert!(!q.admits(&old_driver, &[]));
    }

    #[test]
    fn the_search_asks_for_the_choice_and_prices_the_disk() {
        let body = query(8, 17.0).body();
        assert_eq!(body["gpu_name"], json!({"eq": "RTX PRO 6000 Max-Q"}));
        assert_eq!(body["num_gpus"], json!({"eq": 8}));
        assert_eq!(body["dph_total"], json!({"lte": 17.0}));
        assert_eq!(body["reliability"], json!({"gte": 0.95}));
        assert_eq!(
            (body["disk_space"]["gte"].clone(), body["allocated_storage"].clone()),
            (json!(80), json!(80))
        );
        assert_eq!(
            (body["rentable"]["eq"].clone(), body["rented"]["eq"].clone()),
            (json!(true), json!(false))
        );
        assert_eq!(body["type"], "on-demand");
        // $50 per TB is $0.0488 per GB, the unit Vast's filter takes.
        assert_eq!(body["inet_down_cost"], json!({"lte": 50.0 / 1024.0}));
    }

    #[test]
    fn offers_that_do_not_decode_are_left_out_unless_none_do() {
        let mut rows: Vec<Value> = serde_json::from_str(MAXQ).unwrap();
        rows[0]["machine_id"] = Value::Null;
        assert_eq!(decode_offers(rows.clone()).unwrap().len(), 5);
        let broken: Vec<Value> = rows.iter().map(|_| json!({"id": "x"})).collect();
        let err = decode_offers(broken).unwrap_err();
        assert!(
            err.to_string()
                .starts_with("Vast API response did not decode: none of 6"),
            "{err}"
        );
    }

    #[test]
    fn vast_states_map_to_the_ones_the_harness_acts_on() {
        for (actual, intended, current, expected) in [
            (None, Some("running"), Some("running"), PodStatus::Provisioning),
            (Some("created"), Some("running"), None, PodStatus::Starting),
            (Some("loading"), Some("running"), Some("running"), PodStatus::Starting),
            (Some("running"), Some("running"), Some("running"), PodStatus::Running),
            (Some(" Running "), None, None, PodStatus::Running),
            (Some("exited"), Some("running"), None, PodStatus::Exited),
            (Some("running"), Some("stopped"), Some("running"), PodStatus::Exited),
            (Some("running"), Some("running"), Some("stopped"), PodStatus::Exited),
            (Some("offline"), Some("running"), Some("running"), PodStatus::Error),
            (Some("error"), Some("stopped"), None, PodStatus::Error),
            (Some("rebooting"), Some("running"), None, PodStatus::Unknown),
        ] {
            assert_eq!(
                status_of(actual, intended, current),
                expected,
                "{actual:?} {intended:?} {current:?}"
            );
        }
    }

    fn instances() -> Vec<Pod> {
        let page: Value = serde_json::from_str(INSTANCES).unwrap();
        page["instances"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| instance_pod(row).unwrap())
            .collect()
    }

    #[test]
    fn instances_map_to_the_neutral_pod() {
        let pods = instances();
        let running = &pods[0];
        assert_eq!(
            (running.id.as_str(), running.name.as_str(), running.status),
            ("31000001", "sm120-harness-test-vast-0a0b0c", PodStatus::Running)
        );
        assert_eq!(running.cost, Some(0.0521));
        assert_eq!(
            running.ssh_direct(),
            Some(&SshEndpoint {
                host: "203.0.113.20".into(),
                port: 41022
            })
        );
        assert_eq!(running.container().unwrap().uptime, Some(210.0));
        assert!(running.data_center_id.is_none());
        let loading = &pods[1];
        assert_eq!(
            (loading.name.as_str(), loading.status),
            ("sm120-sglang-33422-tp8-1a2b3c", PodStatus::Starting)
        );
        assert!(loading.ssh_direct().is_none() && loading.container().is_none());
        assert_eq!(pods[2].status, PodStatus::Exited);
        let foreign = &pods[3];
        assert_eq!((foreign.name.as_str(), foreign.status), ("", PodStatus::Running));
        assert_eq!(foreign.ssh_direct().unwrap().port, 40900);
    }

    #[test]
    fn a_running_instance_without_a_mapped_port_is_reached_through_the_proxy() {
        let mut row: Value = serde_json::from_str::<Value>(INSTANCES).unwrap()["instances"][0].clone();
        row["ports"] = json!({});
        let pod = instance_pod(&row).unwrap();
        assert_eq!(
            pod.ssh_direct(),
            Some(&SshEndpoint {
                host: "ssh4.vast.ai".into(),
                port: 31022
            })
        );
        row["id"] = Value::Null;
        assert!(matches!(instance_pod(&row), Err(CloudError::Decode(_))));
    }

    #[test]
    fn charges_add_up_every_row_of_the_instance_and_nothing_else() {
        let page: Value = serde_json::from_str(include_str!("../tests/fixtures/vast/charges_page.json")).unwrap();
        let pages = [page["results"].as_array().unwrap().clone()];
        let total = charges_usd(&pages, "31000001").unwrap();
        assert!((total - 0.0517).abs() < 1e-9, "{total}");
        assert_eq!(charges_usd(&pages, "31000002"), Some(3.2));
        assert_eq!(charges_usd(&pages, "3100000"), None);
        assert_eq!(charges_usd(&[], "31000001"), None);
    }

    #[test]
    fn an_error_body_keeps_vasts_code_and_message() {
        let body = r#"{"success": false, "error": "search_quota_exceeded", "msg": "Daily search limit reached."}"#;
        assert_eq!(vast_detail(body), "search_quota_exceeded: Daily search limit reached.");
        assert_eq!(vast_detail("<html>bad gateway</html>"), "<html>bad gateway</html>");
        let quota = CloudError::Http {
            status: 429,
            detail: vast_detail(body),
            retry_after_s: None,
        }
        .vast();
        assert!(search_quota_used(&quota));
        assert!(
            quota
                .to_string()
                .starts_with("Vast API returned 429: search_quota_exceeded")
        );
        let busy = CloudError::Http {
            status: 429,
            detail: vast_detail(r#"{"error": "rate_limited"}"#),
            retry_after_s: Some(2),
        };
        assert!(!search_quota_used(&busy) && busy.retryable());
    }

    #[test]
    fn a_create_reply_names_the_instance_or_refuses() {
        let reply = |json: &str| created_id(serde_json::from_str(json).unwrap());
        assert_eq!(
            reply(r#"{"success": true, "new_contract": 31000001, "instance_api_key": "k"}"#).unwrap(),
            "31000001"
        );
        let taken = reply(r#"{"success": false, "error": "no_such_ask", "msg": "Instance type no longer available"}"#)
            .unwrap_err();
        assert!(!taken.create_outcome_unknown(), "{taken}");
        assert!(taken.to_string().contains("no_such_ask"), "{taken}");
        let garbled = reply(r#"{"success": true}"#).unwrap_err();
        assert!(garbled.create_outcome_unknown(), "{garbled}");
    }
}
