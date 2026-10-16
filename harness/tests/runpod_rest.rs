mod support;

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use sm120_harness::cloud::{
    Cloud, CloudError, CreatePodRequest, GpuRequest, Mounts, PersistentMount, PodStatus, RunpodRest, find_offers,
};
use sm120_harness::spec::{CloudTier, IssueSpec};
use support::FakeClock;

enum Step {
    Reply(u16, &'static str),
    ReplyWithHeader(u16, &'static str, &'static str, &'static str),
    HangUp,
}

#[derive(Debug, Clone)]
struct Seen {
    method: String,
    target: String,
    authorization: String,
    body: String,
}

struct FixtureServer {
    base: String,
    seen: Arc<Mutex<Vec<Seen>>>,
    handle: Option<JoinHandle<()>>,
}

impl FixtureServer {
    /// Serves one step per connection, in order. RunPod's real responses are replayed from tests/fixtures.
    fn start(steps: Vec<Step>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let handle = thread::spawn(move || {
            for step in steps {
                let (stream, _) = listener.accept().unwrap();
                let request = read_request(&stream);
                log.lock().unwrap().push(request);
                match step {
                    Step::Reply(status, body) => respond(stream, status, None, body),
                    Step::ReplyWithHeader(status, name, value, body) => {
                        respond(stream, status, Some((name, value)), body)
                    }
                    Step::HangUp => drop(stream),
                }
            }
        });
        Self {
            base,
            seen,
            handle: Some(handle),
        }
    }

    fn seen(mut self) -> Vec<Seen> {
        self.handle.take().unwrap().join().unwrap();
        self.seen.lock().unwrap().clone()
    }
}

fn read_request(stream: &TcpStream) -> Seen {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap().to_string();
    let target = parts.next().unwrap().to_string();
    let (mut length, mut authorization) = (0, String::new());
    loop {
        let mut header = String::new();
        reader.read_line(&mut header).unwrap();
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        let (name, value) = header.split_once(':').unwrap();
        match name.to_ascii_lowercase().as_str() {
            "content-length" => length = value.trim().parse().unwrap(),
            "authorization" => authorization = value.trim().to_string(),
            _ => {}
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    Seen {
        method,
        target,
        authorization,
        body: String::from_utf8(body).unwrap(),
    }
}

fn respond(mut stream: TcpStream, status: u16, header: Option<(&str, &str)>, body: &str) {
    let content_type = if status >= 400 {
        "application/problem+json"
    } else {
        "application/json"
    };
    let extra = header
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .unwrap_or_default();
    let reply = format!(
        "HTTP/1.1 {status} X\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(reply.as_bytes()).unwrap();
}

const CREATED: &str = include_str!("fixtures/pod_created.json");
const RUNNING: &str = include_str!("fixtures/pod_running.json");
const NOT_FOUND: &str = include_str!("fixtures/problem_404.json");
const BAD_GATEWAY: &str = r#"{"title":"Bad Gateway","status":502,"detail":"upstream timeout"}"#;

fn create_request() -> CreatePodRequest {
    CreatePodRequest {
        name: "sm120-vllm-47990-7f3a2c".into(),
        image: "vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
            .into(),
        entrypoint: vec!["/bin/bash".into(), "-c".into(), "echo hi".into()],
        env: BTreeMap::from([("HARNESS_DEADLINE_EPOCH".into(), "1792180980".into())]),
        ports: vec!["22/tcp".into()],
        disk: 40,
        mounts: Mounts {
            persistent: PersistentMount {
                size: 80,
                path: "/workspace".into(),
            },
        },
        cloud: CloudTier::Secure,
        gpu: GpuRequest {
            id: "NVIDIA RTX PRO 6000 Blackwell Server Edition".into(),
            count: 1,
            min_cuda_version: "13.0".into(),
        },
        start_ssh: true,
    }
}

#[test]
fn creates_a_pod_with_the_v2_body_and_bearer_auth() {
    let server = FixtureServer::start(vec![Step::Reply(201, CREATED)]);
    let clock = FakeClock::at("2026-10-16T17:03:00Z");
    let pod = RunpodRest::new(&server.base, "test-key", &clock)
        .create_pod(&create_request())
        .unwrap();
    assert_eq!(
        (pod.id.as_str(), pod.status, pod.cost),
        ("kfyqcddi6nefdp", PodStatus::Provisioning, Some(2.49))
    );
    assert!(pod.ssh_direct().is_none());
    let seen = server.seen();
    assert_eq!((seen[0].method.as_str(), seen[0].target.as_str()), ("POST", "/v2/pods"));
    assert_eq!(seen[0].authorization, "Bearer test-key");
    let body: serde_json::Value = serde_json::from_str(&seen[0].body).unwrap();
    assert_eq!(body["startSsh"], true);
    assert_eq!(body["gpu"]["minCudaVersion"], "13.0");
    assert_eq!(body["mounts"]["persistent"]["path"], "/workspace");
    assert_eq!(body["cloud"], "SECURE");
    assert_eq!(body["ports"][0], "22/tcp");
}

#[test]
fn never_resends_a_create_after_a_dropped_connection() {
    let server = FixtureServer::start(vec![Step::HangUp]);
    let clock = FakeClock::at("2026-10-16T17:03:00Z");
    let err = RunpodRest::new(&server.base, "k", &clock)
        .create_pod(&create_request())
        .unwrap_err();
    assert!(matches!(err, CloudError::Network(_)), "{err}");
    assert!(err.create_outcome_unknown());
    assert_eq!(server.seen().len(), 1);
}

#[test]
fn never_resends_a_create_after_a_5xx() {
    let server = FixtureServer::start(vec![Step::Reply(502, BAD_GATEWAY)]);
    let clock = FakeClock::at("2026-10-16T17:03:00Z");
    let err = RunpodRest::new(&server.base, "k", &clock)
        .create_pod(&create_request())
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "RunPod API returned 502: Bad Gateway: upstream timeout"
    );
    assert_eq!(server.seen().len(), 1);
}

#[test]
fn resends_a_create_that_was_rate_limited() {
    let server = FixtureServer::start(vec![
        Step::ReplyWithHeader(429, "Retry-After", "7", r#"{"title":"Too Many Requests","status":429}"#),
        Step::Reply(201, CREATED),
    ]);
    let clock = FakeClock::at("2026-10-16T17:03:00Z");
    let pod = RunpodRest::new(&server.base, "k", &clock)
        .create_pod(&create_request())
        .unwrap();
    assert_eq!(pod.id, "kfyqcddi6nefdp");
    assert_eq!(*clock.slept.borrow(), [Duration::from_secs(7)]);
    assert_eq!(server.seen().len(), 2);
}

#[test]
fn get_pod_reads_the_direct_ssh_endpoint_and_treats_404_as_gone() {
    let server = FixtureServer::start(vec![Step::Reply(200, RUNNING), Step::Reply(404, NOT_FOUND)]);
    let clock = FakeClock::at("2026-10-16T17:03:00Z");
    let api = RunpodRest::new(&server.base, "k", &clock);
    let pod = api.get_pod("kfyqcddi6nefdp").unwrap().unwrap();
    let direct = pod.ssh_direct().unwrap();
    assert_eq!(
        (direct.host.as_str(), direct.port, pod.status),
        ("203.0.113.7", 40122, PodStatus::Running)
    );
    assert!(api.get_pod("gone").unwrap().is_none());
    let targets: Vec<String> = server.seen().into_iter().map(|s| s.target).collect();
    assert_eq!(targets, ["/v2/pods/kfyqcddi6nefdp", "/v2/pods/gone"]);
}

#[test]
fn get_retries_a_5xx_then_succeeds() {
    let server = FixtureServer::start(vec![
        Step::Reply(503, BAD_GATEWAY),
        Step::HangUp,
        Step::Reply(200, RUNNING),
    ]);
    let clock = FakeClock::at("2026-10-16T17:03:00Z");
    let pod = RunpodRest::new(&server.base, "k", &clock)
        .get_pod("kfyqcddi6nefdp")
        .unwrap();
    assert!(pod.is_some());
    assert_eq!(*clock.slept.borrow(), [Duration::from_secs(1), Duration::from_secs(2)]);
}

#[test]
fn auth_failures_surface_at_once() {
    let body = r#"{"title":"Unauthorized","status":401,"detail":"invalid api key"}"#;
    let server = FixtureServer::start(vec![Step::Reply(401, body)]);
    let clock = FakeClock::at("2026-10-16T17:03:00Z");
    let err = RunpodRest::new(&server.base, "bad", &clock).list_pods().unwrap_err();
    assert_eq!(err.status(), Some(401));
    assert!(err.to_string().contains("invalid api key"), "{err}");
    assert!(clock.slept.borrow().is_empty());
}

#[test]
fn a_422_lists_each_validation_error() {
    let body = r#"{"title":"Unprocessable Entity","status":422,"detail":"invalid body","errors":["$.gpu.count: must be >= 1"]}"#;
    let server = FixtureServer::start(vec![Step::Reply(422, body)]);
    let clock = FakeClock::at("2026-10-16T17:03:00Z");
    let err = RunpodRest::new(&server.base, "k", &clock)
        .create_pod(&create_request())
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "RunPod API returned 422: Unprocessable Entity: invalid body; $.gpu.count: must be >= 1"
    );
    assert!(!err.create_outcome_unknown());
}

#[test]
fn list_pods_follows_the_cursor() {
    let server = FixtureServer::start(vec![
        Step::Reply(200, include_str!("fixtures/pods_page1.json")),
        Step::Reply(200, include_str!("fixtures/pods_page2.json")),
    ]);
    let clock = FakeClock::at("2026-10-16T17:03:00Z");
    let pods = RunpodRest::new(&server.base, "k", &clock).list_pods().unwrap();
    let names: Vec<&str> = pods.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["sm120-vllm-47990-7f3a2c", "comfyui"]);
    assert_eq!(pods[1].status, PodStatus::Exited);
    let targets: Vec<String> = server.seen().into_iter().map(|s| s.target).collect();
    assert_eq!(targets, ["/v2/pods", "/v2/pods?cursor=cGFnZTI"]);
}

#[test]
fn list_pods_reads_the_recorded_empty_account() {
    let server = FixtureServer::start(vec![Step::Reply(200, include_str!("fixtures/pods_empty.json"))]);
    let clock = FakeClock::at("2026-10-16T17:03:00Z");
    assert!(
        RunpodRest::new(&server.base, "k", &clock)
            .list_pods()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn delete_succeeds_on_204_and_on_404() {
    let server = FixtureServer::start(vec![Step::Reply(204, ""), Step::Reply(404, NOT_FOUND)]);
    let clock = FakeClock::at("2026-10-16T17:03:00Z");
    let api = RunpodRest::new(&server.base, "k", &clock);
    api.delete_pod("kfyqcddi6nefdp").unwrap();
    api.delete_pod("kfyqcddi6nefdp").unwrap();
    let seen = server.seen();
    assert!(
        seen.iter()
            .all(|s| s.method == "DELETE" && s.target == "/v2/pods/kfyqcddi6nefdp")
    );
}

#[test]
fn billing_total_comes_from_the_recorded_metadata() {
    let server = FixtureServer::start(vec![
        Step::Reply(200, include_str!("fixtures/billing_pod.json")),
        Step::Reply(200, include_str!("fixtures/billing_none.json")),
    ]);
    let clock = FakeClock::at("2026-10-16T17:03:00Z");
    let api = RunpodRest::new(&server.base, "k", &clock);
    let billed = api.pod_billing_usd("qkt9wl9ihcqstz").unwrap().unwrap();
    assert!((billed - 1.0983861774438992).abs() < 1e-12);
    assert_eq!(api.pod_billing_usd("zzzznotapod").unwrap(), None);
    let targets: Vec<String> = server.seen().into_iter().map(|s| s.target).collect();
    assert_eq!(
        targets,
        [
            "/v2/billing/pods?podId=qkt9wl9ihcqstz&bucketSize=day",
            "/v2/billing/pods?podId=zzzznotapod&bucketSize=day"
        ]
    );
}

#[test]
fn find_offers_reads_each_cloud_once_and_keeps_preference_order() {
    let server = FixtureServer::start(vec![
        Step::Reply(200, include_str!("fixtures/catalog_secure_x1.json")),
        Step::Reply(200, include_str!("fixtures/catalog_community_x1.json")),
    ]);
    let text = r#"
issue = "vllm-47990"
planned_hours = 2.0
[pod]
image = "vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
count = 1
min_cuda = "13.0"
container_disk_gb = 40
volume_gb = 80
gpus = [
  { id = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", cloud = "COMMUNITY" },
  { id = "NVIDIA RTX 6000 Ada Generation", cloud = "SECURE" },
  { id = "NVIDIA RTX PRO 6000 Blackwell Server Edition", cloud = "SECURE" },
  { id = "NVIDIA RTX PRO 6000 Blackwell Server Edition", cloud = "COMMUNITY" },
]
[server]
package = "vllm"
model = "m"
launch = ["vllm", "serve", "m"]
port = 8000
fail_pattern = "x"
boot_timeout_s = 900
"#;
    let spec = IssueSpec::parse(text, "vllm-47990", |_| true).unwrap();
    let clock = FakeClock::at("2026-10-16T17:03:00Z");
    let offers = find_offers(&RunpodRest::new(&server.base, "k", &clock), &spec.pod).unwrap();
    let picked: Vec<(CloudTier, f64)> = offers.iter().map(|o| (o.cloud, o.rate_usd_hr)).collect();
    assert_eq!(picked, [(CloudTier::Community, 1.69), (CloudTier::Secure, 2.49)]);
    let targets: Vec<String> = server.seen().into_iter().map(|s| s.target).collect();
    let query = "include=AVAILABILITY&product=POD&count=1";
    assert_eq!(
        targets,
        [
            format!("/v2/catalog/gpus?{query}&cloud=SECURE&minCudaVersion=13.0"),
            format!("/v2/catalog/gpus?{query}&cloud=COMMUNITY&minCudaVersion=13.0"),
        ]
    );
}
