//! VastRest against a local server that replays Vast's replies: the request each call sends and how each reply
//! reads. The offers are a real search reply, scrubbed; the instance, charge, and user replies follow the vastai
//! 1.10 CLI's field names with fake values, since the account had no instance to read on 2026-11-04.

mod support;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use sm120_harness::cloud::{Cloud, CloudError, PodStatus};
use sm120_harness::vast::{OfferQuery, VastApi, VastCreateRequest, VastRest};
use support::FakeClock;

#[derive(Debug, Clone)]
struct Seen {
    method: String,
    target: String,
    authorization: String,
    body: String,
}

struct Server {
    base: String,
    seen: Arc<Mutex<Vec<Seen>>>,
    handle: Option<JoinHandle<()>>,
}

impl Server {
    /// One reply per connection, in order.
    fn start(replies: Vec<(u16, &'static str)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let handle = thread::spawn(move || {
            for (status, body) in replies {
                let (mut stream, _) = listener.accept().unwrap();
                log.lock().unwrap().push(read_request(&stream));
                let reply = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
                     Date: Wed, 04 Nov 2026 17:00:00 GMT\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(reply.as_bytes()).unwrap();
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
    let (method, target) = (parts.next().unwrap().to_string(), parts.next().unwrap().to_string());
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

fn decoded(target: &str) -> String {
    let mut out = Vec::new();
    let bytes = target.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            out.push(u8::from_str_radix(&target[i + 1..i + 3], 16).unwrap());
            i += 3;
        } else {
            out.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
            i += 1;
        }
    }
    String::from_utf8(out).unwrap()
}

const OFFERS: &str = concat!("{\"offers\": ", include_str!("fixtures/vast/offers_maxq.json"), "}");
const INSTANCES: &str = include_str!("fixtures/vast/instances_page.json");
const CHARGES: &str = include_str!("fixtures/vast/charges_page.json");
const USER: &str = include_str!("fixtures/vast/user.json");

fn clock() -> FakeClock {
    FakeClock::at("2026-11-04T17:00:00Z")
}

fn query() -> OfferQuery {
    OfferQuery {
        gpu: "RTX PRO 6000 Max-Q".into(),
        count: 1,
        disk_gb: 30,
        max_usd_hr: 2.0,
        min_reliability: 0.95,
        min_inet_down_mbps: 500.0,
        min_cuda: 12.8,
    }
}

#[test]
fn the_search_posts_the_query_and_decodes_every_offer() {
    let server = Server::start(vec![(200, OFFERS)]);
    let clock = clock();
    let offers = VastRest::new(&server.base, "vast-test-key", &clock)
        .offers(&query())
        .unwrap();
    assert_eq!(offers.len(), 6);
    let seen = server.seen();
    assert_eq!(
        (seen[0].method.as_str(), seen[0].target.as_str()),
        ("POST", "/api/v0/bundles/")
    );
    assert_eq!(seen[0].authorization, "Bearer vast-test-key");
    let body: serde_json::Value = serde_json::from_str(&seen[0].body).unwrap();
    assert_eq!(body, query().body());
}

#[test]
fn a_used_up_search_quota_is_not_retried() {
    let quota = r#"{"success": false, "error": "search_quota_exceeded", "msg": "Daily limit reached"}"#;
    let server = Server::start(vec![(429, quota)]);
    let clock = clock();
    let err = VastRest::new(&server.base, "k", &clock).offers(&query()).unwrap_err();
    assert!(err.to_string().contains("search_quota_exceeded"), "{err}");
    assert_eq!(server.seen().len(), 1);
    assert!(clock.slept.borrow().is_empty());
}

#[test]
fn the_instance_list_reads_every_page_and_refuses_a_loop() {
    let page1 = r#"{"instances": [{"id": 1, "label": "a", "actual_status": "running"}], "next_token": "t1"}"#;
    let server = Server::start(vec![(200, page1), (200, INSTANCES)]);
    let clock = clock();
    let pods = VastRest::new(&server.base, "k", &clock).list_pods().unwrap();
    assert_eq!(pods.len(), 5);
    assert_eq!(pods[1].status, PodStatus::Running);
    let seen = server.seen();
    let first = decoded(&seen[0].target);
    assert!(first.starts_with("/api/v1/instances/?select_filters={}"), "{first}");
    assert!(
        first.contains("order_by=[{\"col\":\"id\",\"dir\":\"asc\"}]&limit=25"),
        "{first}"
    );
    assert!(
        decoded(&seen[1].target).ends_with("&after_token=t1"),
        "{}",
        seen[1].target
    );

    let looping = r#"{"instances": [], "next_token": "t1"}"#;
    let server = Server::start(vec![(200, looping), (200, looping)]);
    let err = VastRest::new(&server.base, "k", &clock).list_pods().unwrap_err();
    assert!(
        matches!(err, CloudError::Vast(_)) && err.to_string().contains("came back twice"),
        "{err}"
    );
    server.seen();
}

#[test]
fn one_instance_reads_as_gone_on_null_or_404_and_never_otherwise() {
    let one = r#"{"instances": {"id": 31000001, "label": "sm120-x-0a0b0c", "actual_status": "loading"}}"#;
    let server = Server::start(vec![
        (200, one),
        (200, r#"{"instances": null}"#),
        (404, r#"{"error": "not_found"}"#),
        (400, r#"{"error": "bad_request", "msg": "nope"}"#),
    ]);
    let clock = clock();
    let api = VastRest::new(&server.base, "k", &clock);
    let pod = api.get_pod("31000001").unwrap().unwrap();
    assert_eq!((pod.name.as_str(), pod.status), ("sm120-x-0a0b0c", PodStatus::Starting));
    assert!(api.get_pod("31000001").unwrap().is_none());
    assert!(api.get_pod_once("31000001").unwrap().is_none());
    let err = api.get_pod("31000001").unwrap_err();
    assert_eq!(err.to_string(), "Vast API returned 400: bad_request: nope");
    let seen = server.seen();
    assert_eq!(seen[0].target, "/api/v0/instances/31000001/?owner=me");
}

#[test]
fn destroy_sends_delete_and_takes_404_as_sent() {
    let server = Server::start(vec![(200, r#"{"success": true}"#), (404, "{}"), (200, "")]);
    let clock = clock();
    let api = VastRest::new(&server.base, "k", &clock);
    api.delete_pod("31000001").unwrap();
    api.delete_pod("31000001").unwrap();
    // A reply without a body still counts: only the status says whether the destroy landed.
    api.delete_pod("31000001").unwrap();
    let seen = server.seen();
    assert_eq!(
        (seen[0].method.as_str(), seen[0].target.as_str(), seen[0].body.as_str()),
        ("DELETE", "/api/v0/instances/31000001/", "{}")
    );
}

fn create_request() -> VastCreateRequest {
    VastCreateRequest {
        client_id: "me".into(),
        image: "nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:133c".into(),
        env: [("HARNESS_DEADLINE_EPOCH".to_string(), "1793812500".to_string())].into(),
        disk: 30,
        label: "sm120-harness-test-vast-0a0b0c".into(),
        onstart: "echo hi".into(),
        runtype: "ssh_direc ssh_proxy".into(),
        cancel_unavail: true,
    }
}

#[test]
fn create_puts_the_ask_and_reads_the_new_instance() {
    let ok = r#"{"success": true, "new_contract": 31000001, "instance_api_key": "instance-scoped-secret"}"#;
    let server = Server::start(vec![(200, ok)]);
    let clock = clock();
    let id = VastRest::new(&server.base, "k", &clock)
        .create_instance(9100000, &create_request())
        .unwrap();
    assert_eq!(id, "31000001");
    let seen = server.seen();
    assert_eq!(
        (seen[0].method.as_str(), seen[0].target.as_str()),
        ("PUT", "/api/v0/asks/9100000/")
    );
    let body: serde_json::Value = serde_json::from_str(&seen[0].body).unwrap();
    assert_eq!(
        body,
        serde_json::json!({
            "client_id": "me", "image": "nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:133c",
            "env": {"HARNESS_DEADLINE_EPOCH": "1793812500"}, "disk": 30,
            "label": "sm120-harness-test-vast-0a0b0c", "onstart": "echo hi", "runtype": "ssh_direc ssh_proxy",
            "cancel_unavail": true,
        })
    );
}

#[test]
fn a_taken_offer_is_refused_and_a_lost_reply_is_unknown() {
    let taken = r#"{"success": false, "error": "no_such_ask", "msg": "Instance type no longer available"}"#;
    let server = Server::start(vec![
        (400, taken),
        (200, taken),
        (502, "<html>bad gateway</html>"),
        (200, r#"{"success": true}"#),
    ]);
    let clock = clock();
    let api = VastRest::new(&server.base, "k", &clock);
    for _ in 0..2 {
        let err = api.create_instance(9100000, &create_request()).unwrap_err();
        assert!(
            !err.create_outcome_unknown() && err.to_string().contains("no_such_ask"),
            "{err}"
        );
    }
    for _ in 0..2 {
        let err = api.create_instance(9100000, &create_request()).unwrap_err();
        assert!(err.create_outcome_unknown(), "{err}");
    }
    assert_eq!(
        server.seen().len(),
        4,
        "a create is never retried on anything but a 429"
    );
}

#[test]
fn a_redirect_is_an_error_not_a_request_without_the_key() {
    let server = Server::start(vec![(301, "")]);
    let clock = clock();
    let err = VastRest::new(&server.base, "k", &clock).account().unwrap_err();
    assert_eq!(err.status(), Some(301), "{err}");
    assert_eq!(server.seen().len(), 1);
}

#[test]
fn the_account_is_the_user_id_qualified_by_vast() {
    let server = Server::start(vec![(200, USER)]);
    let clock = clock();
    let api = VastRest::new(&server.base, "k", &clock);
    assert_eq!(api.account().unwrap().as_deref(), Some("vast:424242"));
    assert_eq!(api.clock_skew(), Some(jiff::SignedDuration::ZERO));
    assert_eq!(server.seen()[0].target, "/api/v0/users/current/");
}

#[test]
fn billing_sums_the_instances_charges_across_pages() {
    let page1 =
        r#"{"results": [{"type": "instance", "source": "instance-31000001", "amount": 0.5}], "next_token": "c1"}"#;
    let server = Server::start(vec![(200, page1), (200, CHARGES)]);
    let clock = clock();
    let billed = VastRest::new(&server.base, "k", &clock)
        .pod_billing_usd("31000001")
        .unwrap()
        .unwrap();
    assert!((billed - 0.5517).abs() < 1e-9, "{billed}");
    let seen = server.seen();
    let first = decoded(&seen[0].target);
    assert!(first.starts_with("/api/v0/charges/?select_filters="), "{first}");
    assert!(first.contains("\"type\":{\"in\":[\"instance\"]}"), "{first}");
    assert!(first.contains("&limit=100&format=table"), "{first}");
    assert!(decoded(&seen[1].target).ends_with("&after_token=c1"));
}
