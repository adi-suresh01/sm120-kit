//! The RunPod create body for real issue specs, byte for byte, as `send_json` sends it. A change to how the
//! harness builds it shows up here before it reaches RunPod.

use std::path::Path;

use jiff::Timestamp;
use sm120_harness::ledger::{PodEntry, PodState};
use sm120_harness::pod;
use sm120_harness::spec::{IssueSpec, Provider};

fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

fn entry(spec: &IssueSpec) -> PodEntry {
    let reserved_at: Timestamp = "2026-11-04T17:00:00Z".parse().unwrap();
    let choice = &spec.pod.gpus[0];
    PodEntry {
        name: format!("sm120-{}-0a0b0c", spec.issue),
        provider: Provider::Runpod,
        id: None,
        state: PodState::Reserved,
        gpu: choice.id.clone(),
        count: spec.pod.count,
        cloud: choice.cloud,
        rate_usd_hr: 2.49,
        reserved_usd: 5.478,
        reserved_at,
        deadline: "2026-11-04T19:00:00Z".parse().unwrap(),
        terminated_at: None,
        cost_usd: None,
        cost_source: None,
        placement: None,
        abandoned: false,
    }
}

fn body(spec: &IssueSpec, data_centers: &[String]) -> String {
    serde_json::to_string(&pod::create_request(spec, &entry(spec), data_centers)).unwrap()
}

fn golden(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap()
        .trim_end_matches('\n')
        .to_string()
}

#[test]
fn a_plain_issue_sends_the_body_it_always_did() {
    let spec = IssueSpec::load(repo(), "vllm-47990").unwrap();
    assert_eq!(body(&spec, &[]), golden("runpod_create_vllm-47990.json"));
}

#[test]
fn an_issue_avoiding_data_centers_sends_the_body_it_always_did() {
    let text = include_str!("fixtures/golden/sglang-33422.toml");
    let spec = IssueSpec::parse(text, "sglang-33422", |_| true).unwrap();
    let preferred = ["CA-MTL-3".to_string(), "EU-RO-1".to_string()];
    assert_eq!(body(&spec, &preferred), golden("runpod_create_sglang-33422.json"));
}

#[test]
fn an_issue_with_an_hf_token_sends_the_body_it_always_did() {
    let text = std::fs::read_to_string(repo().join("issues/vllm-47990/harness.toml")).unwrap();
    let text = text.replace("needs_hf_token = false", "needs_hf_token = true");
    let spec = IssueSpec::parse(&text, "vllm-47990", |_| true).unwrap();
    assert_eq!(body(&spec, &[]), golden("runpod_create_vllm-47990_hf_token.json"));
}
