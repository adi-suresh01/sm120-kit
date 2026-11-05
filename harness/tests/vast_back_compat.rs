//! Every issue folder in this repo reads as it did before Vast, and a ledger written by the RunPod-only harness
//! writes back the same fields and values.

mod support;

use std::fs;
use std::path::Path;

use sm120_harness::ledger::{Budget, LedgerStore};
use sm120_harness::spec::{IssueSpec, Provider};
use sm120_harness::vast::VastOffer;
use support::repo_with_issue;

fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

fn issues() -> Vec<String> {
    let mut issues: Vec<String> = fs::read_dir(repo().join("issues"))
        .unwrap()
        .map(|dir| dir.unwrap().file_name().into_string().unwrap())
        .filter(|issue| repo().join("issues").join(issue).join("harness.toml").is_file())
        .collect();
    issues.sort();
    issues
}

#[test]
fn every_issue_spec_parses_and_only_the_vast_smoke_test_leaves_runpod() {
    let issues = issues();
    assert!(issues.len() >= 20, "{issues:?}");
    for issue in &issues {
        let spec = IssueSpec::load(repo(), issue).unwrap_or_else(|e| panic!("{issue}: {e:#}"));
        let on_vast = spec.pod.choices.iter().any(|c| c.provider() == Provider::Vast);
        assert_eq!(on_vast, issue == "harness-test-vast", "{issue}");
        if !on_vast {
            assert_eq!(spec.pod.choices.len(), spec.pod.gpus.len(), "{issue}");
            assert!(!spec.pod.require_p2p && spec.pod.avoid_machines.is_empty(), "{issue}");
        }
    }
}

#[test]
fn every_ledger_reads_and_writes_back_unchanged() {
    let stores = LedgerStore::all(repo()).unwrap();
    assert!(stores.len() >= 15, "{} ledgers", stores.len());
    for store in stores {
        let ledger = store.load().unwrap_or_else(|e| panic!("{}: {e:#}", store.issue()));
        assert!(ledger.vast.pods.is_empty() && ledger.vast.account.is_none());
        for entry in &ledger.pods {
            assert_eq!(ledger.provider_of(&entry.name), Provider::Runpod);
        }
        // Compared as JSON values: serde_json reads some long floats a bit off, as it always did, so the text of
        // a rewrite can differ in the last digit while every field and value stays.
        let on_disk: serde_json::Value = serde_json::from_str(&fs::read_to_string(store.path()).unwrap()).unwrap();
        assert_eq!(serde_json::to_value(&ledger).unwrap(), on_disk, "{}", store.issue());
    }
}

fn offer() -> VastOffer {
    VastOffer {
        offer_id: 9100000,
        machine_id: 70000,
        host_id: 80000,
        gpu: "GTX 1650".into(),
        count: 1,
        rate_usd_hr: 0.05,
        reliability2: Some(0.98),
        reliability: None,
        disk_space: 100.0,
        inet_down: 4000.0,
        internet_down_cost_per_tb: Some(3.0),
        rentable: Some(true),
        rented: Some(false),
        cuda_max_good: Some(13.2),
        geolocation: Some("Oregon, US".into()),
        hosting_type: Some(0),
    }
}

#[test]
fn the_vast_account_is_recorded_with_the_first_vast_pod_and_holds_after() {
    let repo = repo_with_issue("harness-test-vast", &[]);
    let at = "2026-11-04T17:00:00Z".parse().unwrap();
    let store = |account: Option<&str>| {
        LedgerStore::new(repo.path(), "harness-test-vast")
            .with_account(Some("org-sm120"))
            .with_vast_account(account)
    };
    store(Some("vast:424242"))
        .update(|ledger| {
            ledger.budget = Some(Budget {
                usd: 1.0,
                approved_at: at,
                note: "smoke".into(),
            });
            Ok(())
        })
        .unwrap();
    let ledger = store(None).load().unwrap();
    assert_eq!(ledger.vast.account, None, "no Vast pod yet, so nothing to fingerprint");
    store(Some("vast:424242"))
        .update(|ledger| ledger.reserve_vast("sm120-harness-test-vast-0a0b0c", &offer(), 0.25, at))
        .unwrap();
    let ledger = store(None).load().unwrap();
    assert_eq!(ledger.vast.account.as_deref(), Some("vast:424242"));
    assert_eq!(ledger.account.as_deref(), Some("org-sm120"));
    let err = store(Some("vast:1"))
        .update(|ledger| ledger.entry_mut("sm120-harness-test-vast-0a0b0c")?.mark_create_failed())
        .unwrap_err();
    assert!(format!("{err:#}").contains("Vast key belongs to vast:1"), "{err:#}");
    // A command without a Vast key reads nothing on Vast, so it may still write.
    store(None)
        .update(|ledger| ledger.entry_mut("sm120-harness-test-vast-0a0b0c")?.mark_create_failed())
        .unwrap();
}
