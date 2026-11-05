//! Issue folders and ledgers read as they did before Vast, a ledger the RunPod-only harness wrote writes back byte
//! for byte, a ledger from before each entry carried its provider still reads its Vast pods as Vast's, and each
//! provider's account guards that provider's pods only.

#[path = "support/fake_vast.rs"]
mod fake_vast;
mod support;

use std::fs;
use std::path::Path;

use jiff::{SignedDuration, Timestamp};
use sm120_harness::ledger::{
    Budget, Ledger, LedgerStore, PodState, VastAccount, VastBook, VastPlacement, check_accounts, check_vast_accounts,
};
use sm120_harness::spec::{CloudTier, IssueSpec, Provider};
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

fn written(ledger: &Ledger) -> String {
    serde_json::to_string_pretty(ledger).unwrap() + "\n"
}

#[test]
fn every_issue_spec_in_the_repo_parses() {
    let issues = issues();
    assert!(!issues.is_empty());
    for issue in &issues {
        let spec = IssueSpec::load(repo(), issue).unwrap_or_else(|e| panic!("{issue}: {e:#}"));
        // Without a Vast choice, every choice is the RunPod one it always was.
        if spec.pod.vast_choices().next().is_none() {
            assert_eq!(spec.pod.choices.len(), spec.pod.gpus.len(), "{issue}");
        }
    }
}

/// RunPod-only ledgers write back byte for byte; any other reads back as the same ledger.
#[test]
fn every_ledger_in_the_repo_reads_and_writes_back() {
    let stores = LedgerStore::all(repo()).unwrap();
    assert!(!stores.is_empty());
    for store in stores {
        let ledger = store.load().unwrap_or_else(|e| panic!("{}: {e:#}", store.issue()));
        let on_disk = fs::read_to_string(store.path()).unwrap();
        let runpod_only = ledger.vast == VastBook::default() && ledger.pods.iter().all(|p| p.provider.is_runpod());
        if runpod_only {
            assert_eq!(written(&ledger), on_disk, "{}", store.issue());
        } else {
            assert_eq!(
                serde_json::from_str::<Ledger>(&written(&ledger)).unwrap(),
                ledger,
                "{}",
                store.issue()
            );
        }
    }
}

#[test]
fn a_ledger_the_runpod_only_harness_wrote_reads_as_runpods_and_writes_back_byte_for_byte() {
    let text = include_str!("fixtures/ledger_runpod_only.json");
    let repo = repo_with_issue("rp-fixture", &[("harness-ledger.json", text)]);
    let ledger = LedgerStore::new(repo.path(), "rp-fixture").load().unwrap();
    assert!(
        ledger
            .pods
            .iter()
            .all(|p| p.provider == Provider::Runpod && p.placement.is_none() && !p.abandoned)
    );
    assert_eq!(ledger.vast, VastBook::default());
    assert_eq!(written(&ledger), text);
}

#[test]
fn a_ledger_with_the_vast_side_table_reads_its_vast_pods_as_vasts() {
    let text = include_str!("fixtures/ledger_vast_side_table.json");
    let repo = repo_with_issue("vast-fixture", &[("harness-ledger.json", text)]);
    let store = LedgerStore::new(repo.path(), "vast-fixture");
    let ledger = store.load().unwrap();
    let (runpod, vast) = (&ledger.pods[0], &ledger.pods[1]);
    assert_eq!((runpod.provider, runpod.placement), (Provider::Runpod, None));
    assert_eq!(
        (vast.provider, vast.state, vast.placement),
        (
            Provider::Vast,
            PodState::Live,
            Some(VastPlacement {
                offer: 9100001,
                machine: 70001,
                host: 150001
            })
        )
    );
    assert_eq!(ledger.vast.account.as_deref(), Some("vast:424242"));
    // The next write moves the table onto the entry it names.
    let json: serde_json::Value = serde_json::from_str(&written(&ledger)).unwrap();
    assert_eq!(json["vast"], serde_json::json!({"account": "vast:424242"}));
    assert_eq!(json["pods"][1]["provider"], "vast");
    assert_eq!(json["pods"][1]["placement"]["machine"], 70001);
    assert!(json["pods"][0].get("provider").is_none());
    assert_eq!(serde_json::from_value::<Ledger>(json).unwrap(), ledger);
}

fn offer() -> VastOffer {
    VastOffer {
        gpu: "GTX 1650".into(),
        count: 1,
        ..fake_vast::offer(9100000, 70000, 0.05)
    }
}

#[test]
fn the_vast_account_is_recorded_with_the_first_vast_pod_and_holds_after() {
    let repo = repo_with_issue("harness-test-vast", &[]);
    let at = "2026-11-04T17:00:00Z".parse().unwrap();
    let store = |account: Option<&str>| {
        let vast = account.map_or(VastAccount::Absent, |account| VastAccount::Known(account.into()));
        LedgerStore::new(repo.path(), "harness-test-vast")
            .with_account(Some("org-sm120"))
            .with_vast_account(vast)
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
        .update(|ledger| ledger.reserve_vast("sm120-harness-test-vast-0a0b0c", &offer(), 0.25, 0, at))
        .unwrap();
    let ledger = store(None).load().unwrap();
    assert_eq!(ledger.vast.account.as_deref(), Some("vast:424242"));
    assert_eq!(ledger.account.as_deref(), Some("org-sm120"));
    let err = store(Some("vast:1"))
        .update(|ledger| ledger.entry_mut("sm120-harness-test-vast-0a0b0c")?.mark_create_failed())
        .unwrap_err();
    assert!(format!("{err:#}").contains("Vast key belongs to vast:1"), "{err:#}");
    let unknown = LedgerStore::new(repo.path(), "harness-test-vast")
        .with_account(Some("org-sm120"))
        .with_vast_account(VastAccount::Unknown);
    let err = unknown
        .update(|ledger| ledger.entry_mut("sm120-harness-test-vast-0a0b0c")?.mark_create_failed())
        .unwrap_err();
    assert!(format!("{err:#}").contains("could not be read"), "{err:#}");
    // A command without a Vast key reads nothing on Vast, so it may still write.
    store(None)
        .update(|ledger| ledger.entry_mut("sm120-harness-test-vast-0a0b0c")?.mark_create_failed())
        .unwrap();
}

/// A ledger with one Vast pod booked from billing and one RunPod pod live, its accounts recorded as `vast:1` and
/// `org-sm120`.
fn mixed(repo: &Path) -> &'static str {
    let at: Timestamp = "2026-11-04T17:00:00Z".parse().unwrap();
    LedgerStore::new(repo, ISSUE)
        .with_account(Some("org-sm120"))
        .with_vast_account(VastAccount::Known("vast:1".into()))
        .update(|ledger| {
            ledger.budget = Some(Budget {
                usd: 20.0,
                approved_at: at,
                note: "both".into(),
            });
            ledger.reserve_vast("sm120-mixed-1-0a0b0c", &offer(), 1.0, 0, at)?;
            let entry = ledger.entry_mut("sm120-mixed-1-0a0b0c")?;
            entry.mark_live("31000001")?;
            entry.mark_terminated(at)?;
            entry.apply_billing(0.01, at + SignedDuration::from_hours(25));
            ledger.reserve(
                "sm120-mixed-1-0a0b0d",
                &sm120_harness::cloud::Offer {
                    gpu: "g".into(),
                    cloud: CloudTier::Secure,
                    count: 1,
                    rate_usd_hr: 1.0,
                    data_centers: Vec::new(),
                    unlisted_avoided: Vec::new(),
                },
                1.0,
                at,
            )?;
            ledger.entry_mut("sm120-mixed-1-0a0b0d")?.mark_live("p1")
        })
        .unwrap();
    "sm120-mixed-1-0a0b0d"
}

const ISSUE: &str = "mixed-1";

#[test]
fn a_runpod_write_never_needs_the_vast_account_of_settled_vast_pods() {
    let repo = repo_with_issue(ISSUE, &[]);
    let runpod_pod = mixed(repo.path());
    let at = "2026-11-04T19:00:00Z".parse().unwrap();
    LedgerStore::new(repo.path(), ISSUE)
        .with_account(Some("org-sm120"))
        .with_vast_account(VastAccount::Unknown)
        .update(|ledger| ledger.entry_mut(runpod_pod)?.mark_terminated(at))
        .unwrap();
    check_vast_accounts(repo.path(), &VastAccount::Known("vast:2".into())).unwrap();
}

#[test]
fn another_vast_account_still_never_changes_a_vast_pod() {
    let repo = repo_with_issue(ISSUE, &[]);
    mixed(repo.path());
    let at: Timestamp = "2026-11-06T19:00:00Z".parse().unwrap();
    let err = LedgerStore::new(repo.path(), ISSUE)
        .with_account(Some("org-sm120"))
        .with_vast_account(VastAccount::Known("vast:2".into()))
        .update(|ledger| {
            ledger.entry_mut("sm120-mixed-1-0a0b0c")?.apply_billing(5.0, at);
            Ok(())
        })
        .unwrap_err();
    assert!(format!("{err:#}").contains("Vast key belongs to vast:2"), "{err:#}");
}

#[test]
fn a_vast_only_ledger_never_needs_the_runpod_organization() {
    let repo = repo_with_issue("harness-test-vast", &[]);
    let at: Timestamp = "2026-11-04T17:00:00Z".parse().unwrap();
    let store = |runpod: Option<&str>| {
        LedgerStore::new(repo.path(), "harness-test-vast")
            .with_account(runpod)
            .with_vast_account(VastAccount::Known("vast:424242".into()))
    };
    store(Some("org-sm120"))
        .update(|ledger| {
            ledger.budget = Some(Budget {
                usd: 1.0,
                approved_at: at,
                note: "smoke".into(),
            });
            ledger.reserve_vast("sm120-harness-test-vast-0a0b0c", &offer(), 0.25, 0, at)
        })
        .unwrap();
    // RunPod's organization read fails, or the key is another organization's: no RunPod pod here is at stake.
    for runpod in [None, Some("org-other")] {
        check_accounts(repo.path(), runpod).unwrap();
    }
    store(None)
        .update(|ledger| {
            ledger
                .entry_mut("sm120-harness-test-vast-0a0b0c")?
                .mark_live("31000001")
        })
        .unwrap();
    store(Some("org-other"))
        .update(|ledger| ledger.entry_mut("sm120-harness-test-vast-0a0b0c")?.mark_terminated(at))
        .unwrap();
}
