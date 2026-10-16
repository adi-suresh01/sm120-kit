mod support;

use std::fs;
use std::thread;

use sm120_harness::ledger::{Budget, LedgerStore};
use support::repo_with_issue;

fn budget(usd: f64) -> Budget {
    Budget {
        usd,
        approved_at: "2026-10-16T17:02:11Z".parse().unwrap(),
        note: "approved in chat".into(),
    }
}

fn set_budget(store: &LedgerStore, usd: f64) {
    store
        .update(|l| {
            l.budget = Some(budget(usd));
            Ok(())
        })
        .unwrap();
}

#[test]
fn update_creates_the_ledger_and_load_reads_it_back() {
    let repo = repo_with_issue("vllm-47990", &[]);
    let store = LedgerStore::new(repo.path(), "vllm-47990");
    set_budget(&store, 15.0);
    let text = fs::read_to_string(store.path()).unwrap();
    assert!(text.contains("\"approved_at\": \"2026-10-16T17:02:11Z\""), "{text}");
    assert!(text.ends_with("}\n"));
    assert_eq!(store.load().unwrap().budget, Some(budget(15.0)));
    assert!(!store.path().with_extension("json.tmp").exists());
}

#[test]
fn a_failed_change_writes_nothing() {
    let repo = repo_with_issue("vllm-47990", &[]);
    let store = LedgerStore::new(repo.path(), "vllm-47990");
    set_budget(&store, 15.0);
    let before = fs::read_to_string(store.path()).unwrap();
    let result: anyhow::Result<()> = store.update(|l| {
        l.budget = Some(budget(99.0));
        anyhow::bail!("refused")
    });
    assert!(result.is_err());
    assert_eq!(fs::read_to_string(store.path()).unwrap(), before);
}

#[test]
fn a_failed_write_names_the_file() {
    use std::os::unix::fs::PermissionsExt;
    let repo = repo_with_issue("vllm-47990", &[]);
    let store = LedgerStore::new(repo.path(), "vllm-47990");
    set_budget(&store, 15.0);
    let dir = repo.path().join("issues/vllm-47990");
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
    let result = store.update(|l| {
        l.budget = Some(budget(20.0));
        Ok(())
    });
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
    let err = format!("{:#}", result.unwrap_err());
    assert!(err.contains("issues/vllm-47990/harness-ledger.json.tmp"), "{err}");
    assert_eq!(store.load().unwrap().budget, Some(budget(15.0)));
}

#[test]
fn an_unchanged_ledger_is_never_written() {
    let repo = repo_with_issue("vllm-47990", &[]);
    let store = LedgerStore::new(repo.path(), "vllm-47990");
    store.update(|_| Ok(())).unwrap();
    assert!(!store.path().exists());
}

#[test]
fn concurrent_updates_never_lose_a_write() {
    let repo = repo_with_issue("vllm-47990", &[]);
    let root = repo.path().to_path_buf();
    set_budget(&LedgerStore::new(&root, "vllm-47990"), 0.0);
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let root = root.clone();
            thread::spawn(move || {
                let store = LedgerStore::new(&root, "vllm-47990");
                for _ in 0..25 {
                    store
                        .update(|l| {
                            let b = l.budget.as_mut().unwrap();
                            let seen = b.usd;
                            thread::yield_now();
                            b.usd = seen + 1.0;
                            Ok(())
                        })
                        .unwrap();
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    let usd = LedgerStore::new(&root, "vllm-47990")
        .load()
        .unwrap()
        .budget
        .unwrap()
        .usd;
    assert_eq!(usd, 200.0);
}

#[test]
fn refuses_a_ledger_that_names_another_issue() {
    let repo = repo_with_issue(
        "vllm-47990",
        &[(
            "harness-ledger.json",
            r#"{"issue": "other", "budget": null, "pods": []}"#,
        )],
    );
    let err = LedgerStore::new(repo.path(), "vllm-47990").load().unwrap_err();
    assert!(format!("{err:#}").contains("belongs to issue other"), "{err:#}");
}

#[test]
fn all_finds_every_issue_with_a_ledger() {
    let repo = repo_with_issue(
        "b-2",
        &[("harness-ledger.json", r#"{"issue": "b-2", "budget": null, "pods": []}"#)],
    );
    for (issue, ledger) in [("a-1", true), ("c-3", false)] {
        let dir = repo.path().join("issues").join(issue);
        fs::create_dir_all(&dir).unwrap();
        if ledger {
            let body = format!(r#"{{"issue": "{issue}", "budget": null, "pods": []}}"#);
            fs::write(dir.join("harness-ledger.json"), body).unwrap();
        }
    }
    let issues: Vec<String> = LedgerStore::all(repo.path())
        .unwrap()
        .iter()
        .map(|s| s.issue().to_string())
        .collect();
    assert_eq!(issues, ["a-1", "b-2"]);
}
