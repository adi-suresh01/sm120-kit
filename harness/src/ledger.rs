use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use jiff::{SignedDuration, Timestamp};
use serde::{Deserialize, Serialize};

use crate::spec::CloudTier;

/// Headroom on the quoted GPU rate for disk charges and price drift; the deadline is sized so that
/// rate x MARGIN x deadline never exceeds what is left of the budget.
pub const MARGIN: f64 = 1.10;
pub const MIN_DEADLINE: SignedDuration = SignedDuration::from_mins(15);
/// How long a reservation with no pod id waits for its pod to appear in RunPod's list before it is released.
pub const ADOPT_WINDOW: SignedDuration = SignedDuration::from_mins(10);
/// Billing for a finished pod can show up in parts, so a figure below the estimate is only trusted after this.
pub const BILLING_SETTLE: SignedDuration = SignedDuration::from_hours(24);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ledger {
    pub issue: String,
    /// The RunPod organization whose key wrote the ledger, when the key can tell. Another account's key gets 404
    /// for these pods, which must never be read as "terminated".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account: Option<String>,
    pub budget: Option<Budget>,
    pub pods: Vec<PodEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    pub usd: f64,
    pub approved_at: Timestamp,
    pub note: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PodState {
    Reserved,
    Live,
    Terminated,
    CreateFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostSource {
    Estimate,
    Billing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PodEntry {
    pub name: String,
    pub id: Option<String>,
    pub state: PodState,
    pub gpu: String,
    pub count: u32,
    pub cloud: CloudTier,
    pub rate_usd_hr: f64,
    pub reserved_usd: f64,
    pub reserved_at: Timestamp,
    pub deadline: Timestamp,
    pub terminated_at: Option<Timestamp>,
    pub cost_usd: Option<f64>,
    pub cost_source: Option<CostSource>,
}

/// A GPU choice with a live price: what a reservation is made for.
#[derive(Debug, Clone, PartialEq)]
pub struct Offer {
    pub gpu: String,
    pub cloud: CloudTier,
    pub count: u32,
    pub rate_usd_hr: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reservation {
    pub deadline: Timestamp,
    pub reserved_usd: f64,
}

pub fn plan_reservation(
    now: Timestamp,
    planned_hours: f64,
    remaining_usd: f64,
    rate_usd_hr: f64,
) -> Result<Reservation> {
    ensure!(
        rate_usd_hr > 0.0 && rate_usd_hr.is_finite(),
        "rate must be positive, got {rate_usd_hr}"
    );
    let affordable_hours = remaining_usd.max(0.0) / (rate_usd_hr * MARGIN);
    let seconds = (planned_hours.min(affordable_hours) * 3600.0).floor() as i64;
    let length = SignedDuration::from_secs(seconds);
    ensure!(
        length >= MIN_DEADLINE,
        "${remaining_usd:.2} left buys {} min at ${rate_usd_hr:.2}/hr with the 10% margin; at least 15 min is needed",
        seconds / 60
    );
    Ok(Reservation {
        deadline: now + length,
        reserved_usd: rate_usd_hr * MARGIN * seconds as f64 / 3600.0,
    })
}

impl Ledger {
    pub fn new(issue: &str) -> Self {
        Self {
            issue: issue.to_string(),
            account: None,
            budget: None,
            pods: Vec::new(),
        }
    }

    /// A key that cannot name its organization is refused only once the ledger has recorded one: it might belong
    /// to another account, and the fingerprint exists to stop exactly that.
    pub fn check_account(&self, current: Option<&str>) -> Result<()> {
        match (self.account.as_deref(), current) {
            (Some(recorded), Some(current)) if recorded != current => bail!(
                "issues/{}/harness-ledger.json belongs to RunPod organization {recorded}, but RUNPOD_API_KEY belongs to \
                 {current}; that key gets 404 for this ledger's pods, so the harness stops before reading one as gone",
                self.issue
            ),
            (Some(recorded), None) => bail!(
                "issues/{}/harness-ledger.json belongs to RunPod organization {recorded}, and RUNPOD_API_KEY cannot \
                 read GET /v2/organization, so the harness cannot tell it is the same account; use a key with \
                 membership read",
                self.issue
            ),
            _ => Ok(()),
        }
    }

    pub fn spent_usd(&self) -> f64 {
        self.pods
            .iter()
            .filter(|p| p.state == PodState::Terminated)
            .filter_map(|p| p.cost_usd)
            .sum()
    }

    pub fn reserved_usd(&self) -> f64 {
        self.pods
            .iter()
            .filter(|p| matches!(p.state, PodState::Reserved | PodState::Live))
            .map(|p| p.reserved_usd)
            .sum()
    }

    pub fn remaining_usd(&self) -> Result<f64> {
        let Some(budget) = &self.budget else {
            bail!(
                "issue {} has no budget; record the approval with `harness budget set`",
                self.issue
            );
        };
        Ok(budget.usd - self.spent_usd() - self.reserved_usd())
    }

    /// Whether a RunPod pod is accounted for here: a live entry by id, or a reservation (whose id is not known yet)
    /// by name. A terminated entry does not count, because a pod still listed after it was booked keeps billing.
    pub fn tracks(&self, id: &str, name: &str) -> bool {
        self.pods.iter().any(|entry| match entry.state {
            PodState::Live => entry.id.as_deref() == Some(id),
            PodState::Reserved => entry.name == name,
            PodState::Terminated | PodState::CreateFailed => false,
        })
    }

    /// The pod that is reserved or live, if any. v0 allows one per issue.
    pub fn active(&self) -> Option<&PodEntry> {
        self.pods
            .iter()
            .find(|p| matches!(p.state, PodState::Reserved | PodState::Live))
    }

    pub fn entry_mut(&mut self, name: &str) -> Result<&mut PodEntry> {
        match self.pods.iter_mut().find(|p| p.name == name) {
            Some(entry) => Ok(entry),
            None => bail!("ledger for {} has no pod named {name}", self.issue),
        }
    }

    pub fn reserve(&mut self, name: &str, offer: &Offer, planned_hours: f64, now: Timestamp) -> Result<PodEntry> {
        if let Some(active) = self.active() {
            bail!(
                "issue {} already has pod {} ({:?}); run `harness down {}` first",
                self.issue,
                active.name,
                active.state,
                self.issue
            );
        }
        let plan = plan_reservation(now, planned_hours, self.remaining_usd()?, offer.rate_usd_hr)?;
        let entry = PodEntry {
            name: name.to_string(),
            id: None,
            state: PodState::Reserved,
            gpu: offer.gpu.clone(),
            count: offer.count,
            cloud: offer.cloud,
            rate_usd_hr: offer.rate_usd_hr,
            reserved_usd: plan.reserved_usd,
            reserved_at: now,
            deadline: plan.deadline,
            terminated_at: None,
            cost_usd: None,
            cost_source: None,
        };
        self.pods.push(entry.clone());
        Ok(entry)
    }

    pub fn check(&self) -> Result<()> {
        for (i, pod) in self.pods.iter().enumerate() {
            ensure!(
                self.pods[..i].iter().all(|earlier| earlier.name != pod.name),
                "pod name {} appears twice",
                pod.name
            );
            let consistent = match pod.state {
                PodState::Reserved | PodState::CreateFailed => pod.id.is_none(),
                PodState::Live => pod.id.is_some(),
                PodState::Terminated => {
                    pod.id.is_some()
                        && pod.terminated_at.is_some()
                        && pod.cost_usd.is_some()
                        && pod.cost_source.is_some()
                }
            };
            ensure!(
                consistent,
                "pod {} has fields that do not fit state {:?}",
                pod.name,
                pod.state
            );
        }
        ensure!(
            self.pods
                .iter()
                .filter(|p| matches!(p.state, PodState::Reserved | PodState::Live))
                .count()
                <= 1,
            "more than one pod is reserved or live"
        );
        Ok(())
    }
}

impl PodEntry {
    pub fn reserved_rate_usd_hr(&self) -> f64 {
        self.rate_usd_hr * MARGIN
    }

    pub fn estimate_usd(&self, end: Timestamp) -> f64 {
        let seconds = end.duration_since(self.reserved_at).as_secs_f64().max(0.0);
        self.rate_usd_hr * seconds / 3600.0
    }

    /// A `create_failed` entry can still go live: a create that returns after the reaper released the
    /// reservation has made a pod, and tracking it matters more than the budget it no longer holds.
    pub fn mark_live(&mut self, id: &str) -> Result<()> {
        match self.state {
            PodState::Reserved | PodState::CreateFailed => {
                self.id = Some(id.to_string());
                self.state = PodState::Live;
                Ok(())
            }
            PodState::Live if self.id.as_deref() == Some(id) => Ok(()),
            PodState::Live | PodState::Terminated => {
                bail!(
                    "pod {} is {:?} with id {:?}; cannot attach id {id}",
                    self.name,
                    self.state,
                    self.id
                )
            }
        }
    }

    pub fn mark_create_failed(&mut self) -> Result<()> {
        match self.state {
            PodState::Reserved => {
                self.state = PodState::CreateFailed;
                Ok(())
            }
            PodState::Live | PodState::Terminated | PodState::CreateFailed => {
                bail!(
                    "pod {} is {:?}; only a reservation can fail to create",
                    self.name,
                    self.state
                )
            }
        }
    }

    pub fn mark_terminated(&mut self, at: Timestamp) -> Result<()> {
        match self.state {
            PodState::Live => {
                self.state = PodState::Terminated;
                self.terminated_at = Some(at);
                self.cost_usd = Some(self.estimate_usd(at));
                self.cost_source = Some(CostSource::Estimate);
                Ok(())
            }
            PodState::Reserved | PodState::Terminated | PodState::CreateFailed => {
                bail!(
                    "pod {} is {:?}; only a live pod can be terminated",
                    self.name,
                    self.state
                )
            }
        }
    }

    pub fn needs_billing(&self, now: Timestamp) -> bool {
        match (self.state, self.cost_source, self.terminated_at) {
            (PodState::Terminated, Some(CostSource::Estimate), Some(_)) => true,
            (PodState::Terminated, Some(CostSource::Billing), Some(at)) => now.duration_since(at) < BILLING_SETTLE,
            _ => false,
        }
    }

    /// Returns whether the entry changed. Billing replaces the estimate once it is at least as high, or once
    /// it has had a day to settle, so a partial record never understates what was spent.
    pub fn apply_billing(&mut self, billed_usd: f64, now: Timestamp) -> bool {
        let Some(at) = self.terminated_at else { return false };
        if self.state != PodState::Terminated {
            return false;
        }
        let settled = now.duration_since(at) >= BILLING_SETTLE;
        if billed_usd < self.estimate_usd(at) && !settled {
            return false;
        }
        let changed = self.cost_usd != Some(billed_usd) || self.cost_source != Some(CostSource::Billing);
        self.cost_usd = Some(billed_usd);
        self.cost_source = Some(CostSource::Billing);
        changed
    }
}

pub struct LedgerStore {
    issue: String,
    path: PathBuf,
    lock_path: PathBuf,
    account: Option<String>,
}

/// Run before anything reads a 404 as "gone": every ledger that loads must match the key's account. A ledger that
/// does not load is left to the reaper, which reports it.
pub fn check_accounts(repo: &Path, current: Option<&str>) -> Result<()> {
    for store in LedgerStore::all(repo)? {
        if let Ok(ledger) = store.load() {
            ledger.check_account(current)?;
        }
    }
    Ok(())
}

impl LedgerStore {
    pub fn new(repo: &Path, issue: &str) -> Self {
        let dir = repo.join("issues").join(issue);
        Self {
            issue: issue.to_string(),
            path: dir.join("harness-ledger.json"),
            lock_path: dir.join(".harness-ledger.lock"),
            account: None,
        }
    }

    /// Writes refuse a ledger of another account and record this one on a ledger that has none yet.
    pub fn with_account(mut self, account: Option<&str>) -> Self {
        self.account = account.map(str::to_string);
        self
    }

    /// Every issue that has a ledger, in name order.
    pub fn all(repo: &Path) -> Result<Vec<Self>> {
        let issues = repo.join("issues");
        let mut stores = Vec::new();
        for dir in fs::read_dir(&issues).with_context(|| format!("listing {}", issues.display()))? {
            let dir = dir?;
            let Some(issue) = dir.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if dir.path().join("harness-ledger.json").is_file() {
                stores.push(Self::new(repo, &issue));
            }
        }
        stores.sort_by(|a, b| a.issue.cmp(&b.issue));
        Ok(stores)
    }

    pub fn issue(&self) -> &str {
        &self.issue
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Reads without the lock: writes replace the file by rename, so a reader sees the old or the new ledger.
    pub fn load(&self) -> Result<Ledger> {
        let text = match fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Ledger::new(&self.issue)),
            Err(e) => return Err(e).with_context(|| format!("reading {}", self.path.display())),
        };
        let ledger: Ledger = serde_json::from_str(&text).with_context(|| format!("parsing {}", self.path.display()))?;
        ensure!(
            ledger.issue == self.issue,
            "{} belongs to issue {}",
            self.path.display(),
            ledger.issue
        );
        ledger
            .check()
            .with_context(|| format!("checking {}", self.path.display()))?;
        Ok(ledger)
    }

    /// Read-modify-write under an exclusive lock on a separate lock file. The ledger file itself cannot carry
    /// the lock, because each write replaces it with a new inode.
    pub fn update<T>(&self, change: impl FnOnce(&mut Ledger) -> Result<T>) -> Result<T> {
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&self.lock_path)
            .with_context(|| format!("opening {}", self.lock_path.display()))?;
        lock.lock()
            .with_context(|| format!("locking {}", self.lock_path.display()))?;
        let before = self.load()?;
        before.check_account(self.account.as_deref())?;
        let mut ledger = before.clone();
        let out = change(&mut ledger)?;
        if ledger != before {
            if ledger.account.is_none() {
                ledger.account.clone_from(&self.account);
            }
            ledger.check()?;
            self.write(&ledger)?;
        }
        Ok(out)
    }

    fn write(&self, ledger: &Ledger) -> Result<()> {
        let tmp = self.path.with_extension("json.tmp");
        let mut text = serde_json::to_string_pretty(ledger)?;
        text.push('\n');
        let mut file = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        fs::rename(&tmp, &self.path).with_context(|| format!("replacing {}", self.path.display()))?;
        if let Some(dir) = self.path.parent() {
            File::open(dir)?.sync_all()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(s: &str) -> Timestamp {
        s.parse().unwrap()
    }

    fn offer(rate: f64) -> Offer {
        Offer {
            gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition".into(),
            cloud: CloudTier::Secure,
            count: 1,
            rate_usd_hr: rate,
        }
    }

    fn funded(usd: f64) -> Ledger {
        let mut ledger = Ledger::new("vllm-47990");
        ledger.budget = Some(Budget {
            usd,
            approved_at: at("2026-10-16T17:00:00Z"),
            note: "chat".into(),
        });
        ledger
    }

    #[test]
    fn deadline_is_planned_hours_when_the_budget_covers_them() {
        let plan = plan_reservation(at("2026-10-16T17:00:00Z"), 3.0, 15.0, 2.49).unwrap();
        assert_eq!(plan.deadline, at("2026-10-16T20:00:00Z"));
        assert!((plan.reserved_usd - 2.49 * 1.10 * 3.0).abs() < 1e-9);
    }

    #[test]
    fn deadline_shrinks_to_what_the_budget_buys() {
        let plan = plan_reservation(at("2026-10-16T17:00:00Z"), 10.0, 5.0, 2.49).unwrap();
        let hours: f64 = 5.0 / (2.49 * 1.10);
        let expected = at("2026-10-16T17:00:00Z") + SignedDuration::from_secs((hours * 3600.0).floor() as i64);
        assert_eq!(plan.deadline, expected);
        assert!(plan.reserved_usd <= 5.0);
        assert!(plan.reserved_usd > 4.99);
    }

    #[test]
    fn refuses_a_deadline_under_fifteen_minutes() {
        let err = plan_reservation(at("2026-10-16T17:00:00Z"), 2.0, 0.60, 2.49).unwrap_err();
        assert!(err.to_string().contains("at least 15 min"), "{err}");
        assert!(plan_reservation(at("2026-10-16T17:00:00Z"), 0.2, 100.0, 2.49).is_err());
        let fifteen = 2.49 * 1.10 / 4.0;
        assert!(plan_reservation(at("2026-10-16T17:00:00Z"), 2.0, fifteen + 1e-6, 2.49).is_ok());
    }

    #[test]
    fn remaining_counts_terminated_cost_and_open_reservations() {
        let mut ledger = funded(20.0);
        let first = ledger
            .reserve("sm120-vllm-47990-aaaaaa", &offer(2.0), 1.0, at("2026-10-16T17:00:00Z"))
            .unwrap();
        assert!((ledger.remaining_usd().unwrap() - (20.0 - first.reserved_usd)).abs() < 1e-9);
        let entry = ledger.entry_mut("sm120-vllm-47990-aaaaaa").unwrap();
        entry.mark_live("pod1").unwrap();
        entry.mark_terminated(at("2026-10-16T17:30:00Z")).unwrap();
        assert!((ledger.spent_usd() - 1.0).abs() < 1e-9);
        assert!((ledger.remaining_usd().unwrap() - 19.0).abs() < 1e-9);
    }

    #[test]
    fn reserve_needs_a_budget_and_no_active_pod() {
        let mut bare = Ledger::new("vllm-47990");
        let err = bare
            .reserve("a", &offer(2.0), 1.0, at("2026-10-16T17:00:00Z"))
            .unwrap_err();
        assert!(err.to_string().contains("no budget"), "{err}");
        let mut ledger = funded(20.0);
        ledger
            .reserve("a", &offer(2.0), 1.0, at("2026-10-16T17:00:00Z"))
            .unwrap();
        let err = ledger
            .reserve("b", &offer(2.0), 1.0, at("2026-10-16T17:00:00Z"))
            .unwrap_err();
        assert!(err.to_string().contains("already has pod a"), "{err}");
    }

    #[test]
    fn transitions_follow_the_lifecycle() {
        let mut ledger = funded(20.0);
        ledger
            .reserve("a", &offer(2.0), 1.0, at("2026-10-16T17:00:00Z"))
            .unwrap();
        let entry = ledger.entry_mut("a").unwrap();
        assert!(entry.mark_terminated(at("2026-10-16T17:10:00Z")).is_err());
        entry.mark_live("p1").unwrap();
        entry.mark_live("p1").unwrap();
        assert!(entry.mark_live("p2").is_err());
        assert!(entry.mark_create_failed().is_err());
        entry.mark_terminated(at("2026-10-16T17:10:00Z")).unwrap();
        assert!(entry.mark_live("p1").is_err());
        assert_eq!(entry.cost_source, Some(CostSource::Estimate));
    }

    #[test]
    fn a_released_reservation_can_still_go_live() {
        let mut ledger = funded(20.0);
        ledger
            .reserve("a", &offer(2.0), 1.0, at("2026-10-16T17:00:00Z"))
            .unwrap();
        let entry = ledger.entry_mut("a").unwrap();
        entry.mark_create_failed().unwrap();
        entry.mark_live("late").unwrap();
        assert_eq!(entry.state, PodState::Live);
    }

    #[test]
    fn billing_replaces_the_estimate_only_when_it_cannot_understate() {
        let mut ledger = funded(20.0);
        ledger
            .reserve("a", &offer(2.0), 1.0, at("2026-10-16T17:00:00Z"))
            .unwrap();
        let entry = ledger.entry_mut("a").unwrap();
        entry.mark_live("p1").unwrap();
        entry.mark_terminated(at("2026-10-16T18:00:00Z")).unwrap();
        assert!(entry.needs_billing(at("2026-10-16T18:05:00Z")));
        assert!(!entry.apply_billing(0.5, at("2026-10-16T18:05:00Z")));
        assert_eq!(entry.cost_usd, Some(2.0));
        assert!(entry.apply_billing(2.1, at("2026-10-16T18:05:00Z")));
        assert_eq!(
            (entry.cost_usd, entry.cost_source),
            (Some(2.1), Some(CostSource::Billing))
        );
        assert!(entry.needs_billing(at("2026-10-17T17:00:00Z")));
        assert!(entry.apply_billing(1.9, at("2026-10-17T18:00:00Z")));
        assert!(!entry.needs_billing(at("2026-10-17T18:00:00Z")));
    }

    #[test]
    fn check_rejects_inconsistent_entries() {
        let mut ledger = funded(20.0);
        ledger
            .reserve("a", &offer(2.0), 1.0, at("2026-10-16T17:00:00Z"))
            .unwrap();
        ledger.pods[0].state = PodState::Live;
        assert!(ledger.check().is_err());
        ledger.pods[0].id = Some("p1".into());
        ledger.check().unwrap();
        ledger.pods.push(ledger.pods[0].clone());
        assert!(ledger.check().is_err());
    }

    #[test]
    fn an_account_is_checked_only_once_recorded() {
        let mut ledger = funded(20.0);
        ledger.check_account(Some("org-a")).unwrap();
        ledger.check_account(None).unwrap();
        ledger.account = Some("org-a".into());
        ledger.check_account(Some("org-a")).unwrap();
        assert!(ledger.check_account(Some("org-b")).is_err());
        assert!(ledger.check_account(None).is_err());
        let json = serde_json::to_value(&ledger).unwrap();
        assert_eq!(json["account"], "org-a");
        assert!(serde_json::to_value(funded(1.0)).unwrap().get("account").is_none());
    }

    #[test]
    fn serializes_states_and_times_as_the_design_shows() {
        let mut ledger = funded(15.0);
        ledger
            .reserve("sm120-vllm-47990-7f3a2c", &offer(2.49), 3.0, at("2026-10-16T17:03:00Z"))
            .unwrap();
        let json = serde_json::to_value(&ledger).unwrap();
        let pod = &json["pods"][0];
        assert_eq!(pod["state"], "reserved");
        assert_eq!(pod["cloud"], "SECURE");
        assert_eq!(pod["reserved_at"], "2026-10-16T17:03:00Z");
        assert_eq!(pod["deadline"], "2026-10-16T20:03:00Z");
        assert!(pod["id"].is_null() && pod["cost_source"].is_null());
        assert_eq!(serde_json::from_value::<Ledger>(json).unwrap(), ledger);
    }
}
