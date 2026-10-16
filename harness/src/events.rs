use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{Map, Value};

use crate::clock::Clock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Budget,
    Quote,
    Reserve,
    Create,
    Ready,
    Setup,
    VariantStart,
    Verdict,
    Pull,
    Terminate,
    Reap,
    Warn,
    Error,
}

/// Appends to `results/<issue>/events.jsonl` (or `results/harness-events.jsonl` for events that belong to no issue)
/// and, with `mirror`, writes each event to stdout as the same single line, so a monitor needs only
/// `tail -F | grep`.
pub struct Events<'a> {
    results: PathBuf,
    clock: &'a dyn Clock,
    mirror: bool,
}

impl<'a> Events<'a> {
    pub fn new(results: &Path, clock: &'a dyn Clock, mirror: bool) -> Self {
        Self {
            results: results.to_path_buf(),
            clock,
            mirror,
        }
    }

    /// Never fails and never panics: `println!` panics on a closed terminal, and a panic here during `run`'s
    /// unwinding would abort before `down`.
    pub fn emit(&self, issue: Option<&str>, kind: Kind, fields: Value) {
        let line = self.line(issue, kind, fields);
        if self.mirror {
            let _ = writeln!(io::stdout().lock(), "{line}");
        }
        if let Err(e) = self.append(issue, &line) {
            let _ = writeln!(io::stderr().lock(), "warning: could not record event: {e}");
        }
    }

    fn line(&self, issue: Option<&str>, kind: Kind, fields: Value) -> String {
        let mut event = Map::new();
        event.insert("ts".into(), Value::String(self.clock.now().to_string()));
        event.insert("issue".into(), issue.map_or(Value::Null, |i| Value::String(i.into())));
        event.insert("kind".into(), serde_json::to_value(kind).unwrap_or(Value::Null));
        if let Value::Object(extra) = fields {
            event.extend(extra);
        }
        Value::Object(event).to_string()
    }

    fn append(&self, issue: Option<&str>, line: &str) -> std::io::Result<()> {
        let path = match issue {
            Some(issue) => self.results.join(issue).join("events.jsonl"),
            None => self.results.join("harness-events.jsonl"),
        };
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(file, "{line}")
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use jiff::Timestamp;
    use serde_json::json;

    use super::*;

    struct Fixed;

    impl Clock for Fixed {
        fn now(&self) -> Timestamp {
            "2026-10-16T17:03:00Z".parse().unwrap()
        }

        fn sleep(&self, _: Duration) {}
    }

    #[test]
    fn writes_one_ordered_json_line_per_event() {
        let dir = std::env::temp_dir().join(format!("sm120-events-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let events = Events::new(&dir, &Fixed, false);
        events.emit(
            Some("vllm-47990"),
            Kind::Verdict,
            json!({"variant": "a1", "verdict": "BOOT_FAIL"}),
        );
        events.emit(None, Kind::Warn, json!({"detail": "untracked pod"}));
        let issue_log = fs::read_to_string(dir.join("vllm-47990/events.jsonl")).unwrap();
        assert_eq!(
            issue_log,
            "{\"ts\":\"2026-10-16T17:03:00Z\",\"issue\":\"vllm-47990\",\"kind\":\"verdict\",\"variant\":\"a1\",\"verdict\":\"BOOT_FAIL\"}\n"
        );
        let general = fs::read_to_string(dir.join("harness-events.jsonl")).unwrap();
        assert!(general.contains("\"issue\":null,\"kind\":\"warn\""), "{general}");
        fs::remove_dir_all(&dir).unwrap();
    }
}
