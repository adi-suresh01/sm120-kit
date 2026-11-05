//! What a Vast box runs and how `up` checks it: the onstart loader, the entrypoint with Vast's part spliced in, the
//! create body that carries both, and the check that the box's watchdog holds what it needs to destroy the instance.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use jiff::{SignedDuration, Timestamp};

use crate::ledger::PodEntry;
use crate::pod::{Ctx, ENTRYPOINT, POLL, WATCHDOG_LOG, apt_packages, watchdog_armed_prefix};
use crate::session::Target;
use crate::spec::{IssueSpec, Provider};
use crate::vast::VastCreateRequest;

/// Vast's part of the entrypoint, spliced in by `vast_entrypoint`.
pub const ENTRYPOINT_VAST: &str = include_str!("../pod/entrypoint_vast.sh");
/// Vast's `onstart`: a loader for the entrypoint, which travels base64 in `HARNESS_ENTRYPOINT_B64` so that its size
/// never meets a limit on `onstart`. POSIX sh, since which shell Vast runs it with is not documented. The entrypoint
/// runs as `/bin/bash -c <text>`, its command line on RunPod too, which the watchdog check looks for. Tests only:
/// HARNESS_TEST_ROOT prefixes the paths, as in the entrypoint.
pub const VAST_ONSTART: &str = r#"root=${HARNESS_TEST_ROOT:-}
mkdir -p "$root/workspace"
entry=$(printf '%s' "$HARNESS_ENTRYPOINT_B64" | base64 -d) || { echo "harness: HARNESS_ENTRYPOINT_B64 did not decode; this box has no watchdog" >&2; exit 1; }
if command -v setsid >/dev/null 2>&1; then
    setsid /bin/bash -c "$entry" >"$root/workspace/harness-entrypoint.log" 2>&1 </dev/null &
else
    /bin/bash -c "$entry" >"$root/workspace/harness-entrypoint.log" 2>&1 </dev/null &
fi
"#;
/// Vast's own sshd, reached directly where the host maps a port and through Vast's proxy where not.
const VAST_RUNTYPE: &str = "ssh_direc ssh_proxy";
/// How long a Vast read may come back empty right after the create before the instance counts as gone.
pub const VAST_FIRST_READ: SignedDuration = SignedDuration::from_mins(2);

/// entrypoint.sh with Vast's part spliced in before its last line, `main`, so Vast's definitions replace RunPod's
/// while the watchdog stays one piece of code, and RunPod pods keep running the file as it is.
pub fn vast_entrypoint() -> Result<String> {
    let body = ENTRYPOINT
        .strip_suffix("\nmain\n")
        .context("entrypoint.sh no longer ends with `main`, so Vast's part has nowhere to go")?;
    Ok(format!("{body}\n{ENTRYPOINT_VAST}\nmain\n"))
}

/// The create body for a Vast offer. `entrypoint_b64` is `vast_entrypoint` in base64, made once before any
/// reservation so that nothing can fail between the reservation and the create.
pub fn vast_create_request(spec: &IssueSpec, entry: &PodEntry, entrypoint_b64: &str) -> VastCreateRequest {
    let env = BTreeMap::from([
        (
            "HARNESS_DEADLINE_EPOCH".to_string(),
            entry.deadline.as_second().to_string(),
        ),
        ("HARNESS_APT".to_string(), apt_packages(spec, Provider::Vast)),
        ("HF_HOME".to_string(), "/workspace/hf".to_string()),
        ("HARNESS_ENTRYPOINT_B64".to_string(), entrypoint_b64.to_string()),
    ]);
    VastCreateRequest {
        client_id: "me".into(),
        image: spec.pod.image.clone(),
        env,
        disk: spec.pod.vast_disk_gb(),
        label: entry.name.clone(),
        onstart: VAST_ONSTART.into(),
        runtype: VAST_RUNTYPE.into(),
        cancel_unavail: true,
    }
}

/// Standard base64 with padding, as `base64 -d` on the box reads it.
pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, b)| n | u32::from(*b) << (16 - 8 * i));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(char::from(ALPHABET[(n >> (18 - 6 * i) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// What Vast's part of the entrypoint logs once it holds what it needs to destroy this very instance.
fn vast_ready_line(id: &str) -> String {
    format!("vast self-destruct ready for instance {id}")
}

/// Logged once apt is done, since Vast's own sshd answers before the box has the packages the jobs use.
const VAST_SETUP_DONE: &str = "box setup done";
/// What Vast's part of the entrypoint logs when the box cannot destroy itself: no key, no id, or no curl.
const VAST_MISSING: &str = "VAST SELF-DESTRUCT MISSING";
/// The start of the line that reports a read of this instance with its own key: evidence the key works, short of the
/// destroy itself.
const VAST_PROBE: &str = "vast key reads instance";
/// The loader starts the entrypoint seconds after the container; past this there is none.
const VAST_ARM_WAIT: SignedDuration = SignedDuration::from_mins(5);
const VAST_SETUP_WAIT: SignedDuration = SignedDuration::from_mins(15);

/// On Vast the entrypoint runs from the instance's onstart, beside Vast's own init, so it cannot be the container's
/// main process, and something that kills it leaves the box running until `down` or the reaper. It counts as armed
/// while the pid that logged the armed line is alive and runs the entrypoint, and the entrypoint reported the key and
/// id to destroy this instance. Prints SETUP_RUNNING while apt still runs.
fn vast_watchdog_check_script(deadline: Timestamp, id: &str, log: &str, proc: &str) -> String {
    let armed = watchdog_armed_prefix(deadline);
    let ready = vast_ready_line(id);
    format!(
        r#"log={log}; proc={proc}
pid=$(sed -nE 's/^[^ ]* {armed}([0-9]+)$/\1/p' "$log" 2>/dev/null | tail -n 1)
own=""
if [[ -n $pid && -r $proc/$pid/cmdline ]]; then
    own=$(tr '\0' ' ' <"$proc/$pid/cmdline" 2>/dev/null)
fi
if grep -q ' {VAST_MISSING}' "$log" 2>/dev/null; then
    echo SELF_DESTRUCT_MISSING
    grep ' {VAST_MISSING}' "$log" | tail -n 1
elif [[ $own != "/bin/bash -c #!/usr/bin/env bash"* ]] || ! grep -qx '[^ ]* {ready}' "$log" 2>/dev/null; then
    echo WATCHDOG_MISSING
    tail -n 5 "$log" 2>/dev/null || true
    ps -eo pid,ppid,user,args --forest 2>/dev/null | head -n 30 | cut -c1-160 || true
elif ! grep -q ' {VAST_SETUP_DONE}' "$log" 2>/dev/null; then
    echo SETUP_RUNNING
else
    echo WATCHDOG_ARMED
fi
grep ' {VAST_PROBE} ' "$log" 2>/dev/null | tail -n 1 || true"#
    )
}

/// Vast's sshd answers before the onstart has run, so the check repeats until the watchdog reports itself armed
/// and the box set up, or a limit passes; a failed check (no container env yet) counts as not yet.
pub fn wait_vast_watchdog(ctx: &Ctx, issue: &str, entry: &PodEntry, target: &Target) -> Result<()> {
    let id = entry.id.as_deref().context("live pod without an id")?;
    let script = vast_watchdog_check_script(entry.deadline, id, WATCHDOG_LOG, "/proc");
    let started = ctx.clock.now();
    loop {
        let (state, last) = match ctx.remote.run(target, &script) {
            Ok(reply) => (reply.lines().next().unwrap_or("").to_string(), reply),
            Err(e) => (String::new(), format!("{e:#}")),
        };
        let waited = ctx.clock.now().duration_since(started);
        match state.as_str() {
            "WATCHDOG_ARMED" => return check_vast_probe(ctx, issue, entry, &last),
            "SELF_DESTRUCT_MISSING" => bail!(
                "pod {} cannot destroy itself at its deadline, so nothing on the box would end it: {:?}",
                entry.name,
                last.lines().nth(1).unwrap_or("").trim()
            ),
            "SETUP_RUNNING" => ensure!(
                waited < VAST_SETUP_WAIT,
                "pod {}: apt on the box ran past {} min; the watchdog log ends: {:?}",
                entry.name,
                VAST_SETUP_WAIT.as_mins(),
                last.trim()
            ),
            _ => ensure!(
                waited < VAST_ARM_WAIT,
                "pod {} shows no watchdog armed for its deadline {} with the key to destroy instance {id} within {} \
                 min, so nothing on the box would end it; the last check: {:?}",
                entry.name,
                entry.deadline,
                VAST_ARM_WAIT.as_mins(),
                last.trim()
            ),
        }
        ctx.check_interrupted()?;
        ctx.clock.sleep(POLL);
    }
}

/// The box's read of its own instance with its own key, the one sign before the deadline that the destroy can work.
/// A key Vast rejects (401 or 403) ends the box; any other failure only warns, since a read can fail where a destroy
/// would not, and the reaper ends the box past its deadline either way.
fn check_vast_probe(ctx: &Ctx, issue: &str, entry: &PodEntry, reply: &str) -> Result<()> {
    let probe = reply.lines().find(|line| line.contains(VAST_PROBE));
    if probe.is_some_and(|line| line.contains(": http 2")) {
        return Ok(());
    }
    let seen = probe.map_or("no read logged", str::trim);
    ensure!(
        !seen.ends_with(": http 401") && !seen.ends_with(": http 403"),
        "pod {}: Vast rejected the box's own key ({seen}), so it could not destroy itself at its deadline",
        entry.name
    );
    ctx.warn(
        issue,
        format!(
            "pod {}: the box's own Vast key did not read its instance ({seen}), so its self-destruct may be refused; \
             `harness down {issue}` or the reaper still ends it",
            entry.name
        ),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::PodState;

    #[test]
    fn base64_matches_the_standard_alphabet_and_padding() {
        for (plain, encoded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foobar", "Zm9vYmFy"),
            ("\u{0}\u{ff}?>", "AMO/Pz4="),
        ] {
            assert_eq!(base64(plain.as_bytes()), encoded, "{plain:?}");
        }
    }

    #[test]
    fn vasts_entrypoint_is_runpods_with_vasts_part_before_main() {
        let script = vast_entrypoint().unwrap();
        let body = ENTRYPOINT.strip_suffix("\nmain\n").unwrap();
        assert!(script.starts_with(body) && script.ends_with(&format!("{ENTRYPOINT_VAST}\nmain\n")));
        assert!(script.starts_with("#!/usr/bin/env bash\n"));
        assert!(script.contains("log \"watchdog armed for epoch $deadline in pid $$\""));
        assert!(ENTRYPOINT_VAST.contains(&format!("log \"{}", vast_ready_line("$vast_id"))));
        assert!(ENTRYPOINT_VAST.contains(&format!("log \"{VAST_SETUP_DONE}\"")));
        assert_eq!(ENTRYPOINT_VAST.matches(&format!("log \"{VAST_MISSING}: ")).count(), 2);
        assert!(ENTRYPOINT_VAST.contains(&format!("log \"{VAST_PROBE} $vast_id: http $vast_code\"")));
    }

    fn entry() -> PodEntry {
        PodEntry {
            name: "sm120-harness-test-vast-0a0b0c".into(),
            provider: Provider::Vast,
            id: None,
            state: PodState::Reserved,
            gpu: "GTX 1650".into(),
            count: 1,
            cloud: crate::spec::CloudTier::Community,
            rate_usd_hr: 0.05,
            reserved_usd: 0.0137,
            reserved_at: "2026-11-04T17:00:00Z".parse().unwrap(),
            deadline: "2026-11-04T17:15:00Z".parse().unwrap(),
            terminated_at: None,
            cost_usd: None,
            cost_source: None,
            placement: Some(crate::vast::sample_offer(0.05).placement()),
        }
    }

    #[test]
    fn the_vast_create_carries_the_deadline_the_label_and_the_loader() {
        let spec = IssueSpec::parse(
            include_str!("../../issues/harness-test-vast/harness.toml"),
            "harness-test-vast",
            |_| true,
        )
        .unwrap();
        let request = vast_create_request(&spec, &entry(), "ZW50cnk=");
        assert_eq!(request.label, "sm120-harness-test-vast-0a0b0c");
        assert_eq!(request.env["HARNESS_DEADLINE_EPOCH"], "1793812500");
        assert_eq!(request.env["HARNESS_ENTRYPOINT_B64"], "ZW50cnk=");
        assert_eq!(request.env["HARNESS_APT"], "curl procps patch");
        assert!(!request.env.contains_key("HF_TOKEN"));
        assert_eq!(request.disk, spec.pod.vast_disk_gb());
        assert!(request.cancel_unavail);
        assert_eq!(
            (request.client_id.as_str(), request.runtype.as_str()),
            ("me", "ssh_direc ssh_proxy")
        );
        // Each harness variable is read by the loader or the entrypoint under the same name.
        let script = vast_entrypoint().unwrap();
        for key in request.env.keys().filter(|key| key.starts_with("HARNESS_")) {
            let read = format!("${key}");
            let braced = format!("${{{key}");
            assert!(
                [&request.onstart, &script]
                    .iter()
                    .any(|text| text.contains(&read) || text.contains(&braced)),
                "nothing on the box reads {key}"
            );
        }
    }

    fn vast_check(log_lines: &[&str], procs: &[(u32, &str)]) -> Vec<String> {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("sm120-vast-watchdog-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for (pid, cmdline) in procs {
            let p = dir.join("proc").join(pid.to_string());
            std::fs::create_dir_all(&p).unwrap();
            std::fs::write(p.join("cmdline"), cmdline.replace(' ', "\0") + "\0").unwrap();
        }
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("watchdog.log");
        std::fs::write(&log, log_lines.join("\n") + "\n").unwrap();
        let deadline: Timestamp = "2026-10-16T20:03:00Z".parse().unwrap();
        let script = vast_watchdog_check_script(
            deadline,
            "31000001",
            log.to_str().unwrap(),
            dir.join("proc").to_str().unwrap(),
        );
        let out = std::process::Command::new("bash")
            .arg("-c")
            .arg(format!("set -euo pipefail\n{script}"))
            .output()
            .unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn the_vast_watchdog_counts_once_alive_holding_the_key_for_this_instance_and_set_up() {
        const ENTRY: &str = "/bin/bash -c #!/usr/bin/env bash\n# Pod entrypoint";
        let armed = "2026-10-16T19:48:00Z watchdog armed for epoch 1792180980 in pid 812";
        let ready = "2026-10-16T19:48:00Z vast self-destruct ready for instance 31000001";
        let done = "2026-10-16T19:49:10Z box setup done";
        let probe = "2026-10-16T19:48:01Z vast key reads instance 31000001: http 200";
        let alive = [(812, ENTRY)];
        let reply = vast_check(&[ready, armed, probe, done], &alive);
        assert_eq!(reply, ["WATCHDOG_ARMED", probe]);
        assert_eq!(vast_check(&[ready, armed], &alive)[0], "SETUP_RUNNING");
        let no_curl = "2026-10-16T19:49:10Z VAST SELF-DESTRUCT MISSING: curl is not installed";
        assert_eq!(
            vast_check(&[ready, armed, no_curl], &alive)[..2],
            ["SELF_DESTRUCT_MISSING", no_curl]
        );
        for (lines, procs) in [
            // the key or the id never reached the entrypoint
            (vec![armed, done], &alive[..]),
            // ready for some other instance
            (
                vec![
                    "2026-10-16T19:48:00Z vast self-destruct ready for instance 310000012",
                    armed,
                    done,
                ],
                &alive[..],
            ),
            // the watchdog has died since it logged
            (vec![ready, armed, done], &[][..]),
            // a pid that is not the entrypoint
            (vec![ready, armed, done], &[(812, "sshd: /usr/sbin/sshd")][..]),
            // a deadline that is not the ledger's
            (
                vec![
                    ready,
                    "2026-10-16T19:48:00Z watchdog armed for epoch 1792180000 in pid 812",
                    done,
                ],
                &alive[..],
            ),
        ] {
            assert_eq!(vast_check(&lines, procs)[0], "WATCHDOG_MISSING", "{lines:?}");
        }
    }
}
