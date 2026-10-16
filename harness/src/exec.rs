use std::fmt::Write as _;
use std::fs;
use std::io::ErrorKind;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};

use crate::events::Kind;
use crate::pod::{self, AfterFailure, Ctx, LivePod, StillLive};
use crate::reap;
use crate::session::{TRANSFER_LIMIT, Watch, shell_quote, start_job, wait_job};
use crate::spec::{IssueSpec, Variant};

/// The variant as a bash file that `serve_variant.sh` sources. Every value is single-quoted, so spaces, quotes,
/// `$`, backticks, and newlines reach bash unchanged. Patches go as three parallel arrays (file stem, package,
/// strip level); SNAPSHOT_PKGS lists every package this issue's variants patch, so each is restored before every
/// variant whether or not this one patches it.
pub fn render_variant(spec: &IssueSpec, variant: &Variant) -> Result<String> {
    let server = &spec.server;
    let command: Vec<&str> = server
        .launch
        .iter()
        .chain(&server.base_flags)
        .chain(&variant.flags)
        .map(String::as_str)
        .collect();
    let env: Vec<String> = variant
        .env
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect();
    let (gsm8k, bench) = match &variant.quality {
        Some(quality) => (
            quality.gsm8k.to_string(),
            quality.bench.iter().map(|(c, n)| format!("{c}:{n}")).collect(),
        ),
        None => (String::new(), Vec::new()),
    };
    let strips: Vec<String> = variant.patches.iter().map(|patch| patch.strip.to_string()).collect();
    let mut out = String::new();
    for (name, value) in [
        ("VARIANT", variant.name.as_str()),
        ("PACKAGE", server.package.as_str()),
        ("MODEL", server.model.as_str()),
        ("PORT", &server.port.to_string()),
        ("FAIL_PATTERN", &server.fail_pattern),
        ("BOOT_TIMEOUT", &server.boot_timeout_s.to_string()),
        ("CONCURRENCY", &variant.smoke_concurrency.to_string()),
        ("GSM8K", &gsm8k),
    ] {
        writeln!(out, "{name}={}", shell_quote(value))?;
    }
    for (name, values) in [
        (
            "PATCH_FILES",
            variant
                .patches
                .iter()
                .map(|patch| patch.name.as_str())
                .collect::<Vec<_>>(),
        ),
        (
            "PATCH_PKGS",
            variant.patches.iter().map(|patch| patch.package.as_str()).collect(),
        ),
        ("PATCH_STRIPS", strips.iter().map(String::as_str).collect()),
        ("SNAPSHOT_PKGS", spec.snapshot_packages()),
        ("SERVER_CMD", command),
        ("VARIANT_ENV", env.iter().map(String::as_str).collect()),
        ("BENCH", bench.iter().map(String::as_str).collect()),
    ] {
        let quoted: Vec<String> = values.into_iter().map(shell_quote).collect();
        writeln!(out, "{name}=({})", quoted.join(" "))?;
    }
    ensure!(
        !out.contains('\0'),
        "variant {} has a NUL byte, which bash cannot hold",
        variant.name
    );
    Ok(out)
}

/// A variant resolved and rendered.
pub struct Planned<'s> {
    variant: &'s Variant,
    rendered: String,
}

/// Every requested variant, resolved and rendered before anything is reserved, so a typo costs nothing. No names
/// means every variant in file order.
pub fn plan<'s>(spec: &'s IssueSpec, names: &[String]) -> Result<Vec<Planned<'s>>> {
    let variants: Vec<&Variant> = if names.is_empty() {
        spec.variants.iter().collect()
    } else {
        names.iter().map(|name| spec.variant(name)).collect::<Result<_>>()?
    };
    ensure!(
        !variants.is_empty(),
        "issues/{}/harness.toml defines no variants",
        spec.issue
    );
    variants
        .into_iter()
        .map(|variant| {
            Ok(Planned {
                variant,
                rendered: render_variant(spec, variant)?,
            })
        })
        .collect()
}

/// `harness exec`: the pod outlives any error here, so every error names it and says how to end it.
pub fn exec(ctx: &Ctx, spec: &IssueSpec, live: &LivePod, names: &[String]) -> Result<()> {
    let run = || {
        let planned = plan(spec, names)?;
        // A patch or hooks.sh edited since `up` has to reach the pod before the variant that uses it.
        let issue_dir = format!("issues/{}", spec.issue);
        ctx.remote.upload(&live.target, ctx.repo, &[&issue_dir])?;
        run_planned(ctx, spec, live, &planned)
    };
    run().map_err(|e| e.context(StillLive::of(&spec.issue, &live.entry)))
}

fn run_planned(ctx: &Ctx, spec: &IssueSpec, live: &LivePod, planned: &[Planned]) -> Result<()> {
    for variant in planned {
        ctx.check_interrupted()?;
        run_variant(ctx, spec, live, variant)?;
    }
    Ok(())
}

/// How often a running variant's results come back, so a container that dies mid-run (RunPod stops one at about 98%
/// of its memory limit) leaves the run's logs up to then on the laptop.
const PULL_EVERY: Duration = Duration::from_secs(5 * 60);
/// A pull while the variant runs holds up its polls; the pull after it gets the full transfer limit.
const PULL_WHILE_RUNNING: Duration = Duration::from_secs(2 * 60);

fn run_variant(ctx: &Ctx, spec: &IssueSpec, live: &LivePod, planned: &Planned) -> Result<()> {
    let issue = spec.issue.as_str();
    let variant = planned.variant;
    let name = variant.name.as_str();
    let file = format!("results/{issue}/variants/{name}.sh");
    // The first poll can come before serve_variant.sh clears the last attempt, so its progress goes first.
    ctx.remote
        .run(&live.target, &format!("rm -rf results/{issue}/{name}"))?;
    clear_local(ctx, issue, name)?;
    ctx.remote.write_file(&live.target, &file, &planned.rendered)?;
    ctx.events.emit(
        Some(issue),
        Kind::VariantStart,
        json!({"variant": name, "patches": variant.patches, "flags": variant.flags}),
    );
    let job = format!("variant-{name}");
    let command = format!("bash harness/pod/serve_variant.sh issues/{issue} {file} results/{issue}");
    start_job(ctx.remote, &live.target, issue, &job, &command)?;
    let mut verdicts = 0;
    let progress = format!("results/{issue}/{name}/progress.jsonl");
    let so_far = [format!("results/{issue}/{name}")];
    let watch = Watch {
        on_line: &mut |line| {
            if progress_line(ctx, issue, name, line) {
                verdicts += 1;
            }
        },
        every: Some((PULL_EVERY, &mut || {
            if let Err(e) = pod::pull_paths(ctx, issue, &live.target, &so_far, PULL_WHILE_RUNNING) {
                ctx.warn(issue, format!("pull while {name} runs failed: {e:#}"));
            }
        })),
    };
    let waited = wait_job(ctx, issue, &live.entry, &live.target, &job, &progress, watch);
    // Also after a lost job or a stopped pod: whatever the pod still holds is all that is left of the run.
    if let Err(e) = pull_variant(ctx, issue, live, name) {
        ctx.warn(issue, format!("pull failed: {e:#}"));
    }
    let code = waited?;
    // A server crash is a verdict (BOOT_FAIL); a variant that never reached one is a harness error.
    ensure!(
        code == 0 && verdicts == 1,
        "variant {name} ended with exit {code} and {verdicts} verdict lines; see results/{issue}/jobs/{job}.log"
    );
    Ok(())
}

/// One line of serve_variant.sh's progress.jsonl: a phase as it starts, then the verdict as the last line.
/// Returns whether the line was the verdict. A line that is neither is reported, never fatal, since the run it
/// describes still goes on.
fn progress_line(ctx: &Ctx, issue: &str, variant: &str, line: &str) -> bool {
    match serde_json::from_str::<Value>(line) {
        Ok(Value::Object(fields)) if fields.contains_key("verdict") => {
            ctx.events.emit(Some(issue), Kind::Verdict, Value::Object(fields));
            true
        }
        Ok(Value::Object(fields)) if fields.contains_key("phase") => {
            ctx.events.emit(
                Some(issue),
                Kind::Phase,
                json!({"variant": variant, "phase": fields["phase"], "detail": fields.get("detail"), "t": fields.get("t")}),
            );
            false
        }
        _ => {
            ctx.events.emit(
                Some(issue),
                Kind::Warn,
                json!({"detail": "unreadable progress line", "variant": variant, "line": line}),
            );
            false
        }
    }
}

/// Files from an earlier attempt must not mix with this one's. The local copy is cleared before the run rather than
/// at its last pull, since that pull fails on a container that died and would then leave nothing of the run.
fn clear_local(ctx: &Ctx, issue: &str, variant: &str) -> Result<()> {
    let local = ctx.repo.join("results").join(issue).join(variant);
    match fs::remove_dir_all(&local) {
        Err(e) if e.kind() != ErrorKind::NotFound => Err(e).with_context(|| format!("clearing {}", local.display())),
        _ => Ok(()),
    }
}

/// Only this variant's results, the job logs, and the verdict table come back, so a long session does not pull
/// every earlier variant again. It writes over what the pulls during the run brought back and deletes nothing.
fn pull_variant(ctx: &Ctx, issue: &str, live: &LivePod, variant: &str) -> Result<()> {
    let paths = [
        format!("results/{issue}/{variant}"),
        format!("results/{issue}/jobs"),
        format!("results/{issue}/verdicts.jsonl"),
    ];
    pod::pull_paths(ctx, issue, &live.target, &paths, TRANSFER_LIMIT)
}

/// `up`, `exec`, `down`. `down` runs on every way out: errors and Ctrl-C come back here as `Err`, and a panic
/// reaches it through the guard's `Drop`.
pub fn run(ctx: &Ctx, spec: &IssueSpec, names: &[String]) -> Result<()> {
    let planned = plan(spec, names)?;
    // The guard below would end a pod that an earlier `up` left for `exec`, so stop before arming it.
    if let Some(active) = ctx.ledger(&spec.issue).load()?.active() {
        bail!(
            "issue {} already has pod {} ({:?}), and `run` would end it; use `harness exec` on it, or `harness \
             down {}` first",
            spec.issue,
            active.name,
            active.state,
            spec.issue
        );
    }
    let guard = Teardown {
        ctx,
        issue: &spec.issue,
        armed: true,
    };
    let result =
        pod::up(ctx, spec, true, AfterFailure::CallerEnds).and_then(|live| run_planned(ctx, spec, &live, &planned));
    let down = guard.finish();
    match (result, down) {
        (Ok(()), down) => down,
        (Err(e), Ok(())) => Err(e),
        // The teardown's error comes last so a pod it could not end keeps its StillLive context.
        (Err(e), Err(down)) => Err(down.context(format!("{e:#}; then down failed too"))),
    }
}

struct Teardown<'c, 'a> {
    ctx: &'c Ctx<'a>,
    issue: &'c str,
    armed: bool,
}

impl Teardown<'_, '_> {
    fn finish(mut self) -> Result<()> {
        self.armed = false;
        teardown(self.ctx, self.issue)
    }
}

impl Drop for Teardown<'_, '_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = teardown(self.ctx, self.issue);
        }
    }
}

/// The reaper first: a create with an unknown outcome leaves a reservation that it can adopt, so `down` then
/// terminates that pod too.
fn teardown(ctx: &Ctx, issue: &str) -> Result<()> {
    if let Err(e) = reap::reap(ctx) {
        ctx.warn(issue, format!("reaper failed: {e:#}"));
    }
    pod::down(ctx, issue)
}

#[cfg(test)]
mod tests {
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    const SPEC: &str = r#"
issue = "sglang-44291"
planned_hours = 2.0
[pod]
image = "lmsysorg/sglang:dev-cu13@sha256:0a1ba33aa4e032b8c5bcebdc69c45664074c032074add4f7db01629ec1fcc381"
count = 2
min_cuda = "13.0"
container_disk_gb = 80
volume_gb = 320
gpus = [{ id = "NVIDIA RTX PRO 6000 Blackwell Server Edition", cloud = "SECURE" }]
[server]
package = "sglang"
model = "nvidia/Qwen3.8-Flash-Next-NVFP4"
launch = ["python3", "-m", "sglang.launch_server", "--model-path", "nvidia/Qwen3.8-Flash-Next-NVFP4"]
base_flags = ["--json-model-override-args", '{"language_model_only":true}', "--tp-size", "2"]
port = 30000
fail_pattern = "illegal memory access|CUDA error: "
boot_timeout_s = 1800
patch_strip = 2
[variants.a6]
flags = ["--speculative-algorithm", "NEXTN"]
patches = ["ep-dispatcher-runner", { name = "fi-graphs", package = "flashinfer", strip = 1 }]
env = { HF_HUB_OFFLINE = "1", ODD = "it's $HOME `id`\nline2" }
smoke_concurrency = 9
[variants.a6.quality]
gsm8k = 200
bench = [[1, 8], [16, 64]]
[variants.b0]
"#;

    fn spec() -> IssueSpec {
        IssueSpec::parse(SPEC, "sglang-44291", |_| true).unwrap()
    }

    /// Sources the rendered file in real bash and prints every value NUL-separated.
    fn through_bash(rendered: &str) -> Vec<String> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!("sm120-variant-{}-{n}.sh", std::process::id()));
        std::fs::write(&path, rendered).unwrap();
        // The last value is empty unless the variables the pod scripts no longer read are still set.
        let script = r#"source "$1"
for v in "$VARIANT" "$PACKAGE" "$MODEL" "$PORT" "$FAIL_PATTERN" "$BOOT_TIMEOUT" "$CONCURRENCY" "$GSM8K" \
         "${#PATCH_FILES[@]}" "${PATCH_FILES[@]}" "${#PATCH_PKGS[@]}" "${PATCH_PKGS[@]}" \
         "${#PATCH_STRIPS[@]}" "${PATCH_STRIPS[@]}" "${#SNAPSHOT_PKGS[@]}" "${SNAPSHOT_PKGS[@]}" \
         "${#SERVER_CMD[@]}" "${SERVER_CMD[@]}" "${#VARIANT_ENV[@]}" "${VARIANT_ENV[@]}" \
         "${#BENCH[@]}" "${BENCH[@]}" "${PATCHES+PATCHES}${PATCH_STRIP+PATCH_STRIP}"; do
    printf '%s\0' "$v"
done"#;
        let out = Command::new("bash")
            .arg("-c")
            .arg(script)
            .arg("bash")
            .arg(&path)
            .output()
            .unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let text = String::from_utf8(out.stdout).unwrap();
        text.split_terminator('\0').map(str::to_string).collect()
    }

    #[test]
    fn rendered_variant_round_trips_through_bash() {
        let spec = spec();
        let values = through_bash(&render_variant(&spec, spec.variant("a6").unwrap()).unwrap());
        let expected = [
            "a6",
            "sglang",
            "nvidia/Qwen3.8-Flash-Next-NVFP4",
            "30000",
            "illegal memory access|CUDA error: ",
            "1800",
            "9",
            "200",
            "2",
            "ep-dispatcher-runner",
            "fi-graphs",
            "2",
            "sglang",
            "flashinfer",
            "2",
            "2",
            "1",
            "2",
            "sglang",
            "flashinfer",
            "11",
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "nvidia/Qwen3.8-Flash-Next-NVFP4",
            "--json-model-override-args",
            "{\"language_model_only\":true}",
            "--tp-size",
            "2",
            "--speculative-algorithm",
            "NEXTN",
            "2",
            "HF_HUB_OFFLINE=1",
            "ODD=it's $HOME `id`\nline2",
            "2",
            "1:8",
            "16:64",
            "",
        ];
        assert_eq!(values, expected);
    }

    #[test]
    fn a_bare_variant_still_restores_every_package_the_issue_patches() {
        let spec = spec();
        let values = through_bash(&render_variant(&spec, spec.variant("b0").unwrap()).unwrap());
        assert_eq!(&values[6..14], ["1", "", "0", "0", "0", "2", "sglang", "flashinfer"]);
        assert_eq!(&values[values.len() - 3..], ["0", "0", ""]);
    }

    #[test]
    fn refuses_a_nul_byte() {
        let text = SPEC.replace("\"NEXTN\"", "\"NEXT\\u0000N\"");
        let spec = IssueSpec::parse(&text, "sglang-44291", |_| true).unwrap();
        let err = render_variant(&spec, spec.variant("a6").unwrap()).unwrap_err();
        assert!(err.to_string().contains("NUL byte"), "{err}");
    }
}
