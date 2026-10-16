use std::fmt::Write as _;

use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};

use crate::events::Kind;
use crate::pod::{self, Ctx, LivePod};
use crate::reap;
use crate::session::{shell_quote, start_job};
use crate::spec::{IssueSpec, Variant};

/// The variant as a bash file that `serve_variant.sh` sources. Every value is single-quoted, so spaces, quotes,
/// `$`, backticks, and newlines reach bash unchanged.
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
    let mut out = String::new();
    for (name, value) in [
        ("VARIANT", variant.name.as_str()),
        ("PACKAGE", server.package.as_str()),
        ("MODEL", server.model.as_str()),
        ("PORT", &server.port.to_string()),
        ("FAIL_PATTERN", &server.fail_pattern),
        ("BOOT_TIMEOUT", &server.boot_timeout_s.to_string()),
        ("CONCURRENCY", &variant.smoke_concurrency.to_string()),
        ("PATCH_STRIP", &server.patch_strip.to_string()),
        ("GSM8K", &gsm8k),
    ] {
        writeln!(out, "{name}={}", shell_quote(value))?;
    }
    for (name, values) in [
        (
            "PATCHES",
            variant.patches.iter().map(String::as_str).collect::<Vec<_>>(),
        ),
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

pub fn exec(ctx: &Ctx, spec: &IssueSpec, live: &LivePod, names: &[String]) -> Result<()> {
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
    for variant in variants {
        ctx.check_interrupted()?;
        run_variant(ctx, spec, live, variant)?;
    }
    Ok(())
}

fn run_variant(ctx: &Ctx, spec: &IssueSpec, live: &LivePod, variant: &Variant) -> Result<()> {
    let issue = spec.issue.as_str();
    let name = variant.name.as_str();
    let file = format!("results/{issue}/variants/{name}.sh");
    // The first poll can come before serve_variant.sh clears the last attempt, so its verdict goes first.
    ctx.remote
        .run(&live.target, &format!("rm -rf results/{issue}/{name}"))?;
    ctx.remote
        .write_file(&live.target, &file, &render_variant(spec, variant)?)?;
    ctx.events.emit(
        Some(issue),
        Kind::VariantStart,
        json!({"variant": name, "patches": variant.patches, "flags": variant.flags}),
    );
    let job = format!("variant-{name}");
    let command = format!("bash harness/pod/serve_variant.sh issues/{issue} {file} results/{issue}");
    start_job(ctx.remote, &live.target, issue, &job, &command)?;
    let mut verdicts = 0;
    let verdict_file = format!("results/{issue}/{name}/verdict.json");
    let code = pod::wait_job(
        ctx,
        issue,
        &live.entry,
        &live.target,
        &job,
        &verdict_file,
        &mut |line| {
            verdicts += 1;
            emit_verdict(ctx, issue, line);
        },
    )?;
    if let Err(e) = pod::pull(ctx, issue, &live.target) {
        ctx.warn(issue, format!("pull failed: {e:#}"));
    }
    // A server crash is a verdict (BOOT_FAIL); a variant that never reached one is a harness error.
    ensure!(
        code == 0 && verdicts == 1,
        "variant {name} ended with exit {code} and {verdicts} verdict lines; see results/{issue}/jobs/{job}.log"
    );
    Ok(())
}

fn emit_verdict(ctx: &Ctx, issue: &str, line: &str) {
    match serde_json::from_str::<Value>(line) {
        Ok(Value::Object(fields)) => ctx.events.emit(Some(issue), Kind::Verdict, Value::Object(fields)),
        _ => ctx.events.emit(
            Some(issue),
            Kind::Warn,
            json!({"detail": "unreadable verdict line", "line": line}),
        ),
    }
}

/// `up`, `exec`, `down`. `down` runs on every way out: errors and Ctrl-C come back here as `Err`, and a panic
/// reaches it through the guard's `Drop`.
pub fn run(ctx: &Ctx, spec: &IssueSpec, names: &[String]) -> Result<()> {
    let guard = Teardown {
        ctx,
        issue: &spec.issue,
        armed: true,
    };
    let result = pod::up(ctx, spec, true).and_then(|live| exec(ctx, spec, &live, names));
    let down = guard.finish();
    match (result, down) {
        (Ok(()), down) => down,
        (Err(e), Ok(())) => Err(e),
        (Err(e), Err(down)) => bail!("{e:#}; then down failed too: {down:#}"),
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
patches = ["ep-dispatcher-runner"]
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
        let script = r#"source "$1"
for v in "$VARIANT" "$PACKAGE" "$MODEL" "$PORT" "$FAIL_PATTERN" "$BOOT_TIMEOUT" "$CONCURRENCY" "$PATCH_STRIP" \
         "$GSM8K" "${#PATCHES[@]}" "${PATCHES[@]}" "${#SERVER_CMD[@]}" "${SERVER_CMD[@]}" \
         "${#VARIANT_ENV[@]}" "${VARIANT_ENV[@]}" "${#BENCH[@]}" "${BENCH[@]}"; do
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
            "2",
            "200",
            "1",
            "ep-dispatcher-runner",
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
        ];
        assert_eq!(values, expected);
    }

    #[test]
    fn a_bare_variant_renders_empty_lists() {
        let spec = spec();
        let values = through_bash(&render_variant(&spec, spec.variant("b0").unwrap()).unwrap());
        assert_eq!(&values[6..10], ["1", "2", "", "0"]);
        assert_eq!(values.last().unwrap(), "0");
    }

    #[test]
    fn refuses_a_nul_byte() {
        let text = SPEC.replace("\"NEXTN\"", "\"NEXT\\u0000N\"");
        let spec = IssueSpec::parse(&text, "sglang-44291", |_| true).unwrap();
        let err = render_variant(&spec, spec.variant("a6").unwrap()).unwrap_err();
        assert!(err.to_string().contains("NUL byte"), "{err}");
    }
}
