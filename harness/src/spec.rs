use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::de::{Deserializer, MapAccess, Visitor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CloudTier {
    Secure,
    Community,
}

impl CloudTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Secure => "SECURE",
            Self::Community => "COMMUNITY",
        }
    }
}

/// The serving packages the pod-side scripts know how to snapshot, patch, and clean up after.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Package {
    Vllm,
    Sglang,
}

impl Package {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
            Self::Sglang => "sglang",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IssueSpec {
    pub issue: String,
    pub planned_hours: f64,
    pub pod: PodSpec,
    pub server: ServerSpec,
    #[serde(default, deserialize_with = "variants_in_file_order")]
    pub variants: Vec<Variant>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PodSpec {
    pub image: String,
    pub count: u32,
    pub min_cuda: String,
    pub container_disk_gb: u32,
    pub volume_gb: u32,
    require_pcie: Option<bool>,
    #[serde(default)]
    pub apt: Vec<String>,
    #[serde(default)]
    pub needs_hf_token: bool,
    pub gpus: Vec<GpuChoice>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuChoice {
    pub id: String,
    pub cloud: CloudTier,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerSpec {
    pub package: Package,
    pub model: String,
    pub launch: Vec<String>,
    #[serde(default)]
    pub base_flags: Vec<String>,
    pub port: u16,
    pub fail_pattern: String,
    pub boot_timeout_s: u32,
    #[serde(default = "default_patch_strip")]
    pub patch_strip: u32,
}

#[derive(Debug)]
pub struct Variant {
    pub name: String,
    pub flags: Vec<String>,
    pub patches: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub smoke_concurrency: u32,
    pub quality: Option<Quality>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VariantBody {
    #[serde(default)]
    flags: Vec<String>,
    #[serde(default)]
    patches: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default = "default_smoke_concurrency")]
    smoke_concurrency: u32,
    quality: Option<Quality>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Quality {
    pub gsm8k: u32,
    #[serde(default)]
    pub bench: Vec<(u32, u32)>,
}

fn default_patch_strip() -> u32 {
    1
}

fn default_smoke_concurrency() -> u32 {
    1
}

/// `run` with no variant names runs them in the order the file lists them, so the control can come first.
fn variants_in_file_order<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<Variant>, D::Error> {
    struct Ordered;

    impl<'de> Visitor<'de> for Ordered {
        type Value = Vec<Variant>;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a table of variant tables")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut variants = Vec::new();
            while let Some((name, body)) = map.next_entry::<String, VariantBody>()? {
                variants.push(Variant {
                    name,
                    flags: body.flags,
                    patches: body.patches,
                    env: body.env,
                    smoke_concurrency: body.smoke_concurrency,
                    quality: body.quality,
                });
            }
            Ok(variants)
        }
    }

    deserializer.deserialize_map(Ordered)
}

impl IssueSpec {
    pub fn load(repo: &Path, issue: &str) -> Result<Self> {
        let issue_dir = repo.join("issues").join(issue);
        let path = issue_dir.join("harness.toml");
        let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let patches = issue_dir.join("patches");
        Self::parse(&text, issue, |patch| patches.join(format!("{patch}.diff")).is_file())
            .with_context(|| format!("in {}", path.display()))
    }

    pub fn parse(text: &str, issue: &str, patch_exists: impl Fn(&str) -> bool) -> Result<Self> {
        let spec: Self = toml::from_str(text)?;
        spec.validate(issue, &patch_exists)?;
        Ok(spec)
    }

    pub fn variant(&self, name: &str) -> Result<&Variant> {
        match self.variants.iter().find(|v| v.name == name) {
            Some(variant) => Ok(variant),
            None => bail!("issue {} has no variant {name:?}", self.issue),
        }
    }

    fn validate(&self, issue: &str, patch_exists: &dyn Fn(&str) -> bool) -> Result<()> {
        ensure!(
            is_slug(&self.issue),
            "issue {:?} must be lowercase letters, digits, and dashes",
            self.issue
        );
        ensure!(
            self.issue == issue,
            "issue {:?} does not match its folder issues/{issue}",
            self.issue
        );
        ensure!(
            self.planned_hours.is_finite() && self.planned_hours > 0.0,
            "planned_hours must be positive"
        );
        self.pod.validate()?;
        self.server.validate()?;
        for variant in &self.variants {
            variant
                .validate(self.server.package, patch_exists)
                .with_context(|| format!("variant {:?}", variant.name))?;
        }
        Ok(())
    }
}

impl PodSpec {
    pub fn require_pcie(&self) -> bool {
        self.require_pcie.unwrap_or(self.count > 1)
    }

    fn validate(&self) -> Result<()> {
        let digest = self.image.split_once("@sha256:").map(|(_, hex)| hex);
        ensure!(
            digest.is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit())),
            "pod.image must be pinned by digest (name@sha256:<64 hex>), got {:?}",
            self.image
        );
        ensure!(self.count >= 1, "pod.count must be at least 1");
        ensure!(is_major_minor(&self.min_cuda), "pod.min_cuda must look like 13.0");
        ensure!(self.container_disk_gb >= 1, "pod.container_disk_gb must be at least 1");
        // RunPod rejects persistent volumes under 10 GB at create time, after the reservation is written.
        ensure!(self.volume_gb >= 10, "pod.volume_gb must be at least 10");
        ensure!(!self.gpus.is_empty(), "pod.gpus must list at least one GPU");
        for package in &self.apt {
            ensure!(is_apt_name(package), "pod.apt entry {package:?} is not a package name");
        }
        Ok(())
    }
}

impl ServerSpec {
    fn validate(&self) -> Result<()> {
        ensure!(!self.model.is_empty(), "server.model must not be empty");
        ensure!(!self.launch.is_empty(), "server.launch must not be empty");
        ensure!(self.port != 0 && self.port != 22, "server.port must not be 0 or 22");
        ensure!(!self.fail_pattern.is_empty(), "server.fail_pattern must not be empty");
        ensure!(self.boot_timeout_s >= 60, "server.boot_timeout_s must be at least 60");
        Ok(())
    }
}

impl Variant {
    fn validate(&self, package: Package, patch_exists: &dyn Fn(&str) -> bool) -> Result<()> {
        ensure!(
            is_variant_name(&self.name),
            "variant names must be letters, digits, '_' or '-', and not jobs or variants"
        );
        for patch in &self.patches {
            ensure!(is_name(patch), "patch name {patch:?} must be a plain file stem");
            ensure!(
                patch_exists(patch),
                "patch {patch:?} not found under patches/{patch}.diff"
            );
        }
        for key in self.env.keys() {
            ensure!(is_env_key(key), "env key {key:?} is not a shell variable name");
        }
        ensure!(self.smoke_concurrency >= 1, "smoke_concurrency must be at least 1");
        if let Some(quality) = &self.quality {
            // quality.sh drives sglang's run_eval and bench_serving; no vLLM protocol exists yet.
            ensure!(
                package == Package::Sglang,
                "quality runs need server.package = \"sglang\""
            );
            ensure!(quality.gsm8k >= 1, "quality.gsm8k must be at least 1");
            for &(concurrency, prompts) in &quality.bench {
                ensure!(concurrency >= 1 && prompts >= 1, "quality.bench pairs must be positive");
            }
        }
        Ok(())
    }
}

pub fn is_slug(s: &str) -> bool {
    s.bytes()
        .next()
        .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// A variant's results go to results/<issue>/<variant>/, next to the harness's own jobs/ and variants/.
fn is_variant_name(s: &str) -> bool {
    !matches!(s, "jobs" | "variants")
        && s.bytes().next().is_some_and(|b| b.is_ascii_alphanumeric())
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}

fn is_name(s: &str) -> bool {
    s.bytes().next().is_some_and(|b| b.is_ascii_alphanumeric())
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

fn is_env_key(s: &str) -> bool {
    s.bytes().next().is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

fn is_apt_name(s: &str) -> bool {
    s.bytes()
        .next()
        .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'+' | b'-'))
}

fn is_major_minor(s: &str) -> bool {
    s.split_once('.').is_some_and(|(major, minor)| {
        [major, minor]
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3";

    fn example() -> String {
        format!(
            r#"
issue = "vllm-49476"
planned_hours = 2.0

[pod]
image = "vllm/vllm-openai:v0.31.0@sha256:{DIGEST}"
count = 1
min_cuda = "13.0"
container_disk_gb = 40
volume_gb = 80
apt = ["openssh-server", "tmux", "curl"]
gpus = [
  {{ id = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", cloud = "COMMUNITY" }},
  {{ id = "NVIDIA RTX PRO 6000 Blackwell Server Edition", cloud = "SECURE" }},
]

[server]
package = "vllm"
model = "nvidia/Qwen3.6-35B-A3B-NVFP4"
launch = ["vllm", "serve", "nvidia/Qwen3.6-35B-A3B-NVFP4"]
base_flags = ["--max-model-len", "262144"]
port = 8000
fail_pattern = "illegal memory access|cudaErrorIllegalAddress"
boot_timeout_s = 900

[variants.zeta-first]
flags = ["--moe-backend", "marlin"]

[variants.baseline-graphs]
flags = ["--moe-backend", "flashinfer_b12x"]

[variants.fixed-graphs-31g]
flags = ["--gpu-memory-utilization", "0.303"]
patches = ["pr60069", "pr57946"]
env = {{ MEM_BUDGET_GIB = "31.45" }}
smoke_concurrency = 13
"#
        )
    }

    fn parse(text: &str) -> Result<IssueSpec> {
        IssueSpec::parse(text, "vllm-49476", |patch| patch.starts_with("pr"))
    }

    fn error_of(text: &str) -> String {
        format!("{:#}", parse(text).unwrap_err())
    }

    #[test]
    fn parses_the_design_example() {
        let spec = parse(&example()).unwrap();
        assert_eq!(spec.pod.gpus[1].cloud, CloudTier::Secure);
        assert_eq!(spec.server.package, Package::Vllm);
        assert_eq!(spec.server.patch_strip, 1);
        let fixed = spec.variant("fixed-graphs-31g").unwrap();
        assert_eq!(fixed.patches, ["pr60069", "pr57946"]);
        assert_eq!(fixed.env["MEM_BUDGET_GIB"], "31.45");
        assert_eq!(fixed.smoke_concurrency, 13);
        assert_eq!(spec.variant("baseline-graphs").unwrap().smoke_concurrency, 1);
    }

    #[test]
    fn keeps_variants_in_file_order() {
        let spec = parse(&example()).unwrap();
        let names: Vec<&str> = spec.variants.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(names, ["zeta-first", "baseline-graphs", "fixed-graphs-31g"]);
    }

    #[test]
    fn require_pcie_defaults_on_for_multi_gpu() {
        let one = parse(&example()).unwrap();
        assert!(!one.pod.require_pcie());
        let two = parse(&example().replace("count = 1", "count = 2")).unwrap();
        assert!(two.pod.require_pcie());
        let forced = example().replace("count = 1", "count = 2\nrequire_pcie = false");
        assert!(!parse(&forced).unwrap().pod.require_pcie());
    }

    #[test]
    fn rejects_unknown_keys() {
        let err = error_of(&example().replace("port = 8000", "port = 8000\nprot = 1"));
        assert!(err.contains("unknown field `prot`"), "{err}");
        let err = error_of(&example().replace("smoke_concurrency = 13", "smoke_concurency = 13"));
        assert!(err.contains("unknown field `smoke_concurency`"), "{err}");
    }

    #[test]
    fn rejects_an_image_without_a_digest() {
        let err = error_of(&example().replace(&format!("@sha256:{DIGEST}"), ""));
        assert!(err.contains("pinned by digest"), "{err}");
        let err = error_of(&example().replace(DIGEST, "abc"));
        assert!(err.contains("pinned by digest"), "{err}");
    }

    #[test]
    fn rejects_a_missing_patch() {
        let err = error_of(&example().replace("\"pr57946\"", "\"fi4827\""));
        assert!(err.contains("patch \"fi4827\" not found"), "{err}");
    }

    #[test]
    fn rejects_an_issue_that_does_not_match_its_folder() {
        let err = format!("{:#}", IssueSpec::parse(&example(), "vllm-1", |_| true).unwrap_err());
        assert!(err.contains("does not match its folder"), "{err}");
    }

    #[test]
    fn rejects_names_that_would_escape_into_paths_or_shell() {
        let err = error_of(&example().replace("[variants.zeta-first]", "[variants.\"../x\"]"));
        assert!(err.contains("variant names"), "{err}");
        let err = error_of(&example().replace("[variants.zeta-first]", "[variants.jobs]"));
        assert!(err.contains("not jobs or variants"), "{err}");
        let err = error_of(&example().replace("MEM_BUDGET_GIB", "\"BAD-KEY\""));
        assert!(err.contains("not a shell variable name"), "{err}");
        let err = error_of(&example().replace("\"tmux\"", "\"tmux; rm\""));
        assert!(err.contains("not a package name"), "{err}");
    }

    #[test]
    fn rejects_quality_on_vllm() {
        let text = example() + "\n[variants.baseline-graphs.quality]\ngsm8k = 50\n";
        let err = error_of(&text);
        assert!(err.contains("quality runs need server.package"), "{err}");
        let sglang = text.replace("package = \"vllm\"", "package = \"sglang\"");
        let spec = parse(&sglang).unwrap();
        let quality = spec.variant("baseline-graphs").unwrap().quality.as_ref().unwrap();
        assert_eq!(quality.gsm8k, 50);
        assert!(quality.bench.is_empty());
    }

    #[test]
    fn reads_bench_pairs() {
        let text = example().replace("package = \"vllm\"", "package = \"sglang\"")
            + "\n[variants.baseline-graphs.quality]\ngsm8k = 200\nbench = [[1, 8], [16, 64]]\n";
        let spec = parse(&text).unwrap();
        let quality = spec.variant("baseline-graphs").unwrap().quality.as_ref().unwrap();
        assert_eq!(quality.bench, [(1, 8), (16, 64)]);
    }

    #[test]
    fn rejects_a_small_volume() {
        let err = error_of(&example().replace("volume_gb = 80", "volume_gb = 5"));
        assert!(err.contains("volume_gb must be at least 10"), "{err}");
    }

    #[test]
    fn unknown_variant_is_an_error() {
        let spec = parse(&example()).unwrap();
        assert!(
            spec.variant("nope")
                .unwrap_err()
                .to_string()
                .contains("no variant \"nope\"")
        );
    }
}
