//! The harness.toml examples in README.md have to parse, so the documented format cannot drift from the
//! real one.

use sm120_harness::spec::{IssueSpec, Package};

fn toml_blocks(readme: &str) -> Vec<&str> {
    readme
        .split("```toml\n")
        .skip(1)
        .map(|rest| rest.split("```").next().unwrap())
        .collect()
}

#[test]
fn readme_examples_parse() {
    let blocks = toml_blocks(include_str!("../README.md"));
    assert_eq!(blocks.len(), 2);
    let spec = IssueSpec::parse(blocks[0], "vllm-47990", |_| true).unwrap();
    assert_eq!(spec.server.package, Package::Vllm);
    assert_eq!(spec.variants.len(), 2);
    assert_eq!(spec.snapshot_packages(), ["vllm", "flashinfer"]);
    let sglang = blocks[0].replace("package = \"vllm\"", "package = \"sglang\"") + blocks[1];
    let spec = IssueSpec::parse(&sglang, "vllm-47990", |_| true).unwrap();
    assert_eq!(spec.variant("a6").unwrap().quality.as_ref().unwrap().bench.len(), 3);
}
