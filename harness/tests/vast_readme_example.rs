//! The Vast `[pod]` example in README.md has to parse in a full harness.toml, so the documented keys cannot drift
//! from the real ones. It sits in a plain fence: readme_example.rs reads every `toml` fence as a whole file.

use sm120_harness::spec::{IssueSpec, Provider};

const README: &str = include_str!("../README.md");

/// The README's full example with its own `count` and `gpus` taken out and the Vast example's keys put in.
fn with_vast_example() -> String {
    let full = README.split("```toml\n").nth(1).unwrap().split("```").next().unwrap();
    let vast = README
        .split("### Vast choices")
        .nth(1)
        .unwrap()
        .split("```\n[pod]\n")
        .nth(1)
        .unwrap()
        .split("```")
        .next()
        .unwrap();
    let gpus = full.find("gpus = [").unwrap();
    let end = gpus + full[gpus..].find("]\n").unwrap() + 2;
    let without = format!("{}{}", &full[..gpus], &full[end..]);
    let without: String = without
        .lines()
        .filter(|line| !line.starts_with("count = "))
        .map(|line| format!("{line}\n"))
        .collect();
    without.replacen("[pod]\n", &format!("[pod]\n{vast}"), 1)
}

#[test]
fn the_vast_example_parses() {
    let spec = IssueSpec::parse(&with_vast_example(), "vllm-47990", |_| true).unwrap();
    let providers: Vec<Provider> = spec.pod.choices.iter().map(|c| c.provider()).collect();
    assert_eq!(providers, [Provider::Vast, Provider::Runpod]);
    assert_eq!(spec.pod.count, 8);
    assert!(spec.pod.require_p2p);
    assert_eq!(spec.pod.avoid_machines, [156482]);
}
