# shellcheck shell=bash
# Sourced by harness/pod/serve_variant.sh; see harness/README.md for when each hook runs.
hooks_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Drives the live server with the long prompts this variant names in LONG_PROMPTS.
post_smoke() {
    local run_dir=$1 port=$2 model=$3
    progress post_smoke "long prompts: ${LONG_PROMPTS:-none}"
    python3 -I "$hooks_dir/long_prompts.py" "$port" "$model" "$run_dir" /workspace/prompts
}

# Core dumps are tens of GB, so only their names and sizes come back with the results.
post_stop() {
    local run_dir=$1
    ls -la /root/cores >"$run_dir/cores.txt" 2>&1 || true
    grep -E 'SpecDecoding metrics|Mean acceptance length|Avg prompt throughput' "$run_dir/server.log" \
        | tail -n 20 >"$run_dir/spec_metrics.txt" || true
}
