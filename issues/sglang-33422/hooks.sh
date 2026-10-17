# shellcheck shell=bash
# Sourced by harness/pod/serve_variant.sh; see harness/README.md for when each hook runs.
hooks_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Drives the live server with the issue's workloads (prefill_bench.py).
post_smoke() {
    local run_dir=$1 port=$2 model=$3
    progress post_smoke "workloads ${BENCH_WORKLOADS:-w1,w2}, ${BENCH_REPS:-2} reps"
    python3 -I "$hooks_dir/prefill_bench.py" "$port" "$model" "$PACKAGE" "$VARIANT" "$run_dir" /workspace/prompts
}

# The scheduler's own per-chunk rate and cached-token count, to set beside the client's wall-clock number.
post_stop() {
    local run_dir=$1
    grep -E 'Prefill batch|prefill CUDA graph' "$run_dir/server.log" >"$run_dir/prefill_log.txt" || true
}
