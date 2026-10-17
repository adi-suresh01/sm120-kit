# shellcheck shell=bash
# Sourced by harness/pod/serve_variant.sh; see harness/README.md for when each hook runs.
hooks_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Drives the live server with the issue's workloads (prefill_bench.py, a copy of issues/sglang-33422's).
post_smoke() {
    local run_dir=$1 port=$2 model=$3
    progress post_smoke "workloads ${BENCH_WORKLOADS:-w1,w2}, ${BENCH_REPS:-2} reps"
    python3 -I "$hooks_dir/prefill_bench.py" "$port" "$model" "$PACKAGE" "$VARIANT" "$run_dir" /workspace/prompts
}

# vLLM's 10 s prompt throughput and prefix cache hit rate, to set beside the client's wall-clock number.
post_stop() {
    local run_dir=$1
    grep -E 'Avg prompt throughput|Prefix cache hit rate|MoE backend|sparse MLA' "$run_dir/server.log" \
        >"$run_dir/prefill_log.txt" || true
}
