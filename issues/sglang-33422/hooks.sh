# shellcheck shell=bash
# Sourced by harness/pod/serve_variant.sh; see harness/README.md for when each hook runs.
hooks_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# NCCL writes INFO to stdout without flushing (src/debug.cc:376 at 2.30.7), and the ranks end on SIGKILL, so in
# server.log each rank loses its last lines and the ranks interleave mid-line. Its own file is line-buffered
# (debug.cc:240). `if`, not `&&`: this runs under set -e, and a false test would fail every other variant.
pre_launch() {
    local run_dir=$1
    if [[ ${NCCL_DEBUG:-} == INFO ]]; then
        export NCCL_DEBUG_FILE="$run_dir/nccl.%h.%p.log"
    fi
}

# Drives the live server with the issue's workloads: the 11-05 decode rows (decode_bench.py) for a variant that sets
# BENCH_KIND=decode, else the 10-17 prefill workloads (prefill_bench.py).
post_smoke() {
    local run_dir=$1 port=$2 model=$3
    if [[ ${BENCH_KIND:-prefill} == decode ]]; then
        progress post_smoke "decode rows w1 and w2 at c=1 and c=8"
        python3 -I "$hooks_dir/decode_bench.py" "$port" "$VARIANT" "$run_dir" /workspace/prompts
        return
    fi
    progress post_smoke "workloads ${BENCH_WORKLOADS:-w1,w2}, ${BENCH_REPS:-2} reps"
    python3 -I "$hooks_dir/prefill_bench.py" "$port" "$model" "$PACKAGE" "$VARIANT" "$run_dir" /workspace/prompts
}

# The scheduler's own per-chunk rate and cached-token count, to set beside the client's wall-clock number. With
# NCCL_DEBUG=INFO, also the transport NCCL picked for each channel, so a P2P arm's post quotes it instead of its env;
# the per-rank files stay beside nccl.log for anyone checking the condensed one.
post_stop() {
    local run_dir=$1
    grep -E 'Prefill batch|prefill CUDA graph' "$run_dir/server.log" >"$run_dir/prefill_log.txt" || true
    if [[ ${NCCL_DEBUG:-} == INFO ]]; then
        cat "$run_dir"/nccl.*.log >"$run_dir/nccl.log" 2>/dev/null || true
        bash "$hooks_dir/nccl_transport.sh" "$run_dir/nccl.log" >"$run_dir/nccl_transport.txt" || true
    fi
}
