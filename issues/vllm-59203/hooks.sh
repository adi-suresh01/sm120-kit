# shellcheck shell=bash
# Sourced by harness/pod/serve_variant.sh; see harness/README.md for when each hook runs.
hooks_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# NCCL writes INFO to stdout without flushing (src/debug.cc), and the ranks end on SIGKILL, so in server.log each rank
# would lose its last lines and the ranks would interleave mid-line. Its own file is line-buffered. Every arm sets
# NCCL_DEBUG=INFO.
pre_launch() {
    local run_dir=$1
    export NCCL_DEBUG_FILE="$run_dir/nccl.%h.%p.log"
}

# With the FlashInfer variable set, vLLM falls back to NCCL with only a warning when the constructor, the dtype, or
# ubatching rules the backend out, and silently with no CUDA graph sizes (flashinfer_pcie_ipc_all_reduce.py:57-89,
# 105-121, 218-229 at v0.33.0); the rows would then measure plain NCCL under a FlashInfer arm's name. setup() logs the
# Initialized line only after the tune, so the rows wait for it.
post_smoke() {
    local run_dir=$1 port=$2 model=$3
    if [[ ${VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:-} == 1 ]] \
        && ! grep -aq 'Initialized FlashInfer PCIe IPC all-reduce for TP8' "$run_dir/server.log"; then
        echo "FlashInfer PCIe IPC was asked for but never initialized; see server.log" >&2
        return 1
    fi
    progress post_smoke "warmups, then 1K/512 at c=1, 8, 32, 64 and 32K/256 at c=1"
    python3 -I "$hooks_dir/decode_bench.py" "$port" "$model" "$VARIANT" "$run_dir" /workspace/tokenizer
}

# The transport NCCL picked for each channel, so an arm is labeled by what ran rather than by its env; the per-rank
# files stay beside nccl.log for anyone checking the condensed one.
post_stop() {
    local run_dir=$1
    cat "$run_dir"/nccl.*.*.log >"$run_dir/nccl.log" 2>/dev/null || true
    bash "$hooks_dir/nccl_transport.sh" "$run_dir/nccl.log" >"$run_dir/nccl_transport.txt" || true
}
