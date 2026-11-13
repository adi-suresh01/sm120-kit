# shellcheck shell=bash
# Sourced by harness/pod/serve_variant.sh; see harness/README.md for when each hook runs.

# With the FlashInfer variable set, vLLM falls back to NCCL with only a warning when the backend rules itself out
# (flashinfer_pcie_ipc_all_reduce.py:60-85 and 106-122 at f5d93296c), and silently with no capture sizes (:227-229);
# the arm would then hold no workspace under an IPC arm's name. The Initialized line comes only after the slab is
# allocated and tuned (:127-156). Then what each GPU holds on the live server: with the extensible KV cache both arms
# fill the GPU to the same line and the IPC cost shows in the KV cache, with the estimate-based one the KV cache is
# fixed and the cost shows here.
post_smoke() {
    local run_dir=$1
    if [[ ${VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:-} == 1 ]] \
        && ! grep -aq 'Initialized FlashInfer PCIe IPC all-reduce for TP4' "$run_dir/server.log"; then
        echo "FlashInfer PCIe IPC was asked for but never initialized; see server.log" >&2
        return 1
    fi
    progress post_smoke "GPU memory on the live server"
    timeout 60 nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
        >"$run_dir/gpu_mem_live.csv"
}
