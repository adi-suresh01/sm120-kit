#!/usr/bin/env bash
# Runs one server variant of the sglang#44291 stage 1 matrix and appends a row to results/sglang-44291/summary.tsv.
# Usage: run.sh <variant>        TAG=x appends -x to the result label (e.g. TAG=fix after applying the patch)
#
# Variants (TP2, CUDA graphs on, --log-level debug so the all-reduce choice reaches the log):
#   s1       defaults
#   s2       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (the reporter's suspected setting)
#   s3       s2 + SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1 (FlashInfer PCIe-IPC all-reduce, sglang #34528)
#   s4       --disable-custom-all-reduce (NCCL baseline)
#   s1-ep2   s1 on OLMoE-1B-7B with --ep-size 2 (only if s1 and s2 both serve)
#   s2-ep2   s2 on OLMoE-1B-7B with --ep-size 2
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$here/../.." && pwd)
results_dir="$repo_root/results/sglang-44291"
port=${PORT:-30000}

variant=${1:?usage: run.sh <variant>}
model=Qwen/Qwen3-0.6B
alloc_conf=""
extra_env=()
extra=()

case $variant in
    s1) ;;
    s2) alloc_conf=expandable_segments:True ;;
    s3) alloc_conf=expandable_segments:True; extra_env=(SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1) ;;
    s4) extra=(--disable-custom-all-reduce) ;;
    s1-ep2) model=allenai/OLMoE-1B-7B-0924; extra=(--ep-size 2) ;;
    s2-ep2) model=allenai/OLMoE-1B-7B-0924; alloc_conf=expandable_segments:True; extra=(--ep-size 2) ;;
    *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac

label=${variant}${TAG:+-${TAG}}
log="$results_dir/$label/server.log"
mkdir -p "$(dirname "$log")"

# An empty PYTORCH_CUDA_ALLOC_CONF still counts as set to anything that checks for it, so default runs unset it.
if [[ -n $alloc_conf ]]; then
    alloc_env=(PYTORCH_CUDA_ALLOC_CONF="$alloc_conf")
else
    alloc_env=(-u PYTORCH_CUDA_ALLOC_CONF)
fi
cmd=(env "${alloc_env[@]}" "${extra_env[@]}"
    python3 -m sglang.launch_server --model-path "$model" --tp-size 2
    --host 127.0.0.1 --port "$port" --log-level debug "${extra[@]}")

bash "$repo_root/scripts/wait_gpu_idle.sh"
echo ">>> $label: ${cmd[*]}"
# A server wedged in a faulted kernel can block serve_and_smoke's shutdown wait; timeout kills the whole group.
verdict=$(BOOT_TIMEOUT=${BOOT_TIMEOUT:-600} timeout -k 15 900 \
    bash "$repo_root/scripts/serve_and_smoke.sh" "$log" "$port" "$model" -- "${cmd[@]}" || true)
verdict=${verdict:-RESULT=TIMEOUT outer=900s}
echo "$verdict"

# v2 logs its choice at debug level; v1 logs how many graph addresses it registered once capture succeeds.
backend=$(grep -m1 -oE '\[AR\] Using [A-Za-z0-9]+|Registering [0-9]+ cuda graph addresses' "$log" || true)
fault=$(grep -m1 -oE 'failed to get pointer attr|RuntimeError: [^.]*|illegal memory access[^.]*|CUDA error[^.]*|OutOfMemoryError[^.]*' "$log" || true)
# The frame name only says where it failed, so it is the fallback when no error text reached the log.
if [[ -z $fault ]]; then fault=$(grep -m1 -oE 'get_graph_buffer_ipc_meta' "$log" || true); fi

printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$label" "${backend:--}" "$verdict" "${fault:--}" \
    >>"$results_dir/summary.tsv"
