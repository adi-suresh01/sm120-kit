#!/usr/bin/env bash
# Runs one variant of the vllm#49476 experiment matrix and appends the verdict to a summary table.
# Usage: run.sh <variant> | run.sh all
#
# Variants:
#   marlin           control: same model and flags with --moe-backend marlin (reporter: boots and serves)
#   baseline-eager   unpatched v0.31.0, flashinfer_b12x, --enforce-eager
#   baseline-graphs  unpatched v0.31.0, flashinfer_b12x, CUDA graphs (the run the reporter asked for)
#   patched-eager    PR #60069 applied, --enforce-eager (reporter: works on 5090)
#   patched-graphs   PR #60069 applied, CUDA graphs (reporter: Xid 31 at capture batch 2)
#   patched-min      patched-graphs with capture sizes [1,2] and CUDA_LAUNCH_BLOCKING=1 to localize the fault
#
# MEM_BUDGET_GIB=31.45 emulates the 5090's 32 GB budget on a 96 GB PRO 6000 by scaling
# --gpu-memory-utilization. It reproduces vLLM's memory accounting, not the 5090 hardware.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$here/../.." && pwd)
results_dir="$repo_root/results/vllm-49476"
model=nvidia/Qwen3.6-35B-A3B-NVFP4
port=${PORT:-8000}

gpu_util() {
    if [[ -n ${MEM_BUDGET_GIB:-} ]]; then
        local total_mib
        total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
        # Reporter's budget was 0.92 of the card; treat MEM_BUDGET_GIB as the emulated card size.
        python3 -c "print(round(min(0.92, 0.92 * ${MEM_BUDGET_GIB} * 1024 / ${total_mib}), 3))"
    else
        echo 0.92
    fi
}

run_variant() {
    local variant=$1 backend=flashinfer_b12x patch_state=revert
    local -a extra=()
    local -a env_prefix=()

    case $variant in
        marlin)          backend=marlin ;;
        baseline-eager)  extra=(--enforce-eager) ;;
        baseline-graphs) ;;
        patched-eager)   patch_state=apply; extra=(--enforce-eager) ;;
        patched-graphs)  patch_state=apply ;;
        patched-min)
            patch_state=apply
            extra=(--compilation-config '{"cudagraph_capture_sizes":[1,2]}')
            env_prefix=(env CUDA_LAUNCH_BLOCKING=1)
            ;;
        *) echo "unknown variant: $variant" >&2; exit 2 ;;
    esac

    bash "$here/patch.sh" "$patch_state" >/dev/null

    local util label log
    util=$(gpu_util)
    label="${variant}${MEM_BUDGET_GIB:+-${MEM_BUDGET_GIB}g}"
    log="$results_dir/$label/server.log"

    local -a cmd=("${env_prefix[@]}" vllm serve "$model"
        --trust-remote-code
        --moe-backend="$backend"
        --kv-cache-dtype=fp8
        --max-model-len=262144
        --gpu-memory-utilization="$util"
        --max-num-seqs=16
        --port "$port"
        "${extra[@]}")

    echo ">>> $label (patch: $(bash "$here/patch.sh" status), gpu util: $util)"
    local verdict
    verdict=$(bash "$repo_root/scripts/serve_and_smoke.sh" "$log" "$port" "$model" -- "${cmd[@]}" || true)
    echo "$verdict"

    # Xid codes and CUDA errors are what the upstream thread needs; containers usually hide dmesg.
    local fault
    fault=$(grep -m1 -oE 'Xid[^,]*|illegal memory access[^.]*|CUDA error[^.]*|OutOfMemoryError[^.]*' "$log" || true)

    mkdir -p "$results_dir"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$label" "$util" "$verdict" "${fault:--}" \
        >>"$results_dir/summary.tsv"
}

if [[ ${1:-} == all ]]; then
    for v in marlin baseline-eager baseline-graphs patched-eager patched-graphs; do
        run_variant "$v"
    done
    bash "$here/patch.sh" revert >/dev/null
    column -t -s $'\t' "$results_dir/summary.tsv"
else
    run_variant "${1:?usage: run.sh <variant>|all}"
fi
