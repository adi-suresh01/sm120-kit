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
#   fi4827-graphs    patched-graphs plus FlashInfer PR #4827 (graph-referenced workspace/weight pinning)
#   noskip-graphs    baseline-graphs with VLLM_MOE_SKIP_PADDING=0, so padding rows never carry the -1 sentinel
#   guard-graphs     baseline-graphs plus vLLM PR #57946 (padding sentinel guard)
#   fixed-graphs     PR #60069 + PR #57946, CUDA graphs: the candidate full fix for the reporter
#   patched-noskip-graphs  PR #60069 + VLLM_MOE_SKIP_PADDING=0: workaround with one patch and an env var
#
# CONCURRENCY=N sends N parallel smoke requests so decode batches are padded up to a captured graph size.
# TAG=x appends -x to the result label, for runs on a changed environment (e.g. TAG=fi071 after a FlashInfer bump).
#
# MEM_BUDGET_GIB=31.45 emulates the 5090 on a 96 GB PRO 6000: a ballast process leaves only that much free and
# --gpu-memory-utilization is scaled to the same KV budget. It reproduces the memory ceiling, not the 5090's
# SM count (170 vs 188), so workspaces sized per SM differ slightly.
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

# A crashed server can leave an EngineCore child holding memory, which would make the next variant OOM falsely.
wait_gpu_idle() {
    local used
    for _ in $(seq 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
        if (( used < 2048 )); then return 0; fi
        sleep 2
    done
    echo "GPU 0 still has ${used} MiB in use; kill leftover vllm processes first" >&2
    exit 1
}

ballast_pid=""
trap 'if [[ -n $ballast_pid ]]; then kill "$ballast_pid" 2>/dev/null || true; fi' EXIT

start_ballast() {
    local ballast_log=$1
    python3 "$repo_root/scripts/vram_ballast.py" "$MEM_BUDGET_GIB" >"$ballast_log" 2>&1 &
    ballast_pid=$!
    until grep -q '^holding' "$ballast_log"; do
        if ! kill -0 "$ballast_pid" 2>/dev/null; then cat "$ballast_log" >&2; exit 1; fi
        sleep 1
    done
    cat "$ballast_log"
}

stop_ballast() {
    if [[ -z $ballast_pid ]]; then return 0; fi
    kill "$ballast_pid" 2>/dev/null || true
    wait "$ballast_pid" 2>/dev/null || true
    ballast_pid=""
}

run_variant() {
    local variant=$1 backend=flashinfer_b12x patch_state=revert fi_state=revert guard_state=revert
    local -a extra=()
    local -a env_prefix=()

    case $variant in
        marlin)          backend=marlin ;;
        baseline-eager)  extra=(--enforce-eager) ;;
        baseline-graphs) ;;
        patched-eager)   patch_state=apply; extra=(--enforce-eager) ;;
        patched-graphs)  patch_state=apply ;;
        fi4827-graphs)   patch_state=apply; fi_state=apply ;;
        noskip-graphs)   env_prefix=(env VLLM_MOE_SKIP_PADDING=0) ;;
        guard-graphs)    guard_state=apply ;;
        fixed-graphs)    patch_state=apply; guard_state=apply ;;
        patched-noskip-graphs) patch_state=apply; env_prefix=(env VLLM_MOE_SKIP_PADDING=0) ;;
        patched-min)
            patch_state=apply
            extra=(--compilation-config '{"cudagraph_capture_sizes":[1,2]}')
            env_prefix=(env CUDA_LAUNCH_BLOCKING=1)
            ;;
        *) echo "unknown variant: $variant" >&2; exit 2 ;;
    esac

    # pr57946 sits on top of pr60069 in the same file, so it comes off first and goes on last.
    bash "$here/patch.sh" revert pr57946 >/dev/null
    bash "$here/patch.sh" "$patch_state" pr60069 >/dev/null
    bash "$here/patch.sh" "$fi_state" fi4827 >/dev/null
    bash "$here/patch.sh" "$guard_state" pr57946 >/dev/null

    local util label log
    util=$(gpu_util)
    label="${variant}${MEM_BUDGET_GIB:+-${MEM_BUDGET_GIB}g}${CONCURRENCY:+-c${CONCURRENCY}}${TAG:+-${TAG}}"
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

    echo ">>> $label (pr60069: $(bash "$here/patch.sh" status pr60069), fi4827: $(bash "$here/patch.sh" status fi4827)," \
        "pr57946: $(bash "$here/patch.sh" status pr57946), env: ${env_prefix[*]:-none}, gpu util: $util)"
    wait_gpu_idle
    mkdir -p "$(dirname "$log")"
    if [[ -n ${MEM_BUDGET_GIB:-} ]]; then start_ballast "${log%.log}.ballast.log"; fi

    local verdict
    verdict=$(bash "$repo_root/scripts/serve_and_smoke.sh" "$log" "$port" "$model" -- "${cmd[@]}" || true)
    stop_ballast
    echo "$verdict"

    # Xid codes and CUDA errors are what the upstream thread needs; containers usually hide dmesg.
    # The ValueError covers a KV cache too small for max-model-len, which is how a budget overrun shows up
    # when the profile peak itself fits.
    local fault
    fault=$(grep -m1 -oE 'Xid[^,]*|illegal memory access[^.]*|CUDA error[^.]*|OutOfMemoryError[^.]*|ValueError: [^.]*' \
        "$log" || true)

    mkdir -p "$results_dir"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$label" "$util" "$verdict" "${fault:--}" \
        >>"$results_dir/summary.tsv"
}

if [[ ${1:-} == all ]]; then
    for v in marlin baseline-eager baseline-graphs patched-eager patched-graphs; do
        run_variant "$v"
    done
    bash "$here/patch.sh" revert pr57946 >/dev/null
    bash "$here/patch.sh" revert pr60069 >/dev/null
    bash "$here/patch.sh" revert fi4827 >/dev/null
    # The vllm/vllm-openai image has no column(1).
    column -t -s $'\t' "$results_dir/summary.tsv" 2>/dev/null || cat "$results_dir/summary.tsv"
else
    run_variant "${1:?usage: run.sh <variant>|all}"
fi
