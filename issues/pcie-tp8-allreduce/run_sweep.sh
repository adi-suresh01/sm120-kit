#!/usr/bin/env bash
# One command for the decode-size all-reduce sweep on a PCIe GPU box, no model download: which all-reduce vLLM picks
# by itself, vLLM's NCCL path at the P2P level the box serves with, and FlashInfer's PCIe IPC all-reduce, every result
# checked exactly (vllm_ar_bench.py). Every GPU must be idle, so stop serving first. Run it from this folder inside a
# vLLM v0.33.0 or newer environment, for example:
#
#   docker run --rm --gpus all --ipc=host -v "$PWD":/w -w /w --entrypoint bash vllm/vllm-openai:v0.33.0 \
#       run_sweep.sh --hidden 6144
#
# About 15 minutes, 30 at most: each step has a time limit, and a step that hits it still keeps the rows it wrote.
# Results go to --out (default ./ar-sweep): rows.jsonl, one log per step, NCCL's transport lines, and table.txt, which
# is also printed at the end.
#
# Usage: run_sweep.sh --hidden <model hidden size> [--out <dir>] [--p2p-level <NCCL_P2P_LEVEL value|default>]
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
usage="usage: run_sweep.sh --hidden <model hidden size> [--out <dir>] [--p2p-level <NCCL_P2P_LEVEL value|default>]"
hidden=
out=ar-sweep
# What production runs on the boxes this was written for; `default` leaves NCCL's own choice.
p2p_level=SYS
while (( $# )); do
    case $1 in
        --hidden) hidden=${2:?$usage}; shift 2 ;;
        --out) out=${2:?$usage}; shift 2 ;;
        --p2p-level) p2p_level=${2:?$usage}; shift 2 ;;
        *) echo "$usage" >&2; exit 2 ;;
    esac
done
if [[ ! $hidden =~ ^[0-9]+$ ]]; then
    echo "$usage" >&2
    exit 2
fi
# NCCL ignores a value it does not know, without a warning, and runs its defaults (src/graph/paths.cc, ncclGetLevel).
# NODE, nvidia-smi's word, is one of them; NCCL calls that level PHB.
case $p2p_level in
    LOC|NVL|NVB|C2C|PIX|PXB|P2C|PXN|PHB|SYS|default) ;;
    *) echo "NCCL would silently ignore --p2p-level $p2p_level; use PIX, PXB, PHB, SYS or default" >&2; exit 2 ;;
esac

mkdir -p "$out"
out=$(cd "$out" && pwd)
# Rank i is nvidia-smi's GPU i, so the results line up with `nvidia-smi topo -m`.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

used=$(timeout 60 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
gpus=$(grep -c . <<<"$used")
busiest=$(sort -n <<<"$used" | tail -n 1)
if (( gpus < 2 )); then
    echo "found $gpus GPU(s); an all-reduce needs at least 2" >&2
    exit 1
fi
# A serving engine holds most of each GPU's memory; under 2 GiB is the driver and a display at most.
if (( busiest > 2048 )); then
    echo "a GPU has $busiest MiB in use; stop serving on this box first" >&2
    timeout 60 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
    exit 1
fi

nccl_env=(NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS="INIT,ENV" NCCL_DEBUG_FILE="$out/nccl-$p2p_level.nccl.%h.%p.log")
if [[ $p2p_level != default ]]; then
    nccl_env+=(NCCL_P2P_LEVEL="$p2p_level")
fi
printf 'step\texit\tseconds\n' >"$out/runs.tsv"

# Usage: step <arm> <backend> <limit_s> [VAR=value]...   Always returns 0; the exit code goes to runs.tsv.
step() {
    local arm=$1 backend=$2 limit=$3 start rc=0
    shift 3
    echo "$(date +%H:%M:%S) $arm, up to $(( limit / 60 )) min"
    start=$(date +%s)
    timeout -k 20 "$limit" env "$@" torchrun --standalone --nproc_per_node="$gpus" "$here/vllm_ar_bench.py" \
        --backend "$backend" --arm "$arm" --out "$out" --hidden "$hidden" >"$out/$arm.log" 2>&1 || rc=$?
    printf '%s\t%s\t%s\n' "$arm" "$rc" "$(( $(date +%s) - start ))" >>"$out/runs.tsv"
    # A worker left behind by a timeout can keep a GPU busy, and the next step needs every one.
    pkill -9 -f "$here/vllm_ar_bench.py" 2>/dev/null || true
    sleep 5
}

step probe probe 360
step "nccl-$p2p_level" nccl 420 "${nccl_env[@]}"
step fi-pcie-ipc fi_pcie_ipc 840
python3 -I "$here/sweep_table.py" "$out" "$hidden" | tee "$out/table.txt"
