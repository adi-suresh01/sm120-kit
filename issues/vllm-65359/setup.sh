#!/usr/bin/env bash
# All of vllm#65359's measurement, since an 8x box comes free rarely: record the box, build FlashInfer's all-gather
# module once, find the 4+4 split, then run Lei-010's bench in the ordered placement (copy engine admitted), in the
# interleaved one (same GPUs, recursive doubling only, the condition on Lei's host), with NCCL allowed P2P, and once
# more as a repeat. Exits non-zero only when the box cannot answer the question (not 8 GPUs, or no 4+4 split), which
# makes `harness watch` destroy it and look again. Everything else is a result recorded in runs.tsv.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
kit=$(cd "$here/../.." && pwd)
out="$kit/results/vllm-65359"
runs="$out/runs.tsv"
mkdir -p "$out"
printf 'name\trc\tseconds\tlog\n' >"$runs"
# CUDA_VISIBLE_DEVICES below names GPUs by nvidia-smi index, which islands.py reads through NVML in the same order.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

gpus=$(timeout 60 nvidia-smi -L | grep -c '^GPU ' || true)
if [[ $gpus != 8 ]]; then
    echo "this study needs 8 GPUs; nvidia-smi lists ${gpus:-none}" >&2
    exit 1
fi

# The box's watchdog destroys it at HARNESS_DEADLINE_EPOCH, and `down` still has to pull the results, so no run starts
# unless its time limit ends this long before the deadline.
reserve_s=600
deadline=${HARNESS_DEADLINE_EPOCH:-0}
last_rc=0

# Usage: run <name> <limit_s> <log> <command>...   Always returns 0; the outcome is in runs.tsv and $last_rc.
run() {
    local name=$1 limit=$2 log=$3 start
    shift 3
    if (( deadline > 0 && $(date +%s) + limit + reserve_s > deadline )); then
        printf '%s\tskipped\t0\t%s\n' "$name" "${log#"$out"/}" >>"$runs"
        echo "skip $name: under $(( limit + reserve_s )) s left before the deadline"
        last_rc=skipped
        return 0
    fi
    echo "$(date -u +%H:%M:%S) $name (limit ${limit}s)"
    start=$(date +%s)
    last_rc=0
    timeout -k 20 "$limit" "$@" >"$log" 2>&1 || last_rc=$?
    printf '%s\t%s\t%s\t%s\n' "$name" "$last_rc" "$(( $(date +%s) - start ))" "${log#"$out"/}" >>"$runs"
    # A worker torchrun lost track of after a timeout can still spin in a kernel, and the next run needs every GPU.
    pkill -9 -f "$here/ag_bench.py" 2>/dev/null || true
    bash "$kit/harness/pod/wait_gpu_idle.sh" 120 >>"$log" 2>&1 ||
        echo "$name: a GPU is still busy after 120 s" | tee -a "$log"
}

echo "$(date -u +%H:%M:%S) env_extra"
bash "$here/env_extra.sh" "$out" >"$out/env_extra.txt" 2>&1 || echo "env_extra.sh exited $?; see env_extra.txt"

# Eight ranks reaching an unbuilt JIT module at once would each wait on the build inside the bench's phase limit.
run jit-build 900 "$out/jit_build.log" python3 -c \
    'from flashinfer.comm.pcie_ipc_collectives._ag_rs_module import get_pcie_ipc_ag_rs_module as get; get(); print("built")'
if [[ $last_rc != 0 ]]; then
    # The same image fails the same way on every box, so this is a result to pull, not a reason to rent another.
    echo "the all-gather module did not build ($last_rc); no arm can run, see jit_build.log"
    echo "setup done"
    exit 0
fi

islands_rc=0
python3 -I "$here/islands.py" >"$out/islands.txt" 2>&1 || islands_rc=$?
cat "$out/islands.txt"
if [[ $islands_rc != 0 ]]; then
    # islands.py exits 0 with or without a split, so a crash is the image's and would end every box the same way.
    echo "islands.py exited $islands_rc; no arm can run, see islands.txt"
    echo "setup done"
    exit 0
fi
ordered=$(awk '$1 == "ordered" {print $2}' "$out/islands.txt")
interleaved=$(awk '$1 == "interleaved" {print $2}' "$out/islands.txt")
# Lei's precondition: on a box with no 4+4 split the copy engine is never admitted, so nothing here would answer the
# question, and the window is better spent on the next box.
if [[ -z $ordered || -z $interleaved ]]; then
    echo "no ordered 4+4 split on this box; see islands.txt" >&2
    exit 2
fi

# Usage: bench_arm <arm> <limit_s> <gpu order> <phases> [VAR=value]...
bench_arm() {
    local arm=$1 limit=$2 order=$3 phases=$4
    shift 4
    run "$arm" "$limit" "$out/$arm.log" env CUDA_VISIBLE_DEVICES="$order" "$@" \
        torchrun --standalone --nproc_per_node=8 "$here/ag_bench.py" --arm "$arm" --out "$out" --phases "$phases"
}

# NCCL's INFO lines say which transport each channel took (SHM or P2P), which sets the NCCL column Lei compares to.
nccl_info() {
    echo NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV "NCCL_DEBUG_FILE=$out/$1.nccl.%h.%p.log"
}

# The decisive arm first: the copy engine is admitted only here.
# shellcheck disable=SC2046  # nccl_info prints separate VAR=value words on purpose
bench_arm ordered 900 "$ordered" seed,sweep,tune,tuned $(nccl_info ordered)
bench_arm interleaved 900 "$interleaved" seed,sweep,tune,tuned
# On this host NCCL's default P2P level sends every pair through SHM (issues/pcie-tp8-allreduce); with P2P allowed
# it was 1.5-2.2x faster at all-reduce, so the NCCL baseline is shown both ways.
# shellcheck disable=SC2046
bench_arm ordered-nccl-p2p 300 "$ordered" seed NCCL_P2P_LEVEL=SYS $(nccl_info ordered-nccl-p2p)
# Run-to-run spread for the seed table, on the placement the copy engine needs.
bench_arm ordered-repeat 300 "$ordered" seed

python3 -I "$here/summarize.py" "$out" >"$out/summary.txt" 2>&1 || echo "summarize.py exited $?; the raw rows are in $out"
cat "$out/summary.txt"
echo "setup done"
