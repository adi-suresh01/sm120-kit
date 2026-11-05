#!/usr/bin/env bash
# Stage 1, all of it, since an 8x box comes free rarely: record the box, build nccl-tests against the NCCL torch
# loads, run every NCCL and vLLM arm under its own time limit, and write summary.txt. Exits non-zero only when the box
# does not have 8 GPUs, which makes `harness watch` destroy it and look again. Everything else is a result recorded in
# runs.tsv: an arm that fails, hangs, or times out, and a failed nccl-tests build too, since it would fail the same way
# on every box with this image while the vLLM arms, which decide stage 2, do not need it.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
kit=$(cd "$here/../.." && pwd)
out="$kit/results/pcie-tp8-allreduce"
build=/root/nccl-tests
runs="$out/runs.tsv"
mkdir -p "$out/nccl" "$out/vllm" "$out/pairwise"
printf 'name\trc\tseconds\tlog\n' >"$runs"
# Rank and GPU i are nvidia-smi's GPU i in every run, so results, `topo -m`, and the pairwise matrix line up.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

gpus=$(timeout 60 nvidia-smi -L | grep -c '^GPU ' || true)
if [[ $gpus != 8 ]]; then
    echo "this study needs 8 GPUs; nvidia-smi lists ${gpus:-none}" >&2
    exit 1
fi

# The box's watchdog destroys it at HARNESS_DEADLINE_EPOCH, and `up` still has to pull the results after setup, so
# no run starts unless its time limit ends this long before the deadline.
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
    pkill -9 -f "$here/vllm_ar_bench.py" 2>/dev/null || true
    bash "$kit/harness/pod/wait_gpu_idle.sh" 120 >>"$log" 2>&1 ||
        echo "$name: a GPU is still busy after 120 s" | tee -a "$log"
}

# NCCL buffers INFO lines on stdout, which a killed run loses; its debug file is line buffered (src/debug.cc).
# summarize.py reads <log stem>.nccl.*.log beside each run's log.
# Usage: nccl_info <subsystems> <log>   sets the array $info to the env for that run.
nccl_info() {
    info=(NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS="$1" NCCL_DEBUG_FILE="${2%.log}.nccl.%h.%p.log")
}

echo "$(date -u +%H:%M:%S) env_extra"
bash "$here/env_extra.sh" "$out" >"$out/env_extra.txt" 2>&1 || echo "env_extra.sh exited $?; see env_extra.txt"

# 900 s bounds a stalled download or apt, so a bad mirror cannot hold setup to the deadline.
run build/nccl-tests 900 "$out/nccl_tests_build.log" bash "$here/build_nccl_tests.sh" "$build"
nccl_built=$last_rc
if [[ $nccl_built == 0 ]]; then
    # nccl-tests links libnccl.so.2 with no rpath, so this is how it gets the wheel's copy, the one torch loads.
    nccl_ld="$(<"$build/nccl_lib_dir")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    all_reduce=(env LD_LIBRARY_PATH="$nccl_ld" "$build/bin/all_reduce_perf")
else
    echo "nccl-tests did not build ($nccl_built); its arms and pairwise are skipped, see nccl_tests_build.log"
fi
# One thread per GPU, as vLLM runs one process per GPU: one thread launching for all 8 would time its own launch
# loop in eager mode. -T makes nccl-tests report a collective stuck past 90 s before the outer limit kills it.
common=(-t 8 -g 1 -d bfloat16 -o sum -T 90)
pow2=(-b 4K -e 4M -f 2)
# Batch 1 to 256 at hidden 5120 in bf16: the all-reduce sizes of the public TP8 trace (vllm#59203) and beyond.
decode=(-b 10240 -e 2621440 -f 2)
large=(-b 8M -e 256M -f 2 -w 5 -n 20 -c 1)
# 500 warmups so the first, smallest sizes run at full clocks; 1000 timed collectives per size, replayed from a CUDA
# graph as vLLM's decode does; -c 5 checks five results.
small=(-w 500 -n 200 -c 5 -G 5)
# What each arm's tuning run logs; the default arm's logs more, to explain the baseline.
debug_subsys=INIT,ENV

# Usage: nccl_arm <arm> [VAR=value]...
nccl_arm() {
    local arm=$1 d="$out/nccl/$1"
    shift
    [[ $nccl_built == 0 ]] || return 0
    mkdir -p "$d"
    # NCCL_DEBUG=VERSION prints the library's version in every log, tying each number to one build.
    local env=(env NCCL_DEBUG=VERSION "$@")
    run "nccl/$arm/pow2" 120 "$d/pow2.log" "${env[@]}" "${all_reduce[@]}" "${common[@]}" "${pow2[@]}" "${small[@]}"
    # A hang at the smallest sizes would only repeat at the others. Any other non-zero exit can be wrong results,
    # which are data, and the tuning run then shows what NCCL chose.
    if [[ $last_rc == 124 || $last_rc == 137 || $last_rc == skipped ]] || grep -qs 'Test timeout' "$d/pow2.log"; then
        return 0
    fi
    run "nccl/$arm/decode" 120 "$d/decode.log" "${env[@]}" "${all_reduce[@]}" "${common[@]}" "${decode[@]}" \
        "${small[@]}"
    run "nccl/$arm/large" 240 "$d/large.log" "${env[@]}" "${all_reduce[@]}" "${common[@]}" "${large[@]}" -G 1
    # -U reads NCCL's choice per size through a profiler, which could cost time, so it gets its own short run.
    nccl_info "$debug_subsys" "$d/tuning.log"
    run "nccl/$arm/tuning" 90 "$d/tuning.log" "${env[@]}" "${info[@]}" \
        NCCL_TOPO_DUMP_FILE="$d/topo.xml" NCCL_GRAPH_DUMP_FILE="$d/graph.xml" \
        "${all_reduce[@]}" "${common[@]}" "${pow2[@]}" -w 2 -n 5 -c 1 -U 1
}

# Usage: vllm_arm <arm> <backend> <limit_s> [VAR=value]...
vllm_arm() {
    local arm=$1 backend=$2 limit=$3 log="$out/vllm/$1.log"
    shift 3
    info=()
    # The NCCL arms record which transport each channel took, the evidence for or against the SHM fallback.
    if [[ $backend == nccl ]]; then
        nccl_info INIT,ENV "$log"
    fi
    run "vllm/$arm" "$limit" "$log" env "${info[@]}" "$@" \
        torchrun --standalone --nproc_per_node=8 "$here/vllm_ar_bench.py" --backend "$backend" --arm "$arm" \
        --out "$out/vllm"
}

# The default arm first and in full: the baseline every other number is read against, the non-graph baseline, and
# the topology and channel choices that explain it.
debug_subsys=INIT,GRAPH,TUNING,ENV,P2P,SHM
nccl_arm default
debug_subsys=INIT,ENV
if [[ $nccl_built == 0 ]]; then
    eager=(-w 500 -n 1000 -c 5)
    run nccl/default/pow2-eager 120 "$out/nccl/default/pow2-eager.log" env NCCL_DEBUG=VERSION \
        "${all_reduce[@]}" "${common[@]}" "${pow2[@]}" "${eager[@]}"
    run nccl/default/decode-eager 120 "$out/nccl/default/decode-eager.log" env NCCL_DEBUG=VERSION \
        "${all_reduce[@]}" "${common[@]}" "${decode[@]}" "${eager[@]}"
fi

# vLLM's own paths next, in order of what stage 2 hinges on. Each outer limit outlasts the script's own per-step
# watchdog, so a hang ends with every rank's stack in its fault file rather than a bare kill.
vllm_arm probe probe 360
vllm_arm nccl-default nccl 420
vllm_arm custom-ar custom 420
vllm_arm fi-pcie-ipc fi_pcie_ipc 840

nccl_arm p2p-disable NCCL_P2P_DISABLE=1
# NCCL can refuse P2P across CPU sockets or NUMA nodes and fall back to SHM; this allows it on every path.
nccl_arm p2p-sys NCCL_P2P_LEVEL=SYS
nccl_arm proto-ll NCCL_PROTO=LL
# NCCL documents LL128 as unsafe off NVLink; the result columns record whether it refuses, slows, or corrupts.
nccl_arm proto-ll128 NCCL_PROTO=LL128
nccl_arm algo-tree NCCL_ALGO=Tree
nccl_arm p2p-sys-ll NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL

# The two env levers on vLLM's own NCCL path, in case either beats the default.
vllm_arm nccl-p2p-sys nccl 420 NCCL_P2P_LEVEL=SYS
vllm_arm nccl-p2p-sys-ll nccl 420 NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL

# One-hop latency per GPU pair, so a slow NUMA or socket crossing shows up as a matrix; capped at 3 min.
if [[ $nccl_built == 0 ]]; then
    pairs_until=$(( $(date +%s) + 180 ))
    for a in 0 1 2 3 4 5 6 7; do
        for (( b = a + 1; b < 8; b++ )); do
            if (( $(date +%s) > pairs_until )); then
                echo "pairwise: 3 min cap reached at $a-$b"
                break 2
            fi
            log="$out/pairwise/$a-$b.log"
            nccl_info INIT "$log"
            run "pairwise/$a-$b" 30 "$log" env CUDA_VISIBLE_DEVICES="$a,$b" "${info[@]}" LD_LIBRARY_PATH="$nccl_ld" \
                "$build/bin/sendrecv_perf" -t 2 -g 1 -d bfloat16 -b 8 -e 65536 -f 8192 -w 20 -n 100 -c 1 -G 5 -T 20
        done
    done
fi

python3 -I "$here/summarize.py" "$out" || echo "summarize.py exited $?; the raw logs are all in $out"
echo "setup done"
