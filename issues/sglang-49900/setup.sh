#!/usr/bin/env bash
# Stage 1, all of it, on one GPU with synthetic weights: record the box, fetch the published tables, time the shipped
# kernel against its M == 1 specialization and those tables, time the MXFP8 arms, sweep configs, time the tuned arms,
# check the shipped tuner on [32, 32], and write summary.txt. Exits non-zero only when the GPU is not sm_120, since
# every number here is about that card, or when fp8bench does not import, since every step needs it; any other
# failure is a row in runs.tsv and the next step still runs.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
kit=$(cd "$here/../.." && pwd)
out="$kit/results/sglang-49900"
runs="$out/runs.tsv"
mkdir -p "$out"
printf 'name\trc\tseconds\tlog\n' >"$runs"

cap=$(timeout 60 nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')
if [[ $cap != 12.0 ]]; then
    echo "this bench is about sm_120; GPU 0 reports compute capability ${cap:-none}" >&2
    exit 1
fi

# The watchdog ends the pod at HARNESS_DEADLINE_EPOCH and `up` still pulls the results after setup, so no step starts
# unless its limit ends this long before the deadline.
reserve_s=480
deadline=${HARNESS_DEADLINE_EPOCH:-0}

# Usage: run <name> <limit_s> <command>...   Logs to $out/<name>.log. Always returns 0; the outcome is in runs.tsv.
run() {
    local name=$1 limit=$2 log="$out/$1.log" start rc=0
    shift 2
    if (( deadline > 0 && $(date +%s) + limit + reserve_s > deadline )); then
        printf '%s\tskipped\t0\t%s\n' "$name" "${log#"$out"/}" >>"$runs"
        echo "skip $name: under $(( limit + reserve_s )) s left before the deadline"
        return 0
    fi
    echo "$(date -u +%H:%M:%S) $name (limit ${limit}s)"
    start=$(date +%s)
    timeout -k 20 "$limit" "$@" >"$log" 2>&1 || rc=$?
    printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - start ))" "${log#"$out"/}" >>"$runs"
}

# Clocks and power limits move GEMM times, and they are where the Server, Workstation, and Max-Q editions differ.
fields=name,pci.bus_id,power.limit,power.default_limit,clocks.max.sm,clocks.max.mem,driver_version
{
    timeout 60 nvidia-smi --query-gpu="$fields" --format=csv || echo "nvidia-smi --query-gpu failed"
    timeout 60 nvidia-smi -q -d CLOCK,POWER,PERFORMANCE || echo "nvidia-smi -q failed"
    echo "nproc $(nproc)"
} >"$out/env_extra.txt" 2>&1

run env 180 python3 "$here/env_probe.py" "$out"
# Every step below imports fp8bench, so an image where it cannot load would spend the hour logging failures.
if ! (cd "$here" && timeout 300 python3 -c 'import fp8bench') >"$out/import_check.log" 2>&1; then
    echo "fp8bench does not import on this image; see import_check.log" >&2
    exit 1
fi
run fetch_tables 120 python3 "$here/fetch_tables.py" "$out"

# H1 baseline and H2 first, so the issue's own question is answered even if the sweep runs out of time.
run baseline 480 python3 "$here/compare.py" --out "$out" --budget-s 420 \
    --arms shipped default default-nospec pr39872 pro5000
# H4 before the sweep, so neither can squeeze out the other's arms. FlashInfer may build its MXFP8 kernels on first
# use, so these get the longest limit per arm; linear-shipped is their yardstick until the tuned table exists.
run mxfp8 600 python3 "$here/compare.py" --out "$out" --budget-s 540 \
    --arms linear-shipped mxfp8-cutlass mxfp8-b12x
# The sweep only matters through the tuned arms, so it gets what is left once their limit is set aside.
tuned_limit=420
tune_limit=1200
if (( deadline > 0 )); then
    left=$(( deadline - $(date +%s) - reserve_s - tuned_limit ))
    if (( left < tune_limit )); then
        tune_limit=$left
    fi
fi
if (( tune_limit >= 300 )); then
    run tune "$tune_limit" python3 "$here/tune.py" --out "$out" --budget-s $(( tune_limit - 120 ))
else
    printf 'tune\tskipped\t0\ttune.log\n' >>"$runs"
fi
run tuned "$tuned_limit" python3 "$here/compare.py" --out "$out" --budget-s 360 \
    --arms tuned-stock tuned-stock-nospec tuned-splitk linear-tuned
# Dropping M's specialization also drops its multiple-of-16 hint, which prefill-sized M may miss.
run nospec_large_m 300 python3 "$here/compare.py" --out "$out" --budget-s 240 \
    --arms default default-nospec --ms 16 128 2048

# T0: the tuner sglang ships, at the image's commit (v0.5.23), on one [32, 32] shape.
tuner=/root/tuning_block_wise_kernel.py
v0523=https://raw.githubusercontent.com/sgl-project/sglang/297e54fd105e974cdf401e5b4d20253a22c16f90
if curl -fsSL --max-time 60 -o "$tuner" "$v0523/benchmark/kernels/quantization/tuning_block_wise_kernel.py"; then
    run stock_tuner 300 python3 "$tuner" --N 1792 --K 5120 --block-n 32 --block-k 32 --batch-sizes 1 \
        --out-dtype bfloat16 --save-path /root/stock-tuner
else
    printf 'stock_tuner\tdownload-failed\t0\t-\n' >>"$runs"
fi

python3 "$here/summarize.py" "$out" || echo "summarize.py exited $?; it reads only files, so rerun it on the laptop"
echo "setup done"
