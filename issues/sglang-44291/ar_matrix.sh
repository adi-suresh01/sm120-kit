#!/usr/bin/env bash
# Runs ar_repro.py over every mode, allocator, and repeat, keeping the full log so tracebacks survive.
# Usage: ar_matrix.sh [repeats]   writes results/sglang-44291/ar_repro.log and ar_repro.jsonl
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$here/../.." && pwd)
results_dir="$repo_root/results/sglang-44291"
repeats=${1:-2}
log="$results_dir/ar_repro.log"
mkdir -p "$results_dir"
: >"$log"

for rep in $(seq "$repeats"); do
    for conf in "" "expandable_segments:True"; do
        for mode in registered staged pynccl nccl; do
            echo ">>> rep=$rep conf=${conf:-default} mode=$mode" >>"$log"
            # A rank stuck in a faulted kernel would otherwise hold the pod until gloo's two-hour timeout.
            if PYTORCH_CUDA_ALLOC_CONF="$conf" timeout -k 15 300 \
                torchrun --nproc_per_node 2 "$here/ar_repro.py" --mode "$mode" >>"$log" 2>&1; then
                echo "<<< exit=0" >>"$log"
            else
                echo "<<< exit=$?" >>"$log"
            fi
        done
    done
done
grep '^{' "$log" >"$results_dir/ar_repro.jsonl" || true
echo "ar matrix done: $(wc -l <"$results_dir/ar_repro.jsonl") json rows"
