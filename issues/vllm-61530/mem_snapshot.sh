#!/usr/bin/env bash
# Where the memory went on a live server, after GSM8K and before the rows: what each GPU holds, what vLLM says it
# loaded and left for the KV cache, how much host memory the Engram tables pinned, and the host and container totals
# beside the server's processes. The whole-run trace, peaks included, is memory.jsonl from the harness's memory guard.
# Usage: mem_snapshot.sh <run_dir>   (writes <run_dir>/memory_snapshot.txt)
set -euo pipefail

run_dir=${1:?usage: $0 <run_dir>}
out="$run_dir/memory_snapshot.txt"
section() { printf '\n## %s\n' "$1"; }
try() { timeout 60 "$@" || echo "($1 failed: exit $?)"; }

{
    section "GPU memory"
    try nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

    # Each rank logs one Engram line per table (common/engram.py), so the host total is the two tables' sizes summed,
    # times the TP size.
    section "server.log: load, KV cache, Engram"
    grep -aE 'Engram table offloaded|Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|CUDA graph|available host memory|Initialized FlashInfer PCIe IPC' \
        "$run_dir/server.log" | cut -c1-300 || echo "(no matching lines)"

    section "host /proc/meminfo"
    grep -E '^(MemTotal|MemFree|MemAvailable|Cached|Shmem|Mlocked|Unevictable|AnonPages|HugePages_Total|Hugetlb):' \
        /proc/meminfo

    section "container cgroup"
    if [[ -r /sys/fs/cgroup/memory.current ]]; then
        echo "memory.current $(cat /sys/fs/cgroup/memory.current)"
        echo "memory.max $(cat /sys/fs/cgroup/memory.max)"
        grep -E '^(anon|file|shmem|unevictable|file_mapped) ' /sys/fs/cgroup/memory.stat || true
    else
        echo "(no cgroup v2 memory files)"
    fi

    # Pinned pages count in the RSS of the process that mapped them, so the ranks' RSS shows where the tables sit. awk
    # reads to the end instead of `head` closing the pipe: under pipefail a SIGPIPE here would fail the hook and
    # cost the arm its rows.
    section "server processes by RSS (KiB)"
    ps -eo pid,rss,args --sort=-rss | awk '(NR == 1 || /vllm|VLLM|python/) && n++ < 20 {print substr($0, 1, 160)}'
} >"$out" 2>&1
echo "memory snapshot in $out"
