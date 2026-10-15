#!/usr/bin/env bash
# Waits until every GPU is under 2 GiB used. A crashed server can leave a child process holding memory, which
# would make the next run fail for the wrong reason.
# Usage: wait_gpu_idle.sh [timeout_seconds]
set -euo pipefail

timeout=${1:-120}
deadline=$(( $(date +%s) + timeout ))
while true; do
    busiest=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -n 1)
    # An empty reading would compare as 0 and wave a broken driver through as idle.
    if [[ -z $busiest ]]; then echo "nvidia-smi reported no GPUs" >&2; exit 1; fi
    if (( busiest < 2048 )); then exit 0; fi
    if (( $(date +%s) > deadline )); then
        echo "a GPU still has ${busiest} MiB in use after ${timeout}s; kill leftover server processes first" >&2
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
        exit 1
    fi
    sleep 2
done
