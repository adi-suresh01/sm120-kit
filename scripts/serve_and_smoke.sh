#!/usr/bin/env bash
# Boots an OpenAI-compatible server, sends one chat request, and reports a single verdict line:
#   RESULT=OK | RESULT=BOOT_FAIL | RESULT=SMOKE_FAIL | RESULT=TIMEOUT
# Usage: serve_and_smoke.sh <log_file> <port> <model> -- <server command...>
set -euo pipefail

log_file=$1 port=$2 model=$3
shift 3
[[ ${1:-} == "--" ]] && shift

boot_timeout=${BOOT_TIMEOUT:-1800}
mkdir -p "$(dirname "$log_file")"

"$@" >"$log_file" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT

started=$(date +%s)
until curl -sf "http://127.0.0.1:${port}/health" >/dev/null; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid" && code=0 || code=$?
        echo "RESULT=BOOT_FAIL exit_code=${code} boot_seconds=$(( $(date +%s) - started ))"
        exit 1
    fi
    if (( $(date +%s) - started > boot_timeout )); then
        echo "RESULT=TIMEOUT boot_seconds=${boot_timeout}"
        exit 1
    fi
    sleep 5
done
boot_seconds=$(( $(date +%s) - started ))

ask() {
    curl -sf "http://127.0.0.1:${port}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\": \"${model}\", \"max_tokens\": 64, \"temperature\": 0,
             \"messages\": [{\"role\": \"user\", \"content\": \"List the first $1 prime numbers.\"}]}"
}

# CONCURRENCY > 1 sends parallel requests so decode batches get padded up to a captured graph size, which is
# where padding-row bugs live; a single request never exercises them.
concurrency=${CONCURRENCY:-1}
base=${log_file%.log}.response
pids=()
for i in $(seq "$concurrency"); do
    out=$base.json
    if (( i > 1 )); then out=$base.$i.json; fi
    ask $((4 + i)) >"$out" &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done

if (( failed > 0 )); then
    echo "RESULT=SMOKE_FAIL boot_seconds=${boot_seconds} failed_requests=${failed}/${concurrency}"
    exit 1
fi
echo "RESULT=OK boot_seconds=${boot_seconds} requests=${concurrency}"
