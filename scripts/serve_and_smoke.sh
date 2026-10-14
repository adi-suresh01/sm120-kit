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

response=$(curl -sf "http://127.0.0.1:${port}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"${model}\", \"max_tokens\": 64, \"temperature\": 0,
         \"messages\": [{\"role\": \"user\", \"content\": \"List the first five prime numbers.\"}]}" \
    || true)

if [[ -z "$response" ]]; then
    echo "RESULT=SMOKE_FAIL boot_seconds=${boot_seconds}"
    exit 1
fi

echo "$response" >"${log_file%.log}.response.json"
echo "RESULT=OK boot_seconds=${boot_seconds}"
