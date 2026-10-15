#!/usr/bin/env bash
# Boots an OpenAI-compatible server, sends one chat request, and reports a single verdict line:
#   RESULT=OK | RESULT=BOOT_FAIL | RESULT=SMOKE_FAIL | RESULT=POST_FAIL | RESULT=TIMEOUT
# POST_SMOKE="<command>" runs after a passing smoke test while the server is still up; its output goes to
# <log_file minus .log>.post.log.
# FAIL_PATTERN="<ERE>" ends the boot wait as soon as the log matches it, reported as BOOT_FAIL exit_code=pattern.
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
    # A rank that faults can leave the launcher waiting on the other rank until the boot timeout; a matching
    # fatal line ends the wait now.
    if [[ -n ${FAIL_PATTERN:-} ]] && grep -qE "$FAIL_PATTERN" "$log_file"; then
        echo "RESULT=BOOT_FAIL exit_code=pattern boot_seconds=$(( $(date +%s) - started ))"
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
    curl -sf -m "${REQUEST_TIMEOUT:-600}" "http://127.0.0.1:${port}/v1/chat/completions" \
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
# Large checkpoints take minutes to boot, so evaluations reuse this server instead of booting their own.
if [[ -n ${POST_SMOKE:-} ]]; then
    bash -c "$POST_SMOKE" >"${log_file%.log}.post.log" 2>&1 && post=0 || post=$?
    if (( post != 0 )); then
        echo "RESULT=POST_FAIL boot_seconds=${boot_seconds} requests=${concurrency} post_exit=${post}"
        exit 1
    fi
fi
echo "RESULT=OK boot_seconds=${boot_seconds} requests=${concurrency}"
