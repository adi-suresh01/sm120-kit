#!/usr/bin/env bash
# Boots an OpenAI-compatible server, sends one chat request, and reports a single verdict line:
#   RESULT=OK | RESULT=BOOT_FAIL | RESULT=SMOKE_FAIL | RESULT=POST_FAIL | RESULT=TIMEOUT
# POST_SMOKE="<command>" runs after a passing smoke test while the server is still up, in a session of its own; its
# output goes to <log_file minus .log>.post.log. It is stopped as POST_FAIL if the server dies, the log matches
# FAIL_PATTERN, or the log goes quiet for STALL_MINUTES (default 15). Whatever it started is ended when this script
# exits, however it exits.
# FAIL_PATTERN="<ERE>" ends the boot wait as soon as the log matches it, reported as BOOT_FAIL exit_code=pattern.
# PROGRESS_FILE, when set, gets the boot, smoke, and quality phases (see progress.sh).
# BOOT_POLL_S, POST_POLL_S, STOP_POLL_S (default 5, 10, 1) set the polling intervals; tests shorten them.
# Usage: serve_and_smoke.sh <log_file> <port> <model> -- <server command...>
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
log_file=$1 port=$2 model=$3
shift 3
[[ ${1:-} == "--" ]] && shift

boot_timeout=${BOOT_TIMEOUT:-1800}
boot_poll_s=${BOOT_POLL_S:-5}
post_poll_s=${POST_POLL_S:-10}
stop_poll_s=${STOP_POLL_S:-1}
mkdir -p "$(dirname "$log_file")"

# Progress is for display only; a failed write must not change the result.
progress() {
    if [[ -n ${PROGRESS_FILE:-} ]]; then
        bash "$here/progress.sh" "$PROGRESS_FILE" "$@" || true
    fi
}

"$@" >"$log_file" 2>&1 &
server_pid=$!

stop_server() {
    kill "$server_pid" 2>/dev/null || true
    # A launcher stuck in its own shutdown would otherwise hold this script until the caller's outer timeout.
    for _ in $(seq 60); do
        kill -0 "$server_pid" 2>/dev/null || break
        sleep "$stop_poll_s"
    done
    kill -9 "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}

# POST_SMOKE's clients sit several processes down, and a hook can background one or leave one whose parent died; a
# walk of the process tree misses those and races a parent that starts its next step. setsid makes POST_SMOKE's pid
# the id of a process group that holds all of it (as a background job of a shell without job control it is no group
# leader, so setsid does not fork). The outer timeout signals only its own group, so this runs on every exit: an
# orphan left running could hit the next variant's server on the same port.
post_pid=""
stop_post() {
    if [[ -z $post_pid ]] || ! kill -TERM -- "-$post_pid" 2>/dev/null; then
        return 0
    fi
    for _ in $(seq 50); do
        kill -0 -- "-$post_pid" 2>/dev/null || return 0
        sleep 0.1
    done
    kill -KILL -- "-$post_pid" 2>/dev/null || true
}
# The post group goes first: the outer timeout kills this script 30 s after its TERM, and stop_server can take
# longer than that.
trap 'stop_post; stop_server' EXIT
# GNU timeout sends TERM to this script and then to its group; bash dying of that second signal often skips the EXIT
# trap, which would leave the post group running. Exiting on the signal makes the EXIT trap run.
trap 'exit 143' TERM INT HUP

fatal_in_log() {
    [[ -n ${FAIL_PATTERN:-} ]] && grep -qE "$FAIL_PATTERN" "$log_file"
}

started=$(date +%s)
until curl -sf -m 10 "http://127.0.0.1:${port}/health" >/dev/null; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid" && code=0 || code=$?
        progress boot_fail "server exited with code $code"
        echo "RESULT=BOOT_FAIL exit_code=${code} boot_seconds=$(( $(date +%s) - started ))"
        exit 1
    fi
    # A rank that faults can leave the launcher waiting on the other rank until the boot timeout; a matching
    # fatal line ends the wait now.
    if fatal_in_log; then
        progress boot_fail "log matched: $(grep -a -m1 -E "$FAIL_PATTERN" "$log_file" | head -c 300 || true)"
        echo "RESULT=BOOT_FAIL exit_code=pattern boot_seconds=$(( $(date +%s) - started ))"
        exit 1
    fi
    if (( $(date +%s) - started > boot_timeout )); then
        progress boot_fail "no answer on /health within ${boot_timeout}s"
        echo "RESULT=TIMEOUT boot_seconds=${boot_timeout}"
        exit 1
    fi
    sleep "$boot_poll_s"
done
boot_seconds=$(( $(date +%s) - started ))
progress boot_ok "boot_seconds=${boot_seconds}"

ask() {
    curl -sf -m "${REQUEST_TIMEOUT:-600}" "http://127.0.0.1:${port}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\": \"${model}\", \"max_tokens\": 64, \"temperature\": 0,
             \"messages\": [{\"role\": \"user\", \"content\": \"List the first $1 prime numbers.\"}]}"
}

# CONCURRENCY > 1 sends parallel requests; a count that is not itself a captured graph size makes decode batches
# pad up to the next one, which is where padding-row bugs live.
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
    progress smoke_fail "failed_requests=${failed}/${concurrency}"
    echo "RESULT=SMOKE_FAIL boot_seconds=${boot_seconds} failed_requests=${failed}/${concurrency}"
    exit 1
fi
progress smoke_ok "requests=${concurrency}"
# Large checkpoints take minutes to boot, so evaluations reuse this server instead of booting their own. Eval
# clients retry a dead server for up to an hour, so the server is watched while they run.
if [[ -n ${POST_SMOKE:-} ]]; then
    progress quality_start
    setsid bash -c "$POST_SMOKE" >"${log_file%.log}.post.log" 2>&1 &
    post_pid=$!
    stall_minutes=${STALL_MINUTES:-15}
    while kill -0 "$post_pid" 2>/dev/null; do
        reason=""
        if ! kill -0 "$server_pid" 2>/dev/null; then
            reason=server_exit
        elif fatal_in_log; then
            reason=pattern
        elif [[ -n $(find "$log_file" -mmin "+${stall_minutes}") ]]; then
            reason=stall
        fi
        if [[ -n $reason ]]; then
            stop_post
            progress quality_done "post_exit=${reason}"
            echo "RESULT=POST_FAIL boot_seconds=${boot_seconds} requests=${concurrency} post_exit=${reason}"
            exit 1
        fi
        sleep "$post_poll_s"
    done
    wait "$post_pid" && post=0 || post=$?
    progress quality_done "post_exit=${post}"
    if (( post != 0 )); then
        echo "RESULT=POST_FAIL boot_seconds=${boot_seconds} requests=${concurrency} post_exit=${post}"
        exit 1
    fi
fi
echo "RESULT=OK boot_seconds=${boot_seconds} requests=${concurrency}"
