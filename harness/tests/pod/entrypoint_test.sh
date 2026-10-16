#!/usr/bin/env bash
# Runs the entrypoint's deadline watchdog against fake curl and runpodctl, which record their calls instead of
# acting on a pod.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
entrypoint="$here/../../pod/entrypoint.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# The tests source everything except the last line, which must be the bare `main` call.
[[ $(tail -n 1 "$entrypoint") == main ]] || { echo "entrypoint.sh must end with a bare main call" >&2; exit 1; }

mkdir -p "$work/bin"
cat >"$work/bin/curl" <<'EOF'
#!/usr/bin/env bash
printf 'curl %s\n' "$*" >>"$FAKE_LOG"
printf '%s' "$FAKE_CURL_CODE"
EOF
cat >"$work/bin/runpodctl" <<'EOF'
#!/usr/bin/env bash
printf 'runpodctl %s\n' "$*" >>"$FAKE_LOG"
[[ " $FAKE_RUNPODCTL_OK " == *" $2 "* ]]
EOF
chmod +x "$work/bin/curl" "$work/bin/runpodctl"

# Starts the watchdog in the background with the given deadline, fake replies, and 1 s loops.
start_watchdog() {
    local deadline=$1 curl_code=$2 runpodctl_ok=$3
    rm -f "$work/calls.log" "$work/watchdog.log"
    (
        export PATH="$work/bin:$PATH" FAKE_LOG="$work/calls.log" FAKE_CURL_CODE=$curl_code
        export FAKE_RUNPODCTL_OK=$runpodctl_ok HARNESS_DEADLINE_EPOCH=$deadline
        export RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=test-key
        # shellcheck source=/dev/null
        source <(sed '$d' "$entrypoint")
        # shellcheck disable=SC2034  # read by the sourced watchdog
        watchdog_log="$work/watchdog.log" poll_s=1 retry_s=1
        watchdog
    ) &
    watchdog_pid=$!
}

# Waits up to $1 seconds for the watchdog to return.
finishes_within() {
    local i
    for ((i = 0; i < $1 * 10; i++)); do
        kill -0 "$watchdog_pid" 2>/dev/null || { wait "$watchdog_pid"; return 0; }
        sleep 0.1
    done
    kill "$watchdog_pid" 2>/dev/null || true
    return 1
}

fail() {
    echo "FAIL: $*" >&2
    for f in calls.log watchdog.log; do
        echo "--- $f" >&2
        cat "$work/$f" >&2 2>/dev/null || true
    done
    exit 1
}

past=$(( $(date +%s) - 60 ))

start_watchdog "$past" 204 ""
finishes_within 5 || fail "a 204 delete should end the watchdog"
grep -q -- '-X DELETE -H Authorization: Bearer test-key https://api.runpod.io/v2/pods/pod123' "$work/calls.log" \
    || fail "delete call"
! grep -q runpodctl "$work/calls.log" || fail "runpodctl ran after a successful delete"
grep -q 'http 204' "$work/watchdog.log" || fail "delete result not logged"
echo "ok: past deadline deletes the pod"

start_watchdog "$past" 403 "stop"
finishes_within 5 || fail "a successful stop should end the watchdog"
[[ $(cut -d' ' -f1-3 "$work/calls.log" | tr '\n' ',') == "curl -sS -m,runpodctl pod remove,runpodctl pod stop," ]] \
    || fail "fallback order"
grep -q 'runpodctl pod remove: failed' "$work/watchdog.log" || fail "failed remove not logged"
grep -q 'runpodctl pod stop: ok' "$work/watchdog.log" || fail "stop not logged"
echo "ok: falls back to runpodctl remove, then stop"

start_watchdog "$past" 404 "remove"
finishes_within 5 || fail "remove should end the watchdog"
grep -q 'runpodctl pod remove pod123' "$work/calls.log" || fail "a 404 must not count as deleted"
echo "ok: a 404 is not taken as success"

start_watchdog "garbage" 204 ""
finishes_within 5 || fail "a garbled deadline should act at once"
echo "ok: a garbled deadline acts at once"

start_watchdog "$(( $(date +%s) + 3 ))" 204 ""
sleep 1
[[ ! -s $work/calls.log ]] || fail "acted before the deadline"
finishes_within 8 || fail "did not act after the deadline"
echo "ok: waits for the deadline"

start_watchdog "$past" 500 ""
sleep 3
kill "$watchdog_pid" 2>/dev/null || fail "gave up after failed attempts"
wait "$watchdog_pid" 2>/dev/null || true
(( $(grep -c '^curl' "$work/calls.log") >= 2 )) || fail "failed attempts are not retried"
echo "ok: keeps retrying while every attempt fails"
