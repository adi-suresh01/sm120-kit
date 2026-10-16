#!/usr/bin/env bash
# Pod entrypoint, passed to the container as `bash -c <this file>` by `harness up`, so this shell is the container's
# main process: PID 1, or on RunPod the only child of docker-init (PID 1), which exits when this shell does and
# takes the whole container with it. The deadline watchdog runs in this shell itself, so whatever kills it also
# ends the pod, while a forked subshell would carry this whole text as its command line and could die to a
# `pkill -f sshd` alone and leave the pod running. SSH setup, which can hang on apt, runs in the background instead.
# Env: HARNESS_DEADLINE_EPOCH (absolute, so a container restart cannot extend it), HARNESS_APT, PUBLIC_KEY,
#      RUNPOD_POD_ID and RUNPOD_API_KEY (set by RunPod).
# Tests only: HARNESS_TEST_ROOT prefixes every absolute path; HARNESS_WATCHDOG_NAP_S shortens the naps.
# No -e or -u: a failed step or an unset variable must never end this shell, or the watchdog would go with it.
set -o pipefail

root=${HARNESS_TEST_ROOT:-}
watchdog_log=$root/workspace/harness-watchdog.log
nap_s=${HARNESS_WATCHDOG_NAP_S:-30}
# Longer naps would space out the attempts past the deadline.
[[ $nap_s =~ ^[1-9][0-9]?$ ]] && (( nap_s <= 30 )) || nap_s=30

# Writes to the log file and to stderr (the container log), and cannot fail or block the caller: an unwritable
# /workspace, a full disk, or a closed stderr fails only the write, and a write stuck on a hung volume or a stalled
# container log is stuck in a subshell, not in the watchdog.
log() {
    local line
    line="$(date -u +%FT%TZ 2>/dev/null) $*"
    {
        { printf '%s\n' "$line" >>"$watchdog_log"; } 2>/dev/null
        { printf 'harness: %s\n' "$line" >&3; } 3>&2 2>/dev/null
    } &
    return 0
}

# Captures the output instead of redirecting it to the log, because bash skips a command whose redirection fails.
run_logged() {
    local out rc
    out=$("$@" 2>&1)
    rc=$?
    out=${out//$'\n'/ | }
    log "$*: exit $rc${out:+: ${out:0:500}}"
    return "$rc"
}

# runpodctl's HTTP client has no timeout of its own, and a hung call would stall every later round.
bound=()
if command -v timeout >/dev/null 2>&1; then
    bound=(timeout -k 5 60)
fi

# Only a 2xx shows the request landed: RunPod also answers 404 for a pod the key cannot see.
delete_self() {
    if [[ -z ${RUNPOD_API_KEY:-} || -z ${RUNPOD_POD_ID:-} ]]; then
        log "RUNPOD_API_KEY or RUNPOD_POD_ID is not set; skipping the API delete"
        return 1
    fi
    local key=${RUNPOD_API_KEY//\\/\\\\} out code=none
    key=${key//\"/\\\"}
    # The header reaches curl as config on stdin, so the key is never in any process's argv.
    out=$(printf 'header = "Authorization: Bearer %s"\n' "$key" |
        curl -sS -K - --connect-timeout 10 -m 30 -o /dev/null -w 'http=%{http_code}' -X DELETE \
            "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" 2>&1)
    [[ $out =~ http=([0-9]{3}) ]] && code=${BASH_REMATCH[1]}
    out=${out//$'\n'/ | }
    log "curl DELETE /v2/pods/$RUNPOD_POD_ID: http $code${out:+: ${out:0:500}}"
    [[ $code == 2[0-9][0-9] ]]
}

# One round stops at the first call that reports success; the next round tries again regardless.
terminate_round() {
    delete_self && return 0
    run_logged "${bound[@]}" runpodctl pod remove "${RUNPOD_POD_ID:-}" && return 0
    run_logged "${bound[@]}" runpodctl pod stop "${RUNPOD_POD_ID:-}"
}

# Never returns. The deadline is re-read from the clock every nap, so a nap cut short only means an early check,
# and a call that reports success is not proof the pod is going away: only the pod's own death ends this loop.
watchdog() {
    local deadline=${HARNESS_DEADLINE_EPOCH:-} now
    # A missing or garbled deadline means the pod has no approved time left.
    [[ $deadline =~ ^[0-9]+$ ]] || deadline=0
    log "watchdog armed for epoch $deadline in pid $$"
    while true; do
        printf -v now '%(%s)T' -1 2>/dev/null || now=$(date +%s)
        if [[ ! $now =~ ^[0-9]+$ ]]; then
            log "cannot read the clock"
        elif (( now < deadline )); then
            sleep $(( deadline - now < nap_s ? deadline - now : nap_s ))
            continue
        else
            log "past the deadline ($deadline); terminating this pod"
            terminate_round
        fi
        sleep "$nap_s"
    done
}

setup_ssh() {
    # Jobs run over SSH, which starts sessions with a clean environment; they source this to get the pod's env.
    # They never need the API key, so it stays out of the file.
    (umask 077; unset RUNPOD_API_KEY; export -p >"$root/root/container_env.sh") 2>/dev/null ||
        log "writing container_env.sh failed"
    mkdir -p "$root/run/sshd" "$root/root/.ssh"
    chmod 700 "$root/root/.ssh"
    printf '%s\n' "${PUBLIC_KEY:-}" >>"$root/root/.ssh/authorized_keys"
    chmod 600 "$root/root/.ssh/authorized_keys"
    # HARNESS_APT is a space-separated list of package names, validated by the harness.
    # shellcheck disable=SC2086
    run_logged apt-get update -qq &&
        DEBIAN_FRONTEND=noninteractive run_logged apt-get install -y -qq ${HARNESS_APT:-openssh-server curl procps}
    # Some images ship openssh-server with the host keys stripped, and sshd refuses to start without them.
    run_logged ssh-keygen -A
    run_logged "$root/usr/sbin/sshd"
}

main() {
    # Untrapped, a SIGINT that reaches this shell and its nap together (`pkill -INT -f sleep`) breaks the loop and
    # exits the shell, and docker-init forwards TERM and HUP to this shell, which would kill it outright.
    local sig
    for sig in HUP INT QUIT TERM USR1 USR2 ALRM; do
        # shellcheck disable=SC2064  # the name is fixed per trap on purpose
        trap "log 'ignored SIG$sig'" "$sig"
    done
    # A closed stderr must make writes fail, not kill the shell.
    trap : PIPE
    setup_ssh &
    watchdog
}

main
