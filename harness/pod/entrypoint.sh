#!/usr/bin/env bash
# Pod entrypoint, passed to the container as `bash -c <this file>` by `harness up`. Arms the deadline watchdog
# before anything that can hang, then starts sshd.
# Env: HARNESS_DEADLINE_EPOCH (absolute, so a container restart cannot extend it), HARNESS_APT, PUBLIC_KEY,
#      RUNPOD_POD_ID and RUNPOD_API_KEY (set by RunPod).
# No -e: a failed apt or sshd step must not end PID 1, or the watchdog would go with it.
set -uo pipefail

watchdog_log=/workspace/harness-watchdog.log
poll_s=15
retry_s=60

log() {
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$watchdog_log"
}

# 204 is the only proof of success: RunPod also answers 404 for a pod the key cannot see.
delete_self() {
    local code
    code=$(curl -sS -m 30 -o /dev/null -w '%{http_code}' -X DELETE \
        -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
        "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID:-}" 2>>"$watchdog_log")
    log "curl DELETE /v2/pods/${RUNPOD_POD_ID:-}: http ${code:-none}"
    [[ $code == 204 ]]
}

terminate_self() {
    delete_self && return 0
    if runpodctl pod remove "${RUNPOD_POD_ID:-}" >>"$watchdog_log" 2>&1; then
        log "runpodctl pod remove: ok"
        return 0
    fi
    log "runpodctl pod remove: failed"
    if runpodctl pod stop "${RUNPOD_POD_ID:-}" >>"$watchdog_log" 2>&1; then
        log "runpodctl pod stop: ok"
        return 0
    fi
    log "runpodctl pod stop: failed"
    return 1
}

watchdog() {
    local deadline=${HARNESS_DEADLINE_EPOCH:-0}
    # A missing or garbled deadline means the pod has no approved time left.
    [[ $deadline =~ ^[0-9]+$ ]] || deadline=0
    log "watchdog armed for epoch $deadline"
    while (( $(date +%s) < deadline )); do
        sleep "$poll_s"
    done
    log "deadline reached"
    until terminate_self; do
        sleep "$retry_s"
    done
}

main() {
    export -p >/root/container_env.sh
    watchdog &
    mkdir -p /run/sshd /root/.ssh
    chmod 700 /root/.ssh
    printf '%s\n' "${PUBLIC_KEY:-}" >>/root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    # HARNESS_APT is a space-separated list of package names, validated by the harness.
    # shellcheck disable=SC2086
    if ! { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ${HARNESS_APT:-openssh-server curl procps}; }; then
        log "apt-get install failed"
    fi
    /usr/sbin/sshd || log "sshd failed to start"
    sleep infinity
}

main
