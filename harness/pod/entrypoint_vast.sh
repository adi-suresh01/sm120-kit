# shellcheck shell=bash
# shellcheck disable=SC2154  # root and log come from entrypoint.sh, which this is spliced into
# Vast's part of the pod entrypoint. `harness up` splices it into entrypoint.sh just before that file's last line,
# `main`, so the functions here replace RunPod's while the watchdog loop stays the same code. Vast runs the result
# from the instance's onstart, beside Vast's own init and sshd, so here the watchdog is not the container's main
# process: `up` checks instead that it is alive and holds the key to destroy this very instance.
# Env, set by Vast: CONTAINER_ID (the instance id) and CONTAINER_API_KEY (a key scoped to this instance). Some images
# name the instance only in VAST_CONTAINERLABEL, as C.<id>.

mkdir -p "$root/workspace" 2>/dev/null
vast_id=${CONTAINER_ID:-}
if [[ -z $vast_id && ${VAST_CONTAINERLABEL:-} =~ ^C\.([0-9]+)$ ]]; then
    vast_id=${BASH_REMATCH[1]}
fi
vast_key=${CONTAINER_API_KEY:-}
# The jobs never need the key or the entrypoint's own text, and container_env.sh is written from the exported env.
unset CONTAINER_API_KEY HARNESS_ENTRYPOINT_B64
vast_code=none

# Only a 2xx shows the request landed. The key reaches curl as config on stdin, so it is never in any argv.
vast_call() {
    local method=$1 path=$2 data=$3 key=${vast_key//\\/\\\\} out body=()
    key=${key//\"/\\\"}
    [[ -n $data ]] && body=(--data "$data")
    out=$(printf 'header = "Authorization: Bearer %s"\n' "$key" |
        curl -sS -K - --connect-timeout 10 -m 30 -o /dev/null -w 'http=%{http_code}' -X "$method" \
            -H 'Content-Type: application/json' "${body[@]}" "https://console.vast.ai/api/v0/$path" 2>&1)
    vast_code=none
    [[ $out =~ http=([0-9]{3}) ]] && vast_code=${BASH_REMATCH[1]}
    out=${out//$'\n'/ | }
    log "curl $method /api/v0/$path: http $vast_code${out:+: ${out:0:500}}"
    [[ $vast_code == 2[0-9][0-9] ]]
}

delete_self() {
    if [[ -z $vast_key || -z $vast_id ]]; then
        log "CONTAINER_API_KEY or CONTAINER_ID is not set; skipping the API delete"
        return 1
    fi
    vast_call DELETE "instances/$vast_id/" '{}'
}

# Destroy first, since a stopped instance still bills its disk. A stop is the fallback because it ends the GPU charge,
# and the reaper destroys a stopped instance once it is past its deadline.
terminate_round() {
    delete_self && return 0
    [[ -n $vast_key && -n $vast_id ]] && vast_call PUT "instances/$vast_id/" '{"state": "stopped"}'
}

# Vast runs its own sshd with the account's keys, so this only saves the env for the jobs and installs the packages.
# Vast's sshd answers before this ends, so `up` waits for the last line. Slim images ship no curl, the only way this
# box can destroy itself, so a failed install is a box without a self-destruct, said as loudly as a missing key.
setup_ssh() {
    (umask 077; export -p >"$root/root/container_env.sh") 2>/dev/null || log "writing container_env.sh failed"
    # HARNESS_APT is a space-separated list of package names, validated by the harness.
    # shellcheck disable=SC2086
    run_logged apt-get update -qq &&
        DEBIAN_FRONTEND=noninteractive run_logged apt-get install -y -qq ${HARNESS_APT:-curl procps}
    if ! command -v curl >/dev/null 2>&1; then
        log "VAST SELF-DESTRUCT MISSING: curl is not installed, so this box cannot destroy itself at its deadline"
        return 1
    fi
    # A read of this instance with its own key: the one sign before the deadline that the destroy can work.
    if [[ -n $vast_id && -n $vast_key ]]; then
        vast_call GET "instances/$vast_id/?owner=me" ''
        log "vast key reads instance $vast_id: http $vast_code"
    fi
    log "box setup done"
}

if [[ -n $vast_id && -n $vast_key ]]; then
    log "vast self-destruct ready for instance $vast_id"
else
    log "VAST SELF-DESTRUCT MISSING: CONTAINER_ID or CONTAINER_API_KEY is not set, so this box cannot destroy itself" \
        "at its deadline; harness up ends such a box"
fi
