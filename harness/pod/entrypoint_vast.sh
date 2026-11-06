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
vast_reply=""

# Only a 2xx whose body does not say `"success": false` shows the request landed: Vast answers a destroy or a stop it
# will not do with a 200 that says so. The reply stays in vast_reply for the caller and is logged only when the call
# failed, since a read of the instance carries its env. The key reaches curl as config on stdin, so it is never in
# any argv.
vast_call() {
    local method=$1 path=$2 data=$3 key=${vast_key//\\/\\\\} out said body=()
    local refused='"success"[[:space:]]*:[[:space:]]*false'
    key=${key//\"/\\\"}
    [[ -n $data ]] && body=(--data "$data")
    out=$(printf 'header = "Authorization: Bearer %s"\n' "$key" |
        curl -sS -K - --connect-timeout 10 -m 30 -w 'http=%{http_code}' -X "$method" \
            -H 'Content-Type: application/json' "${body[@]}" "https://console.vast.ai/api/v0/$path" 2>&1)
    vast_code=none
    [[ $out =~ http=([0-9]{3})$ ]] && vast_code=${BASH_REMATCH[1]}
    vast_reply=${out%http=*}
    if [[ $vast_code == 2[0-9][0-9] && ! $vast_reply =~ $refused ]]; then
        log "curl $method /api/v0/$path: http $vast_code"
        return 0
    fi
    said=${vast_reply//$'\n'/ | }
    log "curl $method /api/v0/$path: http $vast_code, not done${said:+: ${said:0:300}}"
    return 1
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
    # A read of this instance with its own key: the one sign before the deadline that the destroy can work. Vast
    # answers a key that does not own the instance with a 200 and no instance, so only a reply naming this id counts.
    if [[ -n $vast_id && -n $vast_key ]]; then
        vast_call GET "instances/$vast_id/?owner=me" '' && reply_names_self
        local named=$?
        log "vast key reads instance $vast_id: http $vast_code"
        (( named == 0 )) && log "vast probe ok id=$vast_id"
    fi
    log "box setup done"
}

# Whether the last reply's instance is this one: parsed by python3 where the image has it, else the id has to follow
# an "id" key, which a reply about no instance or another one does not show.
reply_names_self() {
    local pattern="\"id\"[[:space:]]*:[[:space:]]*\"?${vast_id}\"?([^0-9]|\$)"
    [[ $vast_id =~ ^[0-9]+$ ]] || return 1
    if command -v python3 >/dev/null 2>&1; then
        printf '%s' "$vast_reply" | python3 -I -c '
import json, sys
found = json.load(sys.stdin).get("instances")
if isinstance(found, list):
    found = found[0] if len(found) == 1 else None
sys.exit(0 if isinstance(found, dict) and str(found.get("id")) == sys.argv[1] else 1)
' "$vast_id" 2>/dev/null
    else
        [[ $vast_reply =~ $pattern ]]
    fi
}

# sshd's StrictModes refuses every key while authorized_keys, or /root/.ssh or /root above it, belongs to another user
# or is writable by group or others, and some Vast hosts inject the account's key that way: two of three on 2026-11-06
# logged "bad ownership or modes for file /root/.ssh/authorized_keys" for every login, and `up` destroyed both.
# Only what stat says goes to the log, never what the files hold.
ssh_modes() {
    local path said=""
    for path in /root /root/.ssh /root/.ssh/authorized_keys; do
        if [[ -e $root$path || -L $root$path ]]; then
            said+=", $path $(stat -c '%u:%g %a %F' "$root$path" 2>&1)"
        else
            said+=", $path missing"
        fi
    done
    printf '%s' "${said#, }"
}

# A symlink is left alone, since chown and chmod would change the file it points to instead.
fix_ssh_mode() {
    [[ -e $root$1 && ! -L $root$1 ]] || return 0
    chown 0 "$root$1" 2>/dev/null
    chmod "$2" "$root$1" 2>/dev/null
}

# Vast may write the key file after this starts, or again later, so the repair repeats every 3 s, inside the 10 s
# between `up`'s checks, for 20 min: `up` needs SSH from its first check until apt is done, up to 15 min
# (VAST_SETUP_WAIT in vast_box.rs), and starts checking soon after the container does. The modes are logged whenever
# they differ from the last round's, so the log shows what the host left, what the repair made of it, and a later write.
keep_ssh_modes() {
    local round modes last=""
    for ((round = 0; round < 400; round++)); do
        modes=$(ssh_modes)
        [[ $modes == "$last" ]] || log "ssh key modes: $modes"
        last=$modes
        fix_ssh_mode /root go-w
        fix_ssh_mode /root/.ssh 700
        fix_ssh_mode /root/.ssh/authorized_keys 600
        sleep 3
    done
}
keep_ssh_modes &

if [[ -n $vast_id && -n $vast_key ]]; then
    log "vast self-destruct ready for instance $vast_id"
else
    log "VAST SELF-DESTRUCT MISSING: CONTAINER_ID or CONTAINER_API_KEY is not set, so this box cannot destroy itself" \
        "at its deadline; harness up ends such a box"
fi
