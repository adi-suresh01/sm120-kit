#!/usr/bin/env bash
# Runs the entrypoint as Vast gets it (entrypoint.sh with entrypoint_vast.sh spliced in, written by the Rust test to
# $VAST_ENTRYPOINT_FILE) against fake curl, apt-get, sshd, and runpodctl that record their calls instead of acting,
# then the onstart loader ($VAST_ONSTART_FILE) under sh. Each case is its own box with its own root.
set -euo pipefail

work=$(mktemp -d)
declare -A box_pid=()

cleanup() {
    local pid
    for pid in "${box_pid[@]}"; do
        kill -9 -- "-$pid" 2>/dev/null || true
    done
    rm -rf "$work"
}
trap cleanup EXIT

fail() {
    echo "FAIL: $*" >&2
    local dir
    for dir in "$work"/*/; do
        [[ -f $dir/box.out ]] || continue
        echo "--- $(basename "$dir")" >&2
        cat "$dir/calls.log" "$dir/root/workspace/harness-watchdog.log" >&2 2>/dev/null || true
    done
    exit 1
}

mkdir -p "$work/bin"
# Answers FAKE_CURL_CODE_<METHOD>, else FAKE_CURL_CODE, with the body FAKE_CURL_BODY_<METHOD>, else none, so a
# destroy and a stop can be refused apart, by status or in the body of a 200.
cat >"$work/bin/curl" <<'EOF'
#!/usr/bin/env bash
printf 'curl %s\n' "$*" >>"$FAKE_DIR/calls.log"
format="" prev="" method=GET
for arg in "$@"; do
    [[ $prev == -w ]] && format=$arg
    [[ $prev == -X ]] && method=$arg
    [[ $prev == -K && $arg == - ]] && cat >>"$FAKE_DIR/curl_stdin"
    prev=$arg
done
code_var=FAKE_CURL_CODE_$method
code=${!code_var:-$FAKE_CURL_CODE}
body_var=FAKE_CURL_BODY_$method
printf '%s' "${!body_var:-}"
printf '%s' "${format//%\{http_code\}/$code}"
EOF
# chown is faked too: the test is not root, and the call is the evidence.
for tool in apt-get runpodctl sshd ssh-keygen chown; do
    # shellcheck disable=SC2016  # expanded when the fake runs
    printf '#!/usr/bin/env bash\nprintf "%s %%s\\n" "$*" >>"$FAKE_DIR/calls.log"\n' "$tool" >"$work/bin/$tool"
done
# The box has GNU stat; BSD stat (macOS) spells the same fields -f '%u:%g %Lp %HT'.
cat >"$work/bin/stat" <<'EOF'
#!/usr/bin/env bash
if [[ $1 == -c ]] && ! /usr/bin/stat -c '' / >/dev/null 2>&1; then
    format=${2//%a/%Lp}
    shift 2
    exec /usr/bin/stat -f "${format//%F/%HT}" "$@"
fi
exec /usr/bin/stat "$@"
EOF
chmod +x "$work/bin/"*
# A PATH with no curl at all, as on a slim image whose apt failed.
mkdir -p "$work/nocurl"
for tool in bash python3 env date mkdir sleep; do
    ln -s "$(command -v "$tool")" "$work/nocurl/$tool"
done
ln -s "$work/bin/apt-get" "$work/nocurl/apt-get"
# A PATH with curl but no python3, as on a slim CUDA image: the probe falls back to matching the reply's text.
mkdir -p "$work/nopy"
for tool in bash env date mkdir sleep cat; do
    ln -s "$(command -v "$tool")" "$work/nopy/$tool"
done
ln -s "$work/bin/apt-get" "$work/nopy/apt-get"
ln -s "$work/bin/curl" "$work/nopy/curl"
python=$(command -v python3)

entrypoint_text=$(cat "$VAST_ENTRYPOINT_FILE")
default_signals='import os, signal, sys
signal.signal(signal.SIGINT, signal.SIG_DFL)
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])'
key='vast_TESTKEY"quote\slash'
own_instance='{"instances": {"id": 31000001, "machine_id": 70001, "label": "sm120-x-0a0b0c"}}'

# Starts one box. Args: name deadline [VAR=value...] (overrides for the fakes and the box env)
start_box() {
    local name=$1 deadline=$2 dir=$work/$1
    shift 2
    mkdir -p "$dir/root/root"
    (
        export PATH="$work/bin:$PATH" FAKE_DIR=$dir HARNESS_TEST_ROOT=$dir/root HARNESS_WATCHDOG_NAP_S=1 \
            HARNESS_DEADLINE_EPOCH=$deadline CONTAINER_ID=31000001 CONTAINER_API_KEY=$key \
            HARNESS_ENTRYPOINT_B64=bm90IG5lZWRlZCBoZXJl HARNESS_APT='curl procps patch' HF_HOME=/workspace/hf \
            FAKE_CURL_CODE=200 FAKE_CURL_BODY_GET="$own_instance"
        exec env "$@" "$python" -I -c "$default_signals" bash -c "$entrypoint_text" >"$dir/box.out" 2>&1
    ) &
    box_pid[$name]=$!
    disown "$!"
}

calls() {
    grep -E "^curl .*$2" "$work/$1/calls.log" 2>/dev/null | grep -c . || true
}
log_has() {
    grep -q -- "$2" "$work/$1/root/workspace/harness-watchdog.log" 2>/dev/null
}
alive() {
    kill -0 "${box_pid[$1]}" 2>/dev/null
}
mode_of() {
    "$work/bin/stat" -c %a "$1"
}
# Waits up to 10 s for a line in a box's watchdog log.
wait_log() {
    local _
    for _ in $(seq 100); do
        log_has "$1" "$2" && return 0
        sleep 0.1
    done
    return 1
}

# A host whose key injection left the key file and both directories above it writable by others, and one whose key
# file is a symlink to a file outside them.
home=$work/sshfix/root/root
mkdir -p "$home/.ssh" "$work/sshlink/root/root/.ssh"
printf 'ssh-ed25519 AAAAsecret harness\n' >"$home/.ssh/authorized_keys"
chmod 777 "$home" "$home/.ssh"
chmod 666 "$home/.ssh/authorized_keys"
printf 'ssh-ed25519 AAAAsecret harness\n' >"$work/sshlink/elsewhere"
chmod 666 "$work/sshlink/elsewhere"
ln -s "$work/sshlink/elsewhere" "$work/sshlink/root/root/.ssh/authorized_keys"

now=$(date +%s)
past=$((now - 60))
soon=$((now + 8))
start_box destroy "$past"
start_box fallback "$past" FAKE_CURL_CODE_DELETE=403
start_box refused "$past" FAKE_CURL_BODY_DELETE='{"success": false, "error": "no_such_instance"}'
start_box refusedboth "$past" FAKE_CURL_BODY_DELETE='{"success":false}' \
    FAKE_CURL_BODY_PUT=$'{\n  "success" :\tfalse,\n  "msg": "busy"\n}'
start_box nokey "$past" CONTAINER_API_KEY=
start_box label "$past" CONTAINER_ID= VAST_CONTAINERLABEL=C.456
start_box future "$soon"
start_box nocurl "$((now + 600))" PATH="$work/nocurl"
start_box notmine "$((now + 600))" FAKE_CURL_BODY_GET='{"instances": null}'
start_box other "$((now + 600))" FAKE_CURL_BODY_GET='{"instances": {"id": 310000012}}'
start_box nopy "$((now + 600))" PATH="$work/nopy" FAKE_CURL_BODY_GET='{"instances":{"machine_id":5,"id":31000001}}'
start_box nopyother "$((now + 600))" PATH="$work/nopy" FAKE_CURL_BODY_GET='{"instances":{"id":310000012,"x":31000001}}'
start_box sshfix "$((now + 600))"
start_box sshlink "$((now + 600))"

leaked=""
for ((i = 0; i < 8; i++)); do
    procs=$(ps -A -ww -o args=)
    [[ $procs != *TESTKEY* ]] || leaked=yes
    sleep 0.2
done
(( $(date +%s) < soon )) || fail "the setup phase ran past the deadline; the before-deadline checks mean nothing"
(( $(calls future '-X DELETE') == 0 )) || fail "future destroyed itself before its deadline"
alive future || fail "future: the entrypoint died"
echo "ok: nothing is destroyed before the deadline"

until (( $(date +%s) >= soon + 2 )); do
    sleep 0.2
done

delete_url='-X DELETE -H Content-Type: application/json --data \{\} https://console.vast.ai/api/v0/instances/31000001/'
(( $(calls destroy "$delete_url") >= 2 )) || fail "destroy did not keep destroying itself through the Vast API"
(( $(calls future "$delete_url") >= 1 )) || fail "future did not destroy itself after its deadline"
log_has destroy 'vast self-destruct ready for instance 31000001' || fail "the ready line is missing"
log_has destroy 'vast key reads instance 31000001: http 200' || fail "the key probe is not logged"
log_has destroy 'vast probe ok id=31000001' || fail "a read that names this instance did not count"
log_has destroy 'curl DELETE /api/v0/instances/31000001/: http 200' || fail "the destroy result is not logged"
! grep -q '^runpodctl' "$work/destroy/calls.log" || fail "a Vast box ran runpodctl"
echo "ok: destroys itself through the Vast API at the deadline, every round, and logs it"

stop_url='-X PUT .*--data \{"state": "stopped"\} https://console.vast.ai/api/v0/instances/31000001/'
(( $(calls fallback "$stop_url") >= 1 )) || fail "a refused destroy did not fall back to a stop"
(( $(calls destroy "$stop_url") == 0 )) || fail "a destroy that went through was followed by a stop"
echo "ok: a refused destroy falls back to a stop"

(( $(calls refused "$stop_url") >= 1 )) || fail "a destroy refused in a 200 did not fall back to a stop"
log_has refused 'curl DELETE /api/v0/instances/31000001/: http 200, not done: {"success": false' ||
    fail "a destroy refused in a 200 is not logged as not done"
(( $(calls refusedboth "$delete_url") >= 2 && $(calls refusedboth "$stop_url") >= 2 )) ||
    fail "a box whose destroy and stop are both refused in a 200 stopped trying"
log_has refusedboth 'curl PUT /api/v0/instances/31000001/: http 200, not done' ||
    fail "a stop refused in a 200 is not logged as not done"
echo "ok: a 200 that says success false is a refusal, for the destroy and for the stop"

(( $(calls nokey '') == 0 )) || fail "curl ran without a key"
log_has nokey 'VAST SELF-DESTRUCT MISSING' || fail "a missing key is not said out loud"
alive nokey || fail "nokey: the entrypoint died"
echo "ok: a missing key is logged loudly, and nothing calls Vast"

(( $(calls label 'instances/456/') >= 1 )) || fail "the instance id was not read from VAST_CONTAINERLABEL"
echo "ok: VAST_CONTAINERLABEL names the instance when CONTAINER_ID is missing"

log_has nocurl 'VAST SELF-DESTRUCT MISSING: curl is not installed' || fail "a box without curl is not said out loud"
! log_has nocurl 'box setup done' || fail "a box without curl reported its setup done"
alive nocurl || fail "nocurl: the entrypoint died"
echo "ok: a box without curl says it cannot destroy itself and never reports its setup done"

for name in notmine other nopyother; do
    wait_log "$name" 'box setup done' || fail "$name: setup never reported done"
    log_has "$name" 'vast key reads instance 31000001: http 200' || fail "$name: the key probe is not logged"
    ! log_has "$name" 'vast probe ok' || fail "$name: a read that names no instance, or another, counted"
done
wait_log nopy 'box setup done' || fail "nopy: setup never reported done"
log_has nopy 'vast probe ok id=31000001' || fail "without python3, a read that names this instance did not count"
! grep -q 'vast_TESTKEY\|machine_id' "$work"/*/root/workspace/harness-watchdog.log ||
    fail "the key or a successful read's body reached the watchdog log"
echo "ok: the probe counts only a read that names this very instance, with or without python3"

[[ -z $leaked ]] || fail "the instance key showed up in a process's argv"
if grep -l TESTKEY "$work"/*/calls.log; then
    fail "the instance key reached a fake binary's argv"
fi
[[ $(head -n 1 "$work/destroy/curl_stdin") == 'header = "Authorization: Bearer vast_TESTKEY\"quote\\slash"' ]] ||
    fail "curl did not get the key as config on stdin"
env_file=$work/destroy/root/root/container_env.sh
grep -q 'HARNESS_DEADLINE_EPOCH=' "$env_file" || fail "container_env.sh lost the deadline"
! grep -q -e CONTAINER_API_KEY -e TESTKEY -e HARNESS_ENTRYPOINT_B64 "$env_file" ||
    fail "container_env.sh holds the key or the entrypoint"
log_has destroy 'box setup done' || fail "setup never reported done"
grep -q '^apt-get install' "$work/destroy/calls.log" || fail "apt did not install the packages"
! grep -q -e '^sshd' -e '^ssh-keygen' "$work/destroy/calls.log" || fail "Vast's own sshd was started again"
echo "ok: the key stays out of argv and container_env.sh; setup installs packages and leaves sshd to Vast"

id='[0-9][0-9]*:[0-9][0-9]*'
id_mode="$id [0-7][0-7]*"
wait_log sshfix "ssh key modes: /root $id 777 .*, /root/.ssh $id 777 .*, /root/.ssh/authorized_keys $id 666 " ||
    fail "the modes the host left are not logged"
wait_log sshfix "ssh key modes: /root $id 755 .*, /root/.ssh $id 700 .*, /root/.ssh/authorized_keys $id 600 " ||
    fail "the repaired modes are not logged"
[[ $(mode_of "$home") == 755 && $(mode_of "$home/.ssh") == 700 && $(mode_of "$home/.ssh/authorized_keys") == 600 ]] ||
    fail "sshd's modes were not repaired"
for path in "$home" "$home/.ssh" "$home/.ssh/authorized_keys"; do
    grep -qxF "chown 0 $path" "$work/sshfix/calls.log" || fail "$path was not given to root"
done
log_has destroy "ssh key modes: /root $id_mode .*, /root/.ssh missing, /root/.ssh/authorized_keys missing$" ||
    fail "missing paths are not logged as missing"
echo "ok: the key file's modes are logged, then repaired for sshd; missing paths are only logged"

# Vast writing the key file again later.
chmod 664 "$home/.ssh/authorized_keys"
for _ in $(seq 100); do
    [[ $(mode_of "$home/.ssh/authorized_keys") == 600 ]] && break
    sleep 0.1
done
[[ $(mode_of "$home/.ssh/authorized_keys") == 600 ]] || fail "a later write of the key file was not repaired"
! grep -rq AAAAsecret "$work/sshfix/box.out" "$work/sshfix/root/workspace" ||
    fail "the key file's contents reached the log"
echo "ok: the repair repeats, and the log never holds the key"

[[ $(mode_of "$work/sshlink/elsewhere") == 666 ]] || fail "the repair followed a symlink to another file"
! grep -q "^chown 0 .*/authorized_keys$" "$work/sshlink/calls.log" || fail "chown ran on a symlinked key file"
[[ $(mode_of "$work/sshlink/root/root/.ssh") == 700 ]] || fail "the .ssh above a symlinked key file was not repaired"
grep -qi "/root/.ssh/authorized_keys $id_mode symbolic link" "$work/sshlink/root/workspace/harness-watchdog.log" ||
    fail "a symlinked key file is not logged as one"
echo "ok: a symlinked key file is logged and left alone"

for name in "${!box_pid[@]}"; do
    alive "$name" || fail "$name: the entrypoint exited"
done
echo "ok: every entrypoint is still running"

loader=$work/loader
mkdir -p "$loader"
# The payload records its own command line, which the watchdog check on the box matches.
# shellcheck disable=SC2016  # expanded by the payload's shell, not this one
payload='ps -o args= -p $$ >"$HARNESS_TEST_ROOT/loaded"; sleep 2'
encoded=$(printf '%s' "$payload" | base64 | tr -d '\n')
HARNESS_TEST_ROOT=$loader HARNESS_ENTRYPOINT_B64=$encoded sh "$VAST_ONSTART_FILE" || fail "the loader failed"
for _ in $(seq 50); do
    [[ -s $loader/loaded ]] && break
    sleep 0.1
done
[[ -s $loader/loaded ]] || fail "the loader did not start the entrypoint"
[[ $(cat "$loader/loaded") == "/bin/bash -c ps -o args="* ]] ||
    fail "the entrypoint does not run as /bin/bash -c <text>: $(cat "$loader/loaded")"
if HARNESS_TEST_ROOT=$loader HARNESS_ENTRYPOINT_B64='%%%' sh "$VAST_ONSTART_FILE" 2>"$loader/err"; then
    fail "the loader passed an entrypoint that does not decode"
fi
grep -q 'did not decode; this box has no watchdog' "$loader/err" || fail "a bad entrypoint is not reported"
if HARNESS_TEST_ROOT=$loader HARNESS_ENTRYPOINT_B64='' sh "$VAST_ONSTART_FILE" 2>"$loader/err"; then
    fail "the loader passed an empty entrypoint"
fi
grep -q 'is empty; this box has no watchdog' "$loader/err" || fail "an empty entrypoint is not reported"
grep -q 'is empty; this box has no watchdog' "$loader/workspace/harness-watchdog.log" ||
    fail "an empty entrypoint is not in the watchdog log, where the arm check shows it"
echo "ok: the onstart loader decodes the entrypoint and starts it as /bin/bash -c <text>"
