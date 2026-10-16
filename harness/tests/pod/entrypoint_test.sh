#!/usr/bin/env bash
# Runs the whole entrypoint as `bash -c <file>`, the way a pod starts it, against fake apt-get, ssh-keygen, sshd,
# timeout, curl, and runpodctl that record their calls instead of acting. Each case is its own pod with its own
# root; all of them run at once so the suite takes seconds.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
entrypoint="$here/../../pod/entrypoint.sh"
work=$(mktemp -d)
declare -A pod_pid=()

# Each pod leads its own process group, which also holds the subshells it orphans (a log write stuck on a FIFO
# outlives the setup shell that started it), so killing the group leaves nothing behind.
cleanup() {
    local pid
    for pid in "${pod_pid[@]}"; do
        kill -9 -- "-$pid" 2>/dev/null || true
    done
    rm -rf "$work"
}
trap cleanup EXIT

fail() {
    echo "FAIL: $*" >&2
    local dir
    for dir in "$work"/*/; do
        [[ -f $dir/pod.out ]] || continue
        echo "--- $(basename "$dir")" >&2
        cat "$dir/calls.log" "$dir/pod.out" >&2 2>/dev/null || true
    done
    exit 1
}

mkdir -p "$work/bin"
cat >"$work/bin/curl" <<'EOF'
#!/usr/bin/env bash
printf 'curl %s\n' "$*" >>"$FAKE_DIR/calls.log"
format="" prev=""
for arg in "$@"; do
    [[ $prev == -w ]] && format=$arg
    [[ $prev == -K && $arg == - ]] && cat >>"$FAKE_DIR/curl_stdin"
    prev=$arg
done
printf '%s' "${format//%\{http_code\}/$FAKE_CURL_CODE}"
EOF
cat >"$work/bin/runpodctl" <<'EOF'
#!/usr/bin/env bash
printf 'runpodctl %s\n' "$*" >>"$FAKE_DIR/calls.log"
case $FAKE_RUNPODCTL in
    ok) echo "pod $3 done" ;;
    stop-only) [[ $2 == stop ]] || { echo "Error: forbidden" >&2; exit 1; } ;;
    hang) exec sleep 30 ;;
    *) echo "Error: forbidden" >&2; exit 1 ;;
esac
EOF
cat >"$work/bin/apt-get" <<'EOF'
#!/usr/bin/env bash
printf 'apt-get %s\n' "$*" >>"$FAKE_DIR/calls.log"
if [[ $FAKE_APT == hang ]]; then exec sleep 30; fi
EOF
cat >"$work/bin/ssh-keygen" <<'EOF'
#!/usr/bin/env bash
printf 'ssh-keygen %s\n' "$*" >>"$FAKE_DIR/calls.log"
EOF
cat >"$work/bin/sshd" <<'EOF'
#!/usr/bin/env bash
printf 'sshd %s\n' "$*" >>"$FAKE_DIR/calls.log"
EOF
# GNU timeout stand-in (macOS has none). It holds every command to 1 s so a hung fake costs one second.
cat >"$work/bin/timeout" <<'EOF'
#!/usr/bin/env bash
while [[ $1 == -* ]]; do
    [[ $1 == -k || $1 == -s ]] && shift
    shift
done
shift
exec perl -e 'alarm 1; exec @ARGV or die "exec $ARGV[0]: $!\n"' -- "$@"
EOF
chmod +x "$work/bin/"*

entrypoint_text=$(cat "$entrypoint")
# A background job inherits SIGINT ignored, but a pod's PID 1 starts with it at the default. setsid makes the pod
# a process group leader for cleanup.
default_signals='import os, signal, sys
signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGQUIT, signal.SIG_DFL)
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])'
key='rpa_TESTKEY"quote\slash'

# Starts one pod. Args: name deadline [VAR=value...] (overrides for the fakes and the pod env)
start_pod() {
    local name=$1 deadline=$2 dir=$work/$1
    shift 2
    [[ -e $dir/root/workspace ]] || mkdir -p "$dir/root/workspace"
    mkdir -p "$dir/root/root" "$dir/root/usr/sbin"
    ln -s "$work/bin/sshd" "$dir/root/usr/sbin/sshd"
    (
        export PATH="$work/bin:$PATH" FAKE_DIR=$dir HARNESS_TEST_ROOT=$dir/root HARNESS_WATCHDOG_NAP_S=1 \
            HARNESS_DEADLINE_EPOCH=$deadline RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=$key \
            PUBLIC_KEY='ssh-ed25519 AAAAtest harness' HARNESS_APT='openssh-server curl procps' HF_HOME=/workspace/hf \
            FAKE_CURL_CODE=204 FAKE_RUNPODCTL=ok FAKE_APT=ok
        exec env "$@" python3 -I -c "$default_signals" bash -c "$entrypoint_text" >"$dir/pod.out" 2>&1
    ) &
    pod_pid[$name]=$!
    # The pods die by SIGKILL at the end; disowned, they do not each print a job notice.
    disown "$!"
}

# The calls a pod made to terminate itself, one per line.
kills() {
    grep -E '^(curl|runpodctl) ' "$work/$1/calls.log" 2>/dev/null || true
}
count() {
    kills "$1" | grep -c -- "$2" || true
}
alive() {
    kill -0 "${pod_pid[$1]}" 2>/dev/null
}

now=$(date +%s)
past=$((now - 60))
soon=$((now + 4))
start_pod delete2xx "$past" FAKE_CURL_CODE=200
start_pod notfound "$past" FAKE_CURL_CODE=404
start_pod fallback "$past" FAKE_CURL_CODE=401 FAKE_RUNPODCTL=stop-only
start_pod nokey "$past" RUNPOD_API_KEY=
start_pod hungctl "$past" FAKE_CURL_CODE=500 FAKE_RUNPODCTL=hang
# A file where /workspace should be makes the watchdog log unopenable.
mkdir -p "$work/badlog/root"
touch "$work/badlog/root/workspace"
start_pod badlog "$past"
# Opening a FIFO that no one reads blocks, like a write to a hung network volume.
mkdir -p "$work/hunglog/root/workspace"
mkfifo "$work/hunglog/root/workspace/harness-watchdog.log"
start_pod hunglog "$past"
start_pod hungapt "$past" FAKE_APT=hang
start_pod garbled "not-a-number"
start_pod future "$soon"
start_pod sleepkill "$soon"
start_pod signals "$soon" FAKE_APT=hang
start_pod subshell "$soon" FAKE_APT=hang
start_pod setup "$((now + 600))"

# Until just before the deadline: kill the naps of one pod, signal another together with its naps (what a
# `pkill -f` matching both does), and SIGKILL a third pod's background setup shell.
leaked=""
nap_signals=(KILL TERM INT HUP)
pod_signals=(TERM HUP INT QUIT USR1)
for ((i = 0; i < 8; i++)); do
    for pid in $(pgrep -P "${pod_pid[sleepkill]}" -x sleep || true); do
        kill -s "${nap_signals[i % 4]}" "$pid" 2>/dev/null || true
    done
    for pid in "${pod_pid[signals]}" $(pgrep -P "${pod_pid[signals]}" -x sleep || true); do
        kill -s "${pod_signals[i % 5]}" "$pid" 2>/dev/null || true
    done
    for pid in $(pgrep -P "${pod_pid[subshell]}" -x bash || true); do
        kill -9 "$pid" 2>/dev/null || true
    done
    # A builtin match, so the check itself puts no key on a command line.
    procs=$(ps -A -ww -o args=)
    [[ $procs != *TESTKEY* ]] || leaked=yes
    sleep 0.2
done
(( $(date +%s) < soon )) || fail "the setup phase ran past the deadline; the before-deadline checks mean nothing"

for name in future sleepkill signals subshell; do
    [[ -z $(kills "$name") ]] || fail "$name acted before the deadline"
    alive "$name" || fail "$name: the entrypoint died"
done
echo "ok: nothing happens before the deadline, through killed naps, signals, and a killed setup shell"
(( $(count hungapt 'curl ') >= 1 )) || fail "a hanging apt-get held up the delete"
echo "ok: a hanging apt-get does not hold up the watchdog"
(( $(count garbled 'curl ') >= 1 )) || fail "a garbled deadline should act at once"
echo "ok: a garbled deadline acts at once"

until (( $(date +%s) >= soon + 2 )); do
    sleep 0.2
done

for name in future sleepkill signals subshell; do
    (( $(count "$name" '-X DELETE') >= 1 )) || fail "$name did not delete after the deadline"
done
echo "ok: deletes after the deadline, also after killed naps, signals, and a killed setup shell"

(( $(count delete2xx 'curl ') >= 2 )) || fail "the watchdog stopped after a 2xx delete"
(( $(count delete2xx 'runpodctl') == 0 )) || fail "runpodctl ran in a round whose delete got a 2xx"
grep -q 'http 200' "$work/delete2xx/root/workspace/harness-watchdog.log" || fail "delete result not logged"
echo "ok: a 2xx delete is logged, and the watchdog keeps going after it"

(( $(count notfound 'runpodctl pod remove pod123') >= 2 )) || fail "a 404 must not count, and remove's exit 0 must not end the loop"
(( $(count notfound 'runpodctl pod stop') == 0 )) || fail "stop ran after a remove that succeeded"
echo "ok: a 404 is not success, and an exit-0 remove does not end the watchdog"

[[ $(kills fallback | head -n 4 | cut -d' ' -f1-3 | tr '\n' ',') == "curl -sS -K,runpodctl pod remove,runpodctl pod stop,curl -sS -K," ]] \
    || fail "fallback order"
grep -q 'runpodctl pod remove pod123: exit 1: Error: forbidden' "$work/fallback/root/workspace/harness-watchdog.log" \
    || fail "failed remove not logged with its output"
echo "ok: falls back to runpodctl remove, then stop, every round"

(( $(count nokey 'curl ') == 0 )) || fail "curl ran without a key"
(( $(count nokey 'runpodctl pod remove') >= 1 )) || fail "a missing key must still try runpodctl"
grep -q 'RUNPOD_API_KEY or RUNPOD_POD_ID is not set' "$work/nokey/root/workspace/harness-watchdog.log" \
    || fail "missing key not logged"
echo "ok: a missing key is logged and runpodctl still runs"

(( $(count hungctl 'curl ') >= 2 )) || fail "a hung runpodctl stalled the watchdog"
echo "ok: a hung runpodctl is cut off and the next round runs"

(( $(count badlog 'curl ') >= 2 )) || fail "an unopenable log stopped the delete"
grep -q 'curl DELETE /v2/pods/pod123: http 204' "$work/badlog/pod.out" || fail "the log should fall back to stderr"
echo "ok: an unopenable log path does not stop the watchdog"

(( $(count hunglog 'curl ') >= 2 )) || fail "a log write that blocks stopped the watchdog"
echo "ok: a log write that blocks does not stop the watchdog"

[[ -z $leaked ]] || fail "the API key showed up in a process's argv"
if grep -l TESTKEY "$work"/*/calls.log; then
    fail "the API key reached a fake binary's argv"
fi
[[ $(head -n 1 "$work/delete2xx/curl_stdin") == 'header = "Authorization: Bearer rpa_TESTKEY\"quote\\slash"' ]] \
    || fail "curl did not get the key as config on stdin"
echo "ok: the API key never appears in argv; curl gets it on stdin"

env_file=$work/setup/root/root/container_env.sh
grep -q 'HF_HOME=' "$env_file" || fail "container_env.sh lost the pod env"
grep -q 'HARNESS_DEADLINE_EPOCH=' "$env_file" || fail "container_env.sh lost the deadline"
! grep -q -e RUNPOD_API_KEY -e TESTKEY "$env_file" || fail "container_env.sh holds the API key"
[[ -n $(find "$env_file" -perm 600) ]] || fail "container_env.sh is readable by others"
[[ $(grep -vE '^(curl|runpodctl) ' "$work/setup/calls.log" | cut -d' ' -f1-2 | tr '\n' ',') \
    == "apt-get update,apt-get install,ssh-keygen -A,sshd ," ]] || fail "setup order"
grep -qx 'ssh-ed25519 AAAAtest harness' "$work/setup/root/root/.ssh/authorized_keys" || fail "authorized_keys"
echo "ok: setup writes container_env.sh without the key, then apt, host keys, sshd"

for name in "${!pod_pid[@]}"; do
    alive "$name" || fail "$name: the entrypoint exited"
done
echo "ok: every entrypoint is still running"
