#!/usr/bin/env bash
# Runs env_report.sh and wait_gpu_idle.sh against a fake nvidia-smi that can hang, the way it does on a wedged GPU.
# A fake timeout records each bound and holds the command to 1 s, so a hang costs a second here instead of 60.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
pod="$here/../../pod"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/bin"
cat >"$work/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
if [[ $FAKE_SMI == hang || ($FAKE_SMI == topo-hang && $1 == topo) ]]; then
    exec sleep 30
fi
echo 0
EOF
cat >"$work/bin/timeout" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$WORK/timeout.log"
while [[ $1 == -* ]]; do
    [[ $1 == -k || $1 == -s ]] && shift
    shift
done
shift
exec perl -e 'alarm 1; exec @ARGV or die "exec $ARGV[0]: $!\n"' -- "$@"
EOF
# GNU tools env_report.sh reads that macOS lacks.
printf '#!/usr/bin/env bash\necho 8\n' >"$work/bin/nproc"
printf '#!/usr/bin/env bash\necho "Mem: 64"\n' >"$work/bin/free"
chmod +x "$work/bin/"*
export PATH="$work/bin:$PATH" WORK="$work"

fail() {
    echo "FAIL: $*" >&2
    cat "$work/out" "$work/timeout.log" >&2 2>/dev/null || true
    exit 1
}

# Runs a pod script with a 5 s cap of its own, so a script that does not bound nvidia-smi fails here instead of
# hanging the suite. Args: FAKE_SMI mode, script, args...
run() {
    local mode=$1
    shift
    rm -f "$work/timeout.log"
    FAKE_SMI=$mode perl -e 'alarm 5; exec @ARGV or die' -- bash "$@" >"$work/out" 2>&1
}

run idle "$pod/wait_gpu_idle.sh" 10 || fail "idle GPUs"
echo "ok: idle GPUs pass"

if run hang "$pod/wait_gpu_idle.sh" 10; then
    fail "a hung nvidia-smi must fail the wait"
fi
grep -q 'did not answer within 60 s' "$work/out" || fail "a hung nvidia-smi is not named"
grep -qx '60 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits' "$work/timeout.log" \
    || fail "the query was not bounded at 60 s"
echo "ok: wait_gpu_idle.sh gives up on a hung nvidia-smi"

run topo-hang "$pod/env_report.sh" || fail "a hung topology query must not end the report"
grep -q '^## Host' "$work/out" || fail "the report stopped at the topology"
[[ $(grep -c '^60 nvidia-smi' "$work/timeout.log") -eq 2 ]] || fail "every nvidia-smi call should be bounded at 60 s"
echo "ok: env_report.sh bounds every nvidia-smi call and finishes past a hung topology query"

if run hang "$pod/env_report.sh"; then
    fail "a hung GPU query must fail the report"
fi
grep -q 'did not answer within 60 s' "$work/out" || fail "a hung nvidia-smi is not named"
echo "ok: env_report.sh fails on a hung GPU query"
