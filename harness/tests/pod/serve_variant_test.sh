#!/usr/bin/env bash
# Runs serve_variant.sh on this machine against fake_server.py and the fake sglang and fakeinfer packages: a
# healthy boot with the quality protocol and a patch to each package, a fatal log line, a hung boot, a failed smoke
# test, and a server that dies mid-evaluation. Also checks restores between runs, the hooks, and progress.jsonl.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
pod="$here/../../pod"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cp -R "$here/fake_site" "$work/site"
mkdir -p "$work/bin" "$work/issue/patches" "$work/snap"
cp "$here/demo.diff" "$work/issue/patches/demo.diff"
# A -p2 patch to the second package, so each patch's own package and strip are exercised.
cat >"$work/issue/patches/infer.diff" <<'EOF'
--- a/pkgs/fakeinfer/core.py
+++ b/pkgs/fakeinfer/core.py
@@ -1 +1 @@
-KERNEL = "upstream"
+KERNEL = "patched"
EOF
printf '#!/usr/bin/env bash\necho 0\n' >"$work/bin/nvidia-smi"
# macOS has no timeout(1); the outer bound is not what these cases test.
if ! command -v timeout >/dev/null; then
    cat >"$work/bin/timeout" <<'EOF'
#!/usr/bin/env bash
while [[ $1 == -* ]]; do
    [[ $1 == -k || $1 == -s ]] && shift
    shift
done
shift
exec "$@"
EOF
fi
# The cleanup's `pkill -9 -f` must never run on this machine, where the pattern could match unrelated processes.
# Parent-scoped kills (-P) only reach this test's own children, so they go through.
cat >"$work/bin/pkill" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$WORK/pkill.log"
if [[ $1 == -P ]]; then
    exec "$REAL_PKILL" "$@"
fi
EOF
chmod +x "$work/bin/"*
REAL_PKILL=$(command -v pkill)
export PATH="$work/bin:$PATH" PYTHONPATH="$work/site" PKG_SNAPSHOT_DIR="$work/snap" WORK="$work" REAL_PKILL
export BOOT_POLL_S=0.2 POST_POLL_S=0.2 STOP_POLL_S=0.1
cat >"$work/issue/hooks.sh" <<'EOF'
pre_launch() { printf '%s|%s|%s\n' "$1" "${FAKE_ENV:-}" "$(cat "$WORK/site/fakeinfer/core.py")" >>"$WORK/pre.txt"; }
post_stop() { printf '%s\n' "$1" >>"$WORK/post.txt"; }
EOF

free_port() {
    python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

# A server flag with quotes, a backslash, and a newline, which the launch phase's detail must carry intact.
note=$'--note=say "hi"\nback\\slash'

# Writes a variant file shaped like `harness exec` renders one. Args: name mode gsm8k env [patch:package:strip...]
write_variant() {
    local name=$1 mode=$2 gsm8k=$3 env=$4 files="" pkgs="" strips="" spec file pkg strip
    shift 4
    for spec in "$@"; do
        IFS=: read -r file pkg strip <<<"$spec"
        files+=" '$file'" pkgs+=" '$pkg'" strips+=" '$strip'"
    done
    cat >"$work/$name.sh" <<EOF
VARIANT='$name'
PACKAGE='sglang'
MODEL='fake/model'
PORT='$(free_port)'
FAIL_PATTERN='illegal memory access'
BOOT_TIMEOUT='1'
CONCURRENCY='2'
GSM8K='$gsm8k'
SERVER_CMD=('python3' '$here/fake_server.py' '$mode' '$note')
VARIANT_ENV=(${env:+"'$env'"})
BENCH=('1:8' '16:64')
SNAPSHOT_PKGS=('sglang' 'fakeinfer')
PATCH_FILES=($files)
PATCH_PKGS=($pkgs)
PATCH_STRIPS=($strips)
EOF
}

serve() {
    bash "$pod/serve_variant.sh" "$work/issue" "$work/$1.sh" "$work/results" >"$work/$1.out" 2>&1 \
        || fail "serve_variant.sh $1 exited non-zero"
}

fail() {
    echo "FAIL: $*" >&2
    cat "$work"/*.out >&2 2>/dev/null || true
    exit 1
}

# Checks one verdict.json field against a Python literal.
field() {
    python3 -I - "$work/results/$1/verdict.json" "$2" "$3" <<'EOF' || fail "$1: $2"
import ast
import json
import sys

path, key, want = sys.argv[1:]
got = json.load(open(path))[key]
if got != ast.literal_eval(want):
    sys.exit(f"{key}: got {got!r}, want {want}")
EOF
}

# Checks a run's progress.jsonl: every line is JSON, the phases come in the given order, the launch detail is the
# exact server command, and the last line is verdict.json itself. Args: variant "phase phase ..."
phases() {
    local launch
    # shellcheck source=/dev/null
    launch=$(source "$work/$1.sh" && printf '%s' "${SERVER_CMD[*]} --host 127.0.0.1 --port $PORT")
    python3 -I - "$work/results/$1" "$2" "$launch" <<'EOF' || fail "$1: progress.jsonl"
import json
import sys

run_dir, want, launch = sys.argv[1:]
lines = open(f"{run_dir}/progress.jsonl", encoding="utf-8").read().splitlines()
rows = [json.loads(line) for line in lines]
got = [row.get("phase") for row in rows[:-1]]
if got != want.split():
    sys.exit(f"phases: got {got}, want {want.split()}")
if not all(type(row["t"]) is int for row in rows[:-1]):
    sys.exit("every phase needs an integer t")
if lines[-1] != open(f"{run_dir}/verdict.json", encoding="utf-8").read().rstrip("\n"):
    sys.exit("the last line is not verdict.json")
details = [row.get("detail") for row in rows if row.get("phase") == "launch"]
if details != [launch]:
    sys.exit(f"launch detail: got {details!r}, want {launch!r}")
EOF
}

write_variant healthy ok 50 'FAKE_ENV=from the variant' demo:sglang:1 infer:fakeinfer:2
serve healthy
field healthy verdict "'OK'"
field healthy fault None
field healthy gsm8k 0.97
field healthy accept 2.9374
field healthy tok_s "{'1': 100.0, '16': 1600.0}"
field healthy verdict_error None
grep -q 'patched' "$work/site/sglang/core.py" || fail "sglang patch not applied"
grep -q 'patched' "$work/site/fakeinfer/core.py" || fail "fakeinfer patch not applied at -p2"
grep -q "^$work/results/healthy|from the variant|" "$work/pre.txt" || fail "pre_launch did not see the run dir and env"
[[ $(wc -l <"$work/post.txt") -eq 1 ]] || fail "post_stop should run once"
phases healthy "restore patch patch idle_wait launch boot_ok smoke_ok quality_start quality_done stop"
echo "ok: healthy boot with quality, a patch to each of two packages, hooks, and progress"

write_variant faulty fatal "" ""
serve faulty
field faulty verdict "'BOOT_FAIL'"
field faulty fault "'CUDA error: an illegal memory access was encountered'"
field faulty where "'driver.py\", line 331, in __call__'"
grep -q 'upstream' "$work/site/sglang/core.py" || fail "the next run did not start from the pristine sglang"
[[ $(tail -n 1 "$work/pre.txt") == "$work/results/faulty||KERNEL = \"upstream\"" ]] \
    || fail "fakeinfer was still patched when the next variant launched"
phases faulty "restore idle_wait launch boot_fail stop"
echo "ok: a fatal log line ends the boot as BOOT_FAIL, and both packages were restored before it"

write_variant stuck hang "" ""
serve stuck
field stuck verdict "'TIMEOUT'"
field stuck boot_s 1
phases stuck "restore idle_wait launch boot_fail stop"
echo "ok: a hung boot ends as TIMEOUT"

write_variant smokefail smoke_fail 50 ""
serve smokefail
field smokefail verdict "'SMOKE_FAIL'"
[[ ! -e $work/results/smokefail/server.post.log ]] || fail "quality ran after a failed smoke test"
phases smokefail "restore idle_wait launch boot_ok smoke_fail stop"
echo "ok: a failed chat request ends as SMOKE_FAIL without running quality"

write_variant postfail post_die 50 'FAKE_EVAL=hang'
serve postfail
field postfail verdict "'POST_FAIL'"
grep -q 'post_exit=server_exit' "$work/results/postfail/verdict.json" || fail "postfail: not stopped for server_exit"
phases postfail "restore idle_wait launch boot_ok smoke_ok quality_start quality_done stop"
grep -q '"detail": "post_exit=server_exit"' "$work/results/postfail/progress.jsonl" || fail "quality_done detail"
# The eval client is a grandchild of serve_and_smoke; it must not keep retrying the dead server.
# shellcheck source=/dev/null
post_port=$(source "$work/postfail.sh" && echo "$PORT")
if pgrep -f "sglang.test.run_eval .*--port $post_port" >/dev/null; then
    fail "the eval client outlived its server"
fi
echo "ok: a server that dies mid-evaluation ends as POST_FAIL and takes the eval client down"

[[ $(wc -l <"$work/results/verdicts.jsonl") -eq 5 ]] || fail "verdicts.jsonl should hold one line per run"
cmp -s <(tail -n 1 "$work/results/verdicts.jsonl") "$work/results/postfail/verdict.json" || fail "verdict.json differs"
if pgrep -f "$here/fake_server.py" >/dev/null; then
    fail "a fake server outlived its run"
fi
[[ $(grep -cxF -- '-9 -f [s]glang[.]launch_server|[s]glang::' "$work/pkill.log") -eq 5 ]] \
    || fail "each run should clean up the server pattern once"
echo "ok: one verdict line per run, one cleanup per run, and no server left behind"
