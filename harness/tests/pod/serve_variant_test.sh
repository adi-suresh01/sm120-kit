#!/usr/bin/env bash
# Runs serve_variant.sh on this machine against fake_server.py and a fake sglang package: a healthy boot with the
# quality protocol, a fatal log line, and a hung boot. Also checks patch restore between runs and the hooks.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
pod="$here/../../pod"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cp -R "$here/fake_site" "$work/site"
mkdir -p "$work/bin" "$work/issue/patches" "$work/snap"
cp "$here/demo.diff" "$work/issue/patches/demo.diff"
printf '#!/usr/bin/env bash\necho 0\n' >"$work/bin/nvidia-smi"
# macOS has no timeout(1); the outer bound is not what these cases test.
if ! command -v timeout >/dev/null; then
    printf '#!/usr/bin/env bash\nshift 3\nexec "$@"\n' >"$work/bin/timeout"
fi
chmod +x "$work/bin/"*
export PATH="$work/bin:$PATH" PYTHONPATH="$work/site" PKG_SNAPSHOT_DIR="$work/snap" WORK="$work"
cat >"$work/issue/hooks.sh" <<'EOF'
pre_launch() { printf '%s %s\n' "$1" "$FAKE_ENV" >>"$WORK/pre.txt"; }
post_stop() { printf '%s\n' "$1" >>"$WORK/post.txt"; }
EOF

free_port() {
    python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

# Writes a variant file shaped like `harness exec` renders one. Args: name mode gsm8k [patch]
write_variant() {
    local name=$1 mode=$2 gsm8k=$3 patch=${4:-}
    cat >"$work/$name.sh" <<EOF
VARIANT='$name'
PACKAGE='sglang'
MODEL='fake/model'
PORT='$(free_port)'
FAIL_PATTERN='illegal memory access'
BOOT_TIMEOUT='1'
CONCURRENCY='2'
PATCH_STRIP='1'
GSM8K='$gsm8k'
PATCHES=(${patch:+"'$patch'"})
SERVER_CMD=('python3' '$here/fake_server.py' '$mode')
VARIANT_ENV=('FAKE_ENV=from the variant')
BENCH=('1:8' '16:64')
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

write_variant healthy ok 50 demo
serve healthy
field healthy verdict "'OK'"
field healthy fault None
field healthy gsm8k 0.97
field healthy accept 2.9374
field healthy tok_s "{'1': 100.0, '16': 1600.0}"
grep -q 'patched' "$work/site/sglang/core.py" || fail "patch not applied"
grep -q "$work/results/healthy from the variant" "$work/pre.txt" || fail "pre_launch did not see the run dir and env"
[[ $(wc -l <"$work/post.txt") -eq 1 ]] || fail "post_stop should run once"
echo "ok: healthy boot with quality, patch, and hooks"

write_variant faulty fatal ""
serve faulty
field faulty verdict "'BOOT_FAIL'"
field faulty fault "'CUDA error: an illegal memory access was encountered'"
field faulty where "'driver.py\", line 331, in __call__'"
grep -q 'upstream' "$work/site/sglang/core.py" || fail "the next run did not start from the pristine package"
echo "ok: a fatal log line ends the boot as BOOT_FAIL"

write_variant stuck hang ""
serve stuck
field stuck verdict "'TIMEOUT'"
field stuck boot_s 1
echo "ok: a hung boot ends as TIMEOUT"

[[ $(wc -l <"$work/results/verdicts.jsonl") -eq 3 ]] || fail "verdicts.jsonl should hold one line per run"
cmp -s <(tail -n 1 "$work/results/verdicts.jsonl") "$work/results/stuck/verdict.json" || fail "verdict.json differs"
if pgrep -f "$here/fake_server.py" >/dev/null; then
    fail "a fake server outlived its run"
fi
echo "ok: one verdict line per run and no server left behind"
