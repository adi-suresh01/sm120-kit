#!/usr/bin/env bash
# Feeds verdict.sh run directories whose files have unexpected shapes. Each must still give exactly one verdict
# line that a strict JSON reader accepts, with verdict_error saying what could not be parsed.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
verdict="$here/../../pod/verdict.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

fail() {
    echo "FAIL: $*" >&2
    cat "$work/out" >&2 2>/dev/null || true
    exit 1
}

# Runs verdict.sh on run dir $1 and checks the one line it prints: Python literals for the fields in $2 (a dict),
# and a verdict_error that contains $3 (empty: must be null).
check() {
    local name=$1
    bash "$verdict" "$work/$name" "$name" "RESULT=OK boot_seconds=42 requests=1" >"$work/out" || fail "$name: exit"
    python3 -I - "$work/out" "$2" "$3" <<'EOF' || fail "$name"
import ast
import json
import sys

path, want, error = sys.argv[1:]
lines = open(path, encoding="utf-8").read().splitlines()
if len(lines) != 1:
    sys.exit(f"want exactly one line, got {len(lines)}")
# Rust's serde_json refuses lone surrogate escapes, which Python's own reader would let through.
if "\\ud8" in lines[0] or "\\udc" in lines[0] or "\\udd" in lines[0]:
    sys.exit("lone surrogate escape in the line")
got = json.loads(lines[0])
for key, value in ast.literal_eval(want).items():
    if got.get(key) != value:
        sys.exit(f"{key}: got {got.get(key)!r}, want {value!r}")
if error and error not in (got.get("verdict_error") or ""):
    sys.exit(f"verdict_error {got.get('verdict_error')!r} should mention {error!r}")
if not error and got.get("verdict_error") is not None:
    sys.exit(f"verdict_error should be null, got {got['verdict_error']!r}")
EOF
}

mkdir -p "$work/clean"
printf 'Score: 0.950\nSpeculative accept length (per-request, from meta_info): n=50 mean=2.5 min=2 max=3\n' \
    >"$work/clean/gsm8k.txt"
printf '{"max_concurrency": 1, "completed": 8, "output_throughput": 99.94}\n' >"$work/clean/bench.jsonl"
check clean "{'verdict': 'OK', 'boot_s': 42, 'gsm8k': 0.95, 'accept': 2.5, 'tok_s': {'1': 99.9}}" ""
echo "ok: a clean run has a null verdict_error"

mkdir -p "$work/torn"
printf '{"max_concurrency": 1, "completed": 8, "output_throughput": 99.9}\n{"max_concurr' >"$work/torn/bench.jsonl"
check torn "{'verdict': 'OK', 'tok_s': None}" "bench.jsonl: JSONDecodeError"
echo "ok: a half-written bench line"

mkdir -p "$work/keyless"
printf '{"max_concurrency": 1, "completed": 8}\n' >"$work/keyless/bench.jsonl"
check keyless "{'tok_s': None}" "bench.jsonl: KeyError"
echo "ok: a bench record without throughput"

mkdir -p "$work/notalist"
printf '[1, 2]\n' >"$work/notalist/bench.jsonl"
check notalist "{'tok_s': None}" "bench.jsonl: TypeError"
echo "ok: a bench line that is not an object"

mkdir -p "$work/dirbench/bench.jsonl"
check dirbench "{'tok_s': None}" "bench.jsonl:"
echo "ok: a directory where bench.jsonl should be"

mkdir -p "$work/badscore"
printf 'Score: 0.9.1\nSpeculative accept length (per-request, from meta_info): n=50 mean=. min=2 max=3\n' \
    >"$work/badscore/gsm8k.txt"
check badscore "{'gsm8k': None, 'accept': None}" "gsm8k.txt score: not a number: '0.9.1'"
echo "ok: a score that is not a number"

# A log with NUL bytes (grep calls it binary) and a fault message with bytes that are not UTF-8.
mkdir -p "$work/binary"
printf 'boot\0\0\n  File "/x/driver.py", line 3, in launch\nRuntimeError: CUDA error: bad \xff\xfe byte. more\n' \
    >"$work/binary/server.log"
check binary "{'fault': 'CUDA error: bad �� byte', 'where': 'driver.py\", line 3, in launch'}" ""
echo "ok: a binary log with bytes that are not UTF-8"

# No python3 on PATH: the fallback line, built in bash.
mkdir -p "$work/nopython/bin" "$work/plain"
for tool in grep head awk sed; do
    ln -s "$(command -v "$tool")" "$work/nopython/bin/$tool"
done
PATH="$work/nopython/bin" "$BASH" "$verdict" "$work/plain" plain 'RESULT=BOOT_FAIL exit_code=1' >"$work/out" 2>/dev/null \
    || fail "fallback: exit"
python3 -I - "$work/out" <<'EOF' || fail "fallback line"
import json
import sys

lines = open(sys.argv[1]).read().splitlines()
assert len(lines) == 1, lines
got = json.loads(lines[0])
assert got["variant"] == "plain" and got["verdict"] == "BOOT_FAIL", got
assert "python3 failed" in got["verdict_error"], got
EOF
echo "ok: without python3 the line still comes out, from bash"
