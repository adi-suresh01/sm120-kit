#!/usr/bin/env bash
# Prints the verdict line for a finished variant run: serve_and_smoke's RESULT line plus what the run's logs say.
# Usage: verdict.sh <run_dir> <variant> <result_line>
# Output: exactly one JSON object with variant, verdict, boot_s, fault, where, gsm8k, accept, tok_s, result, and
# verdict_error (null, or what could not be parsed). A run is only counted once a verdict line exists, so a log or
# result file of an unexpected shape must still produce one.
set -euo pipefail

usage="usage: $0 <run_dir> <variant> <result_line>"
run_dir=${1:?$usage}
variant=${2:?$usage}
result=${3:?$usage}
log="$run_dir/server.log"
# Logs are bytes, not always UTF-8; under a UTF-8 locale awk aborts on the first bad byte and the column is lost.
export LC_ALL=C

# -a: a crashed process can leave NUL bytes in its log, and grep would then only say "binary file matches".
# CUDA faults first: a benign early ValueError or RuntimeError line must not mask the real fault. head keeps a
# line with several matches from adding newlines.
fault=$(grep -aoE 'illegal memory access|cudaErrorIllegalAddress|CUDA error: [^.]*|Xid [0-9]+' "$log" 2>/dev/null \
    | head -n 1 || true)
if [[ -z $fault ]]; then
    fault=$(grep -aoE 'ValueError: [^.]*|RuntimeError: [^.]*|AssertionError[^.]*|OutOfMemoryError[^.]*' "$log" \
        2>/dev/null | head -n 1 || true)
fi
# Under CUDA_LAUNCH_BLOCKING=1 the deepest Python frame before the first CUDA fault names the kernel's launcher;
# the frame before the first other error is the fallback, matching the order of the fault column.
where=$(awk '
    /File "/ { f = $0 }
    /illegal memory access|cudaErrorIllegalAddress|CUDA error: / && f != "" { c = f; exit }
    /Error: / && g == "" && f != "" { g = f }
    END { w = (c != "" ? c : g); sub(/.*\//, "", w); print w }' "$log" 2>/dev/null || true)
# The per-request mean over GSM8K only; the server log's last value would come from whatever bench ran last.
accept=$(grep -a -m1 -oE 'from meta_info\): n=[0-9]+ mean=[0-9.]+' "$run_dir/gsm8k.txt" 2>/dev/null \
    | sed 's/.*mean=//' || true)
score=$(grep -a -m1 -oE '^Score: [0-9.]+' "$run_dir/gsm8k.txt" 2>/dev/null | awk '{print $2}' || true)

if line=$(python3 -I - "$variant" "$result" "${fault:0:200}" "$where" "$score" "$accept" "$run_dir/bench.jsonl" <<'EOF'
import json
import os
import re
import sys


def text(value):
    # A byte that is not UTF-8 arrives as a lone surrogate, which json.dumps would write as an escape that strict
    # readers such as serde_json reject.
    return value.encode("utf-8", "surrogateescape").decode("utf-8", "replace")


variant, result, fault, where, score, accept, bench = (text(arg) for arg in sys.argv[1:])
errors = []


def number(name, value):
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        errors.append(f"gsm8k.txt {name}: not a number: {value!r}")
        return None


def speeds(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            runs = [json.loads(line) for line in f if line.strip()]
        return {str(run["max_concurrency"]): round(float(run["output_throughput"]), 1) for run in runs}
    # Whatever bench_serving left behind, a half-written line included, costs the speeds and nothing else.
    except Exception as e:
        errors.append(f"bench.jsonl: {type(e).__name__}: {e}")
        return None


words = result.split()
verdict = words[0].removeprefix("RESULT=") if words and words[0].startswith("RESULT=") else "TIMEOUT"
fields = dict(word.split("=", 1) for word in words[1:] if "=" in word)
boot = fields.get("boot_seconds", "")
print(json.dumps({
    "variant": variant,
    "verdict": verdict,
    "boot_s": int(boot) if re.fullmatch(r"[0-9]+", boot) else None,
    "fault": fault or None,
    "where": where or None,
    "gsm8k": number("score", score),
    "accept": number("accept length", accept),
    "tok_s": speeds(bench),
    "result": result,
    "verdict_error": "; ".join(errors) or None,
}))
EOF
) && [[ -n $line ]]; then
    printf '%s\n' "$line"
    exit 0
fi

# python3 itself failed. The run still needs its one line; names and RESULT words are plain ASCII.
verdict=TIMEOUT
if [[ $result =~ ^RESULT=([A-Z_]+) ]]; then
    verdict=${BASH_REMATCH[1]}
fi
printf '{"variant": "%s", "verdict": "%s", "verdict_error": "verdict.sh: python3 failed"}\n' \
    "${variant//[^A-Za-z0-9_-]/_}" "$verdict"
