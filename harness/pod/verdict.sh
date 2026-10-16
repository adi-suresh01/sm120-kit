#!/usr/bin/env bash
# Prints the verdict line for a finished variant run: serve_and_smoke's RESULT line plus what the run's logs say.
# Usage: verdict.sh <run_dir> <variant> <result_line>
# Output: one JSON object with variant, verdict, boot_s, fault, where, gsm8k, accept, tok_s, result.
set -euo pipefail

usage="usage: $0 <run_dir> <variant> <result_line>"
run_dir=${1:?$usage}
variant=${2:?$usage}
result=${3:?$usage}
log="$run_dir/server.log"

# CUDA faults first: a benign early ValueError or RuntimeError line must not mask the real fault. head keeps a
# line with several matches from adding newlines.
fault=$(grep -oE 'illegal memory access|cudaErrorIllegalAddress|CUDA error: [^.]*|Xid [0-9]+' "$log" 2>/dev/null \
    | head -n 1 || true)
if [[ -z $fault ]]; then
    fault=$(grep -oE 'ValueError: [^.]*|RuntimeError: [^.]*|AssertionError[^.]*|OutOfMemoryError[^.]*' "$log" \
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
accept=$(grep -m1 -oE 'from meta_info\): n=[0-9]+ mean=[0-9.]+' "$run_dir/gsm8k.txt" 2>/dev/null | sed 's/.*mean=//' \
    || true)
score=$(grep -m1 -oE '^Score: [0-9.]+' "$run_dir/gsm8k.txt" 2>/dev/null | awk '{print $2}' || true)

python3 -I - "$variant" "$result" "${fault:0:200}" "$where" "$score" "$accept" "$run_dir/bench.jsonl" <<'EOF'
import json
import os
import sys

variant, result, fault, where, score, accept, bench = sys.argv[1:]
words = result.split()
verdict = words[0].removeprefix("RESULT=") if words and words[0].startswith("RESULT=") else "TIMEOUT"
fields = dict(word.split("=", 1) for word in words[1:] if "=" in word)
boot = fields.get("boot_seconds", "")
tok_s = None
if os.path.exists(bench):
    with open(bench) as f:
        runs = [json.loads(line) for line in f if line.strip()]
    tok_s = {str(run["max_concurrency"]): round(run["output_throughput"], 1) for run in runs}
print(json.dumps({
    "variant": variant,
    "verdict": verdict,
    "boot_s": int(boot) if boot.isdigit() else None,
    "fault": fault or None,
    "where": where or None,
    "gsm8k": float(score) if score else None,
    "accept": float(accept) if accept else None,
    "tok_s": tok_s,
    "result": result,
}))
EOF
