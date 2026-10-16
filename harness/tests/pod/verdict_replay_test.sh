#!/usr/bin/env bash
# Replays sglang#44291 stage 2 runs through verdict.sh and compares with the summary.tsv that stage2_run.sh wrote
# from the full logs on the pod. The logs under replay/ are excerpts that keep every line the parsers decide on.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
verdict="$here/../../pod/verdict.sh"
replay="$here/replay"
stage2_fail_pattern='illegal memory access|cudaErrorIllegalAddress|CUDA error: |Received sigquit|SIGQUIT received|Scheduler hit an exception'

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

runs=0
while IFS=$'\t' read -r _ label _ result accept score fault where; do
    got=$(bash "$verdict" "$replay/$label" "$label" "$result")
    python3 -I - "$got" "$result" "$accept" "$score" "$fault" "$where" <<'EOF' || fail "$label"
import json
import sys

got = json.loads(sys.argv[1])
result, accept, score, fault, where = (None if v == "-" else v for v in sys.argv[2:])
want = {
    "result": result,
    "accept": float(accept) if accept else None,
    "gsm8k": float(score) if score else None,
    "fault": fault,
    "where": where,
}
bad = {key: (got[key], value) for key, value in want.items() if got[key] != value}
if bad:
    sys.exit(f"(verdict.sh, stage 2) differ: {bad}")
EOF
    runs=$((runs + 1))
    echo "ok: $label matches stage 2"
done <"$replay/summary.tsv"
(( runs == 5 )) || fail "expected 5 replayed runs, got $runs"

grep -qE "$stage2_fail_pattern" "$replay/a2/server.log" || fail "the fail pattern misses a2's fault"
echo "ok: the stage 2 fail pattern matches a2, whose boot it ended"

speeds=$(bash "$verdict" "$replay/a5" a5 "RESULT=OK" | python3 -I -c 'import json, sys; print(sorted(json.load(sys.stdin)["tok_s"]))')
[[ $speeds == "['1', '16', '64']" ]] || fail "a5 tok_s keys: $speeds"
echo "ok: a5 reports output tok/s per concurrency"
