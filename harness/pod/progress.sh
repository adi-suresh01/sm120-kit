#!/usr/bin/env bash
# Appends one phase line to a variant's progress.jsonl, which `harness exec` tails to show where a run is.
# Usage: progress.sh <progress_file> <phase> [detail]
# Line: {"phase": "<phase>", "t": <epoch seconds>, "detail": "<detail>"}; detail may hold any text, newlines too.
set -euo pipefail

usage="usage: $0 <progress_file> <phase> [detail]"
python3 -I - "${1:?$usage}" "${2:?$usage}" "${@:3}" <<'EOF'
import json
import sys
import time

path, phase, *detail = sys.argv[1:]
line = {"phase": phase, "t": int(time.time())}
if detail:
    # A byte that is not UTF-8 arrives as a lone surrogate, which strict readers such as serde_json reject.
    line["detail"] = detail[0].encode("utf-8", "surrogateescape").decode("utf-8", "replace")
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(line) + "\n")
EOF
