#!/usr/bin/env bash
# Measures quality and speed against a running server for the sglang#44291 stage 2 matrix. stage2_run.sh calls
# it through POST_SMOKE, so it reuses the boot.
# Usage: stage2_quality.sh full|accept <port> <out_dir>
#   accept  GSM8K 50: enough decode steps for the scheduler to log accept length
#   full    GSM8K 200, then bench_serving at concurrency 1 / 16 / 64
set -euo pipefail

usage="usage: $0 full|accept <port> <out_dir>"
mode=${1:?$usage} port=${2:?$usage} out=${3:?$usage}
case $mode in
    accept) n=50 ;;
    full) n=200 ;;
    *) echo "$usage" >&2; exit 2 ;;
esac
mkdir -p "$out"

# The cookbook's protocol (5-shot, chat template with thinking on, 16k budget). 64 client threads keep the
# server's running-request cap, not the client, as the limit.
python3 -m sglang.test.run_eval --host 127.0.0.1 --port "$port" --eval-name gsm8k --num-examples "$n" \
    --max-tokens 16384 --num-threads 64 >"$out/gsm8k.txt" 2>&1
grep -E '^Score:' "$out/gsm8k.txt"
# run_eval scores a request that failed every retry, or that the server aborted with a 400, as an empty answer
# and still exits 0, so a server that broke mid-eval would pass as a low score.
if grep -qE 'All retry attempts exhausted|Bad Request Error' "$out/gsm8k.txt"; then
    echo "gsm8k: requests failed against the server" >&2
    exit 1
fi

if [[ $mode == accept ]]; then exit 0; fi

# #36787's protocol (random 1024 in / 512 out); fixed lengths so runs compare exactly.
for pair in 1:8 16:64 64:256; do
    c=${pair%%:*} prompts=${pair##*:}
    python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$port" --dataset-name random \
        --random-input-len 1024 --random-output-len 512 --random-range-ratio 1 \
        --max-concurrency "$c" --num-prompts "$prompts" --seed 1 \
        --output-file "$out/bench.jsonl" >"$out/bench_c$c.txt" 2>&1
    # bench_serving exits 0 and reports throughput over the requests that finished, so a server that dropped
    # requests would only look slow.
    python3 -I - "$out/bench.jsonl" "$prompts" "$c" <<'EOF'
import json
import sys

path, want, c = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(path) as f:
    done = json.loads(f.readlines()[-1])["completed"]
if done != want:
    sys.exit(f"bench c={c}: {done}/{want} requests completed")
EOF
done
