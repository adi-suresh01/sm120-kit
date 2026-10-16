#!/usr/bin/env bash
# Measures quality and speed against a running sglang server. serve_variant.sh passes it to serve_and_smoke.sh as
# POST_SMOKE, so it reuses the boot.
# Usage: quality.sh <port> <out_dir> <gsm8k_examples> [<concurrency>:<prompts> ...]
#   GSM8K first (50 examples are enough decode steps for the scheduler to log accept length), then one
#   bench_serving run per concurrency:prompts pair.
set -euo pipefail

usage="usage: $0 <port> <out_dir> <gsm8k_examples> [<concurrency>:<prompts> ...]"
port=${1:?$usage}
out=${2:?$usage}
n=${3:?$usage}
shift 3
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

# #36787's protocol (random 1024 in / 512 out); fixed lengths so runs compare exactly.
for pair in "$@"; do
    c=${pair%%:*}
    prompts=${pair##*:}
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
