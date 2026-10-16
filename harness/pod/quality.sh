#!/usr/bin/env bash
# Measures quality and speed against a running server. serve_variant.sh passes it to serve_and_smoke.sh as
# POST_SMOKE, so it reuses the boot.
# Usage: quality.sh <package> <port> <model> <out_dir> <gsm8k_examples> [<concurrency>:<prompts> ...]
#   GSM8K first, then one benchmark run per concurrency:prompts pair, with the package's own clients: for sglang,
#   run_eval (50 examples are enough decode steps for the scheduler to log accept length) and bench_serving; for
#   vllm, gsm8k_chat.py and `vllm bench serve`. Either way <out_dir>/gsm8k.txt gets a Score line and bench.jsonl one
#   record per pair, which is what verdict.sh reads.
set -euo pipefail

usage="usage: $0 <package> <port> <model> <out_dir> <gsm8k_examples> [<concurrency>:<prompts> ...]"
here=$(cd "$(dirname "$0")" && pwd)
package=${1:?$usage}
port=${2:?$usage}
model=${3:?$usage}
out=${4:?$usage}
n=${5:?$usage}
shift 5
mkdir -p "$out"

sglang_quality() {
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
}

vllm_quality() {
    # Unlike run_eval, gsm8k_chat.py exits non-zero when a request failed, so nothing needs grepping; tee keeps its
    # summary and any failure in the post log too.
    python3 -I "$here/gsm8k_chat.py" "$port" "$model" "$n" "$out" 2>&1 | tee "$out/gsm8k.txt"

    # The same 1024 in / 512 out as for sglang. In vLLM a range ratio of 0 is what fixes the lengths (sglang's is
    # 1), and --ignore-eos keeps any request from ending early, so every run decodes the same tokens.
    for pair in "$@"; do
        c=${pair%%:*}
        prompts=${pair##*:}
        vllm bench serve --backend openai --host 127.0.0.1 --port "$port" --model "$model" --dataset-name random \
            --random-input-len 1024 --random-output-len 512 --random-range-ratio 0 --ignore-eos \
            --max-concurrency "$c" --num-prompts "$prompts" --seed 1 \
            --save-result --result-dir "$out" --result-filename "bench_c$c.json" >"$out/bench_c$c.txt" 2>&1
        # vllm bench serve also exits 0 over whatever finished. The record joins bench.jsonl before the check, as
        # bench_serving's does, so a short run still reports its speed.
        python3 -I - "$out/bench_c$c.json" "$out/bench.jsonl" "$prompts" "$c" <<'EOF'
import json
import sys

path, jsonl, want, c = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
with open(path) as f:
    run = json.load(f)
with open(jsonl, "a") as f:
    f.write(json.dumps(run) + "\n")
if run["completed"] != want:
    sys.exit(f"bench c={c}: {run['completed']}/{want} requests completed")
EOF
    done
}

case $package in
    sglang) sglang_quality "$@" ;;
    vllm) vllm_quality "$@" ;;
    *) echo "no quality protocol for package $package" >&2; exit 2 ;;
esac
