# shellcheck shell=bash
# Sourced by harness/pod/serve_variant.sh; see harness/README.md for when each hook runs.

# Keeps the server saturated for SOAK_MINUTES. The report's workload was up to 70 concurrent multi-turn requests over
# a shared prefix, so each round reuses 16 long prefixes at concurrency 70 with full-length outputs; one round is a
# few minutes, so the harness's 15-minute quiet watchdog never fires on a live server.
post_smoke() {
    local run_dir=$1 port=$2 model=$3
    local end=$(( $(date +%s) + ${SOAK_MINUTES:-25} * 60 )) round=0
    while (( $(date +%s) < end )); do
        round=$((round + 1))
        progress soak "round $round"
        vllm bench serve --backend vllm --host 127.0.0.1 --port "$port" --model "$model" \
            --dataset-name prefix_repetition --prefix-repetition-prefix-len 4096 \
            --prefix-repetition-suffix-len 1024 --prefix-repetition-num-prefixes 16 \
            --prefix-repetition-output-len 1024 --ignore-eos --max-concurrency 70 --num-prompts 600 \
            --seed "$round" >"$run_dir/soak_round$round.txt" 2>&1
        grep -E 'Successful requests|Output token throughput' "$run_dir/soak_round$round.txt"
    done
    echo "soak finished: $round rounds in ${SOAK_MINUTES:-25} min"
}
