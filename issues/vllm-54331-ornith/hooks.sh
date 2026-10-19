# shellcheck shell=bash
# Sourced by harness/pod/serve_variant.sh; see harness/README.md for when each hook runs.

# Resolved now, while the path this file was sourced by is still valid from the current directory.
kit_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Keeps the server saturated for SOAK_MINUTES at the report's concurrency of 70, after checking the attention setup
# matches the report's. text is issues/vllm-54331's workload unchanged; video is soak_video.py.
post_smoke() {
    local run_dir=$1 port=$2 model=$3
    check_attention "$run_dir/server.log"
    case ${SOAK_WORKLOAD:?} in
        text) soak_text "$run_dir" "$port" "$model" ;;
        video)
            progress soak "video, ${SOAK_MINUTES:-25} min"
            python3 "$kit_dir/soak_video.py" --port "$port" --model "$model" --videos /root/videos \
                --minutes "${SOAK_MINUTES:-25}" --concurrency 70 --out "$run_dir/soak_video.jsonl"
            ;;
        *) echo "unknown SOAK_WORKLOAD=$SOAK_WORKLOAD" >&2; return 1 ;;
    esac
}

# The report's log shows fp8 KV with FlashInfer's xqa decode. xqa falls back to native decode without a word when its
# cubins cannot be fetched, and a soak on another path would answer a different question.
check_attention() {
    local log=$1 line
    grep -E 'Setting attention block size to [0-9]+' "$log" | tail -1 || true
    line=$(grep -E 'FlashInfer resolved query dtypes' "$log" | tail -1) || true
    echo "${line:-no FlashInfer attention line in the server log}"
    if [[ $line != *decode_backend=xqa* || $line != *float8_e4m3fn* ]]; then
        echo "attention is not fp8 KV with xqa decode, as in the report; stopping before the soak" >&2
        return 1
    fi
}

# One round is a few minutes, so the harness's 15-minute quiet watchdog never fires on a live server.
soak_text() {
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
