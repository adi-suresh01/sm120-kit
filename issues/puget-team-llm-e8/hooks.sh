# shellcheck shell=bash
# Sourced by harness/pod/serve_variant.sh; see harness/README.md for when each hook runs.

# A 10-23 host ran about 2x slower than another under load with only idle readings to go on, so this samples both
# GPUs every 5 s from launch to kill, into <variant>/gpu_clocks.csv. clocks_event_reasons.active is a bitmask: 0x1
# idle, 0x4 SW power cap, 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal, 0x80 HW power brake. memory.used shows
# that a tp1 variant left GPU 1 idle. The timeout ends a sampler that a killed run left behind.
pre_launch() {
    local run_dir=$1 fields
    fields=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu
    fields+=,memory.used,pcie.link.gen.current,pcie.link.width.current,clocks_event_reasons.active
    timeout $(( BOOT_TIMEOUT + 3700 )) nvidia-smi -l 5 --format=csv --query-gpu="$fields" \
        >"$run_dir/gpu_clocks.csv" 2>&1 </dev/null &
    gpu_sampler=$!
}

# The pack's .env asks for NCCL_P2P_DISABLE=1, but its NVIDIA compose never passes it in, and the 10-19 run modeled
# the file instead of the process, so server_env.txt records the variables each server process really started with.
# The PR #12 review asks what CUDA graph capture costs in memory and startup time on one GPU, so boot_summary.txt
# keeps the log lines that answer it; tqdm's carriage returns can put a log line after a progress bar.
post_smoke() {
    local run_dir=$1 port=$2 model=$3 pid boot_lines
    for pid in $(<"$run_dir/server.pid") $(pgrep -f '^VLLM::'); do
        printf '## pid %s\n' "$pid"
        timeout 10 tr '\0' '\n' <"/proc/$pid/environ" | grep -E '^(NCCL|VLLM|CUDA|TORCH|PYTORCH)_' | sort || true
    done >"$run_dir/server_env.txt" 2>&1
    boot_lines='Model loading took|torch\.compile took|compiled function|Graph capturing finished'
    boot_lines+='|CUDA graph pool memory|CUDA graph memory profiling|Available KV cache memory|GPU KV cache size'
    boot_lines+='|init engine|Mamba cache mode'
    tr '\r' '\n' <"$run_dir/server.log" | grep -a -E "$boot_lines" >"$run_dir/boot_summary.txt" || true
    if [[ -n ${SOAK_MINUTES:-} ]]; then
        run_soak "$run_dir" "$port" "$model"
    fi
}

# vllm#54331 reports hybrid linear-attention models dying 2.5-8 min into saturated serving with CUDA graphs on, which
# the short benchmarks cannot show. This is issues/vllm-54331's soak at concurrency 64: 16 shared 4096-token prefixes
# (so prefix caching stays in use), 1024-token suffixes and outputs, in rounds until SOAK_MINUTES have passed; the
# round in progress then finishes. prefix_repetition sends num_prompts / num_prefixes per prefix, so 256 sends 256.
# vllm bench serve exits 0 over failed requests, and one that died prints no summary, so a request counts only once a
# summary lists it as successful. A round where any request did not ends the soak, keeps the server log's tail, and
# fails the hook (POST_FAIL).
run_soak() {
    local run_dir=$1 port=$2 model=$3 per_round=256 round=0 sent=0 completed=0 start end ok out speed
    start=$(date +%s)
    end=$(( start + SOAK_MINUTES * 60 ))
    echo "soak: concurrency 64, $per_round requests per round, $SOAK_MINUTES min" | tee "$run_dir/soak.txt"
    while (( $(date +%s) < end )); do
        round=$((round + 1))
        out=$run_dir/soak_round$round.txt
        progress soak "round $round"
        vllm bench serve --backend vllm --host 127.0.0.1 --port "$port" --model "$model" \
            --dataset-name prefix_repetition --prefix-repetition-prefix-len 4096 \
            --prefix-repetition-suffix-len 1024 --prefix-repetition-num-prefixes 16 \
            --prefix-repetition-output-len 1024 --ignore-eos --max-concurrency 64 --num-prompts "$per_round" \
            --seed "$round" >"$out" 2>&1 || true
        ok=$(awk '/^Successful requests:/ {print $3}' "$out")
        ok=${ok:-0}
        sent=$((sent + per_round))
        completed=$((completed + ok))
        speed=$(awk '/^(Output token throughput|Mean TPOT)/ {
            sub(/ +$/, ""); gsub(/  +/, " "); printf "%s%s", sep, $0; sep = "; "
        }' "$out")
        echo "round $round: $ok/$per_round completed; ${speed:-no summary}" | tee -a "$run_dir/soak.txt"
        if (( ok < per_round )); then
            tail -n 300 "$run_dir/server.log" >"$run_dir/soak_failure_tail.txt"
            break
        fi
    done
    echo "soak ended: $round rounds in $(( ($(date +%s) - start) / 60 )) min, $completed/$sent requests completed" \
        | tee -a "$run_dir/soak.txt"
    (( completed == sent ))
}

post_stop() {
    local run_dir=$1
    if [[ -n ${gpu_sampler:-} ]]; then
        kill "$gpu_sampler" 2>/dev/null || true
        gpu_sampler=""
    fi
    # The watchdog stops a soak whose server exited, stalled, or logged a fail_pattern line before the soak writes its
    # last line, so the tail is kept here; it ends with the harness's own shutdown of the server.
    if [[ -f $run_dir/soak.txt && ! -f $run_dir/soak_failure_tail.txt ]] \
        && ! grep -q '^soak ended' "$run_dir/soak.txt"; then
        tail -n 300 "$run_dir/server.log" >"$run_dir/soak_failure_tail.txt"
    fi
}
