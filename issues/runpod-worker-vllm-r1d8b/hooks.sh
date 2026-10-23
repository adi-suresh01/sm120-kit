# shellcheck shell=bash
# Sourced by harness/pod/serve_variant.sh; see harness/README.md for when each hook runs.
hooks_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# The longest context a config fit, from vLLM's auto-fit line in that variant's log (the model's own length when it
# all fit), or 0 when the log has none: its boot failed before the KV cache was sized, or the variant has not run.
fit_len() {
    grep -m1 -oE 'reduced from [0-9]+ to [0-9]+|full model context length [0-9]+' "$1/server.log" 2>/dev/null \
        | grep -oE '[0-9]+$' || echo 0
}

# A 10-23 host ran about 2x slower than another under load with only idle readings to go on, so every variant samples
# the GPU every 5 s from launch to kill, into <variant>/gpu_clocks.csv. clocks_event_reasons.active is a bitmask: 0x1
# idle, 0x4 SW power cap, 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal, 0x80 HW power brake. The timeout ends a
# sampler that a killed run left behind.
#
# Band variants (BAND=1) boot at one explicit MAX_MODEL_LEN, written to /workspace/band.yaml for their --config: past
# what worker-default fit, so the worker's OOM path fires, and halfway to what the retry and graphs-fit both fit.
# When only one of them fits more, halfway to that one, so the other's failure there is the finding.
pre_launch() {
    local run_dir=$1 fields results d f g hi len
    fields=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu
    fields+=,pcie.link.gen.current,pcie.link.width.current,clocks_event_reasons.active
    timeout $(( BOOT_TIMEOUT + 3700 )) nvidia-smi -l 5 --format=csv --query-gpu="$fields" \
        >"$run_dir/gpu_clocks.csv" 2>&1 </dev/null &
    gpu_sampler=$!
    if [[ ${BAND:-} != 1 ]]; then
        return 0
    fi
    results=$(dirname "$run_dir")
    # Only a shortened context has a band above it; when the model's own length fit, no length the model allows fails.
    if ! grep -q 'Auto-fit max_model_len: reduced from' "$results/worker-default/server.log" 2>/dev/null; then
        echo "worker-default has no 'Auto-fit max_model_len: reduced from' line, so there is no band to test:" \
            "run worker-default, worker-fallback, and graphs-fit first, and read worker-default/memory_facts.txt" >&2
        return 1
    fi
    d=$(fit_len "$results/worker-default")
    f=$(fit_len "$results/worker-fallback")
    g=$(fit_len "$results/graphs-fit")
    if (( f > d && g > d )); then
        hi=$(( f < g ? f : g ))
    else
        hi=$(( f > g ? f : g ))
    fi
    if (( hi > d )); then
        len=$(( d + (hi - d) / 2 ))
    else
        len=$(( d + 1024 ))
    fi
    printf 'max-model-len: %d\n' "$len" >/workspace/band.yaml
    echo "fits: worker-default $d, worker-fallback $f, graphs-fit $g; band max-model-len $len" | tee "$run_dir/band.txt"
    progress pre_launch "max-model-len $len"
}

# The Hub's env defaults reach vLLM as variables, not flags, so the live server's own environment is the record that
# they applied.
post_smoke() {
    local run_dir=$1
    timeout 10 tr '\0' '\n' <"/proc/$(<"$run_dir/server.pid")/environ" \
        | grep -E '^(PYTORCH_ALLOC_CONF|VLLM_USE_DEEP_GEMM|TOKENIZERS_PARALLELISM)=' | sort >"$run_dir/server_env.txt" \
        || true
}

# What vLLM sized, what the server says it ran with (flags as parsed, eager or the capture sizes, the batched-token
# budget), and what the worker would do with this boot, next to the verdict.
post_stop() {
    local run_dir=$1
    local facts='Auto-fit max_model_len|Cannot auto-fit|Available KV cache memory|CUDA graph memory profiling'
    facts+='|CUDA graph pool memory|Graph capturing finished|GPU KV cache size|Model loading took|Free memory on device'
    facts+='|Chunked prefill is enabled|OutOfMemoryError|No available memory for the cache blocks'
    facts+='|larger than the available KV cache|Found duplicate keys'
    if [[ -n ${gpu_sampler:-} ]]; then
        kill "$gpu_sampler" 2>/dev/null || true
        gpu_sampler=""
    fi
    grep -E "$facts" "$run_dir/server.log" | cut -c1-900 >"$run_dir/memory_facts.txt" || true
    grep -E 'non-default args|with config:' "$run_dir/server.log" >"$run_dir/settings.txt" || true
    python3 -I "$hooks_dir/worker_decision.py" /workspace/worker-vllm/startup_errors.py "$run_dir/server.log" \
        "$WORKER_LAUNCH" "$MODEL" >"$run_dir/worker_decision.json" 2>&1 || true
}
