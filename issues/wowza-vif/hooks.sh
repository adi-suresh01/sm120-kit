# shellcheck shell=bash
# Sourced by harness/pod/serve_variant.sh; see harness/README.md for when each hook runs.

# A 10-23 host ran about 2x slower than another under load with only idle readings to go on, so this samples the GPU
# every 5 s from launch to kill, into <variant>/gpu_clocks.csv. clocks_event_reasons.active is a bitmask: 0x1 idle,
# 0x4 SW power cap, 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal, 0x80 HW power brake. The timeout ends a
# sampler that a killed run left behind.
pre_launch() {
    local run_dir=$1 fields
    fields=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu
    fields+=,pcie.link.gen.current,pcie.link.width.current,clocks_event_reasons.active
    timeout $(( BOOT_TIMEOUT + 3700 )) nvidia-smi -l 5 --format=csv --query-gpu="$fields" \
        >"$run_dir/gpu_clocks.csv" 2>&1 </dev/null &
    gpu_sampler=$!
}

# The A/B rests on what each server says it ran with, not on harness.toml: the engine config line (enforce_eager,
# kv_cache_dtype, enable_prefix_caching, the capture sizes), the KV cache size, and graph capture when there was one.
post_stop() {
    local run_dir=$1
    if [[ -n ${gpu_sampler:-} ]]; then
        kill "$gpu_sampler" 2>/dev/null || true
        gpu_sampler=""
    fi
    grep -E 'non-default args|with config:|KV cache size|Graph capturing finished' "$run_dir/server.log" \
        >"$run_dir/settings.txt" || true
}
