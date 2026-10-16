# shellcheck shell=bash
# Sourced by harness/pod/serve_variant.sh; see harness/README.md for when each hook runs.
hooks_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Versions as the server will import them (PYTHONPATH overlays included), the driver, and the UVA coherence probe.
pre_launch() {
    local run_dir=$1
    progress pre_launch "versions and uva probe"
    {
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
        python3 -c 'from importlib.metadata import version as v
for p in ("vllm", "flashinfer-python", "flashinfer-jit-cache", "compressed-tensors", "transformers"):
    print(p, v(p))'
    } >"$run_dir/versions.txt" 2>&1 || true
    timeout -k 10 300 python3 "$hooks_dir/uva_probe.py" >"$run_dir/uva_probe.json" 2>"$run_dir/uva_probe.err" || true
    bash "$hooks_dir/../../harness/pod/wait_gpu_idle.sh" 120
}

# Runs after GSM8K, so a garbage server still gets its raw output recorded word for word. With LONG_CONTEXT set,
# the server first takes a batch of long prompts, and GSM8K runs again after them (the variant has no quality
# block, so the only score is the one taken on the loaded server).
post_smoke() {
    local run_dir=$1 port=$2 model=$3 model_dir
    if [[ -n ${LONG_CONTEXT:-} ]]; then
        model_dir=$(python3 -I -c "from huggingface_hub import snapshot_download as d
print(d('$model', revision='79a917d6a8034910fa610b097dc61e96eac65d29', local_files_only=True))")
        progress post_smoke "long context: $LONG_CONTEXT"
        python3 -I "$hooks_dir/long_context.py" "$port" "$model" "$run_dir" "$model_dir"
    fi
    python3 -I "$hooks_dir/probe_requests.py" "$port" "$model" "$run_dir"
    if [[ -n ${LONG_CONTEXT:-} ]]; then
        python3 -I "$hooks_dir/../../harness/pod/gsm8k_chat.py" "$port" "$model" 200 "$run_dir"
    fi
}

# Which runner, attention backend, and linear kernel the server actually picked.
post_stop() {
    local run_dir=$1
    local facts='model runner|ModelRunner|attention backend|FlashInfer resolved|Selected .*Kernel|cudagraph_mode'
    grep -i -E "$facts|kv_cache_dtype" "$run_dir/server.log" | head -n 40 >"$run_dir/server_facts.txt" || true
}
