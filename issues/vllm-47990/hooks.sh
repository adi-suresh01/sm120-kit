# Sourced by harness/pod/serve_variant.sh. pre_launch runs after the variant's patches are applied and its env is
# exported, so each check below measures exactly the vLLM that the variant then serves. A failed check is a result,
# not an error: each step's exit code goes to <step>.exit next to its log, and pre_launch always returns 0, because
# under serve_variant.sh's `set -e` a failure here would skip the server boot that is the variant's verdict.
# Variant env switches: ROUTE_PROBE=1 (route.py), KERNEL_SUITE=1 (shape sweep and vLLM's CUTLASS tests),
# GEMM_BENCH=1 (bench_gemm.py), OFFLINE_EVAL=1 (GSM8K and throughput through the offline engine).

hook_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
model=gaunernst/DeepSeek-V2-Lite-Chat-FP8
revision=c6f13916a378a8020921732bda65cddc9a3d13f6

# Runs one step bounded by <seconds>, logging to <run_dir>/<step>.log and recording its exit code. The bounds keep
# one hung step from eating the time the variants after it need before the pod's deadline.
step() {
    local run_dir=$1 name=$2 seconds=$3 rc=0
    shift 3
    # serve_variant.sh's progress(), so `harness exec` shows which step a variant is in.
    if declare -F progress >/dev/null; then
        progress "pre_launch:$name" "$*"
    fi
    echo ">>> pre_launch $name: $*"
    timeout -k 30 "$seconds" "$@" >"$run_dir/$name.log" 2>&1 || rc=$?
    echo "$rc" >"$run_dir/$name.exit"
    echo ">>> pre_launch $name exit $rc"
}

pre_launch() {
    local run_dir=$1
    if [[ ${ROUTE_PROBE:-} == 1 ]]; then
        step "$run_dir" route 300 python3 "$hook_dir/route.py"
    fi
    if [[ ${KERNEL_SUITE:-} == 1 ]]; then
        step "$run_dir" sweep 300 python3 "$hook_dir/sweep.py" "$run_dir/sweep.jsonl"
        # From the tests checkout setup.sh made; it has no vllm/ source dir, so the installed (patched) vLLM is the one
        # imported.
        step "$run_dir" pytest 600 env -C /workspace/vllm-tests python3 -m pytest -q -rA -p no:cacheprovider \
            tests/kernels/quantization/test_block_fp8.py -k cutlass --junitxml="$run_dir/pytest.xml"
    fi
    if [[ ${GEMM_BENCH:-} == 1 ]]; then
        step "$run_dir" bench_gemm 600 python3 "$hook_dir/bench_gemm.py" "$run_dir/bench_gemm.jsonl"
    fi
    if [[ ${OFFLINE_EVAL:-} == 1 ]]; then
        step "$run_dir" eval 1200 python3 "$hook_dir/eval_offline.py" "$model" "$revision" /workspace/gsm8k \
            "$run_dir/eval.json"
    fi
    # The offline engine's worker can take a few seconds to release the GPU after its parent exits, and the server
    # boot right after this would then see less free memory than the other variants.
    bash "$hook_dir/../../harness/pod/wait_gpu_idle.sh" 300 || true
    return 0
}
