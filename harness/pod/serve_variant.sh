#!/usr/bin/env bash
# Runs one variant rendered by `harness exec`: restores the package, applies the variant's patches, waits for idle
# GPUs, boots and smoke-tests the server, runs quality.sh on the live server when asked, and writes one verdict.
# Usage: serve_variant.sh <issue_dir> <variant_file> <results_dir>
#   <variant_file>  sets VARIANT, PACKAGE, MODEL, PORT, FAIL_PATTERN, BOOT_TIMEOUT, CONCURRENCY, PATCH_STRIP,
#                   GSM8K, and the arrays PATCHES, SERVER_CMD, VARIANT_ENV, BENCH
#   <issue_dir>/hooks.sh, when present, may define pre_launch and post_stop; each gets the run directory.
# Writes <results_dir>/<variant>/verdict.json and appends the same line to <results_dir>/verdicts.jsonl.
set -euo pipefail

usage="usage: $0 <issue_dir> <variant_file> <results_dir>"
here=$(cd "$(dirname "$0")" && pwd)
issue_dir=$(cd "${1:?$usage}" && pwd)
variant_file=${2:?$usage}
results_dir=$(mkdir -p "${3:?$usage}" && cd "$3" && pwd)
# shellcheck source=/dev/null
source "$variant_file"

case $PACKAGE in
    # Process titles of the server and its rank children. The brackets keep pkill from matching any command line
    # that merely quotes the pattern, such as an SSH session running this cleanup.
    sglang) server_procs='[s]glang[.]launch_server|[s]glang::' ;;
    vllm) server_procs='[v]llm serve|[V]LLM::' ;;
    *) echo "no process pattern for package $PACKAGE" >&2; exit 2 ;;
esac

run_dir="$results_dir/$VARIANT"
# A rerun under the same name must not inherit the last attempt's logs, scores, or verdict.
rm -rf "${run_dir:?}"
mkdir -p "$run_dir"

if [[ -f $issue_dir/hooks.sh ]]; then
    # shellcheck source=/dev/null
    source "$issue_dir/hooks.sh"
fi

# Scheduler and detokenizer children can outlive the launcher after a fault and hold GPU memory, which would fail
# the next variant's idle check.
cleanup() {
    pkill -9 -f "$server_procs" || true
    if declare -F post_stop >/dev/null; then
        post_stop "$run_dir" || true
    fi
}
trap cleanup EXIT

bash "$here/pkg_snapshot.sh" save "$PACKAGE"
bash "$here/pkg_snapshot.sh" restore "$PACKAGE"
for p in "${PATCHES[@]}"; do
    bash "$here/patch_pkg.sh" apply "$issue_dir/patches/$p.diff" "$PACKAGE" "$PATCH_STRIP" >/dev/null
done
if (( ${#VARIANT_ENV[@]} )); then
    export "${VARIANT_ENV[@]}"
fi

post=""
if [[ -n $GSM8K ]]; then
    post=$(printf '%q ' bash "$here/quality.sh" "$PORT" "$run_dir" "$GSM8K" "${BENCH[@]}")
fi

bash "$here/wait_gpu_idle.sh" 300
if declare -F pre_launch >/dev/null; then
    pre_launch "$run_dir"
fi
echo ">>> $VARIANT patches=[${PATCHES[*]}]: ${SERVER_CMD[*]}"
# The outer bound is the boot plus an hour for the smoke test and quality runs. A server wedged in a faulted
# kernel can block serve_and_smoke's own shutdown, so timeout kills the whole group.
run_timeout=$(( BOOT_TIMEOUT + 3600 ))
result=$(POST_SMOKE=$post FAIL_PATTERN=$FAIL_PATTERN CONCURRENCY=$CONCURRENCY BOOT_TIMEOUT=$BOOT_TIMEOUT \
    timeout -k 30 "$run_timeout" \
    bash "$here/serve_and_smoke.sh" "$run_dir/server.log" "$PORT" "$MODEL" -- \
    "${SERVER_CMD[@]}" --host 127.0.0.1 --port "$PORT" || true)
result=${result:-RESULT=TIMEOUT outer=${run_timeout}s}
echo "$result"
cleanup
trap - EXIT
bash "$here/verdict.sh" "$run_dir" "$VARIANT" "$result" | tee "$run_dir/verdict.json" >>"$results_dir/verdicts.jsonl"
