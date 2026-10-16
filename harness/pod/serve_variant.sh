#!/usr/bin/env bash
# Runs one variant rendered by `harness exec`: restores every snapshotted package, applies the variant's patches,
# waits for idle GPUs, boots and smoke-tests the server, runs quality.sh and the post_smoke hook on the live server
# when asked (see post_smoke.sh), and writes one verdict.
# Usage: serve_variant.sh <issue_dir> <variant_file> <results_dir>
#   <variant_file>  sets VARIANT, PACKAGE, MODEL, PORT, FAIL_PATTERN, BOOT_TIMEOUT, CONCURRENCY, GSM8K, and the
#                   arrays SERVER_CMD, VARIANT_ENV, BENCH, SNAPSHOT_PKGS (the server package first, then every
#                   package any variant of the issue patches), and PATCH_FILES, PATCH_PKGS, PATCH_STRIPS (patch i
#                   is <issue_dir>/patches/<PATCH_FILES[i]>.diff, applied to the import name PATCH_PKGS[i] at
#                   -p<PATCH_STRIPS[i]>)
#   <issue_dir>/hooks.sh, when present, may define pre_launch <run_dir> (before the boot), post_smoke <run_dir>
#                   <port> <model> (on the live server, after quality), and post_stop <run_dir> (after the kill).
# Writes <results_dir>/<variant>/verdict.json, appends the same line to <results_dir>/verdicts.jsonl, and appends
# phase lines (see progress.sh) to <results_dir>/<variant>/progress.jsonl, the verdict line last.
# From the launch to the kill, mem_guard.py watches the container's memory (memory.jsonl, mem_guard.log, and
# memory_guard.txt when it fires, in <results_dir>/<variant>); MEM_GUARD_POLL_S, MEM_GUARD_TRIM_FRACTION, and
# MEM_GUARD_KILL_FRACTION in the variant's env tune it.
set -euo pipefail

usage="usage: $0 <issue_dir> <variant_file> <results_dir>"
here=$(cd "$(dirname "$0")" && pwd)
issue_dir=$(cd "${1:?$usage}" && pwd)
# Absolute, since post_smoke.sh sources it again after pre_launch, which may change directory.
variant_file=$(cd "$(dirname "${2:?$usage}")" && pwd)/$(basename "$2")
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

patch_count=${#PATCH_FILES[@]}
if (( ${#PATCH_PKGS[@]} != patch_count || ${#PATCH_STRIPS[@]} != patch_count )); then
    echo "PATCH_FILES, PATCH_PKGS, and PATCH_STRIPS differ in length" >&2
    exit 2
fi
# Only snapshotted packages are put back before the next variant; a patch to any other would leak into it.
for pkg in "${PATCH_PKGS[@]}"; do
    if [[ " ${SNAPSHOT_PKGS[*]} " != *" $pkg "* ]]; then
        echo "patched package $pkg is not in SNAPSHOT_PKGS" >&2
        exit 2
    fi
done

run_dir="$results_dir/$VARIANT"
# A rerun under the same name must not inherit the last attempt's logs, scores, or verdict.
rm -rf "${run_dir:?}"
mkdir -p "$run_dir"
progress_file="$run_dir/progress.jsonl"

# Progress is for display only; a failed write must not fail the run.
progress() {
    bash "$here/progress.sh" "$progress_file" "$@" || true
}

if [[ -f $issue_dir/hooks.sh ]]; then
    # shellcheck source=/dev/null
    source "$issue_dir/hooks.sh"
fi

guard_pid=""
# A guard left running would write into this run's files after its verdict, and one that never saw a server pid falls
# back to killing by command line, which would hit the next variant's server. One stuck in a read on a wedged FUSE
# mount cannot die until the read returns, so the run does not wait on it past the kill.
stop_guard() {
    if [[ -z $guard_pid ]]; then
        return 0
    fi
    kill "$guard_pid" 2>/dev/null || true
    for _ in $(seq 50); do
        kill -0 "$guard_pid" 2>/dev/null || break
        sleep 0.1
    done
    kill -9 "$guard_pid" 2>/dev/null || true
    guard_pid=""
}

# Scheduler and detokenizer children can outlive the launcher after a fault and hold GPU memory, which would fail
# the next variant's idle check. The guard goes first, so it writes nothing after the verdict.
cleanup() {
    stop_guard
    pkill -9 -f "$server_procs" || true
    if declare -F post_stop >/dev/null; then
        post_stop "$run_dir" || true
    fi
}
trap cleanup EXIT

progress restore "${SNAPSHOT_PKGS[*]}"
for pkg in "${SNAPSHOT_PKGS[@]}"; do
    bash "$here/pkg_snapshot.sh" save "$pkg"
    bash "$here/pkg_snapshot.sh" restore "$pkg"
done
for i in "${!PATCH_FILES[@]}"; do
    progress patch "${PATCH_FILES[i]} -> ${PATCH_PKGS[i]} (-p${PATCH_STRIPS[i]})"
    bash "$here/patch_pkg.sh" apply "$issue_dir/patches/${PATCH_FILES[i]}.diff" "${PATCH_PKGS[i]}" \
        "${PATCH_STRIPS[i]}" >/dev/null
done
if (( ${#VARIANT_ENV[@]} )); then
    export "${VARIANT_ENV[@]}"
fi

post=""
if [[ -n $GSM8K ]] || declare -F post_smoke >/dev/null; then
    post=$(printf '%q ' bash "$here/post_smoke.sh" "$issue_dir" "$variant_file" "$results_dir")
fi

progress idle_wait
bash "$here/wait_gpu_idle.sh" 300
if declare -F pre_launch >/dev/null; then
    pre_launch "$run_dir"
fi
# RunPod stops a container at about 98% of its memory limit and removes it with every log on it. The guard records
# what fills the memory, drops the weights from the page cache, and ends the server before that (see mem_guard.py).
# huggingface_hub keeps the weights in HF_HUB_CACHE, by default HF_HOME's hub.
hub_cache=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
python3 -I "$here/mem_guard.py" "$run_dir" "$$" "$server_procs" "$hub_cache" >"$run_dir/mem_guard.log" 2>&1 &
guard_pid=$!
launch_cmd=("${SERVER_CMD[@]}" --host 127.0.0.1 --port "$PORT")
progress launch "${launch_cmd[*]}"
echo ">>> $VARIANT patches=[${PATCH_FILES[*]}]: ${launch_cmd[*]}"
# The outer bound is the boot plus an hour for the smoke test and quality runs. A server wedged in a faulted
# kernel can block serve_and_smoke's own shutdown, so timeout kills the whole group.
run_timeout=$(( BOOT_TIMEOUT + 3600 ))
result=$(PROGRESS_FILE=$progress_file POST_SMOKE=$post FAIL_PATTERN=$FAIL_PATTERN CONCURRENCY=$CONCURRENCY \
    BOOT_TIMEOUT=$BOOT_TIMEOUT timeout -k 30 "$run_timeout" \
    bash "$here/serve_and_smoke.sh" "$run_dir/server.log" "$PORT" "$MODEL" -- "${launch_cmd[@]}" || true)
result=${result:-RESULT=TIMEOUT outer=${run_timeout}s}
echo "$result"
cleanup
trap - EXIT
progress stop "$result"
verdict=$(bash "$here/verdict.sh" "$run_dir" "$VARIANT" "$result")
printf '%s\n' "$verdict" >"$run_dir/verdict.json"
printf '%s\n' "$verdict" >>"$results_dir/verdicts.jsonl"
# Last, so a reader that sees the verdict here finds verdict.json and verdicts.jsonl already written.
printf '%s\n' "$verdict" >>"$progress_file"
