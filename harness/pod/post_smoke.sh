#!/usr/bin/env bash
# Everything that runs on the live server after a passing smoke test: quality.sh when the variant has a quality
# table, then the issue's post_smoke hook when hooks.sh defines one. serve_variant.sh passes it to serve_and_smoke.sh
# as POST_SMOKE, so the watchdog there (server exit, FAIL_PATTERN, stall) covers both. Both run when both are present,
# quality first, and the exit is the first failure's, so either one failing is POST_FAIL.
# Usage: post_smoke.sh <issue_dir> <variant_file> <results_dir>
#   serve_variant.sh's own arguments, so post_smoke sees what pre_launch does: the variant file's variables, the
#   variant env (exported by serve_variant.sh), and progress. It is called as post_smoke <run_dir> <port> <model>.
set -euo pipefail

usage="usage: $0 <issue_dir> <variant_file> <results_dir>"
here=$(cd "$(dirname "$0")" && pwd)
issue_dir=${1:?$usage}
variant_file=${2:?$usage}
results_dir=${3:?$usage}
# shellcheck source=/dev/null
source "$variant_file"
run_dir="$results_dir/$VARIANT"

# Progress is for display only; a failed write must not fail the run.
progress() {
    bash "$here/progress.sh" "$run_dir/progress.jsonl" "$@" || true
}

if [[ -f $issue_dir/hooks.sh ]]; then
    # shellcheck source=/dev/null
    source "$issue_dir/hooks.sh"
fi

# The hook runs even when quality failed: a GSM8K download that failed must not cost the issue's own evidence from
# a server that took minutes to boot.
status=0
if [[ -n $GSM8K ]]; then
    bash "$here/quality.sh" "$PACKAGE" "$PORT" "$MODEL" "$run_dir" "$GSM8K" "${BENCH[@]}" || status=$?
fi
if declare -F post_smoke >/dev/null; then
    progress hook_start
    # Called behind || the hook would run past a failing command, since bash ignores errexit there even inside a
    # subshell. As a plain command in a subshell it stops at the first one, the way pre_launch does.
    set +e
    (set -e; post_smoke "$run_dir" "$PORT" "$MODEL")
    hook=$?
    set -e
    if (( status == 0 )); then
        status=$hook
    fi
fi
exit "$status"
