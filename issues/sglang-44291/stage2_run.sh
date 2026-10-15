#!/usr/bin/env bash
# Runs one variant of the sglang#44291 stage 2 matrix and appends a row to
# results/sglang-44291-stage2/summary.tsv.
# Usage: stage2_run.sh <variant>
#   QUALITY=full|accept  run stage2_quality.sh against the live server after the smoke test
#   TAG=x                append -x to the result label
#   EXTRA_ARGS="..."     extra server flags (memory levers if a boot runs out of memory)
#   DRY_RUN=1            print the patch set and the server command, change nothing
#
# Variants (reporter's flags, default allocator, --disable-custom-all-reduce unless noted):
#   a0  EP2, no speculation               a1  EP2 + NEXTN
#   a2  a1 + --disable-cuda-graph, CUDA_LAUNCH_BLOCKING=1
#   a3  a1 + #43330                       a4  a1 + #40172
#   a5  a1 with the RadixArk BF16 draft, no patches
#   b0  TP2, no speculation               b1  TP2 + NEXTN
#   b2  b1 + #42566                       b3  b1 with the RadixArk BF16 draft, no patches
#   c-<a4|a5|b2|b3>  that variant + #44448, expandable_segments, custom all-reduce on
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$here/../.." && pwd)
results_dir="$repo_root/results/sglang-44291-stage2"
port=${PORT:-30000}
: "${HF_HOME:?set HF_HOME to the volume disk}"

model=nvidia/Qwen3.8-Flash-Next-NVFP4
model_rev=fc694b54fb0174e0913e6adf86691ef85a4ead47
# Built by stage2_setup.sh: the RadixArk MTP shards with an index trimmed to them.
radixark_dir=/workspace/radixark-mtp
pristine=/workspace/sglang.pristine

nextn=(--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4)
# The same trained MTP head kept in BF16; the cookbook's two-node cell uses it because the NVIDIA export's FP8
# MTP experts cannot be split across two ranks.
radixark=(--speculative-draft-model-path "$radixark_dir" --speculative-draft-model-quantization modelopt_fp4)
ep2=(--tp-size 2 --ep-size 2)
tp2=(--tp-size 2)

variant=${1:?usage: stage2_run.sh <variant>}
# A bad mode would otherwise surface only after a full boot.
case ${QUALITY:-} in
    ""|full|accept) ;;
    *) echo "QUALITY must be full or accept, got $QUALITY" >&2; exit 2 ;;
esac
base=${variant#c-}
patches=() spec=() extra=() launch_env=()
case $base in
    a0) parallel=("${ep2[@]}") ;;
    a1) parallel=("${ep2[@]}"); spec=("${nextn[@]}") ;;
    a2) parallel=("${ep2[@]}"); spec=("${nextn[@]}"); extra=(--disable-cuda-graph); launch_env=(CUDA_LAUNCH_BLOCKING=1) ;;
    a3) parallel=("${ep2[@]}"); spec=("${nextn[@]}"); patches=(pr43330) ;;
    a4) parallel=("${ep2[@]}"); spec=("${nextn[@]}"); patches=(pr40172) ;;
    a5) parallel=("${ep2[@]}"); spec=("${nextn[@]}" "${radixark[@]}") ;;
    b0) parallel=("${tp2[@]}") ;;
    b1) parallel=("${tp2[@]}"); spec=("${nextn[@]}") ;;
    b2) parallel=("${tp2[@]}"); spec=("${nextn[@]}"); patches=(pr42566) ;;
    b3) parallel=("${tp2[@]}"); spec=("${nextn[@]}" "${radixark[@]}") ;;
    *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac

if [[ $variant == c-* ]]; then
    case $base in
        a4|a5|b2|b3) ;;
        *) echo "c- takes a serving NEXTN variant (a4, a5, b2, b3), got $base" >&2; exit 2 ;;
    esac
    patches+=(v1-vmm-staging)
    alloc_env=(PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
    custom_ar=()
else
    # An empty value still counts as set, and newer torch reads PYTORCH_ALLOC_CONF first, so both names go.
    alloc_env=(-u PYTORCH_ALLOC_CONF -u PYTORCH_CUDA_ALLOC_CONF)
    custom_ar=(--disable-custom-all-reduce)
fi

# EXTRA_ARGS is split on whitespace on purpose: it carries whole flags with their values.
# shellcheck disable=SC2206
extra+=(${EXTRA_ARGS:-})

cmd=(env "${alloc_env[@]}" "${launch_env[@]}" HF_HUB_OFFLINE=1
    python3 -m sglang.launch_server --model-path "$model" --revision "$model_rev"
    --moe-runner-backend flashinfer_cutlass --ple-offload-embedding
    --json-model-override-args '{"language_model_only":true}' --reasoning-parser qwen3
    "${parallel[@]}" "${spec[@]}" "${custom_ar[@]}" "${extra[@]}"
    --host 127.0.0.1 --port "$port")

label=${variant}${TAG:+-${TAG}}
run_dir="$results_dir/$label"
post=""
if [[ -n ${QUALITY:-} ]]; then
    post="bash $here/stage2_quality.sh $QUALITY $port $run_dir"
fi

if [[ -n ${DRY_RUN:-} ]]; then
    echo "patches=[${patches[*]}]"
    echo "post=[$post]"
    printf '%q ' "${cmd[@]}"
    echo
    exit 0
fi

# A rerun under the same label must not inherit the last attempt's scores or bench rows.
rm -rf "${run_dir:?}"
mkdir -p "$run_dir"
pkg_root=$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("sglang").origin))')
# #40172 and #43330 both edit standard.py, so reverting one run's set before applying the next can fail; every
# run starts from the untouched tree instead.
rsync -a --delete "$pristine/" "$pkg_root/"
for p in "${patches[@]}"; do
    bash "$repo_root/scripts/patch_pkg.sh" apply "$here/patches/$p.diff" sglang 2 >/dev/null
done

bash "$repo_root/scripts/wait_gpu_idle.sh" 300
echo ">>> $label patches=[${patches[*]}]: ${cmd[*]}"
# 9 is not a captured graph size, so decode batches pad up to the next one, where padding-row bugs live.
# FAIL_PATTERN stops the wait at the first fatal line instead of waiting on a surviving rank. A server wedged in a
# faulted kernel can block serve_and_smoke's shutdown, so timeout kills the whole group.
fail_pattern='illegal memory access|cudaErrorIllegalAddress|CUDA error: |Received sigquit|SIGQUIT received|Scheduler hit an exception'
verdict=$(POST_SMOKE=$post FAIL_PATTERN=$fail_pattern CONCURRENCY=9 BOOT_TIMEOUT=${BOOT_TIMEOUT:-1800} \
    timeout -k 30 "${RUN_TIMEOUT:-5400}" \
    bash "$repo_root/scripts/serve_and_smoke.sh" "$run_dir/server.log" "$port" "$model" -- "${cmd[@]}" || true)
verdict=${verdict:-RESULT=TIMEOUT outer=${RUN_TIMEOUT:-5400}s}
echo "$verdict"
# Scheduler and detokenizer children can outlive the launcher after a fault and hold GPU memory, which would
# fail the next run's idle check. The patterns match sglang's process titles, not this script's path.
pkill -9 -f 'sglang\.launch_server|sglang::' || true

log="$run_dir/server.log"
# CUDA faults first: a benign early ValueError or RuntimeError line must not mask the real fault. head keeps a line
# with several matches from adding newlines to the TSV row.
fault=$(grep -oE 'illegal memory access|cudaErrorIllegalAddress|CUDA error: [^.]*|Xid [0-9]+' "$log" | head -n 1 || true)
if [[ -z $fault ]]; then
    fault=$(grep -oE 'ValueError: [^.]*|RuntimeError: [^.]*|AssertionError[^.]*|OutOfMemoryError[^.]*' "$log" | head -n 1 || true)
fi
fault=${fault:0:200}
# Under CUDA_LAUNCH_BLOCKING=1 the deepest Python frame before the first CUDA fault names the kernel's launcher;
# the frame before the first other error is the fallback, matching the order of the fault column.
where=$(awk '/File "/ { f = $0 } /illegal memory access|cudaErrorIllegalAddress|CUDA error: / && f != "" { c = f; exit } /Error: / && g == "" && f != "" { g = f } END { w = (c != "" ? c : g); sub(/.*\//, "", w); print w }' "$log" || true)
# The per-request mean over GSM8K only; the server log's last value would come from whatever bench ran last.
accept=$(grep -oE 'from meta_info\): n=[0-9]+ mean=[0-9.]+' "$run_dir/gsm8k.txt" 2>/dev/null | sed 's/.*mean=//' || true)
score=$(grep -m1 -oE '^Score: [0-9.]+' "$run_dir/gsm8k.txt" 2>/dev/null | awk '{print $2}' || true)

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$label" "${patches[*]:--}" "$verdict" \
    "${accept:--}" "${score:--}" "${fault:--}" "${where:--}" >>"$results_dir/summary.tsv"
