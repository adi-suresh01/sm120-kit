# shellcheck shell=bash
# Sourced by harness/pod/serve_variant.sh; see harness/README.md for when each hook runs.
# AUTOTUNE_START in a variant's env picks the cache state its boot starts from:
#   cold     no autotune cache at all
#   warm     whatever the previous variant left (same flags, so the same cache key directory)
#   partial  the previous variant's cache with rank 1's file removed
#   skew     the cache AUTOTUNE_FROM's boot left (its cache_after copy) with one entry removed from rank 1's file,
#            which stays loadable

kit_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

autotune_dir() {
    printf '%s/flashinfer/autotune' "${SGLANG_CACHE_DIR:-$HOME/.cache/sglang}"
}

# Copies of the per-rank files let the report tell a reused cache (entries unchanged) from a re-tuned one (every
# profiled entry carries a fresh measured time).
snapshot_cache() {
    local dest=$1 dir
    dir=$(autotune_dir)
    mkdir -p "$dest"
    if [[ -d $dir ]]; then
        cp -a "$dir/." "$dest/"
    fi
}

pre_launch() {
    local run_dir=$1 dir src
    dir=$(autotune_dir)
    case ${AUTOTUNE_START:?set AUTOTUNE_START in the variant env} in
        # Only the autotune tree goes: the Triton, inductor, and CUDA caches beside it stay warm, so a cold boot
        # differs from a warm one only in what the gate sees.
        cold) rm -rf "$dir" ;;
        warm) ;;
        partial) find "$dir" -name 'rank_tp1_*.json' -delete ;;
        # #40420's gate passes two loadable files whatever they hold, so only FlashInfer's per-profile hit
        # agreement stands between this and one rank profiling alone. This arm runs last, since a hang costs the
        # whole boot timeout, and the cold controls before it wipe the tree, so it starts from a saved copy.
        skew)
            src=$(dirname "$run_dir")/${AUTOTUNE_FROM:?set AUTOTUNE_FROM in the variant env}/cache_after
            [[ -d $src ]] || { echo "no $src: run $AUTOTUNE_FROM on this pod first" >&2; return 1; }
            rm -rf "$dir"
            mkdir -p "$dir"
            cp -a "$src/." "$dir/"
            python3 -I - "$dir" <<'EOF'
import json
import sys
from pathlib import Path
dropped = 0
for path in Path(sys.argv[1]).rglob("rank_tp1_*.json"):
    data = json.loads(path.read_text())
    key = min(k for k in data if not k.startswith("_"))
    del data[key]
    path.write_text(json.dumps(data, indent=2))
    print(f"dropped from {path}: {key[:200]}")
    dropped += 1
# Without a dropped entry the boot is a plain warm one, which would read as H6 holding.
if not dropped:
    sys.exit("no rank_tp1 file to drop an entry from")
EOF
            ;;
        *) echo "unknown AUTOTUNE_START=$AUTOTUNE_START" >&2; return 1 ;;
    esac
    snapshot_cache "$run_dir/cache_before"
}

post_stop() {
    local run_dir=$1
    snapshot_cache "$run_dir/cache_after"
    python3 -I "$kit_dir/autotune_report.py" "$run_dir" >"$run_dir/autotune.txt" 2>&1 || true
}
