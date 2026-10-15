#!/usr/bin/env bash
# Applies, reverts, or reports a patch against an installed Python package, so a repro can toggle an upstream
# change on a stock image without rebuilding it.
# Usage: patch_pkg.sh apply|revert|status <patch_file> <package> [strip]
#   strip: patch -p level, relative to the directory that contains <package> (default 1)
set -euo pipefail

usage="usage: $0 apply|revert|status <patch_file> <package> [strip]"
action=${1:?$usage}
patch_file=${2:?$usage}
pkg=${3:?$usage}
strip=${4:-1}

# find_spec locates the package without importing it; importing vllm or sglang takes seconds.
site_root=$(python3 -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('$pkg').origin)))")

# -f answers patch's "reversed patch detected" prompt with "not reversed", so an unapplied patch fails the
# reverse dry run on both GNU and BSD patch instead of being half-applied forward and reported as applied.
is_applied() {
    patch -p"$strip" -R -f --dry-run --silent -d "$site_root" <"$patch_file" >/dev/null 2>&1
}

case $action in
    apply)
        if is_applied; then echo "already applied"; else patch -p"$strip" -N -d "$site_root" <"$patch_file"; fi
        ;;
    revert)
        if is_applied; then patch -p"$strip" -R -d "$site_root" <"$patch_file"; else echo "not applied"; fi
        ;;
    status)
        if is_applied; then echo "applied"; else echo "not applied"; fi
        ;;
    *)
        echo "$usage" >&2
        exit 2
        ;;
esac
