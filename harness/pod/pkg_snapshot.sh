#!/usr/bin/env bash
# Keeps a pristine copy of installed Python packages and puts them back before every variant, so each run's
# patches apply to the untouched tree. Reverting one run's set before the next can fail when two patches edit the
# same file (#40172 and #43330 both edit standard.py).
# Usage: pkg_snapshot.sh save|restore <package>...
#   <package>  import name (sglang, vllm, flashinfer), not the pip distribution name
#   save       copy each package once and record where it is installed; later saves keep the first copy
#   restore    put each saved copy back where it was recorded
# PKG_SNAPSHOT_DIR (default /workspace) holds <package>.pristine and <package>.path.
set -euo pipefail

usage="usage: $0 save|restore <package>..."
action=${1:?$usage}
shift
(( $# )) || { echo "$usage" >&2; exit 2; }
snap_dir=${PKG_SNAPSHOT_DIR:-/workspace}

# find_spec locates the package without importing it; importing vllm or sglang takes seconds.
install_path() {
    python3 - "$1" <<'EOF'
import importlib.util, os, sys
name = sys.argv[1]
spec = importlib.util.find_spec(name)
if spec is None or spec.origin is None:
    sys.exit(f"package not installed: {name}")
# A single-file module's directory is site-packages itself, which a restore would replace wholesale.
if not spec.submodule_search_locations:
    sys.exit(f"{name} is a single-file module, not a package directory; it cannot be snapshotted")
print(os.path.dirname(spec.origin))
EOF
}

save() {
    local pkg=$1 pristine=$snap_dir/$1.pristine path
    [[ -d $pristine ]] && return 0
    path=$(install_path "$pkg")
    mkdir -p "$snap_dir"
    # The path goes first so a snapshot never exists without it, and the copy is renamed into place so an
    # interrupted save never leaves a partial copy to restore from.
    printf '%s\n' "$path" >"$snap_dir/$pkg.path"
    rm -rf "$pristine.tmp"
    cp -a "$path" "$pristine.tmp"
    mv "$pristine.tmp" "$pristine"
}

# The live tree is only touched once a full copy sits next to it, then two renames swap them, so a failed restore
# leaves the package importable. A kill between the renames leaves it missing, and the next restore repairs that
# from the recorded path, which find_spec could no longer find.
restore() {
    local pkg=$1 pristine=$snap_dir/$1.pristine path
    if [[ ! -d $pristine || ! -f $snap_dir/$pkg.path ]]; then
        echo "no snapshot of $pkg in $snap_dir; run save first" >&2
        return 1
    fi
    path=$(<"$snap_dir/$pkg.path")
    rm -rf "$path.new"
    cp -a "$pristine" "$path.new"
    if [[ -e $path ]]; then
        rm -rf "$path.old"
        mv "$path" "$path.old"
    fi
    mv "$path.new" "$path"
    rm -rf "$path.old"
}

for pkg in "$@"; do
    case $action in
        save) save "$pkg" ;;
        restore) restore "$pkg" ;;
        *) echo "$usage" >&2; exit 2 ;;
    esac
done
