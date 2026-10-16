#!/usr/bin/env bash
# Keeps a pristine copy of an installed Python package and puts it back before every variant, so each run's
# patches apply to the untouched tree. Reverting one run's set before the next can fail when two patches edit the
# same file (#40172 and #43330 both edit standard.py).
# Usage: pkg_snapshot.sh save|restore <package>
#   save     copy the package once; later saves keep the first copy
#   restore  replace the installed package with the copy
# PKG_SNAPSHOT_DIR (default /workspace) holds <package>.pristine.
set -euo pipefail

usage="usage: $0 save|restore <package>"
action=${1:?$usage}
pkg=${2:?$usage}
pristine="${PKG_SNAPSHOT_DIR:-/workspace}/$pkg.pristine"

# find_spec locates the package without importing it; importing vllm or sglang takes seconds.
pkg_root=$(python3 - "$pkg" <<'EOF'
import importlib.util, os, sys
spec = importlib.util.find_spec(sys.argv[1])
if spec is None or spec.origin is None:
    sys.exit(f"package not installed: {sys.argv[1]}")
print(os.path.dirname(spec.origin))
EOF
)

case $action in
    save)
        if [[ ! -d $pristine ]]; then
            # Copy, then rename: an interrupted save must never leave a partial copy to restore from.
            rm -rf "$pristine.tmp"
            cp -a "$pkg_root" "$pristine.tmp"
            mv "$pristine.tmp" "$pristine"
        fi
        ;;
    restore)
        if [[ ! -d $pristine ]]; then
            echo "no snapshot at $pristine; run save first" >&2
            exit 1
        fi
        rm -rf "$pkg_root"
        cp -a "$pristine" "$pkg_root"
        ;;
    *)
        echo "$usage" >&2
        exit 2
        ;;
esac
