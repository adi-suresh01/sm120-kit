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

# Both would otherwise surface as "does not apply cleanly", which sends the reader after the wrong problem.
if ! command -v patch >/dev/null; then
    echo "patch(1) is not installed; add patch to the image or to [pod] apt" >&2
    exit 1
fi
if [[ ! -r $patch_file ]]; then
    echo "cannot read patch file: $patch_file" >&2
    exit 1
fi

# find_spec locates the package without importing it; importing vllm or sglang takes seconds.
site_root=$(python3 - "$pkg" <<'EOF'
import importlib.util, os, sys
spec = importlib.util.find_spec(sys.argv[1])
if spec is None or spec.origin is None:
    sys.exit(f"package not installed: {sys.argv[1]}")
print(os.path.dirname(os.path.dirname(spec.origin)))
EOF
)

# -f answers patch's "reversed patch detected" prompt with "not reversed", so an unapplied patch fails the
# reverse dry run on both GNU and BSD patch instead of being half-applied forward and reported as applied.
is_applied() {
    patch -p"$strip" -R -f --dry-run --silent -d "$site_root" <"$patch_file" >/dev/null 2>&1
}

# A forward dry run first keeps apply all-or-nothing, so a package that drifted is never left half-patched. Its
# output names the files and hunks that failed, the only clue to why.
dry_run=""
applies_cleanly() {
    dry_run=$(patch -p"$strip" -N --dry-run -d "$site_root" <"$patch_file" 2>&1)
}

case $action in
    apply)
        if is_applied; then
            echo "already applied"
        elif applies_cleanly; then
            patch -p"$strip" -N -d "$site_root" <"$patch_file"
        else
            printf '%s\n' "$dry_run" >&2
            echo "patch does not apply cleanly (partly applied, or the package changed); reinstall the package" >&2
            exit 1
        fi
        ;;
    revert)
        if is_applied; then
            patch -p"$strip" -R -d "$site_root" <"$patch_file"
        elif applies_cleanly; then
            echo "not applied"
        else
            printf '%s\n' "$dry_run" >&2
            echo "patch is partly applied, or the package changed; reinstall the package" >&2
            exit 1
        fi
        ;;
    status)
        if is_applied; then
            echo "applied"
        elif applies_cleanly; then
            echo "not applied"
        else
            printf '%s\n' "$dry_run" >&2
            echo "partial"
            exit 1
        fi
        ;;
    *)
        echo "$usage" >&2
        exit 2
        ;;
esac
