#!/usr/bin/env bash
# Applies or reverts PR #60069 (shared b12x MoE workspace) on the installed vLLM package.
# The patch is vendored at PR head 5a80ca1d and was dry-run checked against the v0.31.0 source file.
# Usage: patch.sh apply | revert | status
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
patch_file="$here/patches/pr60069-b12x-shared-workspace.diff"
site_root=$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')

is_applied() {
    patch -p1 -R --dry-run --silent -d "$site_root" <"$patch_file" >/dev/null 2>&1
}

case ${1:-status} in
    apply)
        if is_applied; then echo "already applied"; exit 0; fi
        patch -p1 -d "$site_root" <"$patch_file"
        ;;
    revert)
        if ! is_applied; then echo "not applied"; exit 0; fi
        patch -p1 -R -d "$site_root" <"$patch_file"
        ;;
    status)
        if is_applied; then echo "applied"; else echo "not applied"; fi
        ;;
    *)
        echo "usage: $0 apply|revert|status" >&2
        exit 2
        ;;
esac
