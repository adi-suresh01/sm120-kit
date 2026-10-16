#!/usr/bin/env bash
# Applies or reverts a vendored patch on the installed packages.
#   pr60069  vLLM PR #60069 (shared b12x MoE workspace), head 5a80ca1d, dry-run checked against v0.31.0.
#   fi4827   FlashInfer PR #4827 (keep graph-referenced SM12x workspaces and prepared weights alive), head
#            0ab76fd6, rebased onto flashinfer-python 0.7.0.post1: two context comment lines that 0.7.0.post1
#            lacks were dropped from one hunk; the PR's test file is left out.
#   pr57946  vLLM PR #57946 (route the -1 CUDA-graph padding sentinel away from b12x), head a2c8e664, source
#            file only. Applies to v0.31.0 alone and on top of pr60069 (offsets only).
# Usage: patch.sh apply|revert|status [pr60069|fi4827|pr57946]
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)

case ${2:-pr60069} in
    pr60069) patch_file="$here/patches/pr60069-b12x-shared-workspace.diff"; pkg=vllm ;;
    fi4827)  patch_file="$here/patches/fi4827-graph-pinning-on-0.7.0.post1.diff"; pkg=flashinfer ;;
    pr57946) patch_file="$here/patches/pr57946-b12x-padding-sentinel.diff"; pkg=vllm ;;
    *) echo "unknown patch: $2" >&2; exit 2 ;;
esac

exec bash "$here/../../harness/pod/patch_pkg.sh" "${1:-status}" "$patch_file" "$pkg" 1
