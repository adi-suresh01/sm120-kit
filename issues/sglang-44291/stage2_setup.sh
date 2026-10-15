#!/usr/bin/env bash
# Checks the pod, saves a pristine copy of the sglang package, and downloads the checkpoints for the
# sglang#44291 stage 2 matrix, so paid time after this goes to server runs.
# Usage: stage2_setup.sh [--full-radixark]
#   --full-radixark  fetch the whole RadixArk export (fallback if the draft does not load from its MTP shards)
# The draft is served from /workspace/radixark-mtp, built here from the RadixArk snapshot.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the volume disk}"
here=$(cd "$(dirname "$0")" && pwd)
pristine=/workspace/sglang.pristine

# The pinned PLE table is 47.7 GiB; the cookbook says Docker needs an unlimited memlock for it.
echo "memlock: $(ulimit -l)"
free -g
numactl -H || echo "numactl not available; NUMA layout unknown"
df -h /workspace
pip show sglang 2>/dev/null | grep -E '^(Version|Location|Editable project location)' || true

pkg_root=$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("sglang").origin))')
# stage2_run.sh restores this copy before every run, so it must be taken before any patch is applied.
if [[ ! -d $pristine ]]; then
    cp -a "$pkg_root" "$pristine"
fi
echo "pristine copy: $pristine (from $pkg_root)"

python3 - "$here/mtp_shards.py" "${1:-}" <<'EOF'
import json
import os
import subprocess
import sys
import time

from huggingface_hub import hf_hub_download, snapshot_download

mtp_shards, mode = sys.argv[1], sys.argv[2]
NV, NV_REV = "nvidia/Qwen3.8-Flash-Next-NVFP4", "fc694b54fb0174e0913e6adf86691ef85a4ead47"
RA, RA_REV = "RadixArk/Qwen3.8-Flash-Next-NVFP4", "7b719225242aacd3dbd3f9407468c2ee9a9d2594"

start = time.time()
print("nvidia:", snapshot_download(NV, revision=NV_REV), f"{time.time() - start:.0f}s", flush=True)

# The RadixArk repo also carries ~800 per-layer audit files that the loader never reads.
ignore = ["layer-*"]
if mode == "--full-radixark":
    allow = None
else:
    index = hf_hub_download(RA, "model.safetensors.index.json", revision=RA_REV)
    shards = subprocess.check_output([sys.executable, "-I", mtp_shards, index], text=True).split()
    allow = shards + ["*.json", "*.jinja", "*.txt"]
start = time.time()
path = snapshot_download(RA, revision=RA_REV, allow_patterns=allow, ignore_patterns=ignore)
print("radixark:", path, f"{time.time() - start:.0f}s", flush=True)

# sglang refuses a local checkpoint whose index names shards that are not on disk, so the draft gets its own
# directory with an index trimmed to the downloaded shards.
draft_dir = "/workspace/radixark-mtp"
os.makedirs(draft_dir, exist_ok=True)
for name in os.listdir(path):
    dst = os.path.join(draft_dir, name)
    if name != "model.safetensors.index.json" and not os.path.lexists(dst):
        os.symlink(os.path.realpath(os.path.join(path, name)), dst)
with open(os.path.join(path, "model.safetensors.index.json")) as f:
    index = json.load(f)
if allow is not None:
    index["weight_map"] = {k: v for k, v in index["weight_map"].items() if v in shards}
with open(os.path.join(draft_dir, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f)
print("draft dir:", draft_dir, len(set(index["weight_map"].values())), "shards", flush=True)

# The server runs with HF_HUB_OFFLINE=1 and an explicit --revision; a refs/main entry covers any code path that
# resolves the default revision instead.
ref = os.path.join(os.environ["HF_HOME"], "hub", "models--" + NV.replace("/", "--"), "refs", "main")
if not os.path.exists(ref):
    os.makedirs(os.path.dirname(ref), exist_ok=True)
    with open(ref, "w") as f:
        f.write(NV_REV)
EOF
echo "setup done"
