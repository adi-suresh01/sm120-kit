#!/usr/bin/env bash
# Checks the image and both patches, and downloads Qwen3-30B-A3B-NVFP4 to container disk, so paid variant time goes to
# the boots.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
here=$(cd "$(dirname "$0")" && pwd)
pod="$here/../../harness/pod"
model=nvidia/Qwen3-30B-A3B-NVFP4
revision=2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3
want_sglang=7e31007a
# fi-rank-keys.diff is cut against this release's fused_moe/core.py.
want_flashinfer=0.7.4
hub=/root/hf-hub
# The download's chunk cache would otherwise sit under HF_HOME, on the 20 GB volume.
export HF_XET_CACHE=/root/hf-xet

python3 - "$want_sglang" "$want_flashinfer" <<'EOF'
import os
import sys
from importlib.metadata import version
import torch
commit = os.environ.get("SGLANG_BUILD_COMMIT") or version("sglang")
flashinfer = version("flashinfer-python")
print("sglang", version("sglang"), "commit", commit, "flashinfer", flashinfer, "torch", torch.__version__,
      "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
assert sys.argv[1] in commit, f"want sglang at {sys.argv[1]}, got {commit}"
assert flashinfer == sys.argv[2], f"want flashinfer {sys.argv[2]}, got {flashinfer}"
EOF

# A patch that no longer applies would otherwise surface only at its first variant, after the stock boots were paid for.
for spec in "pr40420 sglang 2" "fi-rank-keys flashinfer 1"; do
    read -r name pkg strip <<<"$spec"
    if ! state=$(bash "$pod/patch_pkg.sh" status "$here/patches/$name.diff" "$pkg" "$strip") \
        || [[ $state != "not applied" ]]; then
        echo "patch $name does not apply cleanly to the stock $pkg" >&2
        exit 1
    fi
    echo "patch $name applies to the stock $pkg"
done

# pkg_snapshot.sh copies sglang (6.6k files) and flashinfer (14.7k files, 713 MB) back before every variant. From the
# FUSE /workspace the sglang copy alone took 81 s per variant on 10-23, which 14 variants cannot afford, so the
# pristine copies live on container disk. Every later job sources container_env.sh.
echo 'export PKG_SNAPSHOT_DIR=/root/pkg-snapshots' >>/root/container_env.sh

# Whatever the image ships here is wiped by every cold variant; the listing records what that was.
autotune=${SGLANG_CACHE_DIR:-$HOME/.cache/sglang}/flashinfer/autotune
echo "shipped autotune cache:"
find "$autotune" -type f 2>/dev/null | head -20 || true

if [[ -e $HF_HOME/hub && ! -L $HF_HOME/hub ]]; then
    echo "$HF_HOME/hub is a directory, not the link to $hub; move it away or start a new pod" >&2
    exit 1
fi
mkdir -p "$hub" "$HF_HOME" "$HF_XET_CACHE"
ln -sfn "$hub" "$HF_HOME/hub"

start=$(date +%s)
HF_XET_HIGH_PERFORMANCE=1 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$model', revision='$revision', max_workers=16))" &
dl=$!
# 18.1 GB at 60 MB/s is 5 min, the most setup can spend inside planned_hours.
min_mb_s=60
for _ in $(seq 18); do
    kill -0 "$dl" 2>/dev/null || break
    sleep 5
done
if kill -0 "$dl" 2>/dev/null; then
    # hf_xet stages chunks in its own cache before it writes files under the hub, so both count as progress.
    got=$(du -sbc "$hub" "$HF_XET_CACHE" | tail -1 | cut -f1)
    rate=$(( got / ($(date +%s) - start) / 1000000 ))
    echo "download rate ${rate} MB/s after 90 s"
    if (( rate < min_mb_s )); then
        kill "$dl"
        echo "download at ${rate} MB/s is under ${min_mb_s} MB/s; run harness down" >&2
        exit 1
    fi
fi
wait "$dl"
echo "weights downloaded in $(( $(date +%s) - start )) s"

# The server runs with HF_HUB_OFFLINE=1 and an explicit --revision; a refs/main entry covers any code path that
# resolves the default revision instead.
ref="$hub/models--${model//\//--}/refs/main"
[[ -e $ref ]] || { mkdir -p "$(dirname "$ref")"; printf '%s' "$revision" >"$ref"; }
HF_HUB_OFFLINE=1 python3 -c "import sys; from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1], revision=sys.argv[2]); print('offline config load ok')" "$model" "$revision"
du -sh "$hub"
echo "setup done"
