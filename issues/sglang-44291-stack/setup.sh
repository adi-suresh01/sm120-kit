#!/usr/bin/env bash
# Checks the image, checks every patch applies, and downloads Qwen3.8-Flash-Next-NVFP4 to container disk, so paid
# variant time goes to the matrix.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
here=$(cd "$(dirname "$0")" && pwd)
guard="$here/../../harness/pod/mem_guard.py"
model=nvidia/Qwen3.8-Flash-Next-NVFP4
revision=fc694b54fb0174e0913e6adf86691ef85a4ead47
# The patches are diffs from the stack's base; they were checked against this commit only.
want_sglang=3c560089
hub=/root/hf-hub
# The download's chunk cache would otherwise sit under HF_HOME, on the 20 GB volume.
export HF_XET_CACHE=/root/hf-xet

python3 - "$want_sglang" <<'EOF'
import os
import sys
from importlib.metadata import version
import torch
v = os.environ.get("SGLANG_BUILD_COMMIT") or version("sglang")
print("sglang", version("sglang"), "commit", v, "sglang-kernel", version("sglang-kernel"), "flashinfer",
      version("flashinfer-python"), "torch", torch.__version__, "cuda", torch.version.cuda,
      "gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
assert sys.argv[1] in v, f"want sglang at {sys.argv[1]}, got {v}"
EOF

# A patch that no longer applies would otherwise surface per variant, after a paid boot of the one before it.
site_root=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("sglang").origin)))')
for p in "$here"/patches/*.diff; do
    patch -p2 --dry-run -f -s -d "$site_root" <"$p" >/dev/null || { echo "$(basename "$p") does not apply" >&2; exit 1; }
done
echo "patches apply to $site_root"
# The pinned PLE table alone is 47.7 GiB (stage2.md).
read -r limit _ < <(python3 -I "$guard" --limit)
echo "container memory limit: $limit"
if [[ $limit =~ ^[0-9]+$ ]] && (( limit < 128 * 1024 ** 3 )); then
    echo "container memory limit under 128 GiB; run harness down" >&2
    exit 1
fi
df -h /workspace /root

if [[ -e $HF_HOME/hub && ! -L $HF_HOME/hub ]]; then
    echo "$HF_HOME/hub is a directory, not the link to $hub; move it away or start a new pod" >&2
    exit 1
fi
mkdir -p "$hub" "$HF_HOME"
ln -sfn "$hub" "$HF_HOME/hub"

# The variants export HF_HUB_OFFLINE=1, which bench_serving's random dataset also sees: it samples ShareGPT through
# hf_hub_download, so the file has to be cached now.
sharegpt=(anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json)
python3 -c "import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(sys.argv[1], sys.argv[2], repo_type='dataset'))" "${sharegpt[@]}"

before=$(du -sb "$hub" | cut -f1)
start=$(date +%s)
HF_XET_HIGH_PERFORMANCE=1 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$model', revision='$revision', max_workers=16))" &
dl=$!
# 132.7 GB at 140 MB/s is 16 min, the most setup can spend inside planned_hours.
min_mb_s=140
for _ in $(seq 36); do
    kill -0 "$dl" 2>/dev/null || break
    sleep 5
done
if kill -0 "$dl" 2>/dev/null; then
    rate=$(( ($(du -sb "$hub" | cut -f1) - before) / ($(date +%s) - start) / 1000000 ))
    echo "download rate ${rate} MB/s after 3 min"
    if (( rate < min_mb_s )); then
        kill "$dl"
        echo "download at ${rate} MB/s is under ${min_mb_s} MB/s; run harness down" >&2
        exit 1
    fi
fi
wait "$dl"
echo "weights downloaded in $(( $(date +%s) - start )) s"
# The download's page cache counts against RunPod's container memory limit (docs/runpod.md).
python3 -I "$guard" --drop-cache "$HF_HOME"

# The server runs with HF_HUB_OFFLINE=1 and an explicit --revision; a refs/main entry covers any code path that
# resolves the default revision instead.
ref="$hub/models--${model//\//--}/refs/main"
[[ -e $ref ]] || { mkdir -p "$(dirname "$ref")"; printf '%s' "$revision" >"$ref"; }
# Both client-side loads the quality step makes offline, proven before any paid boot.
HF_HUB_OFFLINE=1 python3 - "$model" "${sharegpt[@]}" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(sys.argv[1])
hf_hub_download(sys.argv[2], sys.argv[3], repo_type="dataset")
print("offline tokenizer and ShareGPT load ok")
EOF
du -sh "$hub"
echo "setup done"
