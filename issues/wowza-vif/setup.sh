#!/usr/bin/env bash
# Records the image and the host's GPU, then downloads the model to container disk, so paid variant time goes to the
# matrix.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
here=$(cd "$(dirname "$0")" && pwd)
guard="$here/../../harness/pod/mem_guard.py"
model=nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8
revision=e9550fdec09682d12e8c3e41548b2e05b16db7c1
hub=/root/hf-hub
# The download's chunk cache would otherwise sit under HF_HOME, on the 20 GB volume.
export HF_XET_CACHE=/root/hf-xet

# transformers too: --trust-remote-code loads the checkpoint's own config classes with the image's version.
python3 - <<'EOF'
from importlib.metadata import version
import torch
print("vllm", version("vllm"), "flashinfer", version("flashinfer-python"), "torch", torch.__version__,
      "cuda", torch.version.cuda, "transformers", version("transformers"),
      "gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
EOF
timeout 60 nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv
# A 10-23 host ran every variant about 2x slower than another with nothing recorded to name why. Idle clocks and
# clock-event reasons here; hooks.sh samples them under load.
timeout 60 nvidia-smi -q -d CLOCK,PERFORMANCE
# Hosts differ within one GPU type, so the data center goes with the numbers.
echo "data center: ${RUNPOD_DC_ID:-not set}"
echo "container memory limit: $(python3 -I "$guard" --limit)"
df -h /workspace /root

if [[ -e $HF_HOME/hub && ! -L $HF_HOME/hub ]]; then
    echo "$HF_HOME/hub is a directory, not the link to $hub; move it away or start a new pod" >&2
    exit 1
fi
mkdir -p "$hub" "$HF_HOME"
ln -sfn "$hub" "$HF_HOME/hub"

before=$(du -sb "$hub" | cut -f1)
start=$(date +%s)
HF_XET_HIGH_PERFORMANCE=1 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$model', revision='$revision', max_workers=16))" &
dl=$!
# 15.4 GB of safetensors at 100 MB/s is under 3 min; slower than that eats the margin planned_hours keeps for a slow
# host.
min_mb_s=100
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
du -sh "$hub"
echo "setup done"
