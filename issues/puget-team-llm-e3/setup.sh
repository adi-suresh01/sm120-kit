#!/usr/bin/env bash
# Records the image, the host's GPU peer-to-peer support, and its GPU clocks, then downloads the model to container
# disk, so paid variant time goes to the matrix.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
here=$(cd "$(dirname "$0")" && pwd)
guard="$here/../../harness/pod/mem_guard.py"
model=cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit
revision=96bde6d5fb87e572cd5e9843fda6af22811ee23f
hub=/root/hf-hub
# The download's chunk cache would otherwise sit under HF_HOME, on the 20 GB volume.
export HF_XET_CACHE=/root/hf-xet

# Recorded so the PR can say whether these hosts offer GPU peer-to-peer at all; this is the same CUDA query
# vLLM's custom all-reduce starts from.
python3 - <<'EOF'
from importlib.metadata import version
import torch
print("vllm", version("vllm"), "flashinfer", version("flashinfer-python"), "torch", torch.__version__,
      "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
print("cuda peer access 0->1:", torch.cuda.can_device_access_peer(0, 1), "1->0:", torch.cuda.can_device_access_peer(1, 0))
EOF
timeout 60 nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv
# A 10-23 host ran every variant about 2x slower than another with nothing recorded to name why. Idle maximum and
# application clocks and clock-event reasons for both GPUs; hooks.sh samples them under load.
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
# 24.5 GB of safetensors at 100 MB/s is about 4 min; slower than that eats the margin planned_hours keeps for a slow
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
