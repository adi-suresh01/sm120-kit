#!/usr/bin/env bash
# Records the pair's peer access as stage 1 does (this is another box), then downloads the model to container disk,
# so paid variant time goes to the matrix.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
here=$(cd "$(dirname "$0")" && pwd)
guard="$here/../../harness/pod/mem_guard.py"
model=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
revision=bee7596271d1495f6992ae224aefde4410e816b8
hub=/root/hf-hub
# The download's chunk cache would otherwise sit under HF_HOME, on the 20 GB volume.
export HF_XET_CACHE=/root/hf-xet

# The variants set PYTORCH_CUDA_ALLOC_CONF, and torch reads PYTORCH_ALLOC_CONF before it: one set in the container
# would hold every variant at the same allocator setting.
if [[ -n ${PYTORCH_ALLOC_CONF:-} ]]; then
    echo "PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF in the container env overrides the variants' allocator setting" >&2
    exit 1
fi

timeout 60 nvidia-smi --query-gpu=index,name,power.limit,pcie.link.gen.max,pcie.link.width.max --format=csv
timeout 60 nvidia-smi topo -p2p w || true
python3 - <<'EOF' || echo "peer access record failed: exit $?" >&2
import os
import torch
from vllm.distributed.device_communicators.all_reduce_utils import gpu_p2p_access_check
print("PYTORCH_CUDA_ALLOC_CONF in the image env:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
print("can_device_access_peer 0->1:", torch.cuda.can_device_access_peer(0, 1),
      "1->0:", torch.cuda.can_device_access_peer(1, 0))
print("vllm IPC probe 0->1:", gpu_p2p_access_check(0, 1), "1->0:", gpu_p2p_access_check(1, 0))
EOF
echo "container memory limit: $(python3 -I "$guard" --limit)"

# /workspace is a FUSE mount on RunPod's secure pods, too slow to boot from; the weights go to container disk.
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
# 21.6 GB at 80 MB/s is 4.5 min, the most setup can spend inside planned_hours.
min_mb_s=80
for _ in $(seq 24); do
    kill -0 "$dl" 2>/dev/null || break
    sleep 5
done
if kill -0 "$dl" 2>/dev/null; then
    rate=$(( ($(du -sb "$hub" | cut -f1) - before) / ($(date +%s) - start) / 1000000 ))
    echo "download rate ${rate} MB/s after 2 min"
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
