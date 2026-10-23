#!/usr/bin/env bash
# Records the image and the host's GPU peer-to-peer support, then downloads the model to container disk, so paid
# variant time goes to the matrix.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
here=$(cd "$(dirname "$0")" && pwd)
guard="$here/../../harness/pod/mem_guard.py"
model=cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit
revision=be51bea8252b091475d3c525fc00461f5c29132e
hub=/root/hf-hub
# The download's chunk cache would otherwise sit under HF_HOME, on the 20 GB volume.
export HF_XET_CACHE=/root/hf-xet

# graphs-p2p only means something where the two GPUs can reach each other directly; this is the same CUDA query
# vLLM's custom all-reduce starts from.
python3 - <<'EOF'
from importlib.metadata import version
import torch
print("vllm", version("vllm"), "flashinfer", version("flashinfer-python"), "torch", torch.__version__,
      "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
print("cuda peer access 0->1:", torch.cuda.can_device_access_peer(0, 1), "1->0:", torch.cuda.can_device_access_peer(1, 0))
EOF
timeout 60 nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv

# The 10-23 recheck ran about 2x slower on a host with both GPUs behind one PCIe switch, so measure the all-reduce
# path itself, with P2P allowed and disabled; NCCL_DEBUG names the transport it picked. A failure here is recorded,
# not fatal: the variants still answer the question.
for p2p in default 1; do
    if [[ $p2p == default ]]; then
        out=$(env -u NCCL_P2P_DISABLE NCCL_DEBUG=INFO timeout 300 torchrun --standalone --nproc_per_node=2 "$here/allreduce_bw.py" 2>&1) || true
    else
        out=$(NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO timeout 300 torchrun --standalone --nproc_per_node=2 "$here/allreduce_bw.py" 2>&1) || true
    fi
    echo "NCCL_P2P_DISABLE=$p2p transports: $(grep -o 'via [A-Za-z/]*' <<<"$out" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    grep '^allreduce ' <<<"$out" || { echo "all-reduce check failed with NCCL_P2P_DISABLE=$p2p:"; tail -15 <<<"$out"; }
done
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
# 80 GB at 100 MB/s is 13 min, the most setup can spend inside planned_hours.
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
