#!/usr/bin/env bash
# Records what the recipe's "no P2P fabric" reading turns on, runs the allocator probe under both settings, then
# downloads the model, so paid variant time goes to the matrix.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
here=$(cd "$(dirname "$0")" && pwd)
model=Qwen/Qwen3-0.6B
revision=c1899de289a04d12100db370d81485cdf75e47ca

# The variants set PYTORCH_CUDA_ALLOC_CONF, and torch reads PYTORCH_ALLOC_CONF before it: one set in the container
# would hold every variant at the same allocator setting.
if [[ -n ${PYTORCH_ALLOC_CONF:-} ]]; then
    echo "PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF in the container env overrides the variants' allocator setting" >&2
    exit 1
fi

# Max-Q (300 W) or a 600 W edition, and the link each GPU trained at. The harness keeps `topo -p2p r` itself.
timeout 60 nvidia-smi --query-gpu=index,name,power.limit,pcie.link.gen.max,pcie.link.width.max --format=csv
timeout 60 nvidia-smi topo -p2p w || true
python3 - <<'EOF' || echo "peer access record failed: exit $?" >&2
import os
import torch
from vllm.distributed.device_communicators.all_reduce_utils import gpu_p2p_access_check
print("PYTORCH_CUDA_ALLOC_CONF in the image env:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
# What vLLM's custom all-reduce gate reads at its default VLLM_SKIP_P2P_CHECK=1.
print("can_device_access_peer 0->1:", torch.cuda.can_device_access_peer(0, 1),
      "1->0:", torch.cuda.can_device_access_peer(1, 0))
# The gate's VLLM_SKIP_P2P_CHECK=0 path: a real cudaIpcOpenMemHandle and peer write between two processes, on plain
# cudaMalloc memory. Passing here means the fabric carries CUDA IPC.
print("vllm IPC probe 0->1:", gpu_p2p_access_check(0, 1), "1->0:", gpu_p2p_access_check(1, 0))
EOF
for conf in expandable_segments:True expandable_segments:False; do
    PYTORCH_CUDA_ALLOC_CONF=$conf python3 "$here/vmm_ipc_probe.py" || echo "vmm_ipc_probe $conf: exit $?" >&2
done

python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$model', revision='$revision'))"
echo "setup done"
