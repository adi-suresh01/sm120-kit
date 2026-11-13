#!/usr/bin/env bash
# Everything before the first boot, cheapest check first: 4 GPUs, no inherited NCCL or all-reduce setting, /dev/shm
# for NCCL, Qwen3-8B at the pinned revision, then ws_probe.py, which measures FlashInfer's PCIe IPC workspace at the
# arms' shapes and fails the box when the workspace cannot be built or its tune hangs, before an arm pays for that. A
# non-zero exit makes `harness watch` destroy the box and rent another.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
deadline=${HARNESS_DEADLINE_EPOCH:?the harness sets HARNESS_DEADLINE_EPOCH}
here=$(cd "$(dirname "$0")" && pwd)
kit=$(cd "$here/../.." && pwd)
out="$kit/results/vllm-ipc-kv-sizing"
model=Qwen/Qwen3-8B
revision=b968826d9c46dd6066d109eabc6255188de91218
# The seven arms and down take about 32 min (README.md, Box and hours); a download still running 38 min before the
# deadline would cost the last arms.
arms_min=38
# Weights on container disk through a link at HF_HOME/hub: on RunPod /workspace is a FUSE mount, and on Vast both
# paths are the same disk (issues/vllm-61530/setup.sh). The download's chunk cache would otherwise fill the volume.
hub=/root/hf-hub
export HF_XET_CACHE=/root/hf-xet
mkdir -p "$out/ws_probe" "$HF_HOME" "$hub"

# setup.log keeps the reason; watch destroys the box and stays off its machine.
fail() {
    echo "$1" >&2
    exit 1
}

if [[ -e $HF_HOME/hub && ! -L $HF_HOME/hub ]]; then
    fail "$HF_HOME/hub is a directory, not the link to $hub; move it away or start a new box"
fi
ln -sfn "$hub" "$HF_HOME/hub"

gpus=$(timeout 60 nvidia-smi -L | grep -c '^GPU ' || true)
if [[ $gpus != 4 ]]; then
    fail "this matrix needs 4 GPUs; nvidia-smi lists ${gpus:-none}"
fi
# The image is pinned by digest; this only puts the build in setup.log.
python3 -I - <<'EOF'
from importlib.metadata import version

import torch

print("vllm", version("vllm"), "flashinfer", version("flashinfer-python"), "torch", torch.__version__,
      "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
EOF

# A setting the image or host exports would leak into every arm and blur what each one changes. NCCL_VERSION is the
# CUDA base image's label, not the NCCL torch loads.
inherited=$(env | grep -E '^(NCCL_|VLLM_ALLREDUCE_|PYTORCH_CUDA_ALLOC_CONF=|PYTORCH_ALLOC_CONF=)' \
    | grep -v '^NCCL_VERSION=' || true)
if [[ -n $inherited ]]; then
    fail "the container already sets: $(tr '\n' ' ' <<<"$inherited")"
fi

# The off arms run NCCL, which falls back to SHM in /dev/shm where P2P is not used; Docker's default is 64 MB, and the
# harness's Vast create asks for no size (issues/sglang-33422).
shm_bytes=$(df -B1 --output=size /dev/shm | tail -n 1 | tr -d ' ')
echo "/dev/shm $(( shm_bytes / 1048576 )) MiB"
if (( shm_bytes < 2 * 1024 * 1024 * 1024 )); then
    fail "/dev/shm is under 2 GiB, too small for NCCL's SHM transport"
fi

limit_s=$(( deadline - $(date +%s) - arms_min * 60 ))
if (( limit_s < 60 )); then
    fail "only $(( limit_s / 60 + arms_min )) min left before the deadline"
fi
start=$(date +%s)
if ! timeout "$limit_s" python3 -I - "$model" "$revision" <<'EOF'
import sys

from huggingface_hub import snapshot_download

snapshot_download(sys.argv[1], revision=sys.argv[2], max_workers=16)
EOF
then
    fail "the download failed or was still running ${arms_min} min before the deadline"
fi
echo "weights downloaded in $(( $(date +%s) - start )) s"

# ws_probe.py's per-step watchdog fires first; this limit catches a hang it cannot see.
if ! timeout -k 20 900 torchrun --standalone --nproc_per_node=4 "$here/ws_probe.py" --out "$out/ws_probe" \
    >"$out/ws_probe/ws_probe.log" 2>&1; then
    tail -n 30 "$out/ws_probe/ws_probe.log" >&2
    fail "ws_probe.py failed or hung; see results/vllm-ipc-kv-sizing/ws_probe/"
fi
# A worker torchrun lost track of can still hold its GPU, which would fail the first arm's idle check.
pkill -9 -f "$here/ws_probe.py" 2>/dev/null || true
grep -E '(warm|e|m): max_numel' "$out/ws_probe/ws_probe.log" || echo "ws_probe.py passed, but its summary did not print"
echo "setup done; $(( (deadline - $(date +%s)) / 60 )) min left to the deadline"
