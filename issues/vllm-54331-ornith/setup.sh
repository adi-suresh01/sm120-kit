#!/usr/bin/env bash
# Records the image, writes the soak's video clips, then downloads the model to container disk, so paid variant time
# goes to the soak.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
here=$(cd "$(dirname "$0")" && pwd)
guard="$here/../../harness/pod/mem_guard.py"
model=ornith-ai/Ornith-1.5-35B-A3B-NVFP4
revision=94e431d9cc47fa1986a7a1a4e9a80f7f118b03aa
hub=/root/hf-hub
# The download's chunk cache would otherwise sit under HF_HOME, on the 20 GB volume.
export HF_XET_CACHE=/root/hf-xet

python3 - <<'EOF'
from importlib.metadata import version
import torch
print("vllm", version("vllm"), "flashinfer", version("flashinfer-python"), "torch", torch.__version__,
      "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
EOF
timeout 60 nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv

# 200 distinct clips, so the encoder keeps seeing new videos for the whole soak (soak_video.py). Each clip has its own
# noise and motion: identical pixels would hash to one multimodal cache entry. 96 frames, because vLLM's OpenCV loader
# samples 32: a clip of exactly 32 frames is passed on as unsampled, and the Qwen3-VL processor then resamples it at
# 2 fps down to 8 frames. A real video takes the 32-frame path, and so do these.
python3 - /root/videos <<'EOF'
import sys
from pathlib import Path
import cv2
import numpy as np
from vllm.multimodal.video import VIDEO_LOADER_REGISTRY

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
w, h, frames = 640, 480, 96
for i in range(200):
    rng = np.random.default_rng(i)
    path = out / f"clip{i:03d}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    base = cv2.resize(rng.integers(0, 256, (60, 80, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    x, y = rng.integers(0, w - 80), rng.integers(0, h - 80)
    dx, dy = rng.integers(-12, 13, 2)
    color = tuple(int(c) for c in rng.integers(0, 256, 3))
    for _ in range(frames):
        frame = base.copy()
        x, y = int(np.clip(x + dx, 0, w - 80)), int(np.clip(y + dy, 0, h - 80))
        cv2.rectangle(frame, (x, y), (x + 80, y + 80), color, -1)
        writer.write(frame)
    writer.release()
# Decoded the way the server will: from bytes, through the loader it uses by default.
got, meta = VIDEO_LOADER_REGISTRY.load("opencv").load_bytes((out / "clip000.mp4").read_bytes(), num_frames=32)
assert got.shape[0] == 32 and not meta["do_sample_frames"], f"loader gave {got.shape[0]} frames, {meta}"
print(f"wrote 200 clips of {frames} frames to {out}; the loader samples {got.shape[0]}, shape {got.shape}")
EOF
du -sh /root/videos
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
# 23 GB at 100 MB/s is 4 min; slower than that and the soak variants lose their time.
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
