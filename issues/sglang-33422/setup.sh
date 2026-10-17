#!/usr/bin/env bash
# Checks the image, builds the prompts, downloads DeepSeek-V4-Flash-0731 to container disk, drops it from the page
# cache, and checks a cold read is fast enough to boot from, so paid variant time goes to the matrix.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
here=$(cd "$(dirname "$0")" && pwd)
guard="$here/../../harness/pod/mem_guard.py"
model=deepseek-ai/DeepSeek-V4-Flash-0731
revision=7872f01b1d1fe23eabc4c98b48bffcef5a386062
# The topk=192 reading in README.md holds for this FlashInfer only; the fallback image (v0.5.21) ships 0.6.18. The
# digest pins sglang itself, whose version string an image built without git metadata reads as 0.0.0.dev0.
want_flashinfer=0.7.1
hub=/root/hf-hub
# The download's chunk cache would otherwise sit under HF_HOME, on the 20 GB volume.
export HF_XET_CACHE=/root/hf-xet

python3 - "$want_flashinfer" <<'EOF'
import sys
from importlib.metadata import version
import torch
print("sglang", version("sglang"), "flashinfer", version("flashinfer-python"), "torch", torch.__version__,
      "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
assert version("flashinfer-python") == sys.argv[1], f"want FlashInfer {sys.argv[1]}"
EOF
# env.txt has the current PCIe link, which reads low at idle; the sglang and vLLM pods are compared on the maximum
# link and the power limit (README.md).
timeout 60 nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv
echo "container memory limit: $(python3 -I "$guard" --limit)"
df -h /workspace /root

# The weights go on container disk: /workspace on secure pods is a FUSE mount, and vLLM spent 7m35s opening the
# vllm-56771 shards there (issues/vllm-56771/README.md).
if [[ -e $HF_HOME/hub && ! -L $HF_HOME/hub ]]; then
    echo "$HF_HOME/hub is a directory, not the link to $hub; move it away or start a new pod" >&2
    exit 1
fi
mkdir -p "$hub" "$HF_HOME"
ln -sfn "$hub" "$HF_HOME/hub"

# Prompts first, so a builder failure stops setup before the 167 GB download.
python3 -I "$here/build_prompts.py" "$model" "$revision" /workspace/prompts

before=$(du -sb "$hub" | cut -f1)
start=$(date +%s)
HF_XET_HIGH_PERFORMANCE=1 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$model', revision='$revision', max_workers=16))" &
dl=$!
# Some hosts reach HF at 15 MB/s (EUR-IS-2 on 10-16); under min_mb_s the download cannot finish inside the budget,
# so stop at 3 min.
min_mb_s=400
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

# Each variant boots from a cold read of the whole checkpoint; under min_read_mb_s that eats the variant budget, so
# the largest shard is read once now, from a cache just dropped, and dropped again after.
min_read_mb_s=500
shard=$(find -L "$hub" -type f -name '*.safetensors' -printf '%s %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
size=$(stat -L -c %s "$shard")
t0=$(date +%s%N)
dd if="$shard" of=/dev/null bs=16M status=none
read_mb_s=$(( size * 1000 / ($(date +%s%N) - t0) ))
python3 -I "$guard" --drop-cache "$shard"
echo "cold read of $(basename "$shard") ($(( size / 1000000 )) MB): ${read_mb_s} MB/s"
if (( read_mb_s < min_read_mb_s )); then
    echo "a cold read at ${read_mb_s} MB/s is under ${min_read_mb_s} MB/s, too slow to boot; run harness down" >&2
    exit 1
fi
du -sh "$hub"
echo "setup done"
