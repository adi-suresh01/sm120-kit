#!/usr/bin/env bash
# Checks the host can hold the model, then downloads DeepSeek-V4.1-Flash and builds the long prompts, so paid
# variant time goes to the matrix.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
here=$(cd "$(dirname "$0")" && pwd)
model=deepseek-ai/DeepSeek-V4.1-Flash
revision=314113f8a7dadb388215396ba5fbee2c610a04d6
min_ram_gb=256

python3 - <<'EOF'
from importlib.metadata import version
import torch
print("vllm", version("vllm"), "flashinfer", version("flashinfer-python"), "torch", torch.__version__,
      "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
assert version("flashinfer-python") == "0.7.2", "the retest is about FlashInfer 0.7.2 (FI #5197)"
EOF

# Engram's two lookup tables (about 189 GiB) go to pinned host memory at TP4; a smaller host fails late in the boot,
# after the download has been paid for.
ram_gb=$(free -g | awk '/^Mem:/ {print $2}')
echo "host RAM ${ram_gb} GB, memlock $(ulimit -l), shm $(df -h /dev/shm | awk 'NR==2 {print $2}')"
df -h /workspace
if (( ram_gb < min_ram_gb )); then
    echo "host RAM ${ram_gb} GB is under ${min_ram_gb} GB; run harness down" >&2
    exit 1
fi
mkdir -p /workspace/cores

# Tokenizer first, so the prompts are built (and checked) before the 510 GB download.
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$model', revision='$revision', allow_patterns=['tokenizer.json', 'tokenizer_config.json', 'config.json'])"
python3 -I "$here/build_prompts.py" "$model" "$revision" /workspace/prompts

start=$(date +%s)
HF_XET_HIGH_PERFORMANCE=1 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$model', revision='$revision', max_workers=16))"
echo "weights downloaded in $(( $(date +%s) - start )) s"
du -sh "$HF_HOME"
echo "setup done"
