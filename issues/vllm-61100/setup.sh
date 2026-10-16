#!/usr/bin/env bash
# Checks the image is the reporter's v0.25.1, builds the package overlays the version-swap variants point
# PYTHONPATH at, and downloads the model, so paid variant time goes to the matrix.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
model=RedHatAI/Apertus-70B-Instruct-2509-FP8-dynamic
revision=79a917d6a8034910fa610b097dc61e96eac65d29
ovl=/workspace/ovl

python3 - <<'EOF'
from importlib.metadata import version
import torch
for p in ("vllm", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache", "compressed-tensors",
          "transformers"):
    print(p, version(p))
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
assert version("vllm") == "0.25.1", "the repro is on the reporter's v0.25.1 image"
EOF
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Overlays instead of installs: the harness only restores patched package folders between variants, so a pip
# install over the image would leak into every later variant. --no-deps keeps the image's torch 2.11.0, which
# vLLM 0.24.0 and 0.25.1 both pin. FlashInfer checks that its three packages match, so all three go together.
rm -rf "$ovl"
python3 -m pip install -q --no-deps --target "$ovl/fi0612" flashinfer-python==0.6.12 flashinfer-cubin==0.6.12
python3 -m pip install -q --no-deps --target "$ovl/fi0612" --index-url https://flashinfer.ai/whl/cu130 \
    "flashinfer-jit-cache==0.6.12+cu130"
python3 -m pip install -q --no-deps --target "$ovl/v0240" vllm==0.24.0
PYTHONPATH="$ovl/v0240:$ovl/fi0612" python3 - <<'EOF'
from importlib.metadata import version
import vllm, flashinfer
print("overlay vllm", version("vllm"), vllm.__file__, "flashinfer", version("flashinfer-python"))
assert version("vllm") == "0.24.0" and version("flashinfer-python") == "0.6.12"
EOF

start=$(date +%s)
HF_XET_HIGH_PERFORMANCE=1 python3 -c "from huggingface_hub import snapshot_download as d
print(d('$model', revision='$revision', max_workers=16))"
echo "weights downloaded in $(( $(date +%s) - start )) s"
echo "setup done"
