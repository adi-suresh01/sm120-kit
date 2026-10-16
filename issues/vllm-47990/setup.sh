#!/usr/bin/env bash
# Checks the image is the v0.31.0 the patches were rebased onto, fetches vLLM's own tests at that tag with PR #48588's
# tests on top, and downloads the model and GSM8K, so paid variant time goes to the matrix.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
here=$(cd "$(dirname "$0")" && pwd)
tests_dir=/workspace/vllm-tests
model=gaunernst/DeepSeek-V2-Lite-Chat-FP8
revision=c6f13916a378a8020921732bda65cddc9a3d13f6

python3 - <<'EOF'
from importlib.metadata import version
import torch
print("vllm", version("vllm"), "torch", torch.__version__, "cuda", torch.version.cuda,
      "gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
assert version("vllm").startswith("0.31.0"), "patches/ are rebased onto v0.31.0"
EOF

# For vLLM's own tests: tests/conftest.py imports tblib.
python3 -m pip install -q pytest tblib

# Sparse, so the checkout has no vllm/ source dir that pytest could import instead of the installed package.
rm -rf "$tests_dir"
git clone -q --depth 1 --branch v0.31.0 --filter=blob:none --sparse https://github.com/vllm-project/vllm.git "$tests_dir"
git -C "$tests_dir" sparse-checkout set --no-cone /tests/ /pyproject.toml
git -C "$tests_dir" apply "$here/pr48588-tests.diff"
# A collection failure only costs the pytest step, so it is reported here and does not stop the run.
(cd "$tests_dir" && python3 -m pytest -q --collect-only -p no:cacheprovider \
    tests/kernels/quantization/test_block_fp8.py -k cutlass | tail -n 3) || echo "WARNING: pytest collection failed"

mkdir -p /workspace/gsm8k
for split in train test; do
    curl -fsSL --retry 3 -o "/workspace/gsm8k/$split.jsonl" \
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/$split.jsonl"
done
wc -l /workspace/gsm8k/test.jsonl

# Python API instead of the CLI: huggingface_hub 1.x renamed huggingface-cli to hf.
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$model', revision='$revision'))"
echo "setup done"
