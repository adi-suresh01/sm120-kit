#!/usr/bin/env bash
# Pins the reporter's stack (vLLM 0.31.0 -> torch 2.13.0, flashinfer-python 0.7.0.post1) and pre-downloads
# the model so paid GPU time is spent on the repro, not on installs.
# Requires: HF_HOME on a persistent volume (weights are ~20 GB).
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the persistent volume}"
model=nvidia/Qwen3.6-35B-A3B-NVFP4

# patch.sh needs patch(1); slim images such as vllm/vllm-openai do not ship it.
command -v patch >/dev/null || { apt-get update && apt-get install -y patch; }

# Skip when already pinned: on the vllm/vllm-openai:v0.31.0 image a reinstall swaps NCCL 2.30.7 for 2.29.7,
# which moves the environment away from the reporter's.
installed=$(python3 -c 'from importlib.metadata import version; print(version("vllm"))' 2>/dev/null || true)
if [[ $installed != 0.31.0 ]]; then
    python3 -m pip install "vllm==0.31.0"
fi

python3 - <<'EOF'
from importlib.metadata import version
expected = {"vllm": "0.31.0", "flashinfer-python": "0.7.0.post1", "torch": "2.13.0"}
for pkg, want in expected.items():
    got = version(pkg)
    flag = "" if got.startswith(want) else "  <-- differs from the reporter"
    print(f"{pkg}=={got}{flag}")
EOF

# Python API instead of the CLI: huggingface_hub 1.x renamed huggingface-cli to hf.
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$model')"
echo "setup done"
