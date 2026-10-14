#!/usr/bin/env bash
# Pins the reporter's stack (vLLM 0.31.0 -> torch 2.13.0, flashinfer-python 0.7.0.post1) and pre-downloads
# the model so paid GPU time is spent on the repro, not on installs.
# Requires: HF_HOME on a persistent volume (weights are ~20 GB).
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the persistent volume}"
model=nvidia/Qwen3.6-35B-A3B-NVFP4

python3 -m pip install --upgrade pip
python3 -m pip install "vllm==0.31.0" "huggingface_hub[cli]"

python3 - <<'EOF'
from importlib.metadata import version
expected = {"vllm": "0.31.0", "flashinfer-python": "0.7.0.post1", "torch": "2.13.0"}
for pkg, want in expected.items():
    got = version(pkg)
    flag = "" if got.startswith(want) else "  <-- differs from the reporter"
    print(f"{pkg}=={got}{flag}")
EOF

huggingface-cli download "$model"
echo "setup done"
