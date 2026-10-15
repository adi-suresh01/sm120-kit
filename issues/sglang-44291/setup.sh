#!/usr/bin/env bash
# Checks what the stage 1 matrix depends on and pre-downloads models so paid time goes to the matrix.
# Usage: setup.sh [--with-olmoe]   (OLMoE is only needed if S1 and S2 both serve; see README)
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the volume disk}"

# v1 is only selected when the GPUs can reach each other; "OK" in every off-diagonal cell is required.
nvidia-smi topo -p2p r || echo "nvidia-smi topo -p2p r is not supported on this host"

# An allocator setting baked into the image would change every variant; run.sh and ar_matrix.sh override it.
env | grep -E '^PYTORCH(_CUDA)?_ALLOC_CONF=' || echo "no allocator conf preset in the image"

# VMM detection in ar_repro.py and in the fix goes through cuda.bindings.
python3 -c "import cuda.bindings.driver; print('cuda.bindings ok')"

# ar_repro.py and the regression test use these sglang internals; an image older than the fix's base lacks them.
python3 -c "from sglang.srt.runtime_context import get_parallel; from sglang.srt.utils.cuda_vmm_utils import is_vmm_pointer; print('sglang internals ok')"

models=(Qwen/Qwen3-0.6B)
if [[ ${1:-} == --with-olmoe ]]; then
    models+=(allenai/OLMoE-1B-7B-0924)
fi
for m in "${models[@]}"; do
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$m')"
done
echo "setup done"
