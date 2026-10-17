#!/usr/bin/env bash
# Puts hermia's FlashInfer 0.6.14 into the image, builds the prompts, downloads DeepSeek-V4-Flash-0731 to container
# disk, drops it from the page cache, and checks a cold read is fast enough to boot from, so paid variant time goes
# to the run.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
here=$(cd "$(dirname "$0")" && pwd)
guard="$here/../../harness/pod/mem_guard.py"
model=deepseek-ai/DeepSeek-V4-Flash-0731
revision=7872f01b1d1fe23eabc4c98b48bffcef5a386062
hub=/root/hf-hub
# The download's chunk cache would otherwise sit under HF_HOME, on the 20 GB volume.
export HF_XET_CACHE=/root/hf-xet

# vLLM 0.25.1 pins FlashInfer 0.6.13, but its sm_120 DSV4 sparse-MLA decode calls an API only 0.6.14 has (hermia's
# setup-venv.sh). It goes in with its dependencies, as hermia installed it (cuda-tile>=1.4.0 is the one change;
# torch stays at the image's). The jit-cache of the same version spares the boot compiling every kernel;
# flashinfer-cubin has no 0.6.14, so it stays at 0.6.13 as in hermia's venv. One variant runs here, so installing
# over the image leaks into nothing.
pip_install() {
    python3 -m pip install --quiet --break-system-packages "$@" || python3 -m pip install --quiet "$@"
}
pip_install flashinfer-python==0.6.14
pip_install --no-deps --index-url https://flashinfer.ai/whl/cu130 "flashinfer-jit-cache==0.6.14+cu130"

python3 - <<'EOF'
from importlib.metadata import version
import torch
print("vllm", version("vllm"), "flashinfer", version("flashinfer-python"), "cubin", version("flashinfer-cubin"),
      "jit-cache", version("flashinfer-jit-cache"), "cuda-tile", version("cuda-tile"), "torch", torch.__version__,
      "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
assert version("vllm") == "0.25.1", "hermia's recipe is vLLM 0.25.1"
assert torch.__version__.startswith("2.11.0"), "the FlashInfer install moved torch off vLLM 0.25.1's pin"
assert version("flashinfer-python") == "0.6.14", "hermia's recipe is FlashInfer 0.6.14"
assert version("flashinfer-jit-cache").startswith("0.6.14"), "a jit-cache of another version would load stale kernels"
EOF
# The MHC-prenorm TileLang kernel compiles at load, and the boot fails without nvcc.
if ! command -v nvcc >/dev/null; then
    echo "nvcc is not on PATH ($PATH); the boot would fail compiling the MHC-prenorm kernel" >&2
    exit 1
fi
nvcc --version | tail -n 1
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

# The boot reads the whole checkpoint cold; under min_read_mb_s that eats the run's budget, so the largest shard is
# read once now, from a cache just dropped, and dropped again after.
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
