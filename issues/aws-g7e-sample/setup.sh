#!/usr/bin/env bash
# Records the image and the host's GPU, installs vLLM 0.32.0 in a venv beside the image's 0.15.1, then downloads both
# checkpoints to container disk, so paid variant time goes to the matrix.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
here=$(cd "$(dirname "$0")" && pwd)
guard="$here/../../harness/pod/mem_guard.py"
hub=/root/hf-hub
# The download's chunk cache would otherwise sit under HF_HOME, on the 20 GB volume.
export HF_XET_CACHE=/root/hf-xet
# harness.toml's VLLM_PYTHON points the current and fp8 variants into this venv.
venv=/root/vllm-0.32.0
# The release's CUDA 12.9 build, the one vllm/vllm-openai:v0.32.0-cu129 runs: both stacks then share the image's CUDA
# 12.9 userspace, and the comparison is vLLM 0.15.1 against 0.32.0, not also CUDA 12.9 against 13.0.
wheel=vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
wheel_sha256=9918be806fb2808c406547ff4f9f43870ac6ff28dc4b179291fcf532145bb50b

# The FlashInfer kernel packages too: without them a server compiles FlashInfer kernels while it boots or serves.
versions() {
    "$1" - <<'EOF'
from importlib.metadata import PackageNotFoundError, version
import torch

def installed(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "absent"

print(*(f"{pkg} {installed(pkg)}" for pkg in ("vllm", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache",
                                              "transformers")),
      "torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(),
      torch.cuda.get_device_name(0))
EOF
}

# A stack whose torch cannot start CUDA on this host is a finding its variants record as a failed boot, so setup goes on
# to install and download what the other variants need.
versions python3 || echo "vllm 0.15.1's torch failed to start CUDA here"
timeout 60 nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv
# A 10-23 host ran every variant about 2x slower than another with nothing recorded to name why, and AWS's g7e runs at
# its own power cap. Idle clocks and clock-event reasons here; hooks.sh samples them under load.
timeout 60 nvidia-smi -q -d CLOCK,PERFORMANCE
# Hosts differ within one GPU type, so the data center goes with the numbers.
echo "data center: ${RUNPOD_DC_ID:-not set}"
# vllm/vllm-openai:v0.15.1 points ld.so at CUDA 12.9's forward-compat libcuda (vLLM #30784), which fails to load on
# newer drivers (#32373); AWS's image deletes that entry. This shows which libcuda the 0.15.1 server will load.
grep -r . /etc/ld.so.conf.d/ || echo "/etc/ld.so.conf.d/ has no entries"
ldconfig -p | grep -F 'libcuda.so.1' || echo "no libcuda.so.1 in the ld.so cache"
echo "container memory limit: $(python3 -I "$guard" --limit)"
df -h /workspace /root

# The venv goes first, and any failure in it ends setup before the 100 GB download: without it current and fp8 measure
# nothing, and shipped alone does not answer the question. The timeouts turn a stalled index into that failure
# instead of the pod's whole deadline.
start=$(date +%s)
mkdir -p /root/wheels
# A release asset's URL spells the wheel's "+" as %2B.
curl -fsSL --retry 3 -m 600 -o "/root/wheels/$wheel" \
    "https://github.com/vllm-project/vllm/releases/download/v0.32.0/${wheel/+/%2B}"
sha256sum -c - <<<"$wheel_sha256  /root/wheels/$wheel"
uv venv --python /usr/bin/python3.12 "$venv"
# How vllm/vllm-openai:v0.32.0-cu129 resolves the same requirements: PyTorch's cu129 index as an extra index under
# UV_INDEX_STRATEGY=unsafe-best-match (its Dockerfile line 843). Best-match is what picks torch's +cu129 build over
# PyPI's CUDA 13 one, and it is needed: the cu129 index also carries requests and setuptools, both older than vLLM's
# floors, which uv's default first-index strategy would refuse to look past.
timeout 1200 uv pip install --no-cache --python "$venv/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu129 "/root/wheels/$wheel"
# A torch that resolved to another CUDA build, a wheel whose kernels cannot link against it, or a torch that cannot
# start CUDA on this driver would fail both 0.32.0 variants at boot.
timeout 300 "$venv/bin/python" -c \
    'import torch, vllm._C; assert torch.version.cuda == "12.9", torch.version.cuda; torch.zeros(1, device="cuda")'
# The release image installs FlashInfer's prebuilt kernels when it is built. Without them 0.32.0 still serves, only
# compiling the kernels it needs on first use, so a failure here is recorded (versions below shows them absent), not
# fatal.
timeout 900 "$venv/bin/flashinfer" download-kernels --cuda-version 12.9 \
    || echo "flashinfer download-kernels failed; see above"
rm -rf /root/wheels
echo "vllm 0.32.0 venv installed in $(( $(date +%s) - start )) s, $(du -sh "$venv" | cut -f1)"
versions "$venv/bin/python"

if [[ -e $HF_HOME/hub && ! -L $HF_HOME/hub ]]; then
    echo "$HF_HOME/hub is a directory, not the link to $hub; move it away or start a new pod" >&2
    exit 1
fi
mkdir -p "$hub" "$HF_HOME"
ln -sfn "$hub" "$HF_HOME/hub"

before=$(du -sb "$hub" | cut -f1)
start=$(date +%s)
HF_XET_HIGH_PERFORMANCE=1 python3 - <<'EOF' &
from huggingface_hub import snapshot_download
for repo, revision in (("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"),
                       ("Qwen/Qwen3-32B-FP8", "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df")):
    print(snapshot_download(repo, revision=revision, max_workers=16))
EOF
dl=$!
# 100 GB of safetensors at 100 MB/s is about 17 min, the most setup can spend inside planned_hours.
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
