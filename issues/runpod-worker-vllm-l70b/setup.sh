#!/usr/bin/env bash
# Records the card and the image, fetches the worker's own startup-error classifier, and downloads the model to
# container disk, so paid variant time goes to the matrix.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
here=$(cd "$(dirname "$0")" && pwd)
guard="$here/../../harness/pod/mem_guard.py"
model=RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic
revision=f50dbad2c84590ca17dc51e207c34321b65ff14b
worker_commit=71d503e6080319e176e77b20515c4aaaf3426b8f
classifier_sha256=c096012a3b8da6807201609b53a5017c15dd2a14f7cea704422679d552555b1c
hub=/root/hf-hub
# The download's chunk cache would otherwise sit under HF_HOME, on the 20 GB volume.
export HF_XET_CACHE=/root/hf-xet

# Free memory at start is what every threshold in this kit is a fraction of.
python3 - <<'EOF'
from importlib.metadata import version
import torch
print("vllm", version("vllm"), "flashinfer", version("flashinfer-python"), "torch", torch.__version__,
      "cuda", torch.version.cuda, "transformers", version("transformers"), torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f"cuda free {free / 2**30:.2f} GiB of {total / 2**30:.2f} GiB")
EOF
timeout 60 nvidia-smi --format=csv --query-gpu=name,memory.total,power.limit,power.max_limit,\
pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max
# A 10-23 host ran every variant about 2x slower than another with nothing recorded to name why. Idle clocks and
# clock-event reasons here; hooks.sh samples them under load.
timeout 60 nvidia-smi -q -d CLOCK,PERFORMANCE
# Hosts differ within one GPU type, so the data center goes with the numbers.
echo "data center: ${RUNPOD_DC_ID:-not set}"
echo "container memory limit: $(python3 -I "$guard" --limit)"
df -h /workspace /root

# hooks.sh runs the worker's own startup_errors.py on every boot log, so the decision it reports is the worker's.
mkdir -p /workspace/worker-vllm
curl -fsSL -o /workspace/worker-vllm/startup_errors.py \
    "https://raw.githubusercontent.com/runpod-workers/worker-vllm/$worker_commit/src/startup_errors.py"
echo "$classifier_sha256  /workspace/worker-vllm/startup_errors.py" | sha256sum -c -

if [[ -e $HF_HOME/hub && ! -L $HF_HOME/hub ]]; then
    echo "$HF_HOME/hub is a directory, not the link to $hub; move it away or start a new pod" >&2
    exit 1
fi
mkdir -p "$hub" "$HF_HOME"
ln -sfn "$hub" "$HF_HOME/hub"

before=$(du -sb "$hub" | cut -f1)
start=$(date +%s)
HF_XET_HIGH_PERFORMANCE=1 python3 -c "from huggingface_hub import snapshot_download
print(snapshot_download('$model', revision='$revision', max_workers=16))" &
dl=$!
# 73 GB of safetensors at 100 MB/s is 12 min, the most setup can spend inside planned_hours.
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
