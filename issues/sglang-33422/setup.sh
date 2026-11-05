#!/usr/bin/env bash
# Checks the image, checks NCCL under each decode arm's P2P setting, builds the prompts, downloads ShareGPT and
# DeepSeek-V4-Flash-0731 to container disk, drops the weights from the page cache, and checks a cold read is fast
# enough to boot from, so paid variant time goes to the matrix. Its last lines say whether d3 joins the exec.
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
setup_start=$(date +%s)
here=$(cd "$(dirname "$0")" && pwd)
guard="$here/../../harness/pod/mem_guard.py"
model=deepseek-ai/DeepSeek-V4-Flash-0731
revision=7872f01b1d1fe23eabc4c98b48bffcef5a386062
# The topk=192 reading in README.md holds for this FlashInfer only; the fallback image (v0.5.21) ships 0.6.18. The
# digest pins sglang itself, whose version string an image built without git metadata reads as 0.0.0.dev0.
want_flashinfer=0.7.1
hub=/root/hf-hub
# On RunPod the download's chunk cache would otherwise sit under HF_HOME, on the 20 GB volume.
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
# link and the power limit (README.md). The default limit tells a Max-Q (300 W) from a card capped down to it.
timeout 60 nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.width.max,power.limit,power.default_limit \
    --format=csv
# NCCL uses P2P on a pair only when NVML reads both read and write as OK (src/graph/paths.cc:383-391), and at SYS it
# logs nothing when write is not (:400); the harness's require_p2p checks the read matrix only.
echo "## P2P read status"
timeout 60 nvidia-smi topo -p2p r || echo "nvidia-smi topo -p2p r failed"
echo "## P2P write status"
timeout 60 nvidia-smi topo -p2p w || echo "nvidia-smi topo -p2p w failed"
echo "container memory limit: $(python3 -I "$guard" --limit)"
df -h /workspace /root /dev/shm
# NCCL's SHM transport, which P2P off uses, lives in /dev/shm; the harness's Vast create asks for no size, and Docker's
# default is 64 MB. 2 GiB is the floor this run takes as room for the buffers of the server's communicators.
shm_bytes=$(df -B1 --output=size /dev/shm | tail -n 1 | tr -d ' ')
if (( shm_bytes < 2 * 1024 * 1024 * 1024 )); then
    echo "/dev/shm is $(( shm_bytes / 1048576 )) MiB, under 2 GiB, too small for NCCL's SHM; run harness down" >&2
    exit 1
fi
# An NCCL_P2P_* the image or host sets would change what every arm measures. NCCL_VERSION is left out: the CUDA base
# image sets it as a label (2.28.3-1), not the NCCL torch loads; the probe's own log names that one below.
env | grep '^NCCL_' | grep -v '^NCCL_VERSION=' || echo "no NCCL_ settings in the environment"

# Each decode arm's NCCL setting once over all GPUs, before the download (nccl_probe.py): the transport NCCL picks per
# pair, its allreduce time, and whether it works at all. All three run before any verdict, so the log tells the cases
# apart. NCCL_CUMEM_ENABLE=0 is what sglang sets, so the lines read as the server's will, and each rank logs to its
# own line-buffered file, as the arms do (hooks.sh, pre_launch).
probe_dir=$here/../../results/sglang-33422/nccl-probe
mkdir -p "$probe_dir"
arms=(d1 d2 d3)
declare -A probe_env=([d1]="NCCL_P2P_DISABLE=0" [d2]="NCCL_P2P_DISABLE=1" [d3]="NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS")
declare -A probe_ok=()
for arm in "${arms[@]}"; do
    read -ra settings <<<"${probe_env[$arm]}"
    echo "## NCCL probe for $arm (${settings[*]})"
    probe_ok[$arm]=0
    if env "${settings[@]}" NCCL_CUMEM_ENABLE=0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH \
        NCCL_DEBUG_FILE="$probe_dir/$arm.%p.nccl" timeout -k 10 180 python3 -I "$here/nccl_probe.py" \
        >"$probe_dir/$arm.log" 2>&1; then
        probe_ok[$arm]=1
    fi
    cat "$probe_dir/$arm".*.nccl >"$probe_dir/$arm.nccl.log" 2>/dev/null || true
    if [[ ${probe_ok[$arm]} == 1 ]]; then
        grep '^allreduce' "$probe_dir/$arm.log"
    else
        tail -n 20 "$probe_dir/$arm.log" "$probe_dir/$arm.nccl.log" >&2
        echo "NCCL with ${settings[*]} failed or hung within 180 s" >&2
    fi
    bash "$here/nccl_transport.sh" "$probe_dir/$arm.nccl.log"
done
# A rank killed in a hang can keep holding its GPU, and d1's idle wait would then fail after the 30 min download.
if [[ " ${probe_ok[*]} " == *" 0 "* ]] && ! bash "$here/../../harness/pod/wait_gpu_idle.sh" 60; then
    echo "a failed NCCL probe left a GPU busy; run harness down" >&2
    exit 1
fi
if [[ ${probe_ok[d1]} != 1 || ${probe_ok[d2]} != 1 ]]; then
    echo "NCCL under d1's or d2's setting failed or hung (above); run harness down and avoid this machine" >&2
    exit 1
fi

# The weights go on container disk: /workspace on secure RunPod pods is a FUSE mount, and vLLM spent 7m35s opening
# the vllm-56771 shards there (issues/vllm-56771/README.md). On Vast both are the one disk.
if [[ -e $HF_HOME/hub && ! -L $HF_HOME/hub ]]; then
    echo "$HF_HOME/hub is a directory, not the link to $hub; move it away or start a new pod" >&2
    exit 1
fi
mkdir -p "$hub" "$HF_HOME"
ln -sfn "$hub" "$HF_HOME/hub"

# Prompts first, so a builder failure stops setup before the 167 GB download.
python3 -I "$here/build_prompts.py" "$model" "$revision" /workspace/prompts
# The decode rows' random dataset is ShareGPT text, which bench_serving would otherwise fetch in d1's first row;
# decode_bench.py passes this path.
sharegpt=$(python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('anon8231489123/ShareGPT_Vicuna_unfiltered', 'ShareGPT_V3_unfiltered_cleaned_split.json', repo_type='dataset'))")
ln -sfn "$sharegpt" /workspace/prompts/sharegpt.json

# Disk use, not apparent size, which reads high where the downloader preallocates.
before=$(du -sB1 "$hub" | cut -f1)
start=$(date +%s)
HF_XET_HIGH_PERFORMANCE=1 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$model', revision='$revision', max_workers=16))" &
dl=$!
# Some hosts reach HF at 15 MB/s (EUR-IS-2 on 10-16). 166.9 GB at 60 MB/s takes 46 min, the most setup can spend with
# d1 and d2 still inside planned_hours on a Max-Q (README.md). The floor at 3 min catches a slow host early; the
# ceiling catches one that slows later, since the harness waits on setup until the deadline.
min_mb_s=60
max_s=$(( 46 * 60 ))
floor_checked=0
while kill -0 "$dl" 2>/dev/null; do
    sleep 30
    elapsed=$(( $(date +%s) - start ))
    if (( ! floor_checked && elapsed >= 180 )) && kill -0 "$dl" 2>/dev/null; then
        floor_checked=1
        rate=$(( ($(du -sB1 "$hub" | cut -f1) - before) / elapsed / 1000000 ))
        echo "download rate ${rate} MB/s after $(( elapsed / 60 )) min"
        if (( rate < min_mb_s )); then
            kill "$dl"
            echo "download at ${rate} MB/s is under ${min_mb_s} MB/s; run harness down" >&2
            exit 1
        fi
    fi
    if (( elapsed >= max_s )) && kill -0 "$dl" 2>/dev/null; then
        kill "$dl"
        echo "download still running after $(( max_s / 60 )) min, past what planned_hours allows; run harness down" >&2
        exit 1
    fi
done
wait "$dl"
echo "weights downloaded in $(( $(date +%s) - start )) s"
# bench_serving would load the tokenizer of the model id's main branch; this is the pinned revision's snapshot.
snapshot=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$model', revision='$revision', local_files_only=True))")
ln -sfn "$snapshot" /workspace/prompts/tokenizer
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

# The d3 call (README.md) needs only these last lines: the transport each probe saw per GPU pair (rank = GPU in the
# probe), and the time left.
echo "## NCCL transport per GPU pair in setup's probe"
d1_shm=0
for arm in "${arms[@]}"; do
    pairs=$(grep -oE '[0-9]+\[[0-9]+\] -> [0-9]+\[[0-9]+\] via [A-Z0-9]+/[A-Za-z]+' "$probe_dir/$arm.nccl.log" \
        | sed -E 's/^[0-9]+\[([0-9]+)\] -> [0-9]+\[([0-9]+)\] via /gpu\1->gpu\2 /' | sort -u | paste -sd ',' - \
        | sed 's/,/, /g' || true)
    if [[ $arm == d1 && $pairs == *SHM* ]]; then
        d1_shm=1
    fi
    status=FAILED
    if [[ ${probe_ok[$arm]} == 1 ]]; then
        status=ok
    fi
    echo "$arm (${probe_env[$arm]}): probe $status; ${pairs:-no transport lines}"
done
now=$(date +%s)
if [[ ! ${HARNESS_DEADLINE_EPOCH:-} =~ ^[0-9]+$ ]]; then
    echo "setup took $(( (now - setup_start) / 60 )) min; HARNESS_DEADLINE_EPOCH is unset, so the time left is unknown"
    exit 0
fi
left_min=$(( (HARNESS_DEADLINE_EPOCH - now) / 60 ))
echo "setup took $(( (now - setup_start) / 60 )) min; ${left_min} min left to the deadline"
# README.md's rule: d1, d2, and d3 take about 95 min on a Max-Q, and d3 adds nothing where d1 already runs P2P on
# every pair.
if (( d1_shm && left_min >= 95 )) && [[ ${probe_ok[d3]} == 1 ]]; then
    echo "d3: run it (SHM in d1's probe, d3's probe ok, ${left_min} >= 95 min left)"
    echo "next: harness exec sglang-33422 d1-ep4-p2p-on d2-ep4-p2p-off d3-ep4-p2p-sys"
else
    echo "d3: skip it (it needs SHM in d1's probe, d3's probe ok, and >= 95 min left)"
    echo "next: harness exec sglang-33422 d1-ep4-p2p-on d2-ep4-p2p-off"
fi
