#!/usr/bin/env bash
# Everything before the first boot, cheapest check first, so a box that cannot run the matrix ends before the 510 GB
# download: 8 GPUs and the image's versions, host RAM for the pinned Engram tables, /dev/shm for NCCL's SHM transport,
# no inherited NCCL setting, a disk fast enough to boot from, every arm's all-reduce settings over the 8 GPUs
# (ar_probe.py), and the benchmark client's tokenizer. Then DeepSeek-V4.1-Flash at the pinned revision, stopped as soon
# as it cannot end in time. A non-zero exit makes `harness watch` destroy the box and rent another. The last lines name
# the arms the exec should run (README.md, Run).
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
deadline=${HARNESS_DEADLINE_EPOCH:?the harness sets HARNESS_DEADLINE_EPOCH}
setup_start=$(date +%s)
here=$(cd "$(dirname "$0")" && pwd)
kit=$(cd "$here/../.." && pwd)
out="$kit/results/vllm-59203"
probe_dir="$out/probe"
guard="$kit/harness/pod/mem_guard.py"
model=deepseek-ai/DeepSeek-V4.1-Flash
revision=314113f8a7dadb388215396ba5fbee2c610a04d6
# Every file at the pinned revision, from the Hub's tree listing; the 48 shards are 510,296,708,312 B of it.
total_bytes=510313372827
mkdir -p "$probe_dir" "$HF_HOME"

# setup.log keeps the reason; watch destroys the box and stays off its machine.
fail() {
    echo "$1" >&2
    exit 1
}

gpus=$(timeout 60 nvidia-smi -L | grep -c '^GPU ' || true)
if [[ $gpus != 8 ]]; then
    fail "this matrix needs 8 GPUs; nvidia-smi lists ${gpus:-none}"
fi
# The patch, the probe, and every source line README.md cites are v0.33.0's.
python3 -I - <<'EOF'
from importlib.metadata import version

import torch

print("vllm", version("vllm"), "flashinfer", version("flashinfer-python"), "torch", torch.__version__,
      "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
assert version("vllm") == "0.33.0", "want vLLM 0.33.0"
EOF
bash "$here/env_extra.sh" "$out" >"$out/env_extra.txt" 2>&1 || echo "env_extra.sh exited $?; see env_extra.txt"

# The two Engram tables (shards 47 and 48, 203 GB) go to pinned host memory, about 189 GiB across the 8 ranks at exact
# size; while they load, the ranks also map one 101.5 GB Engram shard between them, beside 8 workers of several GiB
# each. That is about 350 GiB, which has to stay under the memory guard's 80% trim line (its fadvise skips page cache
# a process has mapped), or the guard may end the boot at 93%. A container's limit is the cgroup's, which `free` does
# not show, so this reads it the way the guard does.
min_ram_gib=448
limit=$(python3 -I "$guard" --limit)
ram_bytes=${limit%% *} ram_source=${limit#* }
if [[ $ram_bytes == none ]]; then
    ram_bytes=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) * 1024 ))
    ram_source="MemTotal, since the container has no memory limit: $ram_source"
fi
ram_gib=$(( ram_bytes / 2**30 ))
echo "RAM ${ram_gib} GiB (${ram_source}), memlock $(ulimit -l)"
if (( ram_gib < min_ram_gib )); then
    fail "RAM ${ram_gib} GiB is under ${min_ram_gib} GiB, too little for the pinned Engram tables"
fi

# b0's NCCL runs over SHM, which lives in /dev/shm; Docker's default is 64 MB, and the harness's Vast create asks for
# no size. 2 GiB is room for the buffers of the server's communicators (issues/sglang-33422).
shm_bytes=$(df -B1 --output=size /dev/shm | tail -n 1 | tr -d ' ')
echo "/dev/shm $(( shm_bytes / 1048576 )) MiB"
if (( shm_bytes < 2 * 1024 * 1024 * 1024 )); then
    fail "/dev/shm is under 2 GiB, too small for NCCL's SHM transport"
fi

# A setting the image or host exports would leak into every arm and blur what each one changes. NCCL_VERSION is the
# CUDA base image's label, not the NCCL torch loads.
inherited=$(env | grep -E '^(NCCL_|VLLM_ALLREDUCE_)' | grep -v '^NCCL_VERSION=' || true)
if [[ -n $inherited ]]; then
    fail "the container already sets: $(tr '\n' ' ' <<<"$inherited")"
fi

# The first boot reads the 475 GiB checkpoint cold, so a slow disk eats b0's boot budget. 4 GiB of random bytes (zeros
# would compress on a ZFS host), written, dropped from the page cache, and read back, tells before the download is paid.
# fadvise does not empty ZFS's own cache, so there the read can pass a slow disk; it never refuses a fast one.
min_free_gb=560
min_read_mb_s=1000
free_gb=$(( $(df -B1 --output=avail "$HF_HOME" | tail -n 1) / 1000000000 ))
disk_check="$HF_HOME/disk-read-check"
dd if=/dev/urandom of="$disk_check" bs=16M count=256 iflag=fullblock conv=fsync status=none
python3 -I "$guard" --drop-cache "$disk_check"
t0=$(date +%s%N)
dd if="$disk_check" of=/dev/null bs=16M status=none
read_mb_s=$(( 256 * 16 * 1048576 * 1000 / ($(date +%s%N) - t0) ))
rm -f "$disk_check"
echo "disk: ${free_gb} GB free, cold read ${read_mb_s} MB/s"
if (( free_gb < min_free_gb || read_mb_s < min_read_mb_s )); then
    fail "the disk needs ${min_free_gb} GB free and ${min_read_mb_s} MB/s cold reads"
fi

# Every arm's settings once over all 8 GPUs before the download: the transport NCCL picks per pair, whether the
# all-reduce is exact, and whether it works at all; a setting that hangs here would hang that arm's boot for
# boot_timeout_s. All four run before any verdict, so the log tells the cases apart. The env strings repeat
# harness.toml's per-arm env.
arms=(b0-default b2-sys-fi b1-p2p-sys b3-fi-only)
declare -A probe_env=(
    [b0-default]=""
    [b2-sys-fi]="NCCL_P2P_LEVEL=SYS VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1"
    [b1-p2p-sys]="NCCL_P2P_LEVEL=SYS"
    [b3-fi-only]="VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1"
)
declare -A probe_ok=() transports=()
for arm in "${arms[@]}"; do
    read -ra settings <<<"${probe_env[$arm]}"
    echo "## all-reduce probe: $arm (${probe_env[$arm]:-NCCL defaults})"
    probe_ok[$arm]=0
    # ar_probe.py's own per-step watchdog fires first; this limit catches a hang it cannot see.
    if env "${settings[@]}" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH \
        NCCL_DEBUG_FILE="$probe_dir/$arm.nccl.%h.%p.log" timeout -k 20 600 \
        torchrun --standalone --nproc_per_node=8 "$here/ar_probe.py" --arm "$arm" --out "$probe_dir" \
        >"$probe_dir/$arm.log" 2>&1; then
        probe_ok[$arm]=1
    fi
    # A worker torchrun lost track of after a timeout can still hold its GPU.
    pkill -9 -f "$here/ar_probe.py" 2>/dev/null || true
    cat "$probe_dir/$arm".nccl.*.log >"$probe_dir/$arm.nccl.log" 2>/dev/null || true
    bash "$here/nccl_transport.sh" "$probe_dir/$arm.nccl.log" >"$probe_dir/$arm.transport.txt" || true
    transports[$arm]=$(grep -oE 'via [A-Z0-9]+/[A-Za-z]+' "$probe_dir/$arm.nccl.log" | sort | uniq -c \
        | awk '{printf "%s%s x%s", sep, $3, $1; sep = ", "}' || true)
    if [[ ${probe_ok[$arm]} == 1 ]]; then
        python3 -I - "$probe_dir/$arm.json" <<'EOF' || echo "$arm: probe passed, but its summary did not print"
import json
import sys

r = json.load(open(sys.argv[1]))
p = r["pynccl"]
us = p["eager_us"]
line = f"pynccl (NCCL {p['nccl']}) exact; eager us: batch 1 {us['1']}, batch 8192 {us['8192']}"
if "flashinfer" in r:
    f = r["flashinfer"]
    line += f"; FlashInfer exact, {f['profile']}, set up in {f['setup_s']} s, eager us batch 1 {f['eager_us']['1']}"
print(line)
EOF
    else
        tail -n 20 "$probe_dir/$arm.log" >&2
        # Step lines start with a time; after the watchdog fires, the stack dump follows the last one.
        last_step=$(grep -E '^[0-9]{2}:[0-9]{2}:[0-9]{2} ' "$probe_dir/$arm.fault.rank0.txt" 2>/dev/null \
            | tail -n 1 || true)
        echo "$arm: probe failed or hung; rank 0's last step: ${last_step:-none}" >&2
    fi
    echo "$arm transports: ${transports[$arm]:-no transport lines}"
done
probe_ok[b0-repeat]=${probe_ok[b0-default]}
if [[ " ${probe_ok[*]} " == *" 0 "* ]] && ! bash "$kit/harness/pod/wait_gpu_idle.sh" 60; then
    fail "a failed probe left a GPU busy"
fi
for arm in b0-default b2-sys-fi b1-p2p-sys; do
    if [[ ${probe_ok[$arm]} != 1 ]]; then
        fail "the $arm probe failed or hung (above), and the matrix needs b0, b2, and b1"
    fi
done

# The client's prompts come from the pinned revision's tokenizer, not the main branch the model id would load. The
# snapshot directory fills in with the weights below. The check samples the random dataset the rows use, the way
# `vllm bench serve` does, so a client that cannot build them fails here and not after the first boot.
snapshot=$(python3 -I - "$model" "$revision" <<'EOF'
import sys

from huggingface_hub import snapshot_download

print(snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=["*.json", "*.jinja"]))
EOF
)
ln -sfn "$snapshot" /workspace/tokenizer
python3 -I - /workspace/tokenizer <<'EOF'
import sys

from vllm.benchmarks.datasets.datasets import RandomDataset
from vllm.tokenizers import get_tokenizer

tokenizer = get_tokenizer(sys.argv[1])
print("bench client tokenizer", type(tokenizer).__name__, "vocab", tokenizer.vocab_size)
for input_len in (1024, 32768):
    requests = RandomDataset(random_seed=1).sample(
        tokenizer=tokenizer, num_requests=2, input_len=input_len, output_len=16, range_ratio=0.0
    )
    lens = [r.prompt_len for r in requests]
    print(f"random dataset at {input_len} tokens: prompt_len {lens}")
    assert all(abs(n - input_len) <= 8 for n in lens), f"prompt lengths {lens} are not {input_len}"
EOF

before=$(du -sB1 "$HF_HOME/hub" | cut -f1)
start=$(date +%s)
# A transient Hub error late in an 80-minute download would otherwise cost the whole box; a retry resumes the files
# already on disk.
HF_XET_HIGH_PERFORMANCE=1 python3 -I - "$model" "$revision" <<'EOF' &
import sys

from huggingface_hub import snapshot_download

for attempt in range(1, 4):
    try:
        snapshot_download(sys.argv[1], revision=sys.argv[2], max_workers=16)
        break
    except Exception as e:
        if attempt == 3:
            raise
        print(f"download attempt {attempt} failed ({type(e).__name__}: {e}); resuming", flush=True)
EOF
dl=$!
# After the download: the cache drop 2 min, b0 30, b2 27, and 6 for the second exec and down (README.md). A download
# that ends less than 75 min before the deadline would cost the matrix its main pair, so it stops as soon as its rate
# says so, or at that point.
cutoff=$(( deadline - 75 * 60 ))
rate_checked=0
while kill -0 "$dl" 2>/dev/null; do
    sleep 30
    now=$(date +%s)
    if (( ! rate_checked && now - start >= 180 )) && kill -0 "$dl" 2>/dev/null; then
        rate_checked=1
        # du exits 1 when a file it walks is renamed under it, as finished downloads are; its total is still good.
        got=$(( $( (du -sB1 "$HF_HOME/hub" 2>/dev/null || true) | cut -f1) - before ))
        rate=$(( got / (now - start) ))
        eta=$(( now + (total_bytes - got) / (rate > 0 ? rate : 1) ))
        echo "download at $(( rate / 1000000 )) MB/s after 3 min; at that rate it ends" \
            "$(( (deadline - eta) / 60 )) min before the deadline"
        if (( eta > cutoff )); then
            kill "$dl"
            fail "the download would end under 75 min before the deadline"
        fi
    fi
    if (( now > cutoff )) && kill -0 "$dl" 2>/dev/null; then
        kill "$dl"
        fail "the download is still running 75 min before the deadline"
    fi
done
wait "$dl"
echo "weights downloaded in $(( ($(date +%s) - start) / 60 )) min"
# The download's page cache counts against a container memory limit and would crowd the Engram pin.
python3 -I "$guard" --drop-cache "$HF_HOME"
du -sh "$HF_HOME/hub"
echo "setup done"

# The exec call (README.md, Run) needs only these last lines.
now=$(date +%s)
left_min=$(( (deadline - now) / 60 ))
echo "setup took $(( (now - setup_start) / 60 )) min; ${left_min} min left to the deadline"
if [[ ${transports[b0-default]} != *SHM* ]]; then
    echo "b0's probe ran no SHM connection: NCCL's default already uses P2P on this box, so b1 changes nothing for NCCL"
fi
# Minutes per arm as README.md estimates them: b0 30 with its cold boot, every other arm 27, and 6 for the second exec
# and down. b1 runs whenever it fits; then b3, then b0-repeat, each only while it still fits. b0 goes alone first, so a
# boot that fails costs one boot, not every arm's.
next=(b2-sys-fi)
need=$(( 30 + 27 + 6 ))
for arm in b1-p2p-sys b3-fi-only b0-repeat; do
    if [[ ${probe_ok[$arm]} == 1 ]] && (( left_min >= need + 27 )); then
        next+=("$arm")
        need=$(( need + 27 ))
    fi
done
echo "next: harness exec vllm-59203 b0-default, then if it booted: harness exec vllm-59203 ${next[*]}"
