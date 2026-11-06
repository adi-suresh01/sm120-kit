#!/usr/bin/env bash
# Everything before the first boot, cheapest check first, so a box that cannot run the matrix ends before the 429 GB
# download: 4 GPUs, host RAM for the pinned Engram tables, /dev/shm for NCCL's SHM transport, no inherited NCCL
# setting, a disk fast enough to boot from, the P2P matrix a1 needs, and the benchmark client's tokenizer. Then the
# checkpoint at the pinned revision, stopped as soon as it cannot end in time. A non-zero exit makes `harness watch`
# destroy the box and rent another. The last line names the arms the exec should run (README.md, Run).
set -euo pipefail

: "${HF_HOME:?set HF_HOME to a path on the pod volume}"
deadline=${HARNESS_DEADLINE_EPOCH:?the harness sets HARNESS_DEADLINE_EPOCH}
setup_start=$(date +%s)
here=$(cd "$(dirname "$0")" && pwd)
kit=$(cd "$here/../.." && pwd)
out="$kit/results/vllm-61530"
guard="$kit/harness/pod/mem_guard.py"
model=nvidia/DeepSeek-V4.1-Flash-1017-NVFP4
revision=3ae9c0d3fbbea5370bd06e679d22044db15883f1
# Every file at the pinned revision, from the Hub's tree listing; the 48 shards are 428,986,460,424 B of it.
total_bytes=429008965421
# The weights go on container disk, through a link at HF_HOME/hub: on RunPod /workspace is a FUSE volume that took
# 7m35s just to open lazy views of DSV4.1's 48 shards, and on Vast both paths are the same disk
# (issues/vllm-56771/setup.sh). The download's chunk cache would otherwise sit on that volume too.
hub=/root/hf-hub
export HF_XET_CACHE=/root/hf-xet
mkdir -p "$out" "$HF_HOME" "$hub"
if [[ -e $HF_HOME/hub && ! -L $HF_HOME/hub ]]; then
    echo "$HF_HOME/hub is a directory, not the link to $hub; move it away or start a new box" >&2
    exit 1
fi
ln -sfn "$hub" "$HF_HOME/hub"

# setup.log keeps the reason; watch destroys the box and stays off its machine.
fail() {
    echo "$1" >&2
    exit 1
}

gpus=$(timeout 60 nvidia-smi -L | grep -c '^GPU ' || true)
if [[ $gpus != 4 ]]; then
    fail "this matrix needs 4 GPUs; nvidia-smi lists ${gpus:-none}"
fi
# The image is pinned by digest; this only puts the build in setup.log.
python3 -I - <<'EOF'
from importlib.metadata import version

import torch

print("vllm", version("vllm"), "flashinfer", version("flashinfer-python"), "torch", torch.__version__,
      "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
EOF
bash "$here/env_extra.sh" "$out" >"$out/env_extra.txt" 2>&1 || echo "env_extra.sh exited $?; see env_extra.txt"

# The two Engram tables (shards 47 and 48, 104.8 GB in MXFP4) go to pinned host memory, about 97 GiB across the 4
# ranks at exact size. At the end of the load the worst case also holds shard 48 mapped (48.8 GiB) and shard 47 still
# in page cache, since the guard trims at most once a minute and fadvise skips mapped pages (mem_guard.py), beside
# about 20 GiB of workers and the API server: about 215 GiB, which has to stay under the guard's 93% kill line. 280
# leaves 45 GiB of margin. A container's limit is the cgroup's, which `free` does not show, so this reads it the way
# the guard does.
min_ram_gib=280
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

# a0's NCCL runs over SHM where P2P is off, which lives in /dev/shm; Docker's default is 64 MB, and the harness's
# Vast create asks for no size. 2 GiB is room for the buffers of the server's communicators (issues/sglang-33422).
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

# The first boot reads the 400 GiB checkpoint cold, so a slow disk eats a0's boot budget. 4 GiB of random bytes (zeros
# would compress on a ZFS host), written, dropped from the page cache, and read back, tells before the download is paid.
# fadvise does not empty ZFS's own cache, so there the read can pass a slow disk; it never refuses a fast one.
min_free_gb=450
min_read_mb_s=1000
free_gb=$(( $(df -B1 --output=avail "$hub" | tail -n 1) / 1000000000 ))
disk_check="$hub/disk-read-check"
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

# FlashInfer's PCIe IPC workspace aborts the boot without peer access on every pair (vllm#65192 is the open fallback),
# so a1 runs only where `topo -p2p r` reads OK for all 12 ordered pairs; a0 and f0 do not need it.
p2p=$(timeout 60 nvidia-smi topo -p2p r || true)
echo "$p2p" >"$out/p2p-read.txt"
# Matrix rows are the ones holding the X self entry; the header row (GPU names, sometimes wrapped in ANSI codes)
# has none.
read -r pairs ok_pairs < <(awk '
    $1 ~ /^GPU[0-9]+$/ {
        self = 0
        for (i = 2; i <= NF; i++) if ($i == "X") self = 1
        if (self) for (i = 2; i <= NF; i++) if ($i != "X") { n++; if ($i == "OK") ok++ }
    }
    END {print n + 0, ok + 0}' <<<"$p2p")
echo "P2P read: ${ok_pairs}/${pairs} pairs OK"
fi_ok=0
if (( pairs == 12 && ok_pairs == 12 )); then
    fi_ok=1
fi

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

before=$(du -sB1 "$hub" | cut -f1)
start=$(date +%s)
# A transient Hub error late in a long download would otherwise cost the whole box; a retry resumes the files
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
# After the download: the cache drop 2 min, a0 30, a1 25, and 6 for the second exec and down (README.md). A download
# that ends less than 65 min before the deadline would cost the matrix its main pair, so it stops as soon as its rate
# says so, or at that point.
cutoff=$(( deadline - 65 * 60 ))
rate_checked=0
while kill -0 "$dl" 2>/dev/null; do
    sleep 30
    now=$(date +%s)
    if (( ! rate_checked && now - start >= 180 )) && kill -0 "$dl" 2>/dev/null; then
        rate_checked=1
        # du exits 1 when a file it walks is renamed under it, as finished downloads are; its total is still good.
        got=$(( $( (du -sB1 "$hub" 2>/dev/null || true) | cut -f1) - before ))
        rate=$(( got / (now - start) ))
        eta=$(( now + (total_bytes - got) / (rate > 0 ? rate : 1) ))
        echo "download at $(( rate / 1000000 )) MB/s after 3 min; at that rate it ends" \
            "$(( (deadline - eta) / 60 )) min before the deadline"
        if (( eta > cutoff )); then
            kill "$dl"
            fail "the download would end under 65 min before the deadline"
        fi
    fi
    if (( now > cutoff )) && kill -0 "$dl" 2>/dev/null; then
        kill "$dl"
        fail "the download is still running 65 min before the deadline"
    fi
done
wait "$dl"
echo "weights downloaded in $(( ($(date +%s) - start) / 60 )) min"
# The download's page cache counts against a container memory limit and would crowd the Engram pin.
python3 -I "$guard" --drop-cache "$hub" "$HF_XET_CACHE"
du -sh "$hub"
echo "setup done"

# The exec call (README.md, Run) needs only these last lines.
now=$(date +%s)
left_min=$(( (deadline - now) / 60 ))
echo "setup took $(( (now - setup_start) / 60 )) min; ${left_min} min left to the deadline"
# Minutes per arm as README.md estimates them: a0 30 with its cold boot, every other arm 25, and 6 for the second exec
# and down. a0 goes alone first, so a boot that fails costs one boot, not every arm's. Its FlashInfer twin runs only on
# a box with P2P on every pair: a1 after an a0 that booted, f1 after f0 when a0 could not fit its KV cache.
if (( fi_ok )) && (( left_min >= 30 + 25 + 6 )); then
    after_ok=a1-fp8idx-fi after_fit=f0-fp8idx-lm
    (( left_min >= 30 + 25 + 25 + 6 )) && after_fit+=" f1-fp8idx-lm-fi"
else
    (( fi_ok )) || echo "a1 and f1 left out: P2P is not OK on every pair"
    after_ok="(nothing else fits)" after_fit=f0-fp8idx-lm
fi
echo "next: harness exec vllm-61530 a0-fp8idx; if it booted: harness exec vllm-61530 ${after_ok};" \
    "if it could not fit its KV cache: harness exec vllm-61530 ${after_fit}"
