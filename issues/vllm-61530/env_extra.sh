#!/usr/bin/env bash
# Records what env_report.sh leaves out and an all-reduce number depends on: PCIe links, power and clock limits, P2P
# write and atomics support, CPU sockets and NUMA layout, the host's IOMMU settings, and the NCCL torch loads.
# env_report.sh already has the GPU names, driver, `topo -m`, and package versions; the harness keeps the P2P read
# matrix. A copy of issues/vllm-59203/env_extra.sh, so the boxes read side by side, plus the NVIDIA driver's
# registry and peer settings, which decide whether SM-issued peer loads take the direct BAR path (helixml/ramjet#32).
# Usage: env_extra.sh <results_dir>   (writes nvidia-smi-q.txt there, the rest to stdout)
set -euo pipefail

out=${1:?usage: $0 <results_dir>}
section() { printf '\n## %s\n' "$1"; }
# A wedged GPU can hang nvidia-smi; a missing tool or field is noted and the record goes on.
try() { timeout 60 "$@" || echo "($1 failed: exit $?)"; }

section "GPUs: PCI address, PCIe link, power, clocks"
fields=(index name pci.bus_id vbios_version pcie.link.gen.max pcie.link.gen.current pcie.link.width.max
    pcie.link.width.current power.limit power.default_limit power.max_limit clocks.max.sm clocks.max.memory
    persistence_mode compute_mode)
try nvidia-smi --query-gpu="$(IFS=,; echo "${fields[*]}")" --format=csv
try nvidia-smi -q >"$out/nvidia-smi-q.txt"
echo "full nvidia-smi -q in nvidia-smi-q.txt"

# #47979 reports AccessSupported=1 but NativeAtomicSupported=0 on every PRO 6000 pair; torch's symmetric-memory
# barrier needs peer atomics, vLLM's custom all-reduce flags do not.
for cap in w a n; do
    section "P2P matrix: topo -p2p $cap"
    try nvidia-smi topo -p2p "$cap"
done

section "CPU"
try lscpu
section "NUMA"
if command -v numactl >/dev/null; then
    try numactl -H
else
    grep -H . /sys/devices/system/node/node*/cpulist || true
fi

section "NVIDIA driver registry and peer settings"
grep -E '^(RegistryDwords|DmaRemapPeerMmio|EnableResizableBar|EnablePCIeGen3|EnableStreamMemOPs):' \
    /proc/driver/nvidia/params 2>/dev/null || echo "(no /proc/driver/nvidia/params)"

section "Host kernel and IOMMU"
cat /proc/driver/nvidia/version 2>/dev/null || true
cat /proc/cmdline
echo "iommu groups: $(find /sys/kernel/iommu_groups -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)"

section "NCCL and the packages env_report.sh does not list"
python3 -I - <<'EOF'
from importlib.metadata import PackageNotFoundError, version

import torch

for pkg in ("flashinfer-cubin", "flashinfer-jit-cache", "nvidia-nccl-cu13"):
    try:
        print(f"{pkg}=={version(pkg)}")
    except PackageNotFoundError:
        print(f"{pkg}: not installed")
print("torch cuda", torch.version.cuda, "nccl", ".".join(map(str, torch.cuda.nccl.version())))
with open("/proc/self/maps") as maps:
    print("libnccl mapped by torch:", sorted({line.split()[-1] for line in maps if "libnccl.so" in line}))
print("gpus visible to torch:", torch.cuda.device_count())
EOF
