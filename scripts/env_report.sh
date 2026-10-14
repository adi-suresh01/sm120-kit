#!/usr/bin/env bash
# Prints the hardware and software facts every upstream issue comment needs.
set -euo pipefail

section() { printf '\n## %s\n' "$1"; }

section "GPUs"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv

section "Topology"
# PCIe-only repros need PHB/NODE/SYS between GPUs; NV# means NVLink and invalidates the repro.
nvidia-smi topo -m || true

section "CUDA"
nvcc --version 2>/dev/null | tail -n 1 || echo "nvcc not installed"

section "Python packages"
python3 - <<'EOF'
from importlib.metadata import version, PackageNotFoundError
for pkg in ("vllm", "torch", "flashinfer-python", "sglang", "triton", "transformers"):
    try:
        print(f"{pkg}=={version(pkg)}")
    except PackageNotFoundError:
        print(f"{pkg}: not installed")
EOF

section "Host"
uname -r
nproc
free -g | head -n 2
