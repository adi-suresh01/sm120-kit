#!/usr/bin/env bash
# Prints the hardware and software facts every upstream issue comment needs.
set -euo pipefail

section() { printf '\n## %s\n' "$1"; }

section "GPUs"
# A wedged GPU can hang nvidia-smi forever; a healthy one answers in seconds.
if ! timeout 60 nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv; then
    echo "nvidia-smi failed or did not answer within 60 s" >&2
    exit 1
fi

section "Topology"
# PCIe-only repros need PHB/NODE/SYS between GPUs; NV# means NVLink and invalidates the repro.
timeout 60 nvidia-smi topo -m || true

section "CUDA"
nvcc --version 2>/dev/null | tail -n 1 || echo "nvcc not installed"

section "Python packages"
# Plain CUDA images, like the harness test pod's, have no Python.
if command -v python3 >/dev/null; then
    python3 - <<'EOF'
from importlib.metadata import version, PackageNotFoundError
for pkg in ("vllm", "torch", "flashinfer-python", "sglang", "triton", "transformers"):
    try:
        print(f"{pkg}=={version(pkg)}")
    except PackageNotFoundError:
        print(f"{pkg}: not installed")
EOF
else
    echo "python3 not installed"
fi

section "Host"
uname -r
nproc
free -g | head -n 2
