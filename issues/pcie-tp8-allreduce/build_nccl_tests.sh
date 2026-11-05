#!/usr/bin/env bash
# Builds nccl-tests' all_reduce_perf and sendrecv_perf against the NCCL that torch loads in this image, so its
# numbers and vLLM's come from one library. Writes that library's directory to <build_dir>/nccl_lib_dir.
# Usage: build_nccl_tests.sh <build_dir>
set -euo pipefail

build=${1:?usage: $0 <build_dir>}
# v2.21.1 (2026-10-05). Its -U prints the algorithm, protocol, and channel count NCCL picked per size, in full from
# NCCL 2.31.
sha=afd59abab774a6a4b492eb344bd0c845a0b7bb4d
cuda_home=/usr/local/cuda

# vLLM's image installs cuda-nvcc-13-0 for FlashInfer's JIT (docker/Dockerfile:801-816 at v0.33.0), and that package
# depends on build-essential and cuda-cudart-dev, so this only runs on an image that dropped them.
if [[ ! -x $cuda_home/bin/nvcc ]] || ! command -v g++ >/dev/null || ! command -v make >/dev/null; then
    cuda_dash=$(python3 -I -c 'import torch; print(torch.version.cuda.replace(".", "-"))')
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "cuda-nvcc-$cuda_dash" g++ make
fi
"$cuda_home/bin/nvcc" --version | tail -n 2

# torch maps the nvidia-nccl wheel's libnccl.so.2 by soname; /proc/self/maps names the file it actually loaded, which
# is also the one vLLM's pynccl gets when it opens libnccl.so.2 in the same process.
read -r lib version cuda_major < <(python3 -I -c '
import torch
version = ".".join(map(str, torch.cuda.nccl.version()))
paths = sorted({line.split()[-1] for line in open("/proc/self/maps") if "libnccl.so" in line})
assert len(paths) == 1, paths
print(paths[0], version, torch.version.cuda.split(".")[0])
')
libdir=$(dirname "$lib")
echo "torch loads NCCL $version from $lib"

# nccl-tests' common.h includes nccl_device.h from NCCL 2.28 on.
need=(nccl.h)
if printf '2.28.0\n%s\n' "$version" | sort -V -C; then
    need+=(nccl_device.h)
fi
inc="$libdir/../include"
for h in "${need[@]}"; do
    [[ -f $inc/$h ]] || inc=""
done
if [[ -z $inc ]]; then
    # NVIDIA's CUDA apt repo, which the image already uses, carries the same NCCL release built for a later CUDA
    # minor. Only its headers are used: the binaries still load the wheel's library through LD_LIBRARY_PATH.
    echo "the wheel has no ${need[*]}; taking NCCL $version headers from libnccl-dev"
    apt-get update -qq
    pkg=$(apt-cache madison libnccl-dev | awk -v want="$version-1+cuda$cuda_major." '
        index($3, want) == 1 { print $3; exit }')
    if [[ -z $pkg ]]; then
        echo "no libnccl-dev $version for CUDA $cuda_major in the apt repo" >&2
        exit 1
    fi
    mkdir -p "$build/deb"
    (cd "$build/deb" && apt-get download "libnccl-dev=$pkg" && dpkg -x libnccl-dev_*.deb .)
    inc="$build/deb/usr/include"
fi

# NCCL_HOME needs include/ and an unversioned lib/libnccl.so for -lnccl; the wheel ships only libnccl.so.2.
mkdir -p "$build/nccl/lib" "$build/src"
ln -sfn "$(cd "$inc" && pwd)" "$build/nccl/include"
ln -sf "$lib" "$build/nccl/lib/libnccl.so"

# A stalled transfer fails here, not at setup.sh's outer limit.
curl -fsSL --retry 3 --connect-timeout 20 --max-time 300 "https://codeload.github.com/NVIDIA/nccl-tests/tar.gz/$sha" \
    -o "$build/nccl-tests.tar.gz"
sha256sum "$build/nccl-tests.tar.gz"
tar -xzf "$build/nccl-tests.tar.gz" -C "$build/src" --strip-components=1

# One architecture keeps the build to a couple of minutes; every GPU this issue rents is sm_120.
make -C "$build/src/src" -j"$(nproc)" \
    BUILDDIR="$build/bin" NCCL_HOME="$build/nccl" CUDA_HOME="$cuda_home" \
    NVCC_GENCODE="-gencode=arch=compute_120,code=sm_120" \
    "$build/bin/all_reduce_perf" "$build/bin/sendrecv_perf"

LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$build/bin/all_reduce_perf" | grep -E 'nccl|cudart'
printf '%s\n' "$libdir" >"$build/nccl_lib_dir"
echo "nccl-tests $sha built against NCCL $version"
