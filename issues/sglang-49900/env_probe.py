"""Writes <out>/env.json: the GPU as torch and sglang see it, the package versions, which block-FP8 [32, 32] tables
the installed sglang ships for this device (none means every shape runs the default config), and which MXFP8
backends FlashInfer reports for it."""

import json
import sys
from importlib.metadata import version
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.gemm import fp8_kernel
from sglang.srt.utils import get_device_name


def main():
    out = Path(sys.argv[1])
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    device = get_device_name().replace(" ", "_")
    configs = Path(fp8_kernel.__file__).parent / "configs"
    env = {
        "device": torch.cuda.get_device_name(0),
        "capability": f"{major}.{minor}",
        "sms": props.multi_processor_count,
        "l2_mb": round(getattr(props, "L2_cache_size", 0) / 2**20, 1),
        "memory_gib": round(props.total_memory / 2**30, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "sglang": version("sglang"),
        "flashinfer": version("flashinfer-python"),
        "fp8_kernel": str(Path(fp8_kernel.__file__).resolve()),
        "device_tables": sorted(
            p.name for p in configs.iterdir() if f"device_name={device}," in p.name and "block_shape=[32, 32]" in p.name
        ),
    }
    try:
        from flashinfer import mm_mxfp8

        env["mm_mxfp8"] = {b: mm_mxfp8.is_backend_supported(b, major * 10 + minor) for b in ("cutlass", "b12x")}
    except Exception as e:  # informational; the mxfp8 arms report their own failure
        env["mm_mxfp8"] = repr(e)[:200]
    (out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
