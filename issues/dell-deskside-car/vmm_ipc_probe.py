"""Exports one PyTorch allocation over CUDA IPC, the call custom_all_reduce.cuh:164 makes on every CUDA-graph buffer.

PyTorch reads PYTORCH_CUDA_ALLOC_CONF when CUDA starts, so run one process per setting. cudaIpcGetMemHandle acts on
this GPU's own memory, so peer access plays no part in the result.
"""

import ctypes
import os

import torch
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "unset")
# A fresh 64 MiB tensor starts a segment of its own, so its pointer is an allocation base, as the .cuh passes.
buf = torch.empty(64 << 20, dtype=torch.uint8, device="cuda:0")
ptr = buf.data_ptr()
# The segment's own flag shows whether the setting took, so a pass under True cannot be a setting that never applied.
segments = torch.cuda.memory._snapshot().get("segments", [])
expandable = next(
    (s.get("is_expandable") for s in segments if s["address"] <= ptr < s["address"] + s["total_size"]), None
)
try:
    CudaRTLibrary().cudaIpcGetMemHandle(ctypes.c_void_p(ptr))
    print(f"vmm_ipc_probe {conf} (segment expandable: {expandable}): cudaIpcGetMemHandle ok")
except RuntimeError as err:
    print(f"vmm_ipc_probe {conf} (segment expandable: {expandable}): cudaIpcGetMemHandle failed: {err}")
