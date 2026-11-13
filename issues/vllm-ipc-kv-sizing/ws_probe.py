"""FlashInfer's PCIe IPC all-reduce workspace on its own, at the shapes the server arms use, before any server boots.
For each shape: the slab FlashInfer's own workspace_size reports (what create_shared_buffer cudaMallocs on each rank),
and what building the workspace takes from the device per rank (cudaMemGetInfo with torch's cache emptied), so
README.md's H2 can split an arm's KV loss into the slab and anything else.

- warm: an 8-token workspace first, which pays the one-time costs (the JIT module and its kernels loading), so the
  shapes after it read only what each workspace adds.
- e: vLLM's own wrapper, set up the way kernel_warmup does it before graph capture
  (flashinfer_pcie_ipc_all_reduce.py:95-156 at f5d93296c): the slab, the tune over the e arms' capture sizes, the
  prepare. 480 x 4096 elements is the reporter's DSV4.1 DSpark shape, 384 x 5120. torch's allocated peak during setup
  is the tune's transient.
- m: the m and s arms' 8192 x 4096 slab alone; the m1 boot pays its tune.

Rank 0 writes <out>/ws_probe.json and prints one line per shape. The IPC kernels spin on peer flags with no timeout,
so a step past its limit dumps every thread's stack to <out>/fault.rank<N>.txt and exits. Every rank leaves through
os._exit after a gloo barrier, since destroy_process_group hung after NCCL use (issues/pcie-tp8-allreduce/README.md).
Usage: torchrun --standalone --nproc_per_node=4 ws_probe.py --out <dir>
"""

import argparse
import faulthandler
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist

HIDDEN = 4096  # Qwen3-8B
# What vLLM captures with --max-cudagraph-capture-size 480 (config/vllm.py:2491-2503 at f5d93296c); the tune covers
# every size, the slab is sized by the last.
E_SIZES = (1, 2, 4, *range(8, 256, 8), *range(256, 481, 16))
M_TOKENS = 8192
# Covers a cold JIT build of the module on the first shape.
STEP_LIMIT_S = 600
MIB = 1 << 20


def free_bytes() -> int:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0]


def measure(build) -> dict:
    """build() returns (workspace, destroy). Free memory is read with torch's cache emptied, so torch blocks the tune
    left cached do not count as held."""
    from flashinfer.comm.pcie_ipc_ar import get_pcie_ipc_comm_module

    before = free_bytes()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    workspace, destroy = build()
    seconds = time.monotonic() - start
    transient = torch.cuda.max_memory_allocated() - base
    held = before - free_bytes()
    row = {
        "max_numel": workspace.max_numel,
        "slab_bytes": get_pcie_ipc_comm_module().workspace_size(
            workspace.world_size, workspace.max_numel, workspace.elem_size, workspace.max_blocks
        ),
        "memop_supported": workspace.memop_supported,
        "profile": workspace.profile,
        "held_bytes": held,
        "torch_transient_bytes": transient,
        "seconds": round(seconds, 1),
    }
    destroy()
    row["kept_after_destroy_bytes"] = before - free_bytes()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    fault_file = open(args.out / f"fault.rank{rank}.txt", "w")

    def step(name: str) -> None:
        faulthandler.cancel_dump_traceback_later()
        fault_file.write(f"{time.strftime('%H:%M:%S')} {name}\n")
        fault_file.flush()
        faulthandler.dump_traceback_later(STEP_LIMIT_S, exit=True, file=fault_file)

    step("init")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="gloo")
    cpu_group = dist.new_group(backend="gloo")
    # vLLM builds the workspace on the TP group's NCCL group and tunes over its CPU group
    # (cuda_communicator.py:140-149). NCCL allocates its buffers on the first collective, so that runs before any
    # baseline is read.
    nccl_group = dist.new_group(backend="nccl", device_id=device)
    dist.all_reduce(torch.zeros(1, device=device), group=nccl_group)

    from flashinfer.comm import PcieIpcAllReduceWorkspace

    from vllm.distributed.device_communicators.flashinfer_pcie_ipc_all_reduce import FlashInferPcieIpcAllReduce

    def bare(tokens: int):
        def build():
            workspace = PcieIpcAllReduceWorkspace(group=nccl_group, max_numel=tokens * HIDDEN, dtype=torch.bfloat16)
            return workspace, workspace.destroy

        return build

    def wrapped():
        comm = FlashInferPcieIpcAllReduce(group=nccl_group, tune_group=cpu_group, device=device)
        comm.setup(hidden_dim=HIDDEN, dtype=torch.bfloat16, capture_sizes=E_SIZES, tune_cache=args.out / "tune_e.json")
        if not comm.initialized:
            raise RuntimeError("vLLM's wrapper left the workspace uninitialized; see the log")
        return comm.workspace, comm.destroy

    rows = {}
    for name, build in (("warm", bare(8)), ("e", wrapped), ("m", bare(M_TOKENS))):
        step(name)
        rows[name] = measure(build)

    step("gather")
    per_rank = [None] * dist.get_world_size()
    dist.all_gather_object(per_rank, rows, group=cpu_group)
    if rank == 0:
        report = {
            "gpu": torch.cuda.get_device_name(device),
            "vllm": version("vllm"),
            "flashinfer": version("flashinfer-python"),
            "hidden": HIDDEN,
            "per_rank": per_rank,
        }
        (args.out / "ws_probe.json").write_text(json.dumps(report, indent=1) + "\n")
        for name in rows:
            shape = [r[name] for r in per_rank]
            print(
                f"{name}: max_numel {shape[0]['max_numel']}, slab {shape[0]['slab_bytes'] / MIB:.1f} MiB, held"
                f" {min(s['held_bytes'] for s in shape) / MIB:.1f}-{max(s['held_bytes'] for s in shape) / MIB:.1f}"
                f" MiB, torch transient {max(s['torch_transient_bytes'] for s in shape) / MIB:.1f} MiB, kept after"
                f" destroy {max(s['kept_after_destroy_bytes'] for s in shape) / MIB:.1f} MiB,"
                f" {max(s['seconds'] for s in shape)} s, memop {shape[0]['memop_supported']}",
                flush=True,
            )
    step("exit barrier")
    dist.barrier(group=cpu_group)
    os._exit(0)


if __name__ == "__main__":
    main()
