"""One arm's all-reduce settings on this box, over all 8 GPUs, before any server boots: vLLM v0.33.0's pynccl
communicator (what every TP8 all-reduce runs through on PCIe unless FlashInfer takes it) and, when
VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1, FlashInfer's PCIe IPC workspace, set up and tuned as vLLM does before graph
capture (flashinfer_pcie_ipc_all_reduce.py:95-156 at v0.33.0).

setup.sh runs it once per arm, under that arm's env and a hard time limit, with NCCL_DEBUG=INFO and NCCL_DEBUG_FILE.
NCCL reads NCCL_P2P_LEVEL once per process, so each arm is its own run. A setting that hangs or fails here would hang
or fail that arm's server boot, where a hang costs boot_timeout_s, so setup ends the box before the 510 GB download
instead. Every rank adds (rank + 1) times a small integer pattern that changes on every check, so the exact sum is
known and exact in bf16, and one wrong element fails the probe. The times are eager launches, which carry launch
overhead the server's CUDA graphs do not: they compare arms on this box, not with stage 1's graph numbers.

Rank 0 writes <out>/<arm>.json. Each rank arms faulthandler before every step: a step past its limit writes every
thread's stack to <out>/<arm>.fault.rank<N>.txt and exits. Every rank leaves through os._exit after a gloo barrier,
since destroy_process_group hung after NCCL use in stage 1 (issues/pcie-tp8-allreduce/README.md, Results).
Usage: torchrun --standalone --nproc_per_node=8 ar_probe.py --arm <arm> --out <dir>
"""

import argparse
import faulthandler
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HIDDEN = 5120  # DeepSeek-V4.1-Flash
# Decode batches, then one prefill chunk at vLLM's default max_num_batched_tokens for a 96 GB GPU behind the OpenAI
# server (engine/arg_utils.py:2983-2995 at v0.33.0).
PYNCCL_BATCHES = (1, 8, 32, 128, 8192)
# vLLM sets the workspace up for every CUDA graph capture size, up to 256 with --max-num-seqs 128 (twice that, capped
# at 512); the powers of two cover the decode batches the rows reach and the workspace's largest size.
FI_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CHECKS = 20
TIMED = {1: 500, 8192: 20}
INIT_LIMIT_S = 300
STEP_LIMIT_S = 120
# Covers a FlashInfer JIT build of the PCIe IPC module on a cold cache; the tune itself took 14 s in stage 1.
FI_SETUP_LIMIT_S = 480


class Watchdog:
    """faulthandler's timer runs without the GIL, so it fires even while the main thread waits on a kernel that
    never finishes."""

    def __init__(self, path: Path) -> None:
        # Left open: faulthandler writes to it until the process ends.
        self.file = open(path, "w")

    def arm(self, seconds: int, step: str) -> None:
        faulthandler.cancel_dump_traceback_later()
        self.file.write(f"{time.strftime('%H:%M:%S')} {step} (limit {seconds} s)\n")
        self.file.flush()
        faulthandler.dump_traceback_later(seconds, exit=True, file=self.file)


def pattern(rows: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's input and the exact sum over all ranks. Values stay at or under 4 x 36, inside bf16's exact
    integers."""
    rank, world = dist.get_rank(), dist.get_world_size()
    base = (torch.arange(rows * HIDDEN, device=device).view(rows, HIDDEN) + 3 * k) % 4 + 1
    return (base * (rank + 1)).to(torch.bfloat16), (base * (world * (world + 1) // 2)).to(torch.bfloat16)


def check(reduce, rows: int, device: torch.device) -> int:
    wrong = 0
    for k in range(CHECKS):
        inp, expected = pattern(rows, k, device)
        wrong += int(reduce(inp).ne(expected).sum())
    return wrong


def eager_us(reduce, rows: int, device: torch.device, cpu_group) -> float:
    """The slowest rank's mean, since a collective ends together on every rank."""
    inp, _ = pattern(rows, 0, device)
    for _ in range(10):
        reduce(inp)
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    start = time.perf_counter()
    for _ in range(TIMED[rows]):
        reduce(inp)
    torch.cuda.synchronize()
    per_rank = [None] * dist.get_world_size()
    dist.all_gather_object(per_rank, (time.perf_counter() - start) / TIMED[rows] * 1e6, group=cpu_group)
    return round(max(per_rank), 1)


def total_wrong(wrong: int, cpu_group) -> int:
    per_rank = [None] * dist.get_world_size()
    dist.all_gather_object(per_rank, wrong, group=cpu_group)
    return sum(per_rank)


def probe_pynccl(dog: Watchdog, device: torch.device, cpu_group) -> dict:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    # The constructor vLLM's TP communicator runs (cuda_communicator.py:108): NCCL's init and transport choice.
    dog.arm(INIT_LIMIT_S, "pynccl init")
    comm = PyNcclCommunicator(group=cpu_group, device=device)
    if comm.disabled:
        return {"ok": False, "reason": "pynccl disabled"}

    def reduce(x: torch.Tensor) -> torch.Tensor:
        return comm.all_reduce(x, stream=torch.cuda.current_stream())

    result = {"nccl": comm.nccl.ncclGetVersion(), "wrong": {}, "eager_us": {}}
    for rows in PYNCCL_BATCHES:
        dog.arm(STEP_LIMIT_S, f"pynccl check batch {rows}")
        result["wrong"][rows] = total_wrong(check(reduce, rows, device), cpu_group)
        if rows in TIMED:
            dog.arm(STEP_LIMIT_S, f"pynccl time batch {rows}")
            result["eager_us"][rows] = eager_us(reduce, rows, device, cpu_group)
    result["ok"] = not any(result["wrong"].values())
    return result


def probe_flashinfer(dog: Watchdog, device: torch.device, cpu_group, tune_cache: Path) -> dict:
    from vllm.distributed.device_communicators import flashinfer_pcie_ipc_all_reduce as fi_ipc

    # vLLM builds the workspace on the TP group's NCCL group and tunes over the CPU group (cuda_communicator.py:144).
    dog.arm(INIT_LIMIT_S, "flashinfer nccl group")
    nccl_group = dist.new_group(backend="nccl", device_id=device)
    dist.all_reduce(torch.zeros(1, device=device), group=nccl_group)
    comm = fi_ipc.FlashInferPcieIpcAllReduce(group=nccl_group, tune_group=cpu_group, device=device)
    if comm.disabled:
        return {"ok": False, "reason": "FlashInferPcieIpcAllReduce disabled itself at construction (see the log)"}
    dog.arm(FI_SETUP_LIMIT_S, "flashinfer setup and tune")
    start = time.monotonic()
    comm.setup(hidden_dim=HIDDEN, dtype=torch.bfloat16, capture_sizes=FI_BATCHES, tune_cache=tune_cache)
    ws = comm.workspace
    result = {
        "setup_s": round(time.monotonic() - start, 1),
        "profile": getattr(ws, "profile", None),
        "profile_reason": getattr(ws, "profile_reason", None),
        "wrong": {},
        "eager_us": {},
    }
    if not comm.initialized:
        return {**result, "ok": False, "reason": "workspace not initialized after setup"}
    for rows in FI_BATCHES:
        inp, _ = pattern(rows, 0, device)
        if not comm.should_use(inp):
            return {**result, "ok": False, "reason": f"declined batch {rows}"}
        dog.arm(STEP_LIMIT_S, f"flashinfer check batch {rows}")
        result["wrong"][rows] = total_wrong(check(comm.all_reduce, rows, device), cpu_group)
    dog.arm(STEP_LIMIT_S, "flashinfer time batch 1")
    result["eager_us"][1] = eager_us(comm.all_reduce, 1, device, cpu_group)
    dog.arm(STEP_LIMIT_S, "flashinfer destroy")
    comm.destroy()
    result["ok"] = not any(result["wrong"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    rank = int(os.environ["RANK"])
    dog = Watchdog(args.out / f"{args.arm}.fault.rank{rank}.txt")
    dog.arm(INIT_LIMIT_S, "init")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="gloo")
    cpu_group = dist.new_group(backend="gloo")

    use_fi = os.environ.get("VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC") == "1"
    report = {
        "arm": args.arm,
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("NCCL_", "VLLM_ALLREDUCE"))},
        "pynccl": probe_pynccl(dog, device, cpu_group),
    }
    if use_fi:
        report["flashinfer"] = probe_flashinfer(dog, device, cpu_group, args.out / f"{args.arm}.fi_tune.json")
    report["ok"] = report["pynccl"]["ok"] and (not use_fi or report["flashinfer"]["ok"])

    dog.arm(STEP_LIMIT_S, "exit barrier")
    dist.barrier(group=cpu_group)
    if rank == 0:
        report["seconds"] = round(time.monotonic() - started, 1)
        (args.out / f"{args.arm}.json").write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
