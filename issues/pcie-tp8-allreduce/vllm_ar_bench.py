"""All-reduce at decode sizes on 8 PCIe GPUs through vLLM v0.33.0's own communicators, in CUDA graphs, checked exactly.

setup.sh and run_sweep.sh run one backend per torchrun, so a hang in one cannot take the others with it:

  probe        which all-reduce vLLM picks by itself at TP8 on this box, and why; forces nothing
  nccl         pynccl (what vLLM uses here) and torch.distributed; run once per NCCL env arm
  custom       vLLM's custom all-reduce with its NVLink gate bypassed, one-shot and two-shot
  fi_pcie_ipc  the opt-in FlashInfer PCIe IPC backend (VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC, vllm#53576)

Each size is captured as LAYERS all-reduces on separate buffers in one graph, the shape of a decode step. Every rank
adds (rank + 1) times a small integer pattern that changes on every replay, so the exact sum is known, every partial
sum is exact in bf16, and a rank that reads a peer's previous buffer shows up as a mismatch. Some replays start late on
some ranks, so the barriers also see ranks arrive out of order.

Sizes are nccl-tests' powers of two from 4 KB to 4 MB plus decode batches 1 to 256 at --hidden, the served model's
hidden size; FlashInfer accepts only the decode shapes.

Rank 0 appends one JSON row per backend and size to <out>/rows.jsonl. Each rank arms faulthandler before every step:
a step that runs past its limit writes every thread's stack to <out>/<arm>.fault.rank<N>.txt, under a line naming the
step, and exits, so a hang is recorded instead of holding the box.

Usage: torchrun --standalone --nproc_per_node=8 vllm_ar_bench.py --backend nccl --arm nccl-default --out <dir> \
           [--hidden 5120]
"""

import argparse
import contextlib
import faulthandler
import json
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist

DEFAULT_HIDDEN = 5120  # DeepSeek-V4.1-Flash, the model of the public TP8 trace (vllm#59203)
DECODE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
POW2_BYTES = tuple(4096 << i for i in range(11))  # 4 KB .. 4 MB, nccl-tests' ladder
LAYERS = 8
WARMUP_REPLAYS = 20
# Tens of microseconds at the PRO 6000's clocks: a rank that sleeps arrives at the barrier after its peers.
JITTER_CYCLES = 50_000
# Covers importing vLLM in 8 processes at once and building the groups.
INIT_LIMIT_S = 300
STEP_LIMIT_S = 120
# FlashInfer may JIT-build its PCIe IPC module, then tunes every capture size.
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

    def disarm(self) -> None:
        faulthandler.cancel_dump_traceback_later()


@dataclass
class Run:
    arm: str
    out: Path
    args: argparse.Namespace
    dog: Watchdog
    device: torch.device
    cpu_group: dist.ProcessGroup
    nccl_group: dist.ProcessGroup
    side: torch.cuda.Stream
    # Custom all-reduce IPC-registers every graph input, so no graph or input may be freed before the run ends.
    keep: list = field(default_factory=list)
    closers: list = field(default_factory=list)

    def emit(self, **row) -> None:
        """Rank 0 only; the numbers in a row were already gathered from every rank."""
        if dist.get_rank() != 0:
            return
        line = json.dumps({"arm": self.arm, **row})
        print(line, flush=True)
        with open(self.out / "rows.jsonl", "a") as f:
            f.write(line + "\n")


def comm_env() -> dict:
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(("NCCL_", "VLLM_", "CUDA_"))}


def shapes(hidden: int) -> list[tuple[int, int]]:
    decode = [(b, hidden) for b in DECODE_BATCHES]
    return sorted(decode + [(1, n // 2) for n in POW2_BYTES], key=lambda s: s[0] * s[1])


def measure(run: Run, reduce_fn: Callable, shape: tuple[int, int], capture_ctx: Callable, inplace: bool):
    """Captures LAYERS all-reduces of `shape`, checks them over check_iters replays, then times replays."""
    args, side = run.args, run.side
    rank, world = dist.get_rank(), dist.get_world_size()
    rows, cols = shape
    idx = torch.arange(rows * cols, device=run.device).view(1, rows, cols)
    layer = torch.arange(LAYERS, device=run.device).view(LAYERS, 1, 1)
    total = world * (world + 1) // 2
    # Pattern k serves the replays with t % 4 == k, so consecutive replays always differ. Values stay at or under
    # 4 * total (144 at 8 ranks), well inside the integers bf16 holds exactly (256).
    bases = [(idx + layer + 3 * k) % 4 + 1 for k in range(4)]
    inputs = [(b * (rank + 1)).to(torch.bfloat16) for b in bases]
    expected = [(b * total).to(torch.bfloat16) for b in bases]
    del bases
    graph_inp = inputs[0].clone()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        for _ in range(3):
            for i in range(LAYERS):
                reduce_fn(graph_inp[i])
        with capture_ctx(), torch.cuda.graph(graph, stream=side):
            outs = [reduce_fn(graph_inp[i]) for i in range(LAYERS)]
        if any(o is None for o in outs):
            raise RuntimeError("the backend declined the input during capture")

        wrong = torch.zeros((), dtype=torch.int64, device=run.device)
        for t in range(args.check_iters):
            k = t % 4
            graph_inp.copy_(inputs[k])
            if (t + rank) % 5 == 0:
                torch.cuda._sleep(JITTER_CYCLES)
            graph.replay()
            wrong += torch.stack(outs).ne(expected[k]).sum()

        for _ in range(WARMUP_REPLAYS):
            graph.replay()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(args.trials):
            start.record()
            for _ in range(args.replays):
                graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / (args.replays * LAYERS))
        checks = args.check_iters
        if not inplace:
            # The timed replays reran the last input back to back with no refill between them, so a race under
            # sustained load leaves wrong sums here. An in-place backend compounded its buffers instead.
            wrong += torch.stack(outs).ne(expected[(args.check_iters - 1) % 4]).sum()
            checks += 1
        side.synchronize()

    per_rank = [None] * world
    dist.all_gather_object(per_rank, (statistics.median(times), min(times), max(times), int(wrong.item())),
                           group=run.cpu_group)
    # A collective ends together on every rank; the slowest rank's median is what a decode step waits for.
    time_us = max(r[0] for r in per_rank)
    algbw = rows * cols * 2 / (time_us * 1e-6) / 1e9
    fields = {
        "time_us": round(time_us, 2),
        "time_us_min": round(min(r[1] for r in per_rank), 2),
        "time_us_max": round(max(r[2] for r in per_rank), 2),
        "algbw_gbps": round(algbw, 3),
        # nccl-tests' bus bandwidth, so the two tools' columns compare.
        "busbw_gbps": round(algbw * 2 * (world - 1) / world, 3),
        "wrong": sum(r[3] for r in per_rank),
        "checked_elements": checks * LAYERS * rows * cols * world,
    }
    return fields, (graph, graph_inp, outs)


def sweep(run: Run, name: str, reduce_fn: Callable, accepts: Callable, capture_ctx: Callable, inplace: bool) -> bool:
    """Returns False after an error, when the communicator can no longer be trusted."""
    for shape in shapes(run.args.hidden):
        rows, cols = shape
        row = {"backend": name, "size_bytes": rows * cols * 2, "rows": rows, "cols": cols, "graph": True,
               "inplace": inplace}
        if not accepts(torch.empty(shape, dtype=torch.bfloat16, device=run.device)):
            run.emit(**row, status="declined")
            continue
        run.dog.arm(STEP_LIMIT_S, f"{name} {row['size_bytes']} B")
        try:
            fields, objs = measure(run, reduce_fn, shape, capture_ctx, inplace)
        except Exception as e:
            # Recorded, and the sweep stops: a failed capture leaves custom all-reduce and FlashInfer half-registered.
            run.emit(**row, status=f"error: {type(e).__name__}: {e}")
            return False
        run.keep.append(objs)
        run.emit(**row, **fields, status="ok")
    return True


def run_probe(run: Run) -> None:
    from vllm.distributed.device_communicators import custom_all_reduce as car
    from vllm.distributed.device_communicators import flashinfer_pcie_ipc_all_reduce as fi_ipc
    from vllm.distributed.device_communicators.all_reduce_utils import SYMM_MEM_ALL_REDUCE_MAX_SIZES
    from vllm.distributed.device_communicators.flashinfer_all_reduce import FlashInferAllReduce
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.device_communicators.symm_mem import SymmMemCommunicator
    from vllm.platforms import current_platform

    world = dist.get_world_size()
    capability = current_platform.get_device_capability().as_version_str()
    physical = [current_platform.visible_device_id_to_physical_device_id(i) for i in range(world)]
    # The constructors vLLM's TP communicator runs at startup (cuda_communicator.py:106-166), nothing forced. Each
    # returns disabled at its first failed check, before it allocates anything shared.
    enabled = {
        "flashinfer_ar": not FlashInferAllReduce(group=run.cpu_group, device=run.device).disabled,
        "custom_ar": not car.CustomAllreduce(group=run.cpu_group, device=run.device).disabled,
        "symm_mem": not SymmMemCommunicator(group=run.cpu_group, device=run.device).disabled,
    }
    pynccl = PyNcclCommunicator(group=run.cpu_group, device=run.device)
    enabled["pynccl"] = not pynccl.disabled
    with open("/proc/self/maps") as maps:
        libnccl = sorted({line.split()[-1] for line in maps if "libnccl.so" in line})
    run.emit(
        event="probe",
        capability=capability,
        physical_ids=physical,
        nvml_nvlink_fully_connected=current_platform.is_fully_connected(physical),
        peer_access=[[i == j or torch.cuda.can_device_access_peer(i, j) for j in range(world)] for i in range(world)],
        multicast=car._has_local_multicast_support(run.device),
        symm_mem_table_has_capability=capability in SYMM_MEM_ALL_REDUCE_MAX_SIZES,
        flashinfer_has_pcie_ipc=fi_ipc._pcie_ipc_available,
        enabled=enabled,
        # CudaCommunicator.all_reduce tries them in this order (cuda_communicator.py:362-438); the PCIe IPC backend
        # goes first, but only with VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1.
        vllm_default_pick=next((name for name, on in enabled.items() if on), "torch.distributed"),
        torch_nccl=".".join(map(str, torch.cuda.nccl.version())),
        pynccl_nccl=None if pynccl.disabled else pynccl.nccl.ncclGetVersion(),
        libnccl=libnccl,
        env=comm_env(),
    )


def run_nccl(run: Run) -> None:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    pynccl = PyNcclCommunicator(group=run.cpu_group, device=run.device)
    run.emit(backend="pynccl", event="init", disabled=pynccl.disabled,
             nccl=None if pynccl.disabled else pynccl.nccl.ncclGetVersion(), env=comm_env())

    def torch_reduce(x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, group=run.nccl_group)
        return x

    if not pynccl.disabled:
        sweep(run, "pynccl", lambda x: pynccl.all_reduce(x, stream=torch.cuda.current_stream()), lambda x: True,
              contextlib.nullcontext, inplace=False)
    sweep(run, "torch", torch_reduce, lambda x: True, contextlib.nullcontext, inplace=True)


def run_custom(run: Run) -> None:
    from vllm.platforms import current_platform

    # custom_all_reduce.py:253-273 (v0.33.0) turns custom all-reduce off above two GPUs unless NVML reports one-hop
    # NVLink between every pair (platforms/cuda.py:816-838); that is the only check bypassed here, and the P2P test
    # after it (:274-288) still runs. The same flag gates should_custom_ar (:509-511) and the default kernel choice
    # (csrc/custom_all_reduce.cuh:300), so patching the one place it is computed covers all three.
    type(current_platform).is_fully_connected = classmethod(lambda cls, ids: True)
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    ca = CustomAllreduce(group=run.cpu_group, device=run.device)
    run.emit(backend="custom_ar", event="init", disabled=ca.disabled, max_size=getattr(ca, "max_size", None),
             env=comm_env())
    if ca.disabled:
        return
    run.closers.append(ca.close)
    for algo in ("1stage", "2stage"):
        # The launcher reads this on every call (csrc/custom_all_reduce.cuh:270-286), so each capture bakes in its
        # kernel.
        os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] = algo
        # Both algorithms share one communicator, whose graph-buffer registration a failed capture leaves half done.
        if not sweep(run, f"custom_ar_{algo}", ca.custom_all_reduce, ca.should_custom_ar, ca.capture, inplace=False):
            return


def run_fi_pcie_ipc(run: Run) -> None:
    from vllm.distributed.device_communicators import flashinfer_pcie_ipc_all_reduce as fi_ipc

    # vLLM builds it on the TP group's NCCL group and tunes on the CPU group (cuda_communicator.py:139-148), then sets
    # it up before graph capture from the model's hidden size and the capture sizes
    # (flashinfer_pcie_ipc_all_reduce.py:202-239). Only (batch, hidden) inputs reach it; the rest are declined.
    comm = fi_ipc.FlashInferPcieIpcAllReduce(group=run.nccl_group, tune_group=run.cpu_group, device=run.device)
    # destroy() does nothing without a workspace, and frees one a failed tune left behind.
    run.closers.append(comm.destroy)
    setup_s, status = None, "ok"
    if not comm.disabled:
        run.dog.arm(FI_SETUP_LIMIT_S, "fi_pcie_ipc setup and tune")
        start = time.monotonic()
        try:
            comm.setup(hidden_dim=run.args.hidden, dtype=torch.bfloat16, capture_sizes=DECODE_BATCHES,
                       tune_cache=run.out / "fi_pcie_ipc_tune.json")
        except Exception as e:
            # A JIT build, topology probe, or tune that fails is the answer for this backend. FlashInfer raises
            # the same error on every rank, so all of them reach teardown together.
            status = f"error: {type(e).__name__}: {e}"
        setup_s = round(time.monotonic() - start, 1)
    ws = comm.workspace
    run.emit(backend="fi_pcie_ipc", event="init", flashinfer_has_pcie_ipc=fi_ipc._pcie_ipc_available,
             initialized=comm.initialized, setup_s=setup_s, profile=getattr(ws, "profile", None),
             profile_reason=getattr(ws, "profile_reason", None), memop=getattr(ws, "memop_supported", None),
             env=comm_env(), status=status)
    if status != "ok" or not comm.initialized:
        return
    sweep(run, "fi_pcie_ipc", comm.all_reduce, comm.should_use, comm.capture, inplace=False)


BACKENDS = {"probe": run_probe, "nccl": run_nccl, "custom": run_custom, "fi_pcie_ipc": run_fi_pcie_ipc}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN, help="the served model's hidden size")
    parser.add_argument("--check-iters", type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--replays", type=int, default=100)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dog = Watchdog(args.out / f"{args.arm}.fault.rank{os.environ['RANK']}.txt")
    dog.arm(INIT_LIMIT_S, "init")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="gloo")
    cpu_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl", device_id=device)
    # NCCL cannot create its communicator inside a graph capture.
    dist.all_reduce(torch.zeros(1, device=device), group=nccl_group)
    run = Run(args.arm, args.out, args, dog, device, cpu_group, nccl_group, torch.cuda.Stream(device))

    BACKENDS[args.backend](run)

    dog.arm(STEP_LIMIT_S, "teardown")
    for close in run.closers:
        close()
    dist.barrier(group=cpu_group)
    dog.disarm()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
