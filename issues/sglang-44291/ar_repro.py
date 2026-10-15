"""Minimal repro for sglang#44291: v1 CustomAllreduce under CUDA-graph capture on 2x PCIe GPUs.

Run with 2 ranks. The allocator mode comes from the caller's environment, e.g.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node 2 ar_repro.py --mode registered

Modes:
  registered  v1 as shipped: graph inputs are IPC-registered after capture
  staged      v1 with graph inputs copied into its pre-registered IPC buffer (proposed fix path)
  pynccl      sglang's PyNcclCommunicator, what a server uses inside CUDA graphs without custom all-reduce
  nccl        torch.distributed all_reduce on an NCCL group, captured the same way

Every rank prints JSON lines: first which all-reduce class sglang's dispatcher picks on this host, then one
line per size with rank, mode, vmm, kb, ok, max_abs_err, us, error.
"""

import argparse
import contextlib
import datetime
import json
import os

import torch
import torch.distributed as dist

LAYERS = 8  # all-reduces per captured graph, roughly a few decoder layers
REPLAYS = 200


def init_groups():
    import sglang.srt.distributed.parallel_state as ps
    from sglang.srt.runtime_context import get_parallel

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    # CustomAllreduce's eligibility checks read sglang's world group; this mirrors sglang's own AR tests.
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    device = torch.device(f"cuda:{local_rank}")
    nccl_group = dist.new_group(backend="nccl", device_id=device)
    return device, coord.cpu_group, nccl_group


def emit(**row):
    print(json.dumps({"rank": dist.get_rank(), **row}), flush=True)


def capture(reduce_fn, graph_inp, ctx):
    graph = torch.cuda.CUDAGraph()
    outs = []
    with ctx:
        with torch.cuda.graph(graph):
            for i in range(LAYERS):
                outs.append(reduce_fn(graph_inp[i]))
    torch.cuda.synchronize()
    return graph, outs


def max_err(outs, ref):
    return (torch.stack(outs).float() - ref.float()).abs().max().item()


def run_size(mode, comm, device, nccl_group, kb):
    numel = kb * 1024 // 2  # bf16
    graph_inp = torch.zeros((LAYERS, numel), dtype=torch.bfloat16, device=device)
    if mode == "nccl":
        def reduce_fn(x):
            dist.all_reduce(x, group=nccl_group)
            return x
        ctx = contextlib.nullcontext()
    elif mode == "pynccl":
        reduce_fn = comm.outplace_all_reduce
        ctx = comm.change_state(enable=True)
    else:
        comm.tms_cudagraph = mode == "staged"
        reduce_fn = comm.custom_all_reduce
        ctx = comm.capture()
    graph, outs = capture(reduce_fn, graph_inp, ctx)
    if any(o is None for o in outs):
        raise RuntimeError(f"{mode} declined the input")

    torch.manual_seed(1234 + dist.get_rank())
    # Small integers stay exact in bf16 after a 2-rank sum, so the check can demand zero error.
    x = torch.randint(0, 16, (LAYERS, numel), device=device).to(torch.bfloat16)
    ref = x.clone()
    dist.all_reduce(ref, group=nccl_group)
    graph_inp.copy_(x)
    graph.replay()
    torch.cuda.synchronize()
    err = max_err(outs, ref)

    for _ in range(10):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000 / (REPLAYS * LAYERS)

    if mode != "nccl":
        # Out-of-place modes recompute from the same input every replay, so a race would show up here;
        # nccl reduces graph_inp in place and its values compound across replays.
        err = max(err, max_err(outs, ref))
    return err, us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["registered", "staged", "pynccl", "nccl"], required=True)
    # Decode with a small model reduces a few KB per layer; that is where a staging copy costs the most.
    parser.add_argument("--sizes-kb", default="2,4,8,16,64,256,1024,4096,8192")
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes_kb.split(",")]

    device, cpu_group, nccl_group = init_groups()
    from sglang.srt.distributed.device_communicators import custom_all_reduce as car
    from sglang.srt.utils.cuda_vmm_utils import is_vmm_pointer

    # The probe must stay alive while the driver is asked about its pointer.
    probe = torch.empty(1, dtype=torch.uint8, device=device)
    vmm = is_vmm_pointer(probe.data_ptr())
    del probe
    # Same collective calls on every rank, so it is safe before the modes diverge.
    emit(mode=args.mode, vmm=vmm, server_pick=car.dispatch_custom_allreduce(cpu_group, device).__name__)
    # Create the NCCL communicator outside capture; NCCL cannot initialize inside a graph.
    dist.all_reduce(torch.zeros(1, device=device), group=nccl_group)

    comm = None
    if args.mode == "pynccl":
        from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

        comm = PyNcclCommunicator(group=cpu_group, device=device)
    elif args.mode in ("registered", "staged"):
        try:
            comm = car.CustomAllreduce(cpu_group, device)
        except Exception as e:
            # Init IPC-shares plain cudaMalloc buffers, so failing here points at H2, not H1.
            emit(mode=args.mode, vmm=vmm, ok=False, error=f"init {type(e).__name__}: {e}")
            raise
        if comm.disabled:
            from sglang.srt.distributed.device_communicators import custom_all_reduce_utils

            emit(
                mode=args.mode,
                vmm=vmm,
                ok=False,
                error="CustomAllreduce disabled at init",
                custom_ar_lib=car.ops.IS_CUSTOM_AR_AVAILABLE,
                p2p=str(getattr(custom_all_reduce_utils, "_gpu_p2p_access_cache", None)),
            )
            return

    try:
        for kb in sizes:
            try:
                err, us = run_size(args.mode, comm, device, nccl_group, kb)
                emit(mode=args.mode, vmm=vmm, kb=kb, ok=err == 0, max_abs_err=err, us=round(us, 2))
            except Exception as e:
                emit(mode=args.mode, vmm=vmm, kb=kb, ok=False, error=f"{type(e).__name__}: {e}")
                # A failed registration leaves v1's graph-buffer state half-built and the CUDA error flag set.
                break
        # A rank stuck in a faulted kernel never arrives; the timeout makes the other rank exit so torchrun
        # tears both down instead of waiting out gloo's two-hour default.
        dist.monitored_barrier(group=cpu_group, timeout=datetime.timedelta(seconds=120), wait_all_ranks=True)
    finally:
        # Close before destroying the groups: v1's destructor still queries its process group.
        if comm is not None and hasattr(comm, "close"):
            comm.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
