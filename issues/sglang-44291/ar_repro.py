"""Minimal repro for sglang#44291: v1 CustomAllreduce under CUDA-graph capture on 2x PCIe GPUs.

Run with 2 ranks. The allocator mode comes from the caller's environment, e.g.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node 2 ar_repro.py --mode registered

Modes:
  registered  v1 as shipped: graph inputs are IPC-registered after capture
  staged      v1 with graph inputs copied into its pre-registered IPC buffer (proposed fix path)
  nccl        torch.distributed all_reduce on an NCCL group, captured the same way (baseline)

Every rank prints one JSON line per size: rank, mode, vmm, kb, ok, max_abs_err, us, error.
"""

import argparse
import contextlib
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


def run_size(mode, comm, device, nccl_group, kb):
    numel = kb * 1024 // 2  # bf16
    graph_inp = torch.zeros((LAYERS, numel), dtype=torch.bfloat16, device=device)
    if mode == "nccl":
        def reduce_fn(x):
            dist.all_reduce(x, group=nccl_group)
            return x
        ctx = contextlib.nullcontext()
    else:
        comm.tms_cudagraph = mode == "staged"
        reduce_fn = comm.custom_all_reduce
        ctx = comm.capture()
    graph, outs = capture(reduce_fn, graph_inp, ctx)
    if any(o is None for o in outs):
        raise RuntimeError("custom_all_reduce declined the input (should_custom_ar is False)")

    torch.manual_seed(1234 + dist.get_rank())
    # Small integers stay exact in bf16 after a 2-rank sum, so the check can demand zero error.
    x = torch.randint(0, 16, (LAYERS, numel), device=device).to(torch.bfloat16)
    ref = x.clone()
    dist.all_reduce(ref, group=nccl_group)
    graph_inp.copy_(x)
    graph.replay()
    torch.cuda.synchronize()
    err = (torch.stack(outs).float() - ref.float()).abs().max().item()

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
    return err, us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["registered", "staged", "nccl"], required=True)
    parser.add_argument("--sizes-kb", default="16,64,256,1024,4096,8192")
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes_kb.split(",")]

    device, cpu_group, nccl_group = init_groups()
    from sglang.srt.utils.cuda_vmm_utils import is_vmm_pointer

    vmm = is_vmm_pointer(torch.empty(1, dtype=torch.uint8, device=device).data_ptr())
    # Create the NCCL communicator outside capture; NCCL cannot initialize inside a graph.
    dist.all_reduce(torch.zeros(1, device=device), group=nccl_group)

    comm = None
    if args.mode != "nccl":
        from sglang.srt.distributed.device_communicators.custom_all_reduce import CustomAllreduce

        comm = CustomAllreduce(cpu_group, device)
        if comm.disabled:
            # Init already IPC-shares plain cudaMalloc buffers, so this also tells H2 apart.
            emit(mode=args.mode, vmm=vmm, ok=False, error="CustomAllreduce disabled at init")
            return

    for kb in sizes:
        try:
            err, us = run_size(args.mode, comm, device, nccl_group, kb)
            emit(mode=args.mode, vmm=vmm, kb=kb, ok=err == 0, max_abs_err=err, us=round(us, 2))
        except Exception as e:
            emit(mode=args.mode, vmm=vmm, kb=kb, ok=False, error=f"{type(e).__name__}: {e}")
            # A failed registration leaves v1's graph-buffer state half-built; later sizes would mislead.
            break
    dist.barrier(group=cpu_group)


if __name__ == "__main__":
    main()
