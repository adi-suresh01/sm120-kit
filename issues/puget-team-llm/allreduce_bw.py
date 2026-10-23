"""Times NCCL all-reduce between the pod's two GPUs, so a run can tell a slow GPU-to-GPU path from a slow model.

setup.sh launches it under torchrun twice: with NCCL's defaults (what the pack installs) and with P2P disabled (what
the pack's .env asks for but never applies).
"""

import os
import time

import torch
import torch.distributed as dist

# Small messages are what TP decode sends per layer, where latency dominates; the large one is prefill-sized, where
# link bandwidth does.
SIZES = ((8 << 10, 200), (1 << 20, 50), (64 << 20, 20))


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    for nbytes, iters in SIZES:
        x = torch.ones(nbytes // 2, dtype=torch.bfloat16, device="cuda")
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / iters
        # nccl-tests' bus bandwidth: a ring all-reduce moves 2 (n - 1) / n of the buffer over each link.
        busbw = nbytes * 2 * (world - 1) / world / seconds / 1e9
        if rank == 0:
            print(
                f"allreduce NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', 'default')} bytes={nbytes} "
                f"latency_us={seconds * 1e6:.1f} busbw_GBps={busbw:.2f}",
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
