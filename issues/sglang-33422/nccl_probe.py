"""One NCCL allreduce group over every GPU on the box, timed at the message sizes TP4 sends, under the P2P setting in
the environment.

setup.sh runs it once per decode arm's setting, with NCCL_DEBUG=INFO and NCCL_DEBUG_FILE, before the 167 GB
download. NCCL reads NCCL_P2P_DISABLE and NCCL_P2P_LEVEL once per process, so each setting is its own run. A setting
that hangs or fails here would hang or fail that arm's boot, and a hang there costs boot_timeout_s (30 min); on
10-17 P2P on hung a RunPod host's first boot for 15 min (README.md). The times are eager launches, so they carry
launch overhead the server's decode CUDA graphs do not; they compare transports on this box, not with the server's
step time.
Usage: nccl_probe.py
"""

import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# bf16 activations at the model's hidden size 4096: one decode token (c=1), eight (c=8), one 4096-token prefill chunk.
SIZES = {"decode-1": 4096, "decode-8": 8 * 4096, "prefill-4096": 4096 * 4096}
ITERS = {"decode-1": 500, "decode-8": 500, "prefill-4096": 20}


def worker(rank: int, world: int, port: int) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
                            device_id=device)
    for name, numel in SIZES.items():
        # Zeros, so the repeated sums never overflow bf16.
        x = torch.zeros(numel, dtype=torch.bfloat16, device=device)
        for _ in range(10):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        for _ in range(ITERS[name]):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        if rank == 0:
            us = (time.perf_counter() - start) / ITERS[name] * 1e6
            print(f"allreduce {name} ({numel * 2} B) over {world} GPUs: {us:.1f} us", flush=True)
    dist.destroy_process_group()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    world = torch.cuda.device_count()
    mp.spawn(worker, args=(world, free_port()), nprocs=world)
