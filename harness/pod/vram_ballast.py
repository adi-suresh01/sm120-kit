#!/usr/bin/env python3
"""Holds GPU memory until killed so a big card leaves only TARGET_GIB free.

Emulates a smaller card's physical ceiling. vLLM's --gpu-memory-utilization only sizes the KV cache after
profile_run, so scaling it down cannot reproduce an OOM that a small card hits during profiling.
Usage: vram_ballast.py <target_free_gib>   (prints "holding ..." once the memory is held)
"""
import signal
import sys

import torch

target_free = int(float(sys.argv[1]) * 2**30)
free, _ = torch.cuda.mem_get_info()
hold = free - target_free
if hold <= 0:
    sys.exit(f"only {free / 2**30:.2f} GiB free, nothing to hold")

ballast = torch.empty(hold, dtype=torch.uint8, device="cuda")
free_after, _ = torch.cuda.mem_get_info()
print(f"holding {hold / 2**30:.2f} GiB, free now {free_after / 2**30:.2f} GiB", flush=True)
signal.pause()
