"""Shared parts of the sglang#49900 stage 1 kernel bench: DeepSeek-V4.1-Flash's per-rank decode GEMM shapes at TP4,
synthetic block-FP8 operands in the layout sglang hands the Triton kernel, the kernel launch, and CUDA graph timing."""

import functools
import json
import math

import torch
import triton

from sglang.kernels.ops.gemm import fp8_kernel as fk
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.utils import get_device_name

BLOCK = [32, 32]

# Per-rank (N, K) of every block-FP8 linear in one decode step at --tp-size 4, with calls per step (README.md,
# "Shapes"). They add up to 210 calls, the reporter's 10,500 over 50 steps.
SHAPES = {
    (1792, 5120): (40, "attn.wqkv_a, wq_a and wkv fused, replicated"),
    (8192, 1280): (40, "attn.wq_b, column-parallel"),
    (5120, 2048): (40, "attn.wo_b, row-parallel"),
    (1152, 5120): (40, "shared_experts.gate_up, column-parallel"),
    (5120, 576): (40, "shared_experts.down, row-parallel"),
    (4096, 1280): (8, "indexer.wq_b, replicated, 8 index layers"),
    (25600, 6144): (2, "engram.wkv, replicated, layers 1 and 14"),
}
DECODE_MS = (1, 2, 4, 8)

# What w8a8_block_fp8_matmul_triton runs when no table names the device and shape (fp8_kernel.py, main :627-634).
DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 3,
}

# The same kernel source with M left a runtime int32. Triton compiles an int argument equal to 1 as a constant
# (python/src/specialize.cc:274 at v3.8.0), so with one running request the decode graph runs a different binary
# from the one it runs at M=2, under the same config.
BASE_NOSPEC = triton.jit(do_not_specialize=["M"])(fk._w8a8_block_fp8_matmul.fn)


def device_name():
    return get_device_name().replace(" ", "_")


def table_name(n, k, device):
    return f"N={n},K={k},device_name={device},dtype=fp8_w8a8,block_shape=[32, 32].json"


def load_table(path):
    with open(path) as f:
        return {int(m): cfg for m, cfg in json.load(f).items()}


def pick(table, m):
    # w8a8_block_fp8_matmul_triton takes the entry with the nearest M.
    return table[min(table, key=lambda key: abs(key - m))]


def make_weight(n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n // 32, 32, k // 32, 32, device="cuda", generator=gen) * 0.02
    # ue8m0 checkpoints carry power-of-two block scales, so the fp32 scales the kernel reads are powers of two too.
    amax = w.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (w / scale[:, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.reshape(n, k).contiguous(), scale.contiguous()


def weight_copies(q, s):
    """Copies of one weight adding up to twice the L2, each its own allocation. A graph that cycles through them misses
    L2 on every call, as a decode step does when 210 different weights stream through it."""
    l2 = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 128 << 20)
    count = min(64, max(2, math.ceil(2 * l2 / q.numel())))
    return [(q.clone(), s.clone()) for _ in range(count)]


def make_activation(m, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=gen).to(torch.bfloat16)
    # triton_w8a8_block_fp8_linear's branch for ue8m0 checkpoints (fp8_utils.py, main :2114-2118): fp32
    # power-of-two scales, row-major.
    q, s = sglang_per_token_group_quant_fp8(x, BLOCK[1], scale_ue8m0=True)
    return x, q, s


def run_config(config, a, b, a_s, b_s, base_kernel=fk._w8a8_block_fp8_matmul):
    """w8a8_block_fp8_matmul_triton's launch with a given config, under sglang#39872's SM120 rule: a config that names
    SPLIT_K runs _w8a8_block_fp8_matmul_hopper (portable tl.dot, split-K and swap-AB), any other the base kernel.
    Returns the output and the compiled matmul kernel."""
    m, k = a.shape
    n = b.shape[0]
    c = a.new_empty((m, n), dtype=torch.bfloat16)
    split_kernel = "SPLIT_K" in config
    kernel = fk._w8a8_block_fp8_matmul_hopper if split_kernel else base_kernel
    split_k = config["SPLIT_K"] if split_kernel else 1
    partials = torch.empty((split_k, m, n), device=a.device, dtype=torch.float32) if split_k > 1 else c
    blocks = triton.cdiv(m, config["BLOCK_SIZE_M"]) * triton.cdiv(n, config["BLOCK_SIZE_N"])
    compiled = kernel[(blocks, split_k) if split_kernel else (blocks,)](
        a,
        b,
        partials,
        a_s,
        b_s,
        m,
        n,
        k,
        BLOCK[0],
        BLOCK[1],
        a.stride(-2),
        a.stride(-1),
        b.stride(1),
        b.stride(0),
        c.stride(-2),
        c.stride(-1),
        a_s.stride(-2),
        a_s.stride(-1),
        b_s.stride(1),
        b_s.stride(0),
        **config,
        needs_masking=k % config["BLOCK_SIZE_K"] != 0,
    )
    if split_k > 1:
        fk._reduce_block_fp8_split_k[(triton.cdiv(m * n, 256),)](partials, c, m * n, split_k, 256)
    return c, compiled


def config_calls(config, q_x, s_x, copies, base_kernel=fk._w8a8_block_fp8_matmul):
    return [functools.partial(run_config, config, q_x, w, s_x, s, base_kernel) for w, s in copies]


# References are float64, so neither TF32 nor an accumulation order enters them.
def dequant_weight_t(q_w, s_w):
    return (q_w.double() * s_w.double().repeat_interleave(BLOCK[0], dim=0).repeat_interleave(BLOCK[1], dim=1)).T


def dequant_act(q_x, s_x):
    return q_x.double() * s_x.double().repeat_interleave(BLOCK[1], dim=1)


def rel_err(out, ref):
    return ((out.double() - ref).norm() / ref.norm()).item()


def _timed(fn, count):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(count):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def graph_us(calls, target_ms):
    """Per-call time of `calls` captured back to back in one CUDA graph and replayed for about target_ms, as the
    server replays its decode graph. Includes the gap between kernels, about a microsecond each."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for call in calls:
            call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    one = _timed(graph.replay, 1)
    replays = max(2, min(200, math.ceil(target_ms / max(one, 1e-3))))
    return _timed(graph.replay, replays) * 1000.0 / (replays * len(calls))


def eager_us(calls, target_ms):
    """Per-call time of the same calls launched one by one from Python, which adds the CPU launch cost wherever it
    exceeds the kernel's."""

    def each():
        for call in calls:
            call()

    each()
    one = _timed(each, 1)
    rounds = max(2, min(200, math.ceil(target_ms / max(one, 1e-3))))
    return _timed(each, rounds) * 1000.0 / (rounds * len(calls))
