"""Padded CUTLASS vs the kernels an unaligned-N layer falls back to, for the shapes vllm#47990 and PR #48588 discuss.

Two levels, both timed as CUDA graphs:
- gemm: CUTLASS vs Triton on pre-quantized operands. CUTLASS gets N zero-padded once here, as PR #48588 does at load
  time, and pays for the output slice copy: the PR author's method, so the numbers compare to theirs.
- layer: `apply_weights` of the CUTLASS, Triton, and Marlin block-FP8 kernels, including each one's activation
  handling. Marlin is weight-only (BF16 activations) and is what PR #47988's guard actually falls through to on
  SM120, since it comes before Triton in vLLM's kernel order and accepts any N. Unaligned N on CUTLASS needs PR
  #48588 here; on stock vLLM that cell records the error instead.
Triton runs with its default config: vLLM ships no tuned block-FP8 config for this GPU.
Usage: python3 bench_gemm.py <out.jsonl>
"""

import json
import sys

import torch
from triton.testing import do_bench_cudagraph

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFp8BlockScaledMMKernel
from vllm.model_executor.kernels.linear.scaled_mm.marlin import MarlinFP8ScaledMMLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig
from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonFp8BlockScaledMMKernel
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    w8a8_triton_block_scaled_mm,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key

BLOCK = 128
FP8 = torch.float8_e4m3fn
# (N, K, what it is). 32 x 1024 is the skinny projection where a reviewer measured tuned Triton ahead of padded CUTLASS.
SHAPES = [
    (576, 7168, "DeepSeek-V3 kv_a_proj_with_mqa"),
    (576, 2048, "DeepSeek-V2-Lite kv_a_proj_with_mqa"),
    (2112, 7168, "DeepSeek-V3 fused q_a + kv_a"),
    (1344, 2048, "GLM-4.7-Flash fused q_a + kv_a"),
    (32, 1024, "skinny projection from the PR review"),
    (512, 7168, "aligned control"),
]
MS = [1, 4, 16, 64, 256, 1024, 4096]
# Names the server's kernel classes use, so a row reads against the server logs.
PARAM_NAMES = ["weight", "weight_scale", "input_scale", "input_scale_ub"]


def time_us(fn):
    ms = do_bench_cudagraph(fn, rep=100, quantiles=[0.5])
    return 1000 * (ms[0] if isinstance(ms, (list, tuple)) else ms)


def weights(n, k, gen):
    info = torch.finfo(FP8)
    w = ((torch.rand(n, k, generator=gen, device="cuda") - 0.5) * 2 * info.max).clamp(info.min, info.max).to(FP8)
    scale = torch.rand(-(-n // BLOCK), -(-k // BLOCK), generator=gen, device="cuda") * 1e-2
    return w, scale


def gemm_fns(m, n, k, w, w_scale, gen):
    a = torch.randn(m, k, generator=gen, device="cuda")
    # use_ue8m0=False as CutlassFp8BlockScaledMMKernel quantizes; the default follows DeepGEMM's E8M0 setting.
    a_fp8, a_scale_cm = per_token_group_quant_fp8(a, BLOCK, column_major_scales=True, use_ue8m0=False)
    a_scale = a_scale_cm.contiguous()
    n_pad = -(-n // BLOCK) * BLOCK
    w_pad = torch.nn.functional.pad(w.view(torch.uint8), (0, 0, 0, n_pad - n)).view(FP8)

    def cutlass():
        out = ops.cutlass_scaled_mm(a_fp8, w_pad.T, out_dtype=torch.bfloat16, scale_a=a_scale_cm, scale_b=w_scale.T)
        return out[:, :n].contiguous() if n_pad != n else out

    def triton():
        return w8a8_triton_block_scaled_mm(a_fp8, w, a_scale, w_scale, [BLOCK, BLOCK], torch.bfloat16)

    return {"triton": triton, "cutlass": cutlass}


def layer_fns(m, n, k, w, w_scale, gen):
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=create_fp8_quant_key(static=True, group_shape=GroupShape(BLOCK, BLOCK)),
        activation_quant_key=create_fp8_quant_key(static=False, group_shape=GroupShape(1, BLOCK)),
        weight_shape=(n, k),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    x = torch.randn(m, k, generator=gen, device="cuda", dtype=torch.bfloat16)
    fns = {}
    for name, kernel in (
        ("triton", TritonFp8BlockScaledMMKernel(config)),
        ("cutlass", CutlassFp8BlockScaledMMKernel(config)),
        ("marlin", MarlinFP8ScaledMMLinearKernel(config, layer_param_names=PARAM_NAMES)),
    ):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(w_scale.clone(), requires_grad=False)
        layer.input_scale = None
        # What Fp8LinearMethod.create_weights sets and Marlin's repack reads.
        layer.output_size_per_partition = n
        layer.input_size_per_partition = k
        layer.logical_widths = [n]
        layer.orig_dtype = torch.bfloat16
        layer.weight_block_size = [BLOCK, BLOCK]
        try:
            kernel.process_weights_after_loading(layer)
        except (RuntimeError, AssertionError) as e:
            fns[name] = e
            continue
        fns[name] = lambda kernel=kernel, layer=layer: kernel.apply_weights(layer, x)
    return fns


def rel(got, ref):
    return (torch.mean(torch.abs(got.float() - ref.float())) / torch.mean(torch.abs(ref.float()))).item()


def measure(fns):
    """Times each callable; Triton is the reference for the error column. A failure is recorded per kernel."""
    out = {}
    ref = fns["triton"]()
    for name, fn in fns.items():
        try:
            if isinstance(fn, Exception):
                raise fn
            got = fn()
            torch.cuda.synchronize()
            out[name] = {"us": time_us(fn), "rel_vs_triton": rel(got, ref)}
        except (RuntimeError, AssertionError) as e:
            out[name] = {"error": str(e).splitlines()[0][:300] if str(e) else type(e).__name__}
    return out


def cell(r, name):
    return f"{r[name]['us']:8.1f}" if "us" in r.get(name, {}) else f"{'err':>8}"


def main():
    out_path = sys.argv[1]
    gen = torch.Generator(device="cuda").manual_seed(0)
    print(f"{'N':>5} {'K':>5} {'M':>5} | gemm us: {'triton':>8} {'cutlass':>8} | layer us: {'triton':>8} {'cutlass':>8} "
          f"{'marlin':>8}")
    with open(out_path, "w") as f, set_current_vllm_config(VllmConfig()):
        for n, k, what in SHAPES:
            w, w_scale = weights(n, k, gen)
            for m in MS:
                row = {"n": n, "k": k, "m": m, "shape": what,
                       "gemm": measure(gemm_fns(m, n, k, w, w_scale, gen)),
                       "layer": measure(layer_fns(m, n, k, w, w_scale, gen))}
                f.write(json.dumps(row) + "\n")
                g, lay = row["gemm"], row["layer"]
                print(f"{n:>5} {k:>5} {m:>5} |          {cell(g, 'triton')} {cell(g, 'cutlass')} |           "
                      f"{cell(lay, 'triton')} {cell(lay, 'cutlass')} {cell(lay, 'marlin')}", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
