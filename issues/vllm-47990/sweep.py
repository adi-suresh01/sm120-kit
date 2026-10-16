"""Shape sweep from vllm#47990, run under whatever vLLM the variant installed.

Each case goes through two entry points: `wrapper` is vLLM's module-level `cutlass_scaled_mm`, which PR #48588 pads per
call, and `raw` is `ops.cutlass_scaled_mm` with the operands exactly as `CutlassFp8BlockScaledMMKernel` passes them, so
a run shows both whether the integration layer handles a shape and whether the SM120 kernel itself still rejects it.
Results are compared against an fp32 dequant reference built here, independent of vLLM's quant helpers.
Usage: python3 sweep.py <out.jsonl>
"""

import json
import sys
from collections import defaultdict

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import cutlass_scaled_mm
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

BLOCK = 128
# 576 is DeepSeek kv_a_proj_with_mqa (512 + 64), 2112 adds q_lora_rank 1536 (DeepSeek-V3 fused q_a + kv_a), 1344 adds
# q_lora_rank 768 (GLM-4.7-Flash); 64 and 192 are the reviewer's extra off-block widths; the rest are aligned controls.
NS = [64, 192, 512, 576, 1344, 1536, 2112, 7168]
KS = [1024, 2048, 7168]
MS = [1, 16, 32, 64, 83, 128, 256, 512, 1024]
FP8 = torch.float8_e4m3fn


def reference(a_fp8, a_scale, b_fp8, b_scale):
    m, k = a_fp8.shape
    n = b_fp8.shape[0]
    a = a_fp8.float() * a_scale.repeat_interleave(BLOCK, dim=1)[:, :k]
    b = b_fp8.float() * b_scale.repeat_interleave(BLOCK, dim=0)[:n].repeat_interleave(BLOCK, dim=1)[:, :k]
    return a @ b.T


def rel_err(out, ref):
    return (torch.mean(torch.abs(out.float() - ref)) / torch.mean(torch.abs(ref))).item()


def run_case(path, m, n, k, gen):
    info = torch.finfo(FP8)
    a = (torch.rand(m, k, generator=gen, device="cuda") - 0.5) * 2 * info.max
    b = ((torch.rand(n, k, generator=gen, device="cuda") - 0.5) * 2 * info.max).clamp(info.min, info.max).to(FP8)
    b_scale = torch.rand(-(-n // BLOCK), -(-k // BLOCK), generator=gen, device="cuda") * 1e-2
    # As the kernel class quantizes: column-major fp32 scales, no E8M0 rounding (the default follows DeepGEMM's
    # setting). The reference takes the same values row-major.
    a_fp8, a_scale_cm = per_token_group_quant_fp8(a, BLOCK, column_major_scales=True, use_ue8m0=False)
    a_scale = a_scale_cm.contiguous()
    if path == "wrapper":
        out = cutlass_scaled_mm(a_fp8, b, a_scale_cm, b_scale, [BLOCK, BLOCK], torch.bfloat16)
    else:
        out = ops.cutlass_scaled_mm(a_fp8, b.T, out_dtype=torch.bfloat16, scale_a=a_scale_cm, scale_b=b_scale.T)
    torch.cuda.synchronize()
    if tuple(out.shape) != (m, n):
        return {"status": "bad_shape", "shape": list(out.shape)}
    return {"status": "ok", "rel_err": rel_err(out, reference(a_fp8, a_scale, b, b_scale))}


def main():
    out_path = sys.argv[1]
    gen = torch.Generator(device="cuda").manual_seed(0)
    summary = defaultdict(lambda: defaultdict(int))
    worst = defaultdict(float)
    with open(out_path, "w") as f:
        for path in ("wrapper", "raw"):
            for n in NS:
                for k in KS:
                    for m in MS:
                        try:
                            res = run_case(path, m, n, k, gen)
                        except RuntimeError as e:
                            msg = str(e).splitlines()[0][:300]
                            res = {"status": "invalid_status" if "Invalid status" in msg else "error", "msg": msg}
                        res.update(path=path, m=m, n=n, k=k)
                        f.write(json.dumps(res) + "\n")
                        summary[(path, n)][res["status"]] += 1
                        if res["status"] == "ok":
                            worst[(path, n)] = max(worst[(path, n)], res["rel_err"])
    print(f"{'path':8} {'N':>5} {'N%128':>5}  results over M x K ({len(MS)} x {len(KS)})   max rel err")
    for (path, n), counts in summary.items():
        shown = ", ".join(f"{s}={c}" for s, c in sorted(counts.items()))
        err = f"{worst[(path, n)]:.1e}" if counts.get("ok") else "-"
        print(f"{path:8} {n:>5} {n % BLOCK:>5}  {shown:40} {err}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
