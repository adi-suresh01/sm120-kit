"""Times each arm (README.md, "Arms") on every TP4 decode shape and M, in a CUDA graph with the weights cycled past
L2, checks its output against the exact product, and appends one JSON row per (arm, shape, M, mode) to
<out>/compare.jsonl."""

import argparse
import functools
import json
import re
import time
from pathlib import Path

import torch

import fp8bench as fb

ARMS = (
    "shipped",
    "default",
    "default-nospec",
    "pr39872",
    "pro5000",
    "tuned-stock",
    "tuned-stock-nospec",
    "tuned-splitk",
    "linear-shipped",
    "linear-tuned",
    "mxfp8-cutlass",
    "mxfp8-b12x",
)
TABLE_SETS = ("pr39872", "pro5000", "tuned-stock", "tuned-splitk")
EAGER_ARMS = {"shipped", "default", "default-nospec"}
KERNEL_TOL = 1e-2
# Linear-level arms quantize the activation themselves; their error against the bf16 product is mostly that
# quantization (about 2.7e-2 on every path in sglang#45262).
LINEAR_TOL = 5e-2
# Where a published table has no entry for a TP4 shape, the table for the same module at TP8 stands in.
TP8_SIBLING = {
    (8192, 1280): (4096, 1280),
    (5120, 2048): (5120, 1024),
    (1152, 5120): (576, 5120),
    (5120, 576): (5120, 288),
}


def tables_in(directory):
    tables = {}
    for path in Path(directory).glob("N=*,K=*,*.json"):
        n, k = (int(v) for v in re.match(r"N=(\d+),K=(\d+),", path.name).groups())
        tables[(n, k)] = fb.load_table(path)
    return tables


def published(tables, n, k):
    if (n, k) in tables:
        return tables[(n, k)], (n, k)
    sibling = TP8_SIBLING.get((n, k))
    if sibling in tables:
        return tables[sibling], sibling
    return None, None


class Shape:
    """One shape's weight copies, all equal in value, and the dequantized weight the references use."""

    def __init__(self, n, k):
        self.n, self.k = n, k
        q, s = fb.make_weight(n, k, seed=7 * n + k)
        self.copies = fb.weight_copies(q, s)
        self.w_deq_t = fb.dequant_weight_t(q, s)
        self.mx_scales = None


def shipped_calls(act, shape):
    _, q_x, s_x = act
    return [
        functools.partial(fb.fk.w8a8_block_fp8_matmul_triton, q_x, w, s_x, s, fb.BLOCK, torch.bfloat16)
        for w, s in shape.copies
    ]


def linear_triton(x, w, s, config):
    q_x, s_x = fb.sglang_per_token_group_quant_fp8(x, fb.BLOCK[1], scale_ue8m0=True)
    if config is None:
        return fb.fk.w8a8_block_fp8_matmul_triton(q_x, w, s_x, s, fb.BLOCK, torch.bfloat16)
    return fb.run_config(config, q_x, w, s_x, s)[0]


def mxfp8_calls(backend, act, shape):
    # Imported here so a stack without the MXFP8 route loses only these two arms.
    from flashinfer import block_scale_interleave

    from sglang.srt.layers.quantization.fp8_utils import (
        block_fp8_scale_to_mxfp8_e8m0,
        flashinfer_mxfp8_blockscaled_linear,
    )

    if shape.mx_scales is None:
        # What Fp8LinearMethod keeps for a block-FP8 weight served as MXFP8 (fp8.py, main :1012-1031, :1113-1124).
        shape.mx_scales = [
            block_scale_interleave(
                block_fp8_scale_to_mxfp8_e8m0(s, (shape.n, shape.k), fb.BLOCK).contiguous()
            ).contiguous()
            for _, s in shape.copies
        ]
    linear = functools.partial(
        flashinfer_mxfp8_blockscaled_linear, output_dtype=torch.bfloat16, backend=backend, pin_tactic=True
    )
    return [functools.partial(linear, act[0], w, sw) for (w, _), sw in zip(shape.copies, shape.mx_scales)]


def build(arm, shape, m, act, tables):
    """(calls, level, config, table shape), or calls=None with the reason where the config would be."""
    n, k = shape.n, shape.k
    _, q_x, s_x = act
    if arm == "shipped":
        return shipped_calls(act, shape), "kernel", None, None
    if arm in ("default", "default-nospec"):
        base = fb.BASE_NOSPEC if arm == "default-nospec" else fb.fk._w8a8_block_fp8_matmul
        return fb.config_calls(fb.DEFAULT_CONFIG, q_x, s_x, shape.copies, base), "kernel", fb.DEFAULT_CONFIG, None
    if arm in ("pr39872", "pro5000", "tuned-stock", "tuned-stock-nospec", "tuned-splitk"):
        source = arm.removesuffix("-nospec")
        table, table_shape = published(tables[source], n, k)
        if table is None:
            return None, "kernel", f"no {source} table for {n}x{k}", None
        config = fb.pick(table, m)
        base = fb.BASE_NOSPEC if arm.endswith("-nospec") else fb.fk._w8a8_block_fp8_matmul
        return fb.config_calls(config, q_x, s_x, shape.copies, base), "kernel", config, table_shape
    if arm in ("linear-shipped", "linear-tuned"):
        config = None
        if arm == "linear-tuned":
            table, _ = published(tables["tuned-splitk"], n, k)
            if table is None:
                return None, "linear", f"no tuned-splitk table for {n}x{k}", None
            config = fb.pick(table, m)
        return [functools.partial(linear_triton, act[0], w, s, config) for w, s in shape.copies], "linear", config, None
    if arm == "mxfp8-b12x" and k % 128:
        return None, "linear", "K % 128 != 0; main serves this shape on CUTLASS (fp8_utils.py:977-983)", None
    return mxfp8_calls(arm.removeprefix("mxfp8-"), act, shape), "linear", None, None


def measure(arm, shape, m, act, refs, tables, ptx_dir):
    calls, level, config, table_shape = build(arm, shape, m, act, tables)
    row = {"arm": arm, "n": shape.n, "k": shape.k, "m": m, "level": level, "table_shape": table_shape}
    if calls is None:
        return [{**row, "mode": "graph", "us": None, "note": config}]
    rows = [{**row, "mode": "graph", "us": fb.graph_us(calls, target_ms=100)}]
    if arm in EAGER_ARMS:
        rows.append({**row, "mode": "eager", "us": fb.eager_us(calls, target_ms=100)})
    result = calls[0]()
    out, compiled = result if isinstance(result, tuple) else (result, None)
    err = fb.rel_err(out, refs[level])
    check = {"rel_err": err, "ok": err <= (KERNEL_TOL if level == "kernel" else LINEAR_TOL), "config": config}
    if compiled is not None:
        check.update(
            regs=getattr(compiled, "n_regs", None),
            spills=getattr(compiled, "n_spills", None),
            shared=getattr(compiled.metadata, "shared", None),
        )
        # The M=1 and M=2 binaries of one config, side by side for H2; the first shape stands for all.
        if arm in ("default", "default-nospec") and (shape.n, shape.k) == next(iter(fb.SHAPES)) and m in (1, 2):
            (ptx_dir / f"{arm}-m{m}.ptx").write_text(compiled.asm.get("ptx", ""))
    return [{**r, **check} for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    parser.add_argument("--ms", nargs="+", type=int, default=list(fb.DECODE_MS))
    parser.add_argument("--budget-s", type=float, required=True)
    args = parser.parse_args()
    stop_at = time.monotonic() + args.budget_s
    tables = {name: tables_in(args.out / "tables" / name) for name in TABLE_SETS}
    ptx_dir = args.out / "ptx"
    ptx_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out / "compare.jsonl", "a") as log:
        for n, k in fb.SHAPES:
            if time.monotonic() > stop_at:
                print(f"budget spent: {n}x{k} and the shapes after it are not measured for {args.arms}", flush=True)
                break
            shape = Shape(n, k)
            for m in args.ms:
                act = fb.make_activation(m, k, seed=m)
                refs = {
                    "kernel": fb.dequant_act(act[1], act[2]) @ shape.w_deq_t,
                    "linear": act[0].double() @ shape.w_deq_t,
                }
                for arm in args.arms:
                    try:
                        rows = measure(arm, shape, m, act, refs, tables, ptx_dir)
                    except Exception as e:  # one arm failing on one shape must not end the run
                        error = repr(e)[:400]
                        rows = [{"arm": arm, "n": n, "k": k, "m": m, "mode": "graph", "us": None, "error": error}]
                    for row in rows:
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        brief = {key: row.get(key) for key in ("arm", "n", "k", "m", "mode", "us", "rel_err", "error")}
                        print(json.dumps(brief), flush=True)
            del shape
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
