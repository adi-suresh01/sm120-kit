"""Sweeps Triton configs for each TP4 decode shape at M = 1, 2, 4, 8 and writes two sglang config tables per shape:
tuned-stock (base kernel only, so stock sglang loads it as is) and tuned-splitk (adds split-K and swap-AB on
_w8a8_block_fp8_matmul_hopper, which on SM120 needs sglang#39872's dispatch rule). Every timing goes to
<out>/tune.jsonl."""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import torch
import triton

import fp8bench as fb

KERNEL_TOL = 1e-2
# BLOCK_SIZE_K stays 32: the kernel steps the scales once per group_k // BLOCK_SIZE_K tiles, which is zero above 32.
# The shipped tuner's CUDA space has only 64 and 128 and keeps sizes that divide block_k, so it has nothing to try
# for [32, 32] (README.md, T0). BLOCK_SIZE_M 16 is the smallest tl.dot tile; 64 is the shipped default's.
STOCK = [
    {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1, "num_warps": w, "num_stages": s}
    for bm, bn, w, s in itertools.product((16, 64), (32, 64, 128, 256), (2, 4, 8), (2, 3, 4, 5))
] + [fb.DEFAULT_CONFIG]
SPLITK = [
    {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": w,
        "num_stages": s,
        "SPLIT_K": sk,
        "SWAP_AB": swap,
    }
    for bn, w, s, sk, swap in itertools.product((32, 64, 128), (2, 4, 8), (2, 3, 4), (1, 2, 4, 8, 16), (False, True))
]
SPLITS = (2, 4, 8, 16)
FINALISTS = 5


class OutOfTime(Exception):
    pass


# Across these shapes the only argument values that change Triton's key are M == 1 and whether the scale stride K/32
# is a multiple of 16, so these four cover every binary a config needs.
KEY_CLASSES = [(m, k) for m in (1, 2) for k in (5120, 1280)]
_operands = {}


def _precompile(task):
    """Compiles one config on a small N, or one split-K reduce, so the timed sweep finds the binary in Triton's disk
    cache. Returns (config index, None for a reduce; whether it compiled and launched)."""
    index = task[1] if task[0] == "gemm" else None
    try:
        if task[0] == "reduce":
            _, elements, splits = task
            parts = torch.zeros((splits, elements), device="cuda")
            out = torch.empty(elements, device="cuda", dtype=torch.bfloat16)
            fb.fk._reduce_block_fp8_split_k[(triton.cdiv(elements, 256),)](parts, out, elements, splits, 256)
        else:
            _, _, config, m, k = task
            if (m, k) not in _operands:
                q_w, s_w = fb.make_weight(256, k, seed=0)
                _, q_x, s_x = fb.make_activation(m, k, seed=0)
                _operands[(m, k)] = (q_x, q_w, s_x, s_w)
            fb.run_config(config, *_operands[(m, k)])
        torch.cuda.synchronize()
        return index, True
    except Exception:  # out of shared memory or registers: the config leaves the sweep
        return index, False


def worker_count():
    """At most 16, one per CPU the container may use, and one per 4 GiB of its memory: each worker holds torch, sglang,
    and a CUDA context, and RunPod removes a container at about 98% of its limit, results and all."""
    cpus = len(os.sched_getaffinity(0))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            memory = min(memory, int(Path(path).read_text()))
            break
        except (OSError, ValueError):  # no such file, or "max"
            continue
    return max(1, min(16, cpus, memory // (4 << 30)))


def precompile(configs, log, limit_s):
    """Returns the configs whose binaries all compiled within limit_s. A worker that dies loses its task without
    failing the pool, so the limit, not the pool, decides when to stop waiting."""
    start = time.monotonic()
    # Builds sglang's activation quant kernel once here, if it is built on first use, so the workers below only load it.
    fb.make_activation(1, 1280, seed=0)
    # Reduces first, since they are few and every split-K config needs them; then configs in list order, so a cut-off
    # precompile keeps the base-kernel configs, the ones stock sglang loads.
    ns = sorted({n for n, _ in fb.SHAPES})
    tasks = [("reduce", m * n, sk) for n in ns for m in fb.DECODE_MS for sk in SPLITS]
    tasks += [("gemm", i, c, m, k) for i, c in enumerate(configs) for m, k in KEY_CLASSES]
    workers = worker_count()
    compiled, failed = [0] * len(configs), set()
    with mp.get_context("spawn").Pool(workers) as pool:
        # chunksize 1: a larger one returns a plain generator, which has no next(timeout).
        results = pool.imap_unordered(_precompile, tasks)
        try:
            for _ in tasks:
                index, ok = results.next(timeout=max(0.0, start + limit_s - time.monotonic()))
                if index is not None:
                    compiled[index] += ok
                    if not ok:
                        failed.add(index)
        except mp.TimeoutError:
            pass
    kept = [c for i, c in enumerate(configs) if compiled[i] == len(KEY_CLASSES)]
    row = {"stage": "precompile", "configs": len(configs), "kept": len(kept), "failed": sorted(failed)}
    write(log, {**row, "workers": workers, "s": time.monotonic() - start})
    return kept


def write(log, row):
    log.write(json.dumps(row) + "\n")
    log.flush()


def check_time(stop_at):
    if time.monotonic() > stop_at:
        raise OutOfTime


def tune_shape(n, k, configs, log, stop_at):
    """{family: {M: config}} for one shape: every config screened briefly, the fastest few per family timed for longer
    and checked against the exact product."""
    q, s = fb.make_weight(n, k, seed=7 * n + k)
    copies = fb.weight_copies(q, s)
    w_deq_t = fb.dequant_weight_t(q, s)
    best = {"stock": {}, "splitk": {}}
    for m in fb.DECODE_MS:
        _, q_x, s_x = fb.make_activation(m, k, seed=m)
        ref = fb.dequant_act(q_x, s_x) @ w_deq_t
        screened = []
        for config in configs:
            check_time(stop_at)
            row = {"stage": "screen", "n": n, "k": k, "m": m, "config": config}
            try:
                us = fb.graph_us(fb.config_calls(config, q_x, s_x, copies), target_ms=10)
            except Exception as e:  # a config that compiled on the small N can still fail here; skip it
                write(log, {**row, "us": None, "error": repr(e)[:300]})
                continue
            write(log, {**row, "us": us})
            screened.append((us, config))
        finals = {}
        for family in best:
            members = [t for t in screened if family == "splitk" or "SPLIT_K" not in t[1]]
            passing = []
            for _, config in sorted(members, key=lambda t: t[0])[:FINALISTS]:
                key = json.dumps(config, sort_keys=True)
                if key not in finals:
                    check_time(stop_at)
                    us = fb.graph_us(fb.config_calls(config, q_x, s_x, copies), target_ms=100)
                    out, _ = fb.run_config(config, q_x, copies[0][0], s_x, copies[0][1])
                    finals[key] = (us, fb.rel_err(out, ref))
                    row = {"stage": "final", "n": n, "k": k, "m": m, "config": config}
                    write(log, {**row, "us": us, "rel_err": finals[key][1]})
                us, err = finals[key]
                if err <= KERNEL_TOL:
                    passing.append((us, config))
            # No finalist within tolerance leaves this M on the shipped default rather than on a wrong answer.
            best[family][m] = min(passing, key=lambda t: t[0])[1] if passing else fb.DEFAULT_CONFIG
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget-s", type=float, required=True)
    args = parser.parse_args()
    stop_at = time.monotonic() + args.budget_s
    device = fb.device_name()
    # Most weight bytes per decode step first, so a short budget still covers what the step spends most on.
    order = sorted(fb.SHAPES, key=lambda nk: -fb.SHAPES[nk][0] * nk[0] * nk[1])
    with open(args.out / "tune.jsonl", "a") as log:
        # Half the budget at most, so a slow or stuck compile still leaves the screens time for the heaviest shapes.
        configs = precompile(STOCK + SPLITK, log, args.budget_s / 2)
        if not configs:
            # Tables of nothing but the default would read as tuned.
            sys.exit("no config compiled in time; no tables written")
        for n, k in order:
            try:
                best = tune_shape(n, k, configs, log, stop_at)
            except OutOfTime:
                print(f"budget spent during {n}x{k}; it and the shapes after it have no tuned table", flush=True)
                break
            for family, by_m in best.items():
                table = {str(m): config for m, config in by_m.items()}
                # Above the tuned range, nearest-M lookup must hand back the shipped default, or every prefill chunk
                # would run the M=8 entry.
                table["16"] = fb.DEFAULT_CONFIG
                path = args.out / "tables" / f"tuned-{family}" / fb.table_name(n, k, device)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(table, indent=4) + "\n")
            print(f"{n}x{k} tuned", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
