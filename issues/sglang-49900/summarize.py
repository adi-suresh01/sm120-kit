"""Builds <out>/summary.txt and <out>/compare.csv from compare.jsonl, tune.jsonl, env.json, runs.tsv, and the stock
tuner's log. Reads only files, so it runs anywhere the results were pulled to."""

import csv
import json
import sys
from pathlib import Path

# Per decode step at TP4, from the reporter's profile: 1820 ms and 1097 ms of _w8a8_block_fp8_matmul over 50 steps.
REPORTED_STEP_MS = {1: 36.4, 2: 21.9}
# fp8bench.SHAPES's calls per step, copied because fp8bench imports torch and sglang, which a laptop may not have.
CALLS = {
    (1792, 5120): 40,
    (8192, 1280): 40,
    (5120, 2048): 40,
    (1152, 5120): 40,
    (5120, 576): 40,
    (4096, 1280): 8,
    (25600, 6144): 2,
}
ARM_ORDER = (
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
# main serves a K % 128 != 0 shape on CUTLASS when b12x is asked for, so b12x's step total does the same.
STAND_IN = {"mxfp8-b12x": "mxfp8-cutlass"}
CSV_FIELDS = (
    "arm", "n", "k", "m", "mode", "level", "us", "rel_err", "ok", "regs", "spills", "shared", "table_shape",
    "config", "note", "error",
)


def rows_of(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def short(config):
    if not isinstance(config, dict):
        return "-"
    text = f"{config['BLOCK_SIZE_M']}x{config['BLOCK_SIZE_N']} w{config['num_warps']} s{config['num_stages']}"
    if "SPLIT_K" in config:
        text += f" sk{config['SPLIT_K']}" + (" swap" if config.get("SWAP_AB") else "")
    return text


def fmt(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def main():
    out = Path(sys.argv[1])
    rows = rows_of(out / "compare.jsonl")
    # A later run of the same measurement replaces the earlier one.
    latest = {(r["arm"], r["n"], r["k"], r["m"], r["mode"]): r for r in rows}
    with open(out / "compare.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in latest.values():
            writer.writerow({**r, "config": json.dumps(r.get("config")), "table_shape": r.get("table_shape")})

    def us(arm, n, k, m, mode="graph"):
        r = latest.get((arm, n, k, m, mode))
        return None if r is None else r.get("us")

    arms = [a for a in ARM_ORDER if any(key[0] == a for key in latest)]
    lines = []
    env = json.loads((out / "env.json").read_text()) if (out / "env.json").exists() else {}
    lines.append(
        f"sglang#49900 stage 1: {env.get('device', '?')}, {env.get('sms', '?')} SMs, L2 {env.get('l2_mb', '?')} MB"
    )
    lines.append(
        f"sglang {env.get('sglang', '?')}, torch {env.get('torch', '?')}, triton {env.get('triton', '?')}, "
        f"flashinfer {env.get('flashinfer', '?')}; tables for this device in the image: {env.get('device_tables', '?')}"
    )

    lines += ["", "Per-call time (us), CUDA graph, weights cycled past L2. M = running requests. Linear arms"]
    lines += ["include the activation quant. M1/M2 above 1 means one request is slower per call than two.", ""]
    lines.append(f"{'shape':<12}{'arm':<20}{'M=1':>9}{'M=2':>9}{'M=4':>9}{'M=8':>9}{'M1/M2':>8}")
    for n, k in CALLS:
        for arm in arms:
            t = [us(arm, n, k, m) for m in (1, 2, 4, 8)]
            if all(v is None for v in t):
                continue
            ratio = t[0] / t[1] if t[0] and t[1] else None
            lines.append(f"{f'{n}x{k}':<12}{arm:<20}" + "".join(f"{fmt(v):>9}" for v in t) + f"{fmt(ratio, 2):>8}")

    lines += ["", "Dense FP8 GEMM time per decode step (ms), calls per step x per-call time, over all 7 shapes."]
    lines += [f"Reporter's profile: {REPORTED_STEP_MS[1]} at M=1, {REPORTED_STEP_MS[2]} at M=2. k/7 means only k shapes"]
    lines += ["were measured. mxfp8-b12x takes mxfp8-cutlass's time on shapes with K % 128 != 0, as main serves them.", ""]
    lines.append(f"{'arm':<20}{'M=1':>9}{'M=2':>9}{'M=4':>9}{'M=8':>9}")
    for arm in arms:
        cells = []
        for m in (1, 2, 4, 8):
            parts = [
                us(STAND_IN[arm], n, k, m) if arm in STAND_IN and k % 128 else us(arm, n, k, m) for n, k in CALLS
            ]
            done = [CALLS[nk] * v for nk, v in zip(CALLS, parts) if v is not None]
            cells.append(fmt(sum(done) / 1000, 2) if len(done) == len(CALLS) else f"{len(done)}/7")
        lines.append(f"{arm:<20}" + "".join(f"{c:>9}" for c in cells))

    lines += ["", "H2: the default config with M specialized (stock JIT) and as a runtime int (nospec)."]
    lines.append(f"{'shape':<12}{'M':>5}{'stock g':>9}{'nospec g':>9}{'stock e':>9}{'nospec e':>9}  regs/spills/shared")
    for n, k in CALLS:
        for m in sorted({key[3] for key in latest if key[1:3] == (n, k) and key[0] == "default"}):
            meta = []
            for arm in ("default", "default-nospec"):
                r = latest.get((arm, n, k, m, "graph"), {})
                meta.append(f"{r.get('regs', '-')}/{r.get('spills', '-')}/{r.get('shared', '-')}")
            timing = [us(a, n, k, m, mode) for mode in ("graph", "eager") for a in ("default", "default-nospec")]
            lines.append(f"{f'{n}x{k}':<12}{m:>5}" + "".join(f"{fmt(v):>9}" for v in timing) + "  " + " vs ".join(meta))

    tables = {}
    for family in ("stock", "splitk"):
        for path in (out / "tables" / f"tuned-{family}").glob("N=*.json"):
            n, k = (int(p.split("=")[1]) for p in path.name.split(",")[:2])
            tables[(family, n, k)] = json.loads(path.read_text())
    if tables:
        lines += ["", "Tuned configs (BLOCK_M x BLOCK_N, warps, stages, split-K, swap-AB), stock | splitk tables."]
        for n, k in CALLS:
            if ("stock", n, k) not in tables:
                continue
            cells = [f"M={m}: {short(tables[('stock', n, k)].get(str(m)))} | "
                     f"{short(tables.get(('splitk', n, k), {}).get(str(m)))}" for m in (1, 2, 4, 8)]
            lines.append(f"{f'{n}x{k}':<12}" + "; ".join(cells))

    bad = [r for r in latest.values() if r.get("error") or r.get("ok") is False]
    lines += ["", f"Rows with an error or over tolerance: {len(bad)}"]
    lines += [f"  {r['arm']} {r['n']}x{r['k']} M={r['m']}: {r.get('error') or 'rel_err ' + fmt(r.get('rel_err'), 4)}"
              for r in bad]

    tune_rows = rows_of(out / "tune.jsonl")
    pre = [r for r in tune_rows if r.get("stage") == "precompile"]
    if pre:
        p = pre[-1]
        lines.append(f"Sweep: {p['configs']} configs, {p.get('kept', '?')} compiled in time, {len(p['failed'])} failed "
                     f"to compile, precompile {p['s']:.0f} s on {p['workers']} workers; "
                     f"{sum(r.get('stage') == 'screen' for r in tune_rows)} screens")

    tuner_log = out / "stock_tuner.log"
    if tuner_log.exists():
        tail = [ln for ln in tuner_log.read_text(errors="replace").splitlines() if ln.strip()][-1:]
        lines.append(f"Stock tuner on [32, 32] (T0), last line: {tail[0] if tail else '(empty log)'}")
    runs = out / "runs.tsv"
    if runs.exists():
        failed = [ln for ln in runs.read_text().splitlines()[1:] if ln.split("\t")[1] != "0"]
        lines += [f"Runs not ending 0: {len(failed)}"] + [f"  {ln}" for ln in failed]

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
