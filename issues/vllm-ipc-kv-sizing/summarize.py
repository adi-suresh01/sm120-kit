"""README.md's Results tables from a pulled results/vllm-ipc-kv-sizing/: per arm, what vLLM logged at each sizing step
and what the GPUs held on the live server; then each IPC arm against its control, beside FlashInfer's slab size and
what the workspace held in ws_probe.json. Each worker value is rank 0's line (Worker_TP0), else the first match; KV
tokens come from the engine's one line.
Usage: python3 -I summarize.py results/vllm-ipc-kv-sizing
"""

import csv
import json
import re
import sys
from pathlib import Path

ARMS = ("e0-off", "e1-ipc", "e2-off", "m0-off", "m1-ipc", "s0-off-est", "s1-ipc-est")
# (control, IPC arm, ws_probe shape)
PAIRS = (
    ("e0-off", "e1-ipc", "e"),
    ("e2-off", "e1-ipc", "e"),
    ("m0-off", "m1-ipc", "m"),
    ("s0-off-est", "s1-ipc-est", "m"),
)
# Qwen3-8B's KV per token on one GPU at TP4: 36 layers x K and V x 2 KV heads x 128 x 2 bytes.
KV_BYTES_PER_TOKEN = 36 * 2 * 2 * 128 * 2
GIB = 1 << 30
MIB = 1 << 20
FIELDS = {
    "estimate_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "ipc_max_tokens": r"Initialized FlashInfer PCIe IPC all-reduce for TP\d+, hidden_dim=\d+, dtype=\S+,"
    r" max_tokens=(\d+)",
    "graph_pool_gib": r"CUDA graph pool memory: ([\d.]+) GiB \(actual\)",
    "capture_gib": r"Graph capturing finished in \d+ secs, took (-?[\d.]+) GiB",
    "warmup_free_gib": r"Memory after warmup: ([\d.]+) GiB free",
    "transient_gib": r"([\d.]+) GiB kept for transient activations",
    "kv_after_warmup_gib": r"KV cache blocks \(([\d.]+) GiB\)",
    "kv_tokens": r"GPU KV cache size: ([\d,]+) tokens",
}
# The extensible fallback and the startup check at util 0.98 change what an arm measures without an OOM.
ERROR = re.compile(
    r"out of memory|OutOfMemoryError|No available memory for the cache blocks|To serve at least one request"
    r"|peer access is not supported|Disabling the extensible KV cache|is less than desired GPU memory utilization"
)


def arm_row(arm_dir: Path) -> dict:
    log = (arm_dir / "server.log").read_text(errors="replace") if (arm_dir / "server.log").exists() else ""
    # Workers log in whatever order they finish, and ranks differ by tens of MiB, about the e slab's size; one rank
    # keeps every arm's columns on the same GPU.
    rank0 = "\n".join(line for line in log.splitlines() if "Worker_TP0 pid=" in line)
    row = {}
    for field, pattern in FIELDS.items():
        match = re.search(pattern, rank0) or re.search(pattern, log)
        row[field] = float(match.group(1).replace(",", "")) if match else None
    verdict = arm_dir / "verdict.json"
    row["verdict"] = json.loads(verdict.read_text())["verdict"] if verdict.exists() else "not run"
    live = arm_dir / "gpu_mem_live.csv"
    row["used_mib"] = [int(r[1]) for r in csv.reader(live.read_text().splitlines())] if live.exists() else None
    error = next((line for line in log.splitlines() if ERROR.search(line)), "")
    row["error"] = error.strip()[-160:]
    return row


def cell(value, scale=1.0, digits=2) -> str:
    return "-" if value is None else f"{value * scale:,.{digits}f}"


def main() -> None:
    results = Path(sys.argv[1])
    rows = {arm: arm_row(results / arm) for arm in ARMS}
    print("| Arm | Verdict | IPC max_tokens | Profile estimate GiB | Graph pool GiB | Capture GiB | After warmup:"
          " free / transient / KV GiB | KV tokens | Used MiB per GPU |")
    print("|---|---|---:|---:|---:|---:|---|---:|---|")
    for arm, r in rows.items():
        warmup = " / ".join(cell(r[k]) for k in ("warmup_free_gib", "transient_gib", "kv_after_warmup_gib"))
        used = " ".join(map(str, r["used_mib"])) if r["used_mib"] else "-"
        print(f"| {arm} | {r['verdict']} | {cell(r['ipc_max_tokens'], digits=0)} | {cell(r['estimate_gib'])} |"
              f" {cell(r['graph_pool_gib'])} | {cell(r['capture_gib'])} | {warmup} | {cell(r['kv_tokens'], digits=0)} |"
              f" {used} |")
    for arm, r in rows.items():
        if r["error"]:
            print(f"\n{arm} error: `{r['error']}`")

    probe_file = results / "ws_probe" / "ws_probe.json"
    probe = json.loads(probe_file.read_text())["per_rank"] if probe_file.exists() else None
    print("\n| IPC arm vs control | KV tokens lost | As GiB per GPU | Transient delta GiB | Used delta MiB (max GPU)"
          " | Slab MiB | Workspace held MiB (max rank) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for control, ipc, shape in PAIRS:
        c, i = rows[control], rows[ipc]
        lost = None if c["kv_tokens"] is None or i["kv_tokens"] is None else c["kv_tokens"] - i["kv_tokens"]
        transient = None if c["transient_gib"] is None or i["transient_gib"] is None else (
            i["transient_gib"] - c["transient_gib"])
        used = None if not (c["used_mib"] and i["used_mib"]) else max(
            a - b for a, b in zip(i["used_mib"], c["used_mib"]))
        slab = held = None
        if probe:
            slab = probe[0][shape]["slab_bytes"] / MIB
            held = max(r[shape]["held_bytes"] for r in probe) / MIB
        print(f"| {ipc} vs {control} | {cell(lost, digits=0)} | {cell(lost, KV_BYTES_PER_TOKEN / GIB, 3)} |"
              f" {cell(transient, digits=3)} | {cell(used, digits=0)} | {cell(slab, digits=1)} |"
              f" {cell(held, digits=1)} |")


if __name__ == "__main__":
    main()
