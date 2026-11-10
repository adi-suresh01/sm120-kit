"""Prints run_sweep.sh's results as one table: microseconds per all-reduce at each message size for vLLM's NCCL path
and FlashInfer's PCIe IPC all-reduce, plus wrong-result counts, what vLLM picks by itself, and NCCL's transports.

Usage: python3 -I sweep_table.py <out dir> <hidden size>
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A step killed at its time limit can leave half a line.
            continue
    return rows


def kib(size: int) -> str:
    return f"{size / 1024:g} KiB"


def main() -> None:
    out, hidden = Path(sys.argv[1]), int(sys.argv[2])
    rows = load_rows(out / "rows.jsonl")

    print(f"{'step':<16}{'exit':>5}{'seconds':>9}   (124 = hit its time limit; rows written before it still count)")
    for line in (out / "runs.tsv").read_text().splitlines()[1:]:
        arm, rc, seconds = line.split("\t")
        print(f"{arm:<16}{rc:>5}{seconds:>9}")
    print()

    for r in rows:
        if r.get("event") == "probe":
            peers = all(all(pair) for pair in r["peer_access"])
            print(f"vLLM left alone picks: {r['vllm_default_pick']} (NVLink between all GPUs: "
                  f"{r['nvml_nvlink_fully_connected']}, P2P on every pair: {peers}, "
                  f"FlashInfer PCIe IPC available: {r['flashinfer_has_pcie_ipc']})")
        elif r.get("backend") == "fi_pcie_ipc" and r.get("event") == "init":
            print(f"FlashInfer setup: {r['status']}, initialized {r['initialized']} in {r['setup_s']} s, "
                  f"profile {r['profile']} ({r['profile_reason']})")
    transports = Counter()
    for log in out.glob("nccl-*.nccl.*.log"):
        transports.update(re.findall(r" via ([A-Z0-9]+)", log.read_text(errors="replace")))
    print(f"NCCL channels by transport: {dict(transports) or 'no NCCL debug lines found'}")

    timed, problems = {}, []
    for r in rows:
        if "size_bytes" not in r:
            continue
        if r["status"] == "ok":
            timed[(r["size_bytes"], r["backend"])] = r
        elif r["status"] != "declined":
            problems.append(f"{r['backend']} at {kib(r['size_bytes'])}: {r['status']}")

    print(f"\nMicroseconds per all-reduce (slowest GPU's median, CUDA graphs); batch = tokens at hidden size {hidden}")
    print(f"{'size':>10}{'batch':>7}{'NCCL':>10}{'FlashInfer':>12}{'FI/NCCL':>9}")
    for size in sorted({size for size, _ in timed}):
        nccl, fi = timed.get((size, "pynccl")), timed.get((size, "fi_pcie_ipc"))
        shape = nccl or fi or timed.get((size, "torch"))
        batch = str(shape["rows"]) if shape["cols"] == hidden else "-"
        nccl_us = f"{nccl['time_us']:.1f}" if nccl else "-"
        fi_us = f"{fi['time_us']:.1f}" if fi else "-"
        ratio = f"{fi['time_us'] / nccl['time_us']:.2f}" if nccl and fi else "-"
        print(f"{kib(size):>10}{batch:>7}{nccl_us:>10}{fi_us:>12}{ratio:>9}")

    print()
    for backend in ("pynccl", "torch", "fi_pcie_ipc"):
        done = [r for (_, b), r in timed.items() if b == backend]
        if done:
            wrong = sum(r["wrong"] for r in done)
            checked = sum(r["checked_elements"] for r in done)
            print(f"{backend}: {wrong} wrong of {checked:,} checked values over {len(done)} sizes")
    if problems:
        print("\nDid not finish:")
        for problem in problems:
            print(f"  {problem}")
    print("\nFI/NCCL under 1.00 means FlashInfer is faster. A wrong value, an error, or a step that timed out before its "
          "last size means that backend is not safe on this box.")


if __name__ == "__main__":
    main()
