"""Turns one stage-1 box's logs into allreduce.csv, nccl_tuning.csv, pairwise.csv, and summary.txt, the first file
to read after `harness down`.

Reads what setup.sh leaves in the results directory: runs.tsv (how every run ended), nccl/<arm>/<run>.log
(nccl-tests output), vllm/rows.jsonl and vllm/*.fault.rank*.txt (vllm_ar_bench.py), pairwise/<a>-<b>.log, the NCCL
debug files beside those logs, and the env reports. Usage: python3 -I summarize.py <results_dir>
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

WORLD = 8
# 8 KB is nccl-tests' smallest common point; 10 KB and 80 KB are batch 1 and 8 at hidden 5120 in bf16, the sizes of
# the public TP8 trace (vllm#59203).
SUMMARY_SIZES = (8192, 10240, 81920)
CSV_FIELDS = ("source", "arm", "backend", "graph", "inplace", "size_bytes", "time_us", "algbw_gbps", "busbw_gbps",
              "wrong")


def runs(out: Path) -> list[dict]:
    with open(out / "runs.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def run_status(out: Path, run: dict) -> str:
    if run["rc"] == "skipped":
        return "skipped (deadline)"
    if run["rc"] == "0":
        return "ok"
    if run["rc"] in ("124", "137"):
        return "timeout"
    log = out / run["log"]
    text = log.read_text(errors="replace") if log.is_file() else ""
    if "Test timeout" in text:
        return "hang (nccl-tests -T)"
    if run["name"].startswith("vllm/"):
        arm = run["name"].split("/", 1)[1]
        for fault in sorted((out / "vllm").glob(f"{arm}.fault.rank*.txt")):
            lines = fault.read_text(errors="replace").splitlines()
            hit = next((i for i, line in enumerate(lines) if line.startswith("Timeout (")), None)
            if hit is not None:
                steps = [line for line in lines[:hit] if re.match(r"\d\d:\d\d:\d\d ", line)]
                return f"hang in {steps[-1][9:] if steps else 'an unnamed step'} ({fault.name})"
    return f"exit {run['rc']}"


def nccl_rows(text: str) -> list[list[str]]:
    """nccl-tests' result lines: size, count, type, redop, root, then time/algbw/busbw/#wrong out of place and in
    place, then the -U tuning columns when asked for."""
    rows = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) >= 13 and tokens[0].isdigit() and tokens[1].isdigit() and tokens[2] == "bfloat16":
            rows.append(tokens)
    return rows


def run_text(out: Path, log: str) -> str:
    """A run's log plus the NCCL debug files setup.sh writes beside it, <stem>.nccl.<host>.<pid>.log."""
    path = out / log
    files = [path, *sorted(path.parent.glob(f"{path.stem}.nccl.*.log"))]
    return "\n".join(f.read_text(errors="replace") for f in files if f.is_file())


def transports(text: str) -> str:
    seen = Counter(m.group(1).split("/")[0] for m in re.finditer(r" via (\S+)", text))
    return ", ".join(f"{name} x{n}" for name, n in seen.most_common()) or "not logged"


def parse_nccl(out: Path, all_runs: list[dict], table: list[dict], tuning: list[dict]) -> dict:
    """Returns {arm: {"tuning": {size: (algo, proto, channels)}, "transports": str, "version": str}}."""
    info = {}
    for run in all_runs:
        parts = run["name"].split("/")
        if parts[0] != "nccl" or not (out / run["log"]).is_file():
            continue
        arm, kind = parts[1], parts[2]
        text = run_text(out, run["log"])
        arm_info = info.setdefault(arm, {"tuning": {}, "transports": "not logged", "version": "?"})
        version = re.search(r"NCCL version (\S+)", text)
        if version:
            arm_info["version"] = version.group(1)
        if kind == "tuning":
            arm_info["transports"] = transports(text)
            for t in nccl_rows(text):
                if len(t) >= 16:
                    arm_info["tuning"][int(t[0])] = (t[14], t[15], t[-3])
                    tuning.append({"arm": arm, "size_bytes": t[0], "impl": t[13], "algo": t[14], "proto": t[15],
                                   "channels": t[-3], "warps": t[-2]})
            continue
        for t in nccl_rows(text):
            for inplace, base in ((0, 5), (1, 9)):
                table.append({"source": "nccl-tests", "arm": arm, "backend": "nccl",
                              "graph": 0 if kind.endswith("-eager") else 1, "inplace": inplace,
                              "size_bytes": int(t[0]), "time_us": float(t[base]), "algbw_gbps": float(t[base + 1]),
                              "busbw_gbps": float(t[base + 2]), "wrong": t[base + 3]})
    return info


def parse_vllm(out: Path, table: list[dict]) -> tuple[list[dict], int]:
    """Appends the measured rows; returns the rest (init, probe, declined, errors) and how many lines did not
    parse."""
    path = out / "vllm" / "rows.jsonl"
    events, unreadable = [], 0
    for line in path.read_text(errors="replace").splitlines() if path.is_file() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A run killed mid-write leaves its last line cut short; every other row still counts.
            unreadable += 1
            continue
        if row.get("status") == "ok" and "size_bytes" in row:
            table.append({"source": "vllm", "arm": row["arm"], "backend": row["backend"], "graph": 1,
                          "inplace": int(row["inplace"]), "size_bytes": row["size_bytes"], "time_us": row["time_us"],
                          "algbw_gbps": row["algbw_gbps"], "busbw_gbps": row["busbw_gbps"], "wrong": row["wrong"]})
        else:
            events.append(row)
    return events, unreadable


def parse_pairwise(out: Path, all_runs: list[dict], pairs: list[dict]) -> dict:
    latency = {}
    for run in all_runs:
        if not run["name"].startswith("pairwise/") or not (out / run["log"]).is_file():
            continue
        a, b = (int(x) for x in run["name"].split("/")[1].split("-"))
        text = run_text(out, run["log"])
        via = transports(text)
        for t in nccl_rows(text):
            pairs.append({"gpu_a": a, "gpu_b": b, "size_bytes": int(t[0]), "time_us": float(t[5]),
                          "busbw_gbps": float(t[7]), "transport": via})
            if int(t[0]) == 8:
                latency[(a, b)] = latency[(b, a)] = float(t[5])
    return latency


def write_csv(path: Path, rows: list[dict], fields) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def box_lines(out: Path) -> list[str]:
    env = (out / "env.txt").read_text(errors="replace") if (out / "env.txt").is_file() else ""
    extra = (out / "env_extra.txt").read_text(errors="replace") if (out / "env_extra.txt").is_file() else ""
    gpus = re.findall(r"^\d+, (NVIDIA [^,]+), [^,]+, [^,]+, ([^,]+),", env, re.M)
    names = Counter(name for name, _ in gpus)
    cpu = re.search(r"Model name:\s+(.+)", extra)
    sockets = re.search(r"Socket\(s\):\s+(\d+)", extra)
    numa = re.search(r"available: (\d+) nodes", extra)
    nccl = re.search(r"torch cuda (\S+) nccl (\S+)", extra)
    return [
        "gpus: " + (", ".join(f"{n}x {name}" for name, n in names.items()) or "see env.txt")
        + (f", driver {gpus[0][1]}" if gpus else ""),
        f"cpu: {cpu.group(1) if cpu else '?'}, {sockets.group(1) if sockets else '?'} socket(s), "
        f"{numa.group(1) if numa else '?'} NUMA node(s)",
        f"torch CUDA {nccl.group(1) if nccl else '?'}, NCCL {nccl.group(2) if nccl else '?'} (env_extra.txt)",
    ]


def fmt(value) -> str:
    return "-" if value is None else f"{value:.1f}"


def main() -> None:
    out = Path(sys.argv[1])
    all_runs = runs(out)
    table, tuning, pairs = [], [], []
    nccl_info = parse_nccl(out, all_runs, table, tuning)
    events, unreadable = parse_vllm(out, table)
    latency = parse_pairwise(out, all_runs, pairs)
    write_csv(out / "allreduce.csv", table, CSV_FIELDS)
    write_csv(out / "nccl_tuning.csv", tuning, ("arm", "size_bytes", "impl", "algo", "proto", "channels", "warps"))
    write_csv(out / "pairwise.csv", pairs, ("gpu_a", "gpu_b", "size_bytes", "time_us", "busbw_gbps", "transport"))

    lines = ["pcie-tp8-allreduce stage 1", *box_lines(out), ""]
    probe = next((e for e in events if e.get("event") == "probe"), None)
    if probe:
        peers_ok = all(all(row) for row in probe["peer_access"])
        lines += [
            f"vLLM at TP8 by itself picks {probe['vllm_default_pick']}; enabled {probe['enabled']}",
            f"  NVLink fully connected (NVML) {probe['nvml_nvlink_fully_connected']}, peer access on every pair "
            f"{peers_ok}, multicast {probe['multicast']}, symm-mem table has {probe['capability']} "
            f"{probe['symm_mem_table_has_capability']}, FlashInfer has PCIe IPC {probe['flashinfer_has_pcie_ipc']}",
            f"  NCCL torch {probe['torch_nccl']}, pynccl {probe['pynccl_nccl']}, {probe['libnccl']}",
            "",
        ]

    statuses = {run["name"]: run_status(out, run) for run in all_runs}
    lines.append("all-reduce out of place (torch: in place), CUDA graphs unless marked eager; us at 8 KB / 10 KB / "
                 "80 KB; wrong = mismatched elements over the whole sweep")
    groups = {}
    for row in table:
        if row["inplace"] and row["backend"] != "torch":
            continue
        key = (row["source"], row["arm"], row["backend"], row["graph"])
        groups.setdefault(key, []).append(row)
    clean = []
    for (source, arm, backend, graph), rows in sorted(groups.items()):
        at = {r["size_bytes"]: r["time_us"] for r in rows}
        wrong = sum(int(r["wrong"]) for r in rows if str(r["wrong"]).isdigit())
        unknown = sum(1 for r in rows if not str(r["wrong"]).isdigit())
        if graph and not wrong and not unknown:
            clean += rows
        label = f"{source} {arm} {backend}" + ("" if graph else " eager")
        extra = ""
        if source == "nccl-tests":
            arm_info = nccl_info.get(arm, {})
            picked = arm_info.get("tuning", {}).get(8192)
            extra = (f"  algo/proto/channels at 8 KB {'/'.join(picked) if picked else '?'}, "
                     f"via {arm_info.get('transports', '?')}")
            status = ", ".join(sorted({s for name, s in statuses.items() if name.startswith(f"nccl/{arm}/")}))
        else:
            status = statuses.get(f"vllm/{arm}", "?")
            via = transports(run_text(out, f"vllm/{arm}.log"))
            extra = f"  via {via}" if via != "not logged" else ""
        lines.append(f"{label:<44} {' / '.join(fmt(at.get(s)) for s in SUMMARY_SIZES):<20} wrong {wrong}"
                     + (f" (+{unknown} unchecked rows)" if unknown else "") + f"  [{status}]{extra}")

    shown = {(source, arm) for source, arm, _, _ in groups}
    for run in all_runs:
        kind, arm = run["name"].split("/")[:2]
        source = {"nccl": "nccl-tests", "vllm": "vllm"}.get(kind)
        # The probe measures nothing by design; its findings are printed above.
        if source and arm != "probe" and (source, arm) not in shown:
            shown.add((source, arm))
            lines.append(f"{source} {arm}: no measurements  [{statuses[run['name']]}]")
    for event in events:
        if event.get("status", "").startswith("error") or event.get("disabled") or event.get("initialized") is False:
            lines.append(f"  note: {json.dumps(event)[:300]}")
    if unreadable:
        lines.append(f"  note: {unreadable} line(s) of vllm/rows.jsonl did not parse and were skipped")
    lines.append("")

    baseline = {r["size_bytes"]: r["time_us"] for r in table
                if r["source"] == "vllm" and r["arm"] == "nccl-default" and r["backend"] == "pynccl"}
    # Stage 2's question: how much a vLLM path with no mismatch anywhere in its sweep saves over pynccl, what vLLM
    # runs today. nccl-tests drives 8 GPUs from one process, a different harness, so it is only a reference.
    for size in (10240, 81920):
        at_size = [r for r in clean if r["size_bytes"] == size]
        vllm_rows = [r for r in at_size if r["source"] == "vllm"]
        if vllm_rows and size in baseline:
            best = min(vllm_rows, key=lambda r: r["time_us"])
            lines.append(f"fastest clean vLLM path at {size} B: {best['arm']} {best['backend']} "
                         f"{best['time_us']:.1f} us vs pynccl {baseline[size]:.1f} us = "
                         f"{baseline[size] / best['time_us']:.2f}x")
        reference = [r for r in at_size if r["source"] == "nccl-tests"]
        if reference:
            best = min(reference, key=lambda r: r["time_us"])
            lines.append(f"  nccl-tests reference at {size} B (one process, not comparable 1:1): {best['arm']} "
                         f"{best['time_us']:.1f} us")
    lines.append("")

    if latency:
        lines.append("pairwise sendrecv, 8 B, us (nccl-tests, graph mode)")
        lines.append("     " + "".join(f"{j:>7}" for j in range(WORLD)))
        for i in range(WORLD):
            lines.append(f"{i:>5}" + "".join(f"{fmt(latency.get((i, j))):>7}" if i != j else f"{'X':>7}"
                                             for j in range(WORLD)))
        lines.append("")

    counts = Counter(s.split(" ")[0] for s in statuses.values())
    lines.append("runs: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) + " (runs.tsv)")
    lines += [f"  {name}: {s}" for name, s in statuses.items() if s != "ok"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
