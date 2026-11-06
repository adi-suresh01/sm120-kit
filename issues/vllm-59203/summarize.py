"""The run as markdown: the box's GPUs, every arm by every decode row, with each arm's GSM8K and the NCCL transport its
connections took beside the numbers, then what each arm actually ran (vLLM's all-reduce backends, whether FlashInfer's
workspace initialized, setup's probe). An arm is read by what its logs show, never by its env alone. Run on the laptop
after `harness down`; also writes <results_dir>/summary.md.
Usage: summarize.py <results_dir>
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

# harness.toml's order; b0-default is the baseline every "vs b0" column is read against.
ARMS = ["b0-default", "b2-sys-fi", "b1-p2p-sys", "b3-fi-only", "b0-repeat"]
BASELINE = "b0-default"
# cuda_communicator.py:353-360 at v0.33.0; the TP group's name starts with "tp".
BACKENDS = re.compile(r"Using \[([^\]]*)\] all-reduce backends \(in dispatch order\) for group 'tp")
FI_INIT = "Initialized FlashInfer PCIe IPC all-reduce for TP8"
TRANSPORT = re.compile(r"via ([A-Z0-9]+/[A-Za-z]+)")


def transports(path: Path) -> str:
    if not path.exists():
        return "no NCCL log"
    counts = Counter(TRANSPORT.findall(path.read_text(errors="replace")))
    return ", ".join(f"{name} x{n}" for name, n in counts.most_common()) or "no transport lines"


def backends(log: Path) -> tuple[str, str]:
    if not log.exists():
        return "no server.log", "-"
    text = log.read_text(errors="replace")
    match = BACKENDS.search(text)
    picked = match.group(1).replace("'", "") if match else "not logged"
    return picked, "yes" if FI_INIT in text else "no"


def delta(row: dict, base: dict | None, key: str) -> str:
    """The row's change against b0's same row, in percent."""
    if base is None or not float(base[key]):
        return "-"
    return f"{(float(row[key]) - float(base[key])) / float(base[key]) * 100:+.1f}%"


def table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def gpus(env: Path) -> str:
    """The cards from env_report.sh's GPU table: a Max-Q box decodes slower, so its rows must not pass for a Server
    Edition's."""
    if not env.exists():
        return "no env.txt"
    lines = env.read_text(errors="replace").splitlines()
    if "## GPUs" not in lines:
        return "no GPU table in env.txt"
    names = Counter()
    for line in lines[lines.index("## GPUs") + 2:]:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2 or not fields[0].isdigit():
            break
        names[fields[1]] += 1
    return ", ".join(f"{n}x {name}" for name, n in names.items()) or "no GPU rows in env.txt"


def probe(path: Path) -> str:
    if not path.exists():
        return "not run"
    report = json.loads(path.read_text())
    # A probe that stopped early has no times, only its reason.
    pynccl = report["pynccl"].get("eager_us", {})
    line = f"pynccl {pynccl.get('1', '-')} / {pynccl.get('8192', '-')}"
    if "flashinfer" in report:
        fi = report["flashinfer"]
        line += f"; FI {fi.get('eager_us', {}).get('1', '-')}"
        if "reason" in fi:
            line += f" ({fi['reason']})"
    return line if report["ok"] else f"FAILED: {line}"


def main() -> None:
    results = Path(sys.argv[1])
    arms = [arm for arm in ARMS if (results / arm).is_dir()]
    rows = {}
    for arm in arms:
        path = results / arm / "decode_rows.csv"
        rows[arm] = list(csv.DictReader(path.read_text().splitlines())) if path.exists() else []
    base = {row["row"]: row for row in rows.get(BASELINE, [])}

    lines = [table_row(["Arm", "Row", "Median ITL (ms)", "vs b0", "Mean TPOT (ms)", "Output tok/s", "vs b0",
                        "Median TTFT (ms)", "vs b0", "GSM8K", "NCCL transport (server)"])]
    facts = [table_row(["Arm", "Verdict", "Boot (s)", "vLLM all-reduce, dispatch order", "FlashInfer initialized",
                        "Probe eager us: pynccl batch 1 / 8192; FI batch 1"])]
    for arm in arms:
        verdict_path = results / arm / "verdict.json"
        verdict = json.loads(verdict_path.read_text()) if verdict_path.exists() else {}
        gsm8k = verdict.get("gsm8k")
        arm_cols = [f"{gsm8k:.2f}" if gsm8k is not None else "-", transports(results / arm / "nccl.log")]
        # decode_bench.py writes a row before it checks the row's completions, and post_smoke runs after a failed
        # GSM8K, so a failed arm can still have rows; its cells say so rather than read as a clean run.
        status = verdict.get("verdict", "no verdict")
        arm_cell = arm if status == "OK" else f"{arm} ({status})"
        for i, row in enumerate(rows[arm] or [{"row": "no rows"}]):
            b = base.get(row["row"])
            row_cell = row["row"]
            if "completed" in row and row["completed"] != row["prompts"]:
                row_cell += f" ({row['completed']}/{row['prompts']} completed)"
            cells = ["-"] * 7
            if "median_itl_ms" in row:
                cells = [row["median_itl_ms"], delta(row, b, "median_itl_ms"), row["mean_tpot_ms"],
                         row["output_tok_s"], delta(row, b, "output_tok_s"), row["median_ttft_ms"],
                         delta(row, b, "median_ttft_ms")]
            lines.append(table_row([arm_cell, row_cell, *cells, *(arm_cols if i == 0 else ["", ""])]))
        picked, fi = backends(results / arm / "server.log")
        probe_arm = BASELINE if arm == "b0-repeat" else arm
        boot_s = verdict.get("boot_s")
        facts.append(table_row([arm, verdict.get("verdict", "no verdict"), "-" if boot_s is None else str(boot_s),
                                picked, fi, probe(results / "probe" / f"{probe_arm}.json")]))

    lines.insert(1, "|" + "---|" * 11)
    facts.insert(1, "|" + "---|" * 6)
    text = "\n".join([f"GPUs: {gpus(results / 'env.txt')}", ""] + lines + [""] + facts) + "\n"
    (results / "summary.md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
