"""Tables from ag_bench.py's rows, next to Lei-010's numbers from vllm#65359. Times are rank 0's, as Lei's are.
Usage: summarize.py <results dir>   (prints to stdout)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ARMS = ("ordered", "interleaved", "ordered-nccl-p2p", "ordered-repeat")
# issuecomment-6997879140: 8x Max-Q, 4 NUMA nodes (3+1+3+1), ordered_4plus4 False, FlashInfer on the seed (RD).
LEI = {
    (256, 1): (64.5, 13.5),
    (256, 8): (89.4, 27.6),
    (256, 16): (123.9, 86.7),
    (5120, 1): (172.9, 185.4),
    (5120, 8): (362.0, 1937.0),
    (5120, 16): (734.0, 4086.0),
}

out = Path(sys.argv[1])
rows = defaultdict(list)
for arm in ARMS:
    path = out / f"{arm}.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[arm].append(row)

runs = out / "runs.tsv"
if runs.exists():
    print("## Runs (rc 0 = finished; 124/137 = time limit)\n")
    print(runs.read_text())

for arm in ARMS:
    if not rows[arm]:
        print(f"## {arm}: no rows\n")
        continue
    print(f"## {arm}\n")
    for row in rows[arm]:
        if row["phase"] == "topology":
            print(f"ordered_4plus4={row['ordered_4plus4']} ({row['reason']}), fingerprint {row['fingerprint']}, "
                  f"CUDA_VISIBLE_DEVICES={row['cuda_visible_devices']}, NCCL env {row['nccl_env']}")
            print(f"{row['device']}, torch {row['torch']}, NCCL {row['nccl']}, FlashInfer {row['flashinfer']}\n")

    for phase in ("seed", "tuned"):
        table = [row for row in rows[arm] if row["phase"] == phase]
        if not table:
            continue
        print(f"{phase}: us per call (FI/NCCL < 1 means FlashInfer is faster)\n")
        print("| hidden x rows | KB | NCCL | NCCL into | FI | FI+post | copy | FI/NCCL | FI config | match | "
              "Lei NCCL | Lei FI |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in table:
            us = {key: value["rank0"] for key, value in row["us"].items()}
            lei_nccl, lei_fi = LEI.get((row["hidden"], row["c"]), ("", ""))
            print(f"| {row['hidden']} x {row['rows']} | {row['payload_kb']:.0f} | {us['nccl']:.1f} | "
                  f"{us['nccl_into']:.1f} | {us['fi']:.1f} | {us['fi_post']:.1f} | {us['copy']:.1f} | "
                  f"{us['fi'] / us['nccl']:.2f} | {'/'.join(map(str, row['config']))} | {row['fi_matches_nccl']} | "
                  f"{lei_nccl} | {lei_fi} |")
        print()

    sweep = [row for row in rows[arm] if row["phase"] == "sweep"]
    if sweep:
        print("sweep: best config per variant (rank 0 us; configs that did not match NCCL are counted, not timed)\n")
        print("| hidden x rows | variant | best us | config | configs tried | mismatches |")
        print("|---|---|---|---|---|---|")
        groups = defaultdict(list)
        for row in sweep:
            groups[(row["hidden"], row["rows"], row["config"][0])].append(row)
        for (hidden, n_rows, variant), group in sorted(groups.items()):
            timed = [row for row in group if row["us"]]
            bad = sum(not row["fi_matches_nccl"] for row in group)
            if timed:
                best = min(timed, key=lambda row: row["us"]["rank0"])
                print(f"| {hidden} x {n_rows} | {variant} | {best['us']['rank0']:.1f} | "
                      f"{'/'.join(map(str, best['config'][1:]))} | {len(group)} | {bad} |")
            else:
                print(f"| {hidden} x {n_rows} | {variant} | - | - | {len(group)} | {bad} |")
        print()

    for row in rows[arm]:
        if row["phase"] == "tune":
            print(f"tune: {row['seconds']} s, winners {row['winners']}\n")
