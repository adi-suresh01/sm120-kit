#!/usr/bin/env python3
"""Summarizes one boot's FlashInfer autotune: did the gate drop the cache, did any rank profile, and did the
per-rank cache files carry over unchanged. Usage: autotune_report.py <run_dir>

sglang sets FlashInfer's logger to ERROR, so its "Loaded"/"Saved" lines never reach server.log. Reuse is read from
the files instead: hooks.sh copies the cache before the launch and after the kill, and every profiled entry carries
its measured time, so a re-tuned entry changes even when it picks the same tactic.
"""

import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)([^\]]*)\] (.*)$")
RANK = re.compile(r"\bTP\d+")
# Op names hold colons (trtllm::fused_moe::gemm1); tqdm puts ": " after the description.
TUNING = re.compile(r"\[AutoTuner\]: Tuning (\S+): ")


def parse_log(log: Path) -> None:
    text = log.read_text(errors="replace") if log.exists() else ""
    starts: dict[str, datetime] = {}
    seconds: dict[str, float] = defaultdict(float)
    rounds: dict[str, int] = defaultdict(int)
    drops = 0
    # tqdm redraws its bar with carriage returns, so each redraw is its own record.
    for record in re.split(r"[\r\n]+", text):
        m = LINE.match(record)
        if not m:
            continue
        rank = RANK.search(m[2])
        stamp, rank, msg = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S"), rank[0] if rank else "TP0", m[3]
        if msg.startswith("Running FlashInfer autotune"):
            starts[rank] = stamp
            rounds[rank] += 1
        elif msg.startswith("FlashInfer autotune completed") and rank in starts:
            seconds[rank] += (stamp - starts.pop(rank)).total_seconds()
        elif "per-rank caches disagree" in msg:
            drops += 1
    ops = sorted(set(TUNING.findall(text)))
    print(f"gate_drops={drops}")
    for rank in sorted(rounds):
        print(f"autotune {rank}: rounds={rounds[rank]} seconds={seconds[rank]:.0f}")
    print(f"tuned_ops={','.join(ops) if ops else 'none'}")


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def caches(root: Path) -> dict[str, dict[str, Path]]:
    found: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in sorted(root.rglob("rank_*.json")) if root.is_dir() else []:
        found[str(path.parent.relative_to(root))][path.stem] = path
    return found


def compare(run_dir: Path) -> None:
    before, after = caches(run_dir / "cache_before"), caches(run_dir / "cache_after")
    for key in sorted(set(before) | set(after)):
        print(f"cache {key}")
        for rank in sorted(set(before.get(key, {})) | set(after.get(key, {}))):
            old, new = before.get(key, {}).get(rank), after.get(key, {}).get(rank)
            old_e, new_e = (load(old) if old else None), (load(new) if new else None)
            changed = "-" if old_e is None or new_e is None else sum(
                old_e.get(k) != v for k, v in new_e.items()) + len(old_e.keys() - new_e.keys())
            sha = hashlib.sha256(new.read_bytes()).hexdigest()[:12] if new else "-"
            print(f"  {rank}: before={len(old_e) if old_e is not None else '-'} "
                  f"after={len(new_e) if new_e is not None else '-'} changed={changed} sha={sha}")
        ranks = sorted(after.get(key, {}).items())
        if len(ranks) > 1:
            (name_a, a), (name_b, b) = ranks[0], ranks[1]
            ea, eb = load(a), load(b)
            only_a, only_b = sorted(ea.keys() - eb.keys()), sorted(eb.keys() - ea.keys())
            differ = sum(ea[k] != eb[k] for k in ea.keys() & eb.keys())
            print(f"  {name_a} vs {name_b}: only_{name_a}={len(only_a)} only_{name_b}={len(only_b)} "
                  f"same_key_other_value={differ}")
            # Two keys are enough to see which fields (ranks, shapes) differ between the ranks.
            for k in only_a[:1] + only_b[:1]:
                print(f"    {k[:240]}")


def main() -> None:
    run_dir = Path(sys.argv[1])
    parse_log(run_dir / "server.log")
    compare(run_dir)


if __name__ == "__main__":
    main()
