"""Decode rows on the live server with sglang's own client, the method agreed on sglang#33422 for the P2P pair.

Each row is one `python3 -m sglang.bench_serving --backend sglang` run: the random dataset (ShareGPT text repeated to
length) at a fixed length, greedy with ignore_eos (that backend's defaults at v0.5.22, benchmark/serving.py:695-697
and :2514-2517), OUTPUT_LEN tokens out, its one warmup request, then the radix cache flushed, since the warmup sends
the row's first prompt and would otherwise serve its prefill from cache. sglang fixes the lengths at
--random-range-ratio 1: it draws them from [ratio x target, target] (benchmark/datasets/common.py:56-64), so 0, the
ratio that fixes vLLM's, spreads them from 1 token to the target.
Writes <out_dir>/decode/<row>.jsonl (the client's record with --output-details: per-request TTFT and inter-token
gaps) and <row>.txt (its output), and one line per row to <out_dir>/decode_rows.csv. Exits 1 at the first row that
completes fewer requests than it sent, so the harness reports POST_FAIL.
Usage: decode_bench.py <port> <variant> <out_dir> <prompt_dir>
"""

import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

# 511 gaps per request (over 4,000 per row) hold the median and p99 ITL steady, and at about 90 tok/s a request
# decodes for about 6 s; 1024 would add about 2 min per arm and resolve nothing the gaps do not.
OUTPUT_LEN = 512
# (shape, input tokens, concurrency, prompts). W1 is the reporter's 9 x 90K, so its rows send 9 prompts; W2's send 8,
# and at c=8 their 1.9M tokens fit the 6.87M-token full KV pool the 10-17 boots allocated (README.md), so all 8 run
# at once.
ROWS = [
    ("w1", 90_000, 1, 9),
    ("w1", 90_000, 8, 9),
    ("w2", 238_000, 1, 8),
    ("w2", 238_000, 8, 8),
]
# One seed on every arm, so each row sends the same prompts on all of them and the rows pair.
SEED = 1
FIELDS = [
    "variant", "shape", "input_len", "concurrency", "prompts", "completed", "input_tokens", "output_tokens",
    "cached_tokens", "ttft_mean_ms", "ttft_median_ms", "tpot_mean_ms", "itl_median_ms", "itl_p99_ms",
    "decode_tok_s_min", "decode_tok_s_median", "decode_tok_s_max", "duration_s",
]


def bench(port: int, prompt_dir: Path, row_dir: Path, row: str, input_len: int, concurrency: int,
          prompts: int) -> dict | None:
    record = row_dir / f"{row}.jsonl"
    cmd = [
        sys.executable, "-m", "sglang.bench_serving", "--backend", "sglang",
        "--host", "127.0.0.1", "--port", str(port),
        "--dataset-name", "random", "--dataset-path", str(prompt_dir / "sharegpt.json"),
        # The checkpoint's pinned revision, not the main branch the model id would load.
        "--tokenizer", str(prompt_dir / "tokenizer"),
        "--random-input-len", str(input_len), "--random-output-len", str(OUTPUT_LEN), "--random-range-ratio", "1",
        "--num-prompts", str(prompts), "--max-concurrency", str(concurrency), "--seed", str(SEED),
        "--flush-cache", "--cache-report", "--disable-tqdm",
        "--output-file", str(record), "--output-details",
    ]
    with open(row_dir / f"{row}.txt", "w") as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=row_dir, check=False)
    # The client writes no record when every request failed or it crashed.
    if not record.exists():
        return None
    return json.loads(record.read_text().splitlines()[-1])


def summarize(variant: str, shape: str, input_len: int, concurrency: int, prompts: int, run: dict) -> dict:
    # A request's decode rate is its gaps over their sum: first token to last, as the client saw them.
    rates = [len(gaps) / sum(gaps) for gaps in run["itls"] if gaps and sum(gaps) > 0]
    return {
        "variant": variant,
        "shape": shape,
        "input_len": input_len,
        "concurrency": concurrency,
        "prompts": prompts,
        "completed": run["completed"],
        # input_lens is the server's count; total_input_tokens is the dataset's target.
        "input_tokens": sum(run["input_lens"]),
        "output_tokens": run["total_output_tokens"],
        # Should be 0 on every row: anything else served part of a prefill from cache and shortens TTFT.
        "cached_tokens": sum(n or 0 for n in run["cached_tokens"]),
        "ttft_mean_ms": round(run["mean_ttft_ms"], 1),
        "ttft_median_ms": round(run["median_ttft_ms"], 1),
        "tpot_mean_ms": round(run["mean_tpot_ms"], 2),
        "itl_median_ms": round(run["median_itl_ms"], 2),
        "itl_p99_ms": round(run["p99_itl_ms"], 2),
        "decode_tok_s_min": round(min(rates), 1) if rates else None,
        "decode_tok_s_median": round(statistics.median(rates), 1) if rates else None,
        "decode_tok_s_max": round(max(rates), 1) if rates else None,
        "duration_s": round(run["duration"], 1),
    }


def main() -> None:
    port, variant, out, prompt_dir = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    row_dir = out / "decode"
    row_dir.mkdir(parents=True, exist_ok=True)
    with open(out / "decode_rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        f.flush()
        for shape, input_len, concurrency, prompts in ROWS:
            row = f"{shape}-c{concurrency}"
            run = bench(port, prompt_dir, row_dir, row, input_len, concurrency, prompts)
            if run is None:
                print(row, "FAILED: no record; see", row_dir / f"{row}.txt", flush=True)
                sys.exit(1)
            summary = summarize(variant, shape, input_len, concurrency, prompts, run)
            writer.writerow(summary)
            f.flush()
            print(row, json.dumps(summary), flush=True)
            if run["completed"] != prompts:
                # A crashed engine fails every later row too; one failure is the answer.
                print(row, f"FAILED: {run['completed']}/{prompts} requests completed", flush=True)
                sys.exit(1)


if __name__ == "__main__":
    main()
