"""Decode rows on the live server with vLLM's own client, `vllm bench serve`, sending the same prompts on every arm.

A copy of issues/vllm-59203/decode_bench.py with the same rows and seeds, minus c=64, so the TP4 rows read against
that kit's TP8 rows. Each row is one run on the random dataset at fixed lengths (in vLLM a range ratio of 0 fixes them,
benchmarks/datasets/utils.py get_sampling_params at 4fab5a07), --ignore-eos so every request decodes its full output,
greedy, every request queued at once behind --max-concurrency. Its warmup requests send the row's first prompt
(benchmarks/serve.py:926-945), which would then serve that request's prefill from the prefix cache and shorten its
TTFT, so they stay off and a warmup run with a seed of its own goes before each shape's rows: it takes the first-use
cost of the shape's prefill and decode before anything is measured. Each row has its own seed, the same on every arm,
so the rows pair across arms and no row repeats another's prompts. The 32K shape goes last, so a fault there does not
cost the 1K rows.
Writes <out_dir>/bench/<run>.json (the client's result, per-request TTFTs and ITLs included) and <run>.txt (its
output), and one line per measured row to <out_dir>/decode_rows.csv. Exits 1 at the first run that completes fewer
requests than it sent, so the harness reports POST_FAIL.
Usage: decode_bench.py <port> <model> <variant> <out_dir> <tokenizer_dir>
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

# (run, input tokens, output tokens, concurrency, prompts, seed), in order; runs named warmup-* are not reported. 511
# gaps per 512-token request hold the median ITL steady at c=1 with 8 requests; the c=8 and c=32 rows run two or more
# waves so the median is not one batch's. The 32K row is for TTFT: its prefill all-reduces are one chunk x 5120 x bf16
# (80 MiB at the default 8192-token chunk), and every arm sends them through NCCL, since vLLM sizes the FlashInfer
# workspace to the CUDA graph batches only.
RUNS = [
    ("warmup-in1k-c8", 1024, 512, 8, 16, 101),
    ("in1k-c1", 1024, 512, 1, 8, 11),
    ("in1k-c8", 1024, 512, 8, 32, 12),
    ("in1k-c32", 1024, 512, 32, 64, 13),
    ("warmup-in32k-c1", 32768, 256, 1, 2, 102),
    ("in32k-c1", 32768, 256, 1, 6, 15),
]
FIELDS = [
    "variant", "row", "input_len", "output_len", "concurrency", "prompts", "completed", "failed",
    "output_tok_s", "median_itl_ms", "p90_itl_ms", "p99_itl_ms", "mean_tpot_ms", "median_tpot_ms",
    "median_ttft_ms", "mean_ttft_ms", "p99_ttft_ms", "duration_s",
]


def bench(port: int, model: str, tokenizer: Path, row_dir: Path, row: str, input_len: int, output_len: int,
          concurrency: int, prompts: int, seed: int) -> dict | None:
    cmd = [
        "vllm", "bench", "serve", "--backend", "openai", "--host", "127.0.0.1", "--port", str(port),
        "--model", model, "--tokenizer", str(tokenizer),
        "--dataset-name", "random", "--random-input-len", str(input_len), "--random-output-len", str(output_len),
        "--random-range-ratio", "0", "--ignore-eos", "--temperature", "0",
        "--max-concurrency", str(concurrency), "--num-prompts", str(prompts), "--seed", str(seed),
        "--num-warmups", "0", "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
        "--save-result", "--save-detailed", "--result-dir", str(row_dir), "--result-filename", f"{row}.json",
        "--disable-tqdm",
    ]
    with open(row_dir / f"{row}.txt", "w") as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, check=False)
    # The client writes no result when it crashed before the run ended.
    result = row_dir / f"{row}.json"
    return json.loads(result.read_text()) if result.exists() else None


def summarize(variant: str, row: str, input_len: int, output_len: int, concurrency: int, prompts: int,
              run: dict) -> dict:
    def ms(key: str) -> float:
        return round(run[key], 2)

    return {
        "variant": variant,
        "row": row,
        "input_len": input_len,
        "output_len": output_len,
        "concurrency": concurrency,
        "prompts": prompts,
        "completed": run["completed"],
        "failed": run["failed"],
        "output_tok_s": round(run["output_throughput"], 1),
        "median_itl_ms": ms("median_itl_ms"),
        "p90_itl_ms": ms("p90_itl_ms"),
        "p99_itl_ms": ms("p99_itl_ms"),
        "mean_tpot_ms": ms("mean_tpot_ms"),
        "median_tpot_ms": ms("median_tpot_ms"),
        "median_ttft_ms": round(run["median_ttft_ms"], 1),
        "mean_ttft_ms": round(run["mean_ttft_ms"], 1),
        "p99_ttft_ms": round(run["p99_ttft_ms"], 1),
        "duration_s": round(run["duration"], 1),
    }


def main() -> None:
    port, model, variant = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    out, tokenizer = Path(sys.argv[4]), Path(sys.argv[5])
    row_dir = out / "bench"
    row_dir.mkdir(parents=True, exist_ok=True)
    with open(out / "decode_rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        f.flush()
        for row, input_len, output_len, concurrency, prompts, seed in RUNS:
            run = bench(port, model, tokenizer, row_dir, row, input_len, output_len, concurrency, prompts, seed)
            if run is None:
                print(row, "FAILED: no result; see", row_dir / f"{row}.txt", flush=True)
                sys.exit(1)
            if row.startswith("warmup-"):
                print(row, f"{run['completed']}/{prompts} completed", flush=True)
            else:
                summary = summarize(variant, row, input_len, output_len, concurrency, prompts, run)
                writer.writerow(summary)
                f.flush()
                print(row, json.dumps(summary), flush=True)
            if run["completed"] != prompts:
                # A crashed engine fails every later row too; one failure is the answer.
                print(row, f"FAILED: {run['completed']}/{prompts} requests completed", flush=True)
                sys.exit(1)


if __name__ == "__main__":
    main()
