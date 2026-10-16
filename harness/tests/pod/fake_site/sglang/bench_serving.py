"""Fake `python3 -m sglang.bench_serving`: appends one result line shaped like the real --output-file record."""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--max-concurrency", type=int, required=True)
parser.add_argument("--num-prompts", type=int, required=True)
parser.add_argument("--output-file", required=True)
args, _ = parser.parse_known_args()

record = {
    "max_concurrency": args.max_concurrency,
    "completed": args.num_prompts,
    "output_throughput": 100.0 * args.max_concurrency,
}
with open(args.output_file, "a") as f:
    f.write(json.dumps(record) + "\n")
print(f"Output token throughput (tok/s): {record['output_throughput']:.2f}")
