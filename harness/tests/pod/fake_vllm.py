"""Fake `vllm bench serve`: with --save-result, writes a result file shaped like the real one.

Only the flags quality.sh passes are accepted, each of which vllm bench serve has in v0.25.1 and v0.31.0, so a
misspelled or dropped one fails here as it would on the pod. FAKE_BENCH=drop completes one request short, the way the
real client finishes a run with failed requests and still exits 0; FAKE_BENCH=crash fails the way it does when
--model names no tokenizer it can load.
"""
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser(prog="vllm")
parser.add_argument("command", choices=["bench"])
parser.add_argument("subcommand", choices=["serve"])
for flag in ["--backend", "--host", "--model", "--dataset-name", "--random-range-ratio", "--result-dir",
             "--result-filename"]:
    parser.add_argument(flag, required=True)
for flag in ["--port", "--random-input-len", "--random-output-len", "--max-concurrency", "--num-prompts", "--seed"]:
    parser.add_argument(flag, type=int, required=True)
for flag in ["--ignore-eos", "--save-result"]:
    parser.add_argument(flag, action="store_true")
args = parser.parse_args()
if os.environ.get("FAKE_BENCH") == "crash":
    sys.exit(f"OSError: {args.model} is not a local folder and is not a valid model identifier")

record = {
    "max_concurrency": args.max_concurrency,
    "num_prompts": args.num_prompts,
    "completed": args.num_prompts - (os.environ.get("FAKE_BENCH") == "drop"),
    "output_throughput": 100.0 * args.max_concurrency,
}
if args.save_result:
    with open(os.path.join(args.result_dir, args.result_filename), "w") as f:
        json.dump(record, f)
print(f"Output token throughput (tok/s): {record['output_throughput']:.2f}")
