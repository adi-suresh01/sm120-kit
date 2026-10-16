"""Fake `python3 -m sglang.test.run_eval`: prints the two lines verdict.sh reads, after one request to the server."""
import argparse
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--num-examples", type=int, required=True)
args, _ = parser.parse_known_args()

try:
    urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=5)
except OSError:
    print("All retry attempts exhausted")
print("Score: 0.970")
print(f"Speculative accept length (per-request, from meta_info): n={args.num_examples} mean=2.9374 min=2.4 max=3.4")
