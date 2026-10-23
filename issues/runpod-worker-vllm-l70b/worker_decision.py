#!/usr/bin/env python3
"""Prints what worker-vllm does when its vLLM boot ends the way this log ends, using the worker's own classifier.

The rules are src/main.py's at 71d503e6: it keeps the last 400 lines of vLLM's output; after a failed first launch it
relaunches once with ENFORCE_EAGER=true and MAX_NUM_BATCHED_TOKENS=8192 when startup_errors.memory_shortfall()
matches; otherwise, and after that relaunch fails too, it answers every job with startup_errors.classify()'s
message, or exits 1 for the platform to restart it when classify() has none. main.py checks for a vanished revision
and a full disk before memory, each with a relaunch of its own; no variant here should hit them, so they are only
reported, and a true one means the decision below is not the worker's.
Usage: worker_decision.py <startup_errors.py> <server.log> <first|retry> <model>
"""
import importlib.util
import json
import sys

TAIL_LINES = 400

errors_path, log_path, launch, model = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("startup_errors", errors_path)
errors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)

with open(log_path, errors="replace") as f:
    lines = f.readlines()
tail = "".join(lines[-TAIL_LINES:])
booted = any("Application startup complete" in line for line in lines)
shortfall = errors.memory_shortfall(tail)

message = None
if booted:
    worker = "serves; nothing in its responses says which launch this is"
elif launch == "first" and shortfall:
    worker = "relaunches once with ENFORCE_EAGER=true MAX_NUM_BATCHED_TOKENS=8192"
else:
    message = errors.classify(tail, model=model, oom_retried=launch == "retry")
    worker = "answers every job with message" if message else "exits 1; the platform restarts it on the same config"
print(json.dumps({"launch": launch, "booted": booted, "memory_shortfall": shortfall,
                  "revision_not_found": errors.revision_not_found(tail), "out_of_disk": errors.out_of_disk(tail),
                  "worker": worker, "message": message}, indent=1))
