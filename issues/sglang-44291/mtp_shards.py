#!/usr/bin/env python3
"""Prints the safetensors shards that hold mtp.* tensors, one per line.

Usage: mtp_shards.py <model.safetensors.index.json>

The RadixArk draft head lives in 3 of the export's 206 shards, so fetching only these saves ~120 GB.
"""
import json
import sys

index_path = sys.argv[1]
with open(index_path) as f:
    weight_map = json.load(f)["weight_map"]
shards = sorted({shard for name, shard in weight_map.items() if name.startswith("mtp.")})
if not shards:
    sys.exit(f"no mtp.* tensors in {index_path}")
print("\n".join(shards))
