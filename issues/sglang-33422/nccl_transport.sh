#!/usr/bin/env bash
# What NCCL logged about its transports, from a log written with NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,
# short enough to quote: whether P2P was in use is read from these lines, not from the env the arm set.
# Line formats are NCCL 2.30.7's (transport/p2p.cc:416-437, transport/shm.cc:116, graph/search.cc:1319).
# Usage: nccl_transport.sh <log>
set -euo pipefail

log=${1:?usage: $0 <log>}

grep -m1 -oE 'NCCL version [^ ]+' "$log" || echo "no NCCL version line"
# An unset or "0" NCCL_P2P_DISABLE prints nothing here; 1, or any NCCL_P2P_LEVEL, prints a line.
grep -oE 'NCCL_[A-Z0-9_]+ set by environment to [^ ]+' "$log" | sort -u || true
# NCCL's verdict where NVML reports a pair's P2P read or write as not OK.
grep -oE 'P2P is disabled between connected GPUs [0-9]+ and [0-9]+' "$log" | sort -u || true

echo "## search result per communicator (type: path NCCL saw between GPUs)"
grep -oE 'Pattern [0-9]+, crossNic [0-9]+, nChannels [0-9]+, bw [0-9.]+/[0-9.]+, type [A-Z0-9]+/[A-Z0-9]+' "$log" \
    | sort | uniq -c || true

# A server runs several communicators over the same GPUs, so the counts add theirs up.
echo "## connections per rank pair and transport, rank[gpu]"
grep -oE '[0-9]+\[[0-9]+\] -> [0-9]+\[[0-9]+\] (\[(send|receive)\] )?via [A-Za-z0-9]+(/[A-Za-z0-9]+)*' "$log" \
    | sort | uniq -c || true

echo "## channel lines, deduplicated across processes and communicators"
grep -E 'NCCL INFO Channel [0-9]+(/[0-9]+)? : .* via ' "$log" | sed -E 's/^.*NCCL INFO //' | sort | uniq -c || true
