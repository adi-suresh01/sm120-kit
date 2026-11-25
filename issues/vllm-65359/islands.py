"""Find the 8-GPU split FlashInfer 0.7.3 calls ordered 4+4, and the two GPU orders the bench runs in.

decide_pcie_ipc_ag_rs_topology (flashinfer/comm/pcie_ipc_collectives/_ag_rs_topology.py) admits the copy-engine
all-gather only when logical ranks 0-3 and 4-7 form two islands: every pair inside an island below
NVML_TOPOLOGY_SYSTEM, every pair across at or above it. This checks the same NVML levels for every 4+4 split and prints
two CUDA_VISIBLE_DEVICES values in NVML (PCI bus) order:
  ordered      island A then island B: ordered_4plus4 should come out True
  interleaved  a0,b0,a1,b1,...: each logical half now spans both islands, so the same GPUs come out False
or "none" when no split works. Exits 0 either way; setup.sh decides what "none" means.
"""

import itertools

try:
    import pynvml
except ImportError:
    from vllm.third_party import pynvml

pynvml.nvmlInit()
count = pynvml.nvmlDeviceGetCount()
if count != 8:
    print("none", f"{count} GPUs")
    raise SystemExit(0)

handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
crosses = {
    (i, j): pynvml.nvmlDeviceGetTopologyCommonAncestor(handles[i], handles[j]) >= pynvml.NVML_TOPOLOGY_SYSTEM
    for i in range(count)
    for j in range(count)
    if i != j
}

for island_a in itertools.combinations(range(count), 4):
    # Fixing GPU 0 in island A visits each split once.
    if 0 not in island_a:
        continue
    island_b = tuple(i for i in range(count) if i not in island_a)
    inside_local = all(
        not crosses[i, j] for island in (island_a, island_b) for i, j in itertools.combinations(island, 2)
    )
    across_system = all(crosses[i, j] for i in island_a for j in island_b)
    if inside_local and across_system:
        print("ordered", ",".join(map(str, island_a + island_b)))
        print("interleaved", ",".join(str(gpu) for pair in zip(island_a, island_b) for gpu in pair))
        break
else:
    print("none", "no 4+4 split with every pair inside below SYS and every pair across at SYS")
