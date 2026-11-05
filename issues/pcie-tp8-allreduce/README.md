# TP8 decode all-reduce on 8 PCIe-only RTX PRO 6000: stage 1, communication only

Decode at TP8 on 8x RTX PRO 6000 without NVLink is all-reduce latency bound, and vLLM sends every one of those
all-reduces through NCCL's ring. Stage 1 is a microbenchmark with no model weights: on a box where P2P reads OK on
every GPU pair, how fast can an 8-GPU all-reduce be at decode sizes, through which mechanism, and is it correct?
Stage 2 (a serving change) happens only if stage 1 shows a large, correct win.

## Context

- **What vLLM does at TP8 here.** v0.33.0 builds custom all-reduce for the TP group
  (`cuda_communicator.py:157-166`), then turns it off: above two GPUs it needs NVML to report one-hop NVLink between
  every pair (`custom_all_reduce.py:253-273`, `platforms/cuda.py:816-838`), and PCIe-only boxes fail that. Torch
  symmetric memory is off too: `SYMM_MEM_ALL_REDUCE_MAX_SIZES` has no `12.0` entry (`all_reduce_utils.py:58-83`), and
  it needs multicast (`symm_mem.py:106`), which PCIe does not have. FlashInfer's fused all-reduce has no sm_120 entry
  (`allreduce_rms_fusion.py:104`). So every decode all-reduce is pynccl (`cuda_communicator.py:432-438`), an NCCL ring.
- **Public numbers.** xzwgit's trace on vllm#59203 (8x Server Edition, dual-socket AMD EPYC, vLLM nightly 0.30.1rc1,
  NCCL 2.29.7 and 2.32.3, DeepSeek-V4.1-Flash: hidden 5120, bf16, 40 layers, TP8):
  `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` at 43.8 us per call at batch 1 (10240 B, 81 calls per decode step,
  29.4% of GPU time) and 82.9 us at batch 8 (81920 B, 32.9%); no `cross_device_reduce_*` kernel ran. A cross-node
  16-rank `all_reduce_perf` gave 39.5 us at 8 KB, so the cost is fixed latency, not bandwidth.
- **The box exists.** Lei-010 on sglang#33422 reports `nvidia-smi topo -p2p r` OK on all 28 pairs of an 8x Max-Q box
  with 4 NUMA nodes (GPU0-2, GPU3, GPU4-6, GPU7).
- **Prior art.**
  - vllm#47979 (lucifer1004, draft, stale since 08-25): opt-in custom all-reduce on PCIe plus a PCIe-safe barrier for
    torch symmetric memory. It reports P2P AccessSupported=1 but NativeAtomicSupported=0 on every PRO 6000 pair, and
    that custom all-reduce's flags need no peer atomics, so the NVLink gate is a performance heuristic, not a
    correctness one.
  - vllm#53576 (lucifer1004, merged 09-01, in v0.33.0): opt-in FlashInfer PCIe IPC all-reduce,
    `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`, aimed at exactly this NCCL RING_LL cost. vllm#65192 (aagallo, open)
    makes it fall back instead of aborting on boxes where P2P works only within PCIe-switch pairs; it cites xzwgit
    running it on a full-P2P 8x RTX 6000D box (#60762). We found no TP8 per-call latency for it in those threads.
  - vllm#52555 (Suppressor72, open): custom all-reduce size override, TP2 only. vllm#50941 (thomaslwang, open):
    island-aware hierarchical all-reduce. FlashInfer#7301 (CE ring all-reduce on SM120, merged 10-25) and #6401 (PCIe
    IPC fixes, merged 10-19) came after the 0.7.2 release branch, so the image's FlashInfer 0.7.2.post1 has neither.
    None of these three are measured here.

## Hypotheses

| # | Hypothesis | Confirmed if | Killed if |
|---|---|---|---|
| H1 | NCCL's floor on this box is the ~40 us ring LL latency, whatever the env | every nccl-tests arm and vLLM NCCL arm is within ~15% of default at 8-80 KB | an env arm is well under default |
| H2 | NCCL routes some pairs through host memory (SHM) by default, from the path type and the CPU; `NCCL_P2P_LEVEL=SYS` is a zero-code fix | the default arm's NCCL debug file, or vLLM's `nccl-default`, shows `via SHM` channels and `p2p-sys` is faster | default is P2P on every channel and `p2p-sys` matches it |
| H3 | vLLM's custom all-reduce, forced on at world size 8, is correct over PCIe P2P and at least 2x faster than pynccl at 10-80 KB | `custom-ar` has 0 wrong over every check and one-shot is under half of pynccl | any mismatch, a hang, or under 1.3x |
| H4 | The opt-in FlashInfer PCIe IPC backend already captures that win with an env var | `fi-pcie-ipc` has 0 wrong and is within ~10% of the best custom all-reduce | slower than custom all-reduce, wrong, or hangs |
| H5 | LL128 is unusable on PCIe: NCCL refuses it, runs it slower, or corrupts data (NCCL's docs warn of corruption where it is unsupported) | `proto-ll128` shows any of those | it runs, faster, with 0 wrong |
| H6 | Tree is no better than Ring for 8 GPUs in one node | `algo-tree` is at or above default | `algo-tree` is faster at 8-80 KB |
| H7 | vLLM's symmetric-memory all-reduce cannot run on PCIe sm_120 at v0.33.0 | the probe reports it disabled, with no `12.0` table entry and no multicast | the probe reports it enabled |
| H8 | GPU pairs are not equal: NUMA or socket crossings add latency, and the ring waits on its slowest hop | the pairwise matrix has clearly slower pairs that match `topo -m` | the matrix is flat |

Stage 2 goes ahead if a path with no mismatch anywhere in its sweep runs at half of pynccl's time or less at both
10240 B and 81920 B on the same box. At xzwgit's 81 calls per step and 29.4% share, halving the call time is about a
15% shorter decode step at batch 1.

## Experiment matrix

Everything runs inside `setup.sh`, one run per row below, each under its own `timeout`; a run that fails, hangs, or
times out is recorded in `runs.tsv` and setup goes on. That includes the nccl-tests build (15 min limit): it would
fail the same way on every box with this image, so a failed build skips the nccl-tests arms and pairwise but not the
vLLM arms. Setup exits non-zero only when the box does not show 8 GPUs.

nccl-tests v2.21.1 (`afd59ab`), `all_reduce_perf -t 8 -g 1 -d bfloat16 -o sum`, built for sm_120 against the NCCL
torch loads (the `nvidia-nccl-cu13` wheel's `libnccl.so.2`, found through `/proc/self/maps`), so its numbers and
vLLM's come from one library. One process with one thread per GPU, so no thread launches for 8 GPUs; v2.21.1 captures
graphs per thread (`cudaStreamCaptureModeThreadLocal`), so graph and eager rows share this launch model. It is still
one process, not vLLM's eight, so its numbers are a reference beside vLLM's, not a substitute. Per arm:

| Run | Sizes | Iterations | Graph |
|---|---|---|---|
| pow2 | 4 KB to 4 MB, x2 | 500 warmup (clocks up before the first size), 200 x 5 graph launches, 5 checked | yes (`-G 5`) |
| decode | 10240 B to 2.5 MB, x2 (batch 1 to 256 at hidden 5120) | same | yes |
| large | 8 MB to 256 MB, x2 | 5 warmup, 20, 1 checked | yes (`-G 1`) |
| tuning | 4 KB to 4 MB | 5, with `-U 1` (algo, proto, channels per size) and `NCCL_DEBUG=INFO` transports | no |

| Arm | Env | Extra |
|---|---|---|
| default | none | eager pow2 and decode, 500 warmup and 1000 timed (the non-graph baseline); tuning run logs `INIT,GRAPH,TUNING,ENV,P2P,SHM` and dumps `topo.xml`, `graph.xml` |
| p2p-disable | `NCCL_P2P_DISABLE=1` | |
| p2p-sys | `NCCL_P2P_LEVEL=SYS` | |
| proto-ll | `NCCL_PROTO=LL` | |
| proto-ll128 | `NCCL_PROTO=LL128` | |
| algo-tree | `NCCL_ALGO=Tree` | |
| p2p-sys-ll | `NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL` | |

vLLM v0.33.0's own communicators (`vllm_ar_bench.py`, `torchrun --standalone --nproc_per_node=8`, one backend per
run). Sizes: 4 KB to 4 MB in powers of two plus batch 1 to 256 at hidden 5120 (10240, 20480, 40960, 81920, 163840 B,
...). Each size is captured as 8 all-reduces on separate buffers in one CUDA graph. Every rank adds (rank + 1) times a
small integer pattern that changes on each replay, so the exact sum is known and exact in bf16, and a stale read is a
mismatch; some replays start late on some ranks so the barriers see out-of-order arrivals. 200 checked replays, then 5
timed trials of 100 replays, then one more check after the back-to-back timed replays.

| Arm | Backend | What it forces |
|---|---|---|
| probe | constructs each backend as vLLM does at TP8 | nothing; records which one vLLM picks, NVLink and peer-access matrices, multicast, NCCL versions |
| nccl-default | pynccl, torch.distributed | nothing |
| custom-ar | custom all-reduce, `VLLM_CUSTOM_ALLREDUCE_ALGO=1stage` then `2stage` | `is_fully_connected` patched to True (below) |
| fi-pcie-ipc | `FlashInferPcieIpcAllReduce`, set up and tuned as vLLM does before capture | nothing (the class is used directly instead of the env var) |
| nccl-p2p-sys | pynccl, torch.distributed | `NCCL_P2P_LEVEL=SYS` |
| nccl-p2p-sys-ll | pynccl, torch.distributed | `NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL` |

Every `nccl-*` arm and every nccl-tests tuning and pairwise run sets `NCCL_DEBUG=INFO` with `NCCL_DEBUG_FILE`, since
NCCL line-buffers only that file (`src/debug.cc`) and a killed run would lose INFO lines buffered on stdout;
`summarize.py` counts the transport of each channel (P2P or SHM) from those files.

How custom all-reduce is forced: `custom_all_reduce.py:253-273` computes `fully_connected` from
`current_platform.is_fully_connected` (NVML NVLink status per pair) and returns disabled when it is False above two
GPUs. The same flag gates `should_custom_ar` (`:509-511`) and the default kernel choice in
`csrc/custom_all_reduce.cuh:300` (without it, world size 8 launches no kernel at all). The script patches that one
classmethod to return True before constructing `CustomAllreduce`; the P2P test after the gate (`:274-288`, which
trusts `torch.cuda.can_device_access_peer` since `VLLM_SKIP_P2P_CHECK` defaults to 1, `envs.py:1214`) still runs.
One-shot and two-shot come from `VLLM_CUSTOM_ALLREDUCE_ALGO`, which the launcher reads on every call
(`custom_all_reduce.cuh:270-286`), so each capture bakes in its kernel. vLLM's own
`benchmarks/kernels/benchmark_device_communicators.py` does the same two-algorithm sweep in CUDA graphs but does not
check results and keeps the NVLink gate, so the method is reused, not the file.

Pairwise: nccl-tests `sendrecv_perf -t 2 -g 1` on each of the 28 GPU pairs, 8 B and 64 KB, graph mode, with the transport
NCCL logs; capped at 3 minutes.

Environment additions to the harness's `env.txt` and P2P matrix (`env_extra.sh`): `nvidia-smi -q`, PCIe link
generation and width, power and clock limits, `topo -p2p` write, atomics, and NVLink matrices, `lscpu`, `numactl -H`,
the kernel command line and IOMMU groups, and the NCCL file torch maps.

## Box, time, cost

- `vllm/vllm-openai:v0.33.0` pinned by its manifest-list digest (Docker Hub, pushed 2026-11-02). At that tag:
  CUDA 13.0.3 with nvcc (`cuda-nvcc-13-0`, which pulls in `build-essential`), torch 2.14.0, NCCL 2.31.2 (the
  `nvidia-nccl-cu13` override in `docker/Dockerfile`), FlashInfer 0.7.2.post1, Python 3.12, Ubuntu 24.04.
  `build_nccl_tests.sh` installs nvcc, g++, and make only if they are missing, and takes NCCL headers from
  `libnccl-dev` in the CUDA apt repo only if the wheel lacks them.
- 8x RTX PRO 6000 Max-Q on Vast, falling back to the Server and Workstation Editions (same sm_120, same PCIe
  question), `max_usd_hr = 17.0`, reliability 0.95 or better. `require_p2p = true` refuses any box where a pair reads
  other than OK, and the PCIe gate refuses NVLink.

| Step | Estimate |
|---|---|
| Image pull, 9.7 GB compressed (seconds if the host has it) | 5-10 min |
| Harness gates: SSH, watchdog, apt, topology, P2P, idle GPUs, upload | 3-5 min |
| `env_report.sh`, `env_extra.sh` | under 1 min |
| nccl-tests build, two binaries, sm_120 only | 2-3 min |
| nccl-tests default arm | about 2 min |
| vLLM probe, nccl-default, custom-ar, fi-pcie-ipc (JIT build and tuning) | 1 + 1.5 + 1.5 + 3-5 min |
| nccl-tests, six more arms | about 6 min |
| vLLM nccl-p2p-sys, nccl-p2p-sys-ll | about 3 min |
| Pairwise, 28 pairs | about 2 min, 3 min cap |
| Result pulls and `down` | 2-3 min |
| **Total** | **about 35-50 min** |

`planned_hours = 1.25`. `setup.sh` starts no run whose time limit would end within 10 minutes of the deadline, so a
slow pull or a run of hangs costs the last arms, never the box or the results. `harness quote` on 2026-11-05: no 8x
Max-Q free; first in stock 8x RTX PRO 6000 S on Vast machine 142634 (Bulgaria) at $13.35/hr, reliability 0.991,
downloads $13.33/TB; worst case $18.51. A Max-Q at the $17.00 cap reserves $23.38 plus $0.24 of downloads, so a $25
budget lets `watch` rent any choice. Expected spend at 45 minutes: $10-13.

## Run

```bash
harness quote pcie-tp8-allreduce
harness watch pcie-tp8-allreduce --for 24h; harness down pcie-tp8-allreduce
harness budget show pcie-tp8-allreduce
```

The $25 budget is approved (`harness budget show`). `watch` without a budget would only report the first offer; with
it, `watch` rents, runs `setup.sh`, and pulls the results;
`down` pulls again and destroys the box. `;` rather than `&&`, so `down` runs even when `watch` exits non-zero; with
no box live it does nothing.

## Outputs (`results/pcie-tp8-allreduce/`)

| File | Holds |
|---|---|
| `summary.txt` | the first look: the box, what vLLM picks and why, one line per arm and backend (us at 8 KB / 10 KB / 80 KB, mismatches, how the run ended, transports), the fastest vLLM path with no mismatch against pynccl (nccl-tests' best beside it as a reference), the pairwise matrix, every run that did not end OK |
| `allreduce.csv` | every measurement: source, arm, backend, graph, inplace, size_bytes, time_us, algbw_gbps, busbw_gbps, wrong |
| `nccl_tuning.csv`, `pairwise.csv` | NCCL's algorithm, protocol, and channels per arm and size; sendrecv per pair |
| `runs.tsv` | every run: exit code (124 or 137 is a timeout, `skipped` is the deadline guard), seconds, log |
| `env.txt`, `env_extra.txt`, `nvidia-smi-q.txt`, `p2p-<pod>.txt` | the box |
| `nccl_tests_build.log`, `nccl/<arm>/*.log`, `nccl/default/{topo,graph}.xml` | nccl-tests |
| `vllm/<arm>.log`, `vllm/rows.jsonl`, `vllm/<arm>.fault.rank<N>.txt`, `vllm/fi_pcie_ipc_tune.json` | vLLM runs; a fault file ending in `Timeout (` is a hang, under a line naming the step |
| `pairwise/<a>-<b>.log`, `jobs/setup.log` | per-pair logs; setup's own log |
| `<run log stem>.nccl.<host>.<pid>.log` | NCCL INFO for that run: transports, and for the default arm topology, graph, and tuning |

## Results

Not run yet.
