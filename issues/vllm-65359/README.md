# vllm#65359: FlashInfer PCIe IPC all-gather at TP8, copy engine vs recursive doubling

https://github.com/vllm-project/vllm/issues/65359

## Question

Lei-010 wired FlashInfer's PCIe IPC all-gather into vLLM's DeepSeek-V4.1-Flash gathers on 8x RTX PRO 6000 Max-Q at
TP8 and measured it against NCCL with a standalone bench (issuecomment-6997879140). FlashInfer was 1.4-4.8x faster than
NCCL up to 144 KB per rank, about even at 180 KB, and 5.35x / 5.57x slower at 1.4 MB and 2.9 MB.

That host has 4 NUMA nodes (GPUs 0-2, 3, 4-6, 7), so FlashInfer 0.7.3's topology check returns
`ordered_4plus4 = False` and the TP8 all-gather has one variant, recursive doubling. On an ordered 4+4 host the
policy seed switches to the copy-engine variant at 512 KB per rank and up, which covers exactly the two slow shapes.
Lei's question: is the large-payload gap a property of the kernel or of that host? Lei's precondition: check
`ordered_4plus4` on the rented box before spending the window.

## Reporter setup (issuecomment-6997879140, verbatim where quoted)

- 8x RTX PRO 6000 Blackwell Max-Q, TP8, PCIe, 4 NUMA domains in a 3+1+3+1 layout, `ordered_4plus4 = False`
  ("rank pair (0, 3) does not match logical 4+4 islands"), placement fingerprint `80a930336815094f`
- vLLM 0.34.0rc1.dev154 with FlashInfer 0.7.3, production container
- Bench: "torchrun, 8 ranks, 300 iters after 60 warmup, bf16, `PcieIpcAllGatherWorkspace(max_numel=8M)`"
- Shapes: rows = c x 6 x 3 for c in 1, 8, 16; hidden 256 (Engram) and 5120 (DSpark)

| shape | payload | NCCL | FI raw kernel | ratio |
|---|---|---|---|---|
| [18, 256] bf16 | 9 KB | 64.5 us | 13.5 us | 0.21x |
| [144, 256] bf16 | 72 KB | 89.4 us | 27.6 us | 0.31x |
| [288, 256] bf16 | 144 KB | 123.9 us | 86.7 us | 0.70x |
| [18, 5120] bf16 | 180 KB | 172.9 us | 185.4 us | 1.07x |
| [144, 5120] bf16 | 1.4 MB | 362 us | 1937 us | 5.35x |
| [288, 5120] bf16 | 2.9 MB | 734 us | 4086 us | 5.57x |

## Hypotheses

- H1: on an ordered 4+4 host the seed picks the copy engine at 1.4 MB and 2.9 MB, and it is at or below NCCL there.
  Confirmed if the `ordered` arm's seed rows show `COPY_ENGINE` at those shapes with FI/NCCL <= 1.
  Killed if the copy engine is also several times slower than NCCL.
- H2: the gap comes from recursive doubling itself, not from Lei's 4-NUMA host. Read from two places: the `ordered`
  arm's recursive-doubling sweep rows at 1.4 MB and 2.9 MB (same GPUs and placement as the copy engine), and the
  `interleaved` arm (ranks interleaved across the islands, so `ordered_4plus4 = False` and recursive doubling only,
  as on Lei's host). Confirmed if recursive doubling stays several times slower than the copy engine at every
  geometry; the sweeps show whether any geometry closes it.
- H3: tuning at Lei's row counts (`tune_batches` 18/144/288; the default 1..512 sweep misses all three) changes the
  picture. The tuned rows and winners in both arms answer it.

## Kit

- `islands.py`: reads NVML levels and finds the 4+4 split FlashInfer's check admits; prints the `ordered` and
  `interleaved` GPU orders, or `none`.
- `ag_bench.py`: Lei's bench as written, plus `all_gather_into_tensor`, a bit-exact check against NCCL, a sweep of every
  launchable config, FlashInfer's tuner, and the loop again on the tuner's winners.
- `setup.sh`: env record, one-time JIT build of the all-gather module, `islands.py`, then the arms below. Exits non-zero
  on a box without 8 GPUs or without a 4+4 split, so `harness watch` moves on.
- `summarize.py`: tables per arm next to Lei's numbers, in `summary.txt`.

## Matrix

| arm | GPU order | phases | NCCL | answers |
|---|---|---|---|---|
| ordered | island A then B | seed, sweep, tune, tuned | default (SHM on this host) | H1, H3 |
| interleaved | a0,b0,a1,b1,... | seed, sweep, tune, tuned | default | H2, H3 |
| ordered-nccl-p2p | island A then B | seed | `NCCL_P2P_LEVEL=SYS` | the NCCL baseline with P2P |
| ordered-repeat | island A then B | seed | default | run-to-run spread |

## Hardware

8x RTX PRO 6000 Server Edition on Vast, steered to machine 142634 (dual EPYC 9575F, 2 NUMA nodes, NODE inside GPUs
0-3 and 4-7, SYS across, P2P on all 28 pairs; issues/pcie-tp8-allreduce, 2026-11-05). Server Edition runs at 600 W
where Lei's Max-Q runs at 300-325 W, so absolute times differ; the comparisons that matter are inside one box.
vLLM v0.34.0 image (FlashInfer 0.7.3, torch 2.14.0). `planned_hours = 1.25`.

## Results

### 2026-11-25, 8x RTX PRO 6000 Server Edition, Vast machine 153419 (Romania)

Machine 142634 was outbid by a newer, cheaper listing that was not on `avoid_machines`; setup's check covered it.
Box: dual Xeon Platinum 8568Y+, 4 NUMA nodes, GPUs 0-3 on node 0 and 4-7 on node 2, PCIe switch pairs (0-1, 2-3,
4-5, 6-7 at PIX), NODE inside each half, SYS across. `islands.py`: `ordered 0,1,2,3,4,5,6,7`,
`interleaved 0,4,1,5,2,6,3,7`; FlashInfer agreed (`ordered_4plus4` True, fingerprint `a62e4a6e4d7b7bd6`; interleaved
False, "rank pair (0, 1) does not match logical 4+4 islands"). vLLM v0.34.0 image, FlashInfer 0.7.3, torch
2.14.0+cu130, NCCL 2.30.7. The module loaded from the AOT cache (4 s). All four arms finished (rc 0, 19-132 s each);
no phase limit fired. 468 sweep configs plus every seed and tuned row matched NCCL bit for bit.

The 5120-wide rows, rank 0 us per call (Lei's timing):

| per-rank payload | copy engine (ordered seed) | RD, best of 48 geometries (ordered) | RD at the seed geometry 64x512 (ordered) | RD seed, interleaved | RD tuned, interleaved (1x512) |
|---|---|---|---|---|---|
| 1.4 MB (144 x 5120) | 652 | 1643 (1x512) | 3366 | 4563 | 1175 |
| 2.9 MB (288 x 5120) | 1290 | 3052 (2x512) | 6208 | 9991 | 2365 |

- H1 holds for the variant: with the ordered placement the seed picks `COPY_ENGINE/1/32` at both sizes, 2.4-2.5x
  faster than the best recursive-doubling geometry on the same GPUs and 4.8-5.2x faster than recursive doubling at the
  seed geometry. `ordered-repeat` reproduced it within 1% (652.9 and 1301.6 us). The FI/NCCL part of H1 cannot be read
  on this box (NCCL below).
- H2, reframed by the data: on this box the seed's recursive-doubling geometry (64 blocks x 512 threads at these sizes)
  costs more than the variant does. With `ordered_4plus4` False the tuner's 1x512 is 3.9x / 4.2x faster than the seed.
  Recursive doubling ran faster interleaved than ordered here (1171 vs 1643 us best at 1.4 MB), so the interleaved arm
  is not a worst case; which step crosses sockets in each order was not checked.
- H3 holds: tuning at 18/144/288 took 25.5 s (ordered) and 26.1 s (interleaved) and beat the seed at every shape but
  the two copy-engine rows, by up to 4.2x. Ordered, the tuner also picked the copy engine at 180 KB (110 vs 344 us),
  below the seed's 512 KB threshold.

| shape | ordered seed | ordered tuned | interleaved seed | interleaved tuned |
|---|---|---|---|---|
| 18 x 256 | 23.1 | 15.7 (RD 2x512) | 21.8 | 13.6 (RD 3x256) |
| 144 x 256 | 135.0 | 46.1 (RD 6x64) | 121.9 | 56.7 (RD 2x256) |
| 288 x 256 | 258.0 | 83.3 (RD 1x512) | 264.5 | 112.4 (RD 1x512) |
| 18 x 5120 | 344.1 | 110.0 (CE) | 429.8 | 150.0 (RD 1x512) |
| 144 x 5120 | 652.1 (CE) | 651.6 (CE) | 4562.7 | 1175.3 (RD 1x512) |
| 288 x 5120 | 1289.0 (CE) | 1290.5 (CE) | 9990.6 | 2365.4 (RD 1x512) |

NCCL on this box is not a usable baseline for Lei's ratios: at 72 KB to 2.9 MB it took 0.45-3.3 ms (Lei: 0.09-0.73
ms), not monotonic in size, and the NCCL INFO logs show it split channels between `P2P/CUMEM` and `SHM/direct`. With
`NCCL_P2P_LEVEL=SYS` (every channel P2P) it got 7-8x slower at 1.4 and 2.9 MB (10.7 and 22.6 ms), so cross-socket P2P
is slow on this platform. Only the 9 KB row is near Lei's (33-117 us vs 64.5). The comparisons above are all
FlashInfer against FlashInfer on the same GPUs.
