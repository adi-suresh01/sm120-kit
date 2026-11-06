# vllm#59203 stage 2: what the two all-reduce levers do to TP8 decode of DeepSeek-V4.1-Flash on 8 PCIe-only GPUs

https://github.com/vllm-project/vllm/issues/59203 (xzwgit, 8x RTX PRO 6000 Server Edition, TP8; closed 11-05 when
vllm#60762 merged). Stage 1 (`issues/pcie-tp8-allreduce/`, commit 3946236) measured the all-reduce alone on a Vast 8x
RTX PRO 6000 Server Edition box. This stage serves the real model at TP8 and changes nothing between arms but the
all-reduce env, to answer what the threads now ask: what do `NCCL_P2P_LEVEL=SYS` and
`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` do to end-to-end decode?

## What is known

- **The trace.** xzwgit's torch profile on #59203 (09-30; vLLM 0.30.1rc1.dev382, TP8, 625-token prompt, 160 out):
  a batch-1 decode step is 12.1 ms of GPU time, `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` is 29.4% of it, 81 calls
  per step at 43.8 us each (10 KB payloads); batch 8 is 82.9 us per call, 32.9%. Unprofiled, the same server decodes
  at about 11 ms TPOT, 90 tok/s at c=1 (1K in / 1K out).
- **Stage 1** (vLLM v0.33.0, CUDA graphs, payload batch x 5120 x bf16, us per all-reduce, every cell exact):

  | Batch | default (pynccl, NCCL picks SHM) | `NCCL_P2P_LEVEL=SYS` (P2P/IPC) | FlashInfer PCIe IPC (vllm#53576) |
  |---|---|---|---|
  | 1 | 33.2 | 18.5 | 7.5 |
  | 8 | 38.5 | 21.8 | 25.1 |
  | 32 | 85.7 | 39.8 | 39.1 |
  | 128 | 149.1 | 94.8 | 99.8 |

  NCCL's default `p2pLevel = PATH_PXB` (`src/graph/paths.cc:356`) leaves out pairs that meet at a CPU root port, so on
  a box without PCIe switches "P2P on" runs over SHM, and vLLM turns its own custom all-reduce off above 2 PCIe GPUs.
- **TP4 on SGLang** (`issues/sglang-33422`, 11-05): `NCCL_P2P_LEVEL=SYS` cut TTFT 25-30% but median ITL only 2-3%.
- **Production with FlashInfer IPC.** Lei-010 (8x Max-Q, TP8, vLLM dev253, DSpark k=3, V2 runner) measured c=8
  651.0 -> 689.5 tok/s and batch_iter 30.40 -> 25.07 ms with `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` (vllm#60762
  thread, 10-27). Speculation was on, so the plain decode step was never isolated.
- **Correctness.** On an 8x RTX 6000D, FlashInfer IPC plus DSpark k=7 gave identical garbage at temperature 0 with a
  healthy engine; IPC without speculation was clean (xzwgit, #60762 thread). Speculation is off here, and every arm
  runs GSM8K before its rows.
- **Hosts without full P2P.** FlashInfer's IPC workspace aborts the boot with `peer access is not supported` where a
  pair lacks P2P (aagallo, #59203, 11-06); vllm#65192 (open) makes it fall back. `require_p2p` keeps such boxes out,
  `fail_pattern` ends a boot that hits it anyway, and setup's probe sets the workspace up before any download.

## What v0.33.0 needs for DeepSeek-V4.1-Flash on sm_120 (source at tag v0.33.0, commit 4ba787ec, 10-31)

- **vllm#60762 (64-token sparse-MLA blocks): needed, as `patches/pr60762-dsv41-block64.diff`.** It merged on 11-05
  (7ecc83a9), after the tag. At v0.33.0 `DeepseekV41IndexerBackend` still declares `[128]` on sm_120
  (`v1/attention/backends/mla/indexer.py:366-367`) and `DeepseekV4FlashInferMLASparseBackend` `[128]`
  (`models/deepseek_v41/nvidia/flashinfer_sparse.py:117-118`), the pair that fails on the 0.32.0 release. The patch is
  the merge commit's two files, regenerated against the tag (`git apply --check` clean). With it, the flags are the
  triple our TP4 run booted with:
  `--block-size 64 --kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype":"fp8"}'`.
- **`dsv41-stream-load` (issues/vllm-56771): not needed, and it no longer applies (3 of 8 hunks fail).** Its streaming
  half went upstream as vllm#58857 (merged 10-17): `nvidia/vl_model.py:320` streams the language model in checkpoint
  order. Its eviction half (vllm#62110) is still open; it protected against RunPod stopping a container at 98% of its
  limit with page cache counted. Vast has no such stop, and the memory guard drops the checkpoint's page cache at 80%
  of any limit. Its fadvise skips what a process has mapped, so the pinned tables, the Engram shard the ranks map while
  they load it (101.5 GB), and the workers, about 350 GiB, have to fit under that line: setup refuses a box under
  448 GiB.
- **`PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:64`: kept.** Engram `cpu_offload` defaults to true
  (`config/engram.py:37`), and the tables are allocated with `pin_memory=True`, which torch rounds up to a power of two
  (`models/deepseek_v41/common/engram.py:1122`): about 256 GiB instead of 189 at TP8.
- **The all-reduce path is the plain one on sm_120.** The fused mHC all-reduce needs SM100 and TP4
  (`nvidia/ops/mhc.py:44-72`), and sequence parallel needs expert parallel (`nvidia/model.py:261-271`), so every TP
  all-reduce goes through `CudaCommunicator.all_reduce`: FlashInfer PCIe IPC first when enabled and the input fits,
  then pynccl. FlashInfer's workspace is sized to the CUDA graph capture sizes
  (`flashinfer_pcie_ipc_all_reduce.py:126` and `:227`), so prefill all-reduces go to NCCL in every arm.
- **Flags.** xzwgit's 09-30 TP8 command (the one the trace ran): `--tensor-parallel-size 8 --tokenizer-mode
  deepseek_v41 --engram-config.cpu_offload true --max-model-len 131072 --max-num-seqs 128 --gpu-memory-utilization
  0.92 --language-model-only --trust-remote-code`, the deepseek_v41 tool and reasoning parsers, and the block-size
  triple above in place of its attention backend pin (the backend is auto-selected since). No speculative config, no
  `--kernel-config`, no `VLLM_DEEP_GEMM_WARMUP=skip`: as in the trace's server, JIT work happens at boot, where it
  cannot land inside a measured row. Checkpoint revision `314113f8`, the one issues/vllm-56771 served.

## Hypotheses

| # | Hypothesis | Confirmed if | Killed if |
|---|---|---|---|
| H0 | v0.33.0 plus the #60762 backport boots and serves DSV4.1 at TP8 on sm_120 with these flags | b0 verdict OK, GSM8K >= 0.90 | b0 BOOT_FAIL; the run is then about the boot |
| H1 | Batch-1 decode is all-reduce latency bound: both levers together cut c=1 median ITL by 10% or more (stage 1 saves ~26 us x 81 calls, ~2.1 ms of an ~11 ms step) | b2 in1k-c1 median ITL >= 10% under b0 | under 5% |
| H2 | `NCCL_P2P_LEVEL=SYS` alone helps decode more at TP8 than the 2-3% it gave at TP4 | b1 in1k-c1 median ITL 5% or more under b0, with b0 on SHM and b1 on P2P | under 2%, or b0 already on P2P (then b1 is null by construction) |
| H3 | FlashInfer's edge over NCCL P2P is a small-batch edge, as in stage 1 | b2 beats b1 on median ITL at c=1 by more than at c=32 | b2's lead over b1 is as large at c=32 as at c=1 |
| H4 | Prefill gains come from P2P for NCCL, not from FlashInfer, since no prefill all-reduce reaches FlashInfer | b1 and b2 in32k-c1 median TTFT >= 15% under b0; b3 (if run) within 5% of b0 | b1 within 5% of b0 on TTFT |
| H5 | Neither lever changes outputs without speculation | GSM8K of b1 and b2 within 2/50 of b0, every request completes | an arm 3/50 or more under b0, or POST_FAIL |
| H6 | The FlashInfer arms really run FlashInfer | server.log has the TP group's dispatch list starting `FLASHINFER_PCIE_IPC` and `Initialized FlashInfer PCIe IPC all-reduce for TP8` | a fallback warning, or no Initialized line (the hook then fails the arm before its rows) |

## Arms

One vLLM boot each, same flags, same patch, GSM8K 50 before the rows. The env every arm shares:
`PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:64 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH`, with
`NCCL_DEBUG_FILE` per rank from `hooks.sh` (NCCL's stdout INFO is not flushed; its file is line-buffered). With COLL
left out NCCL logs while it builds communicators, not per collective.

| Arm | Adds | Order | In by default |
|---|---|---|---|
| b0-default | nothing: NCCL defaults | 1, in an exec of its own | yes |
| b2-sys-fi | `NCCL_P2P_LEVEL=SYS VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` | 2 | yes, if b0 booted |
| b1-p2p-sys | `NCCL_P2P_LEVEL=SYS` | 3 | yes, if b0 booted and setup ends with 90+ min left |
| b3-fi-only | `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` | 4 | only if setup ends with 117+ min left |
| b0-repeat | nothing, again last: drift check | 5 | only if setup ends with 144+ min left |

b0 runs alone: every flag here is at vLLM's defaults (FlashInfer autotune, JIT warmups, 8192-token chunks), unlike the
one DSV4.1 boot on sm_120 we have (issues/vllm-56771), so a boot that fails should cost one boot, not three. `exec`
reports BOOT_FAIL as a verdict and goes on to the next variant, hence the separate call. b2 runs before b1: if the
deadline cuts the last arm, b0 against b2 still gives the combined effect, and its 32K row still carries P2P's prefill
effect, while b0 against b1 alone would say nothing about FlashInfer. The optional arms are out of the default plan
because at the ~110 MB/s a 1 Gbps host gives, the download leaves no room for them inside 3.5 h; setup's last lines
add them when a faster download does. b3 comes first because it splits b2 into its two levers; b0-repeat only bounds
drift.

## Per arm

1. `setup.sh` (once per box, before the download): 8 GPUs, vLLM 0.33.0, RAM >= 448 GiB by the container's limit,
   `/dev/shm` >= 2 GiB, no inherited `NCCL_*` or `VLLM_ALLREDUCE_*`, 560 GB free and a 1000 MB/s cold read (4 GiB
   written, dropped, read back), then `ar_probe.py` under each arm's env (below), then the client tokenizer check.
   A failed check exits non-zero, and `harness watch` destroys the box and rents another.
2. Boot, smoke, GSM8K 50 (`harness/pod/gsm8k_chat.py`, 5-shot, chat endpoint, greedy, up to 64 in flight).
3. `post_smoke` in `hooks.sh`: a FlashInfer arm without the Initialized line fails here; then `decode_bench.py`.
4. `post_stop`: the per-rank NCCL files into `nccl.log` and `nccl_transport.txt`.

**Probe** (`ar_probe.py`, `torchrun` over the 8 GPUs, 600 s hard limit plus a faulthandler watchdog per step): vLLM's
own `PyNcclCommunicator`, exact-sum checks at batch 1, 8, 32, 128 and one 8192-token prefill chunk (x 5120 bf16),
eager timings at batch 1 and 8192; under the FlashInfer variable also `FlashInferPcieIpcAllReduce` set up and tuned
as vLLM does before capture, checked at every power of two from 1 to 256 (the server prepares up to 256 with
`--max-num-seqs 128`). Every rank exits through `os._exit` after a gloo
barrier, since `destroy_process_group` hung after NCCL use in stage 1. b0, b2, and b1 must pass or the box is
refused; a failed b3 probe only drops b3. NCCL INFO goes to a file per rank, so each arm's transport is known before
any boot.

**Rows** (`decode_bench.py`, `vllm bench serve --backend openai`, random dataset, `--random-range-ratio 0` fixes the
lengths in vLLM, `--ignore-eos`, `--temperature 0`, percentiles 50/90/99 of TTFT, TPOT, ITL, E2EL):

| Run, in order | In / out | c | Prompts | Seed |
|---|---|---|---|---|
| warmup-in1k-c8 (not reported) | 1024 / 512 | 8 | 16 | 101 |
| in1k-c1 | 1024 / 512 | 1 | 8 | 11 |
| in1k-c8 | 1024 / 512 | 8 | 32 | 12 |
| in1k-c32 | 1024 / 512 | 32 | 64 | 13 |
| in1k-c64 | 1024 / 512 | 64 | 128 | 14 |
| warmup-in32k-c1 (not reported) | 32768 / 256 | 1 | 2 | 102 |
| in32k-c1 | 32768 / 256 | 1 | 6 | 15 |

The same seed per row on every arm, so the rows pair; a seed per row, so no row repeats another's prompts in the
prefix cache. The client's own warmups send the row's first prompt (`benchmarks/serve.py:926-945`) and would serve that
request's prefill from cache, so they stay off and a warmup run of each shape goes before its rows. The 32K shape goes
last: no DSV4.1 server of ours has prefilled in 8192-token chunks (vllm-56771 ran 4096), and a fault there must not
cost the 1K rows. Median ITL is the decode number; mean TPOT at c >= 8 also carries other requests' prefill. Output
tok/s is the client's.

## Box, hours, cost

Vast, 8x RTX PRO 6000: Server Edition first (stage 1's box and the trace's), then Workstation (same 600 W GB202),
then Max-Q (300 W, the sglang#33422 production card; its rows decode slower and are labeled by card).
`require_p2p = true` (all 28 pairs OK), the PCIe gate refuses NVLink, `max_usd_hr = 15`, reliability 0.95,
`min_inet_down_mbps = 800`, downloads at most $10/TB. Image `vllm/vllm-openai:v0.33.0` by digest, the same as stage 1.

The download is the big unknown. Every 8x offer on 11-06 listed 0.86-1.6 Gbps (S x8 Texas at 1.0 and 0.86 Gbps,
$14.04/hr, $2.73/TB; WS x8 Quebec at 1.6 Gbps, $16.00/hr, reliability 0.906), and sglang-33422's Vast box delivered
about 120 MB/s from HF on a host listed at 2.9 Gbps. This plan assumes ~110 MB/s:

| Step | Minutes | Running |
|---|---|---|
| up: create, 9.7 GB image, SSH, apt, watchdog, topology, P2P and idle gates, kit copy | 12 | 12 |
| setup checks, env_extra, 4 probes (stage 1's FlashInfer setup took 14 s), client check | 8 | 20 |
| 510.3 GB at ~110 MB/s, cache drop | 78 | 98 |
| b0: cold boot (475 GiB, 189 GiB pinned, TileLang, DeepGEMM, torch.compile, capture) 20, GSM8K 3, rows 7 | 30 | 128 |
| b2: boot 15 (the measured DSV4.1 loads ran 14-15 min: 815-829 s at TP4, Lei-010's 872 s at TP8), GSM8K 3, rows 7, the second exec | 27 | 155 |
| b1 | 27 | 182 |
| down: pull, destroy | 6 | 188 |

`planned_hours = 3.5` (210 min), 22 min over the expected path for boots slower than estimated. The download stops at
its 3-minute rate check when it would end less than 75 min before the deadline, or at that point, so a slow host costs
about 25 min, never b0 and b2. On the expected path the run is about 3.15 h: $44.20 at $14.04/hr plus $1.43 of
downloads, about $46. On a host that delivers 350 MB/s the download takes 25 min and the run about 2.25 h, about $33.
The `harness quote` worst case reserves 3.5 h x rate x 1.10 plus the downloads.

## Run

```bash
harness quote vllm-59203
harness budget set vllm-59203 <usd> --note <approval>        # nothing rents without it
harness watch vllm-59203 --for 24h && harness exec vllm-59203 b0-default \
  && grep -qE '"verdict": "(OK|POST_FAIL)"' results/vllm-59203/b0-default/verdict.json \
  && harness exec vllm-59203 b2-sys-fi b1-p2p-sys; harness down vllm-59203
python3 -I issues/vllm-59203/summarize.py results/vllm-59203
harness budget show vllm-59203
```

The chain above runs the default three, unattended, and goes on past b0 only when b0's server came up (a POST_FAIL
booted; its rows up to the failure are kept). When someone is watching, read the `next:` line at the end of
`results/vllm-59203/jobs/setup.log` instead, which adds b3 and b0-repeat to the second exec only when they fit. Never
leave the names out: `exec` with none runs every variant in file order until the deadline.

## Stop rules

- Any setup check or a b0/b2/b1 probe fails or hangs: setup exits 1 before the download; `watch` destroys the box
  and avoids the machine (3 such boxes stop the watch).
- The download would end, or is still running, less than 75 min before the deadline: setup exits 1.
- b0 BOOT_FAIL, SMOKE_FAIL, or TIMEOUT: no further arm; `harness down`, and report the fault (H0). With `No common
  block size` or an `attention.hpp` assert the patch did not take; with a hang in FlashInfer autotune or a JIT warmup,
  the next rental uses issues/vllm-56771's `--kernel-config` (autotune and warmups off) in every arm.
- A FlashInfer arm without the Initialized line: POST_FAIL before its rows (H6), not a slow arm.

## Outputs (`results/vllm-59203/`)

| File | Holds |
|---|---|
| `summary.md` | `summarize.py`'s GPU line (the card, from `env.txt`) and tables: arm x row (median ITL, mean TPOT, output tok/s, median TTFT, each vs b0; GSM8K and the server's NCCL transport per arm; an arm's verdict beside its name unless OK, and a row's completions beside it when some failed), then per arm the verdict, boot time, vLLM's all-reduce dispatch list, FlashInfer initialized or not, and the probe's eager times |
| `<arm>/decode_rows.csv`, `<arm>/bench/<row>.json` | one line per row; the client's full result with per-request TTFTs and ITLs |
| `<arm>/server.log`, `verdict.json`, `gsm8k.txt`, `gsm8k_responses.jsonl` | the harness's per-variant record |
| `<arm>/nccl.log`, `nccl_transport.txt`, `nccl.<host>.<pid>.log` | NCCL's INFO lines per rank, and condensed |
| `probe/<arm>.json`, `.log`, `.transport.txt`, `.fault.rank<N>.txt`, `.fi_tune.json` | setup's probe per arm |
| `env.txt`, `env_extra.txt`, `nvidia-smi-q.txt`, `p2p-<pod>.txt`, `jobs/setup.log` | the box, and setup's log with the `next:` line |

Not measured: how many of a step's all-reduces FlashInfer took (inputs other than batch x 5120 bf16 fall through to
NCCL); a torch profile window per arm, as xzwgit took, would show it.

## Results

### 2026-11-06, 8x RTX PRO 6000 Server Edition, Vast machine 57498 (California)

Box: dual-socket AMD EPYC 9554, 2 NUMA nodes, GPUs 0-3 on socket 0 and 4-7 on socket 1 (`NODE` within a socket,
`SYS` across), driver 580.105.08, `topo -p2p r` OK on all 28 pairs, 1.45 TB RAM. vLLM v0.33.0 plus the #60762
backport, TP8, CUDA graphs, DSpark off. Weights came down at 246 MB/s; setup took 40 min. Boots: b0 567 s (cold),
b2 231 s, b1 151 s. Every request in every row completed (0 failed), GSM8K 48/50 in all three arms. One rep per arm,
paired on this box only; c=1 rows are 8 requests, the 32K row 6. 1.24 h on this box, $17.51 (estimate); the issue
spent $23.67 including three boxes that failed before setup (one slow HF download, two SSH key-mode refusals that
harness ad3cf4c now repairs).

| Arm | Row | Median ITL (ms) | vs b0 | Output tok/s | vs b0 | Median TTFT (ms) | vs b0 |
|---|---|---|---|---|---|---|---|
| b0 defaults (SHM) | 1K/512 c=1 | 10.56 | | 92.7 | | 129 | |
| b0 | c=8 | 16.32 | | 465.8 | | 520 | |
| b0 | c=32 | 29.04 | | 978.2 | | 1,292 | |
| b0 | c=64 | 44.50 | | 1,249.9 | | 1,334 | |
| b0 | 32K/256 c=1 | 10.63 | | 57.6 | | 1,775 | |
| b2 SYS + FlashInfer PCIe IPC | 1K/512 c=1 | 8.33 | -21.1% | 116.9 | +26.1% | 126 | -3.0% |
| b2 | c=8 | 15.42 | -5.5% | 488.9 | +5.0% | 591 | +13.7% |
| b2 | c=32 | 24.51 | -15.6% | 1,118.1 | +14.3% | 1,546 | +19.7% |
| b2 | c=64 | 37.70 | -15.3% | 1,393.2 | +11.5% | 1,618 | +21.3% |
| b2 | 32K/256 c=1 | 8.39 | -21.1% | 60.8 | +5.6% | 2,101 | +18.4% |
| b1 SYS (P2P) | 1K/512 c=1 | 9.48 | -10.2% | 103.1 | +11.2% | 125 | -3.4% |
| b1 | c=8 | 14.77 | -9.5% | 502.9 | +8.0% | 594 | +14.2% |
| b1 | c=32 | 25.52 | -12.1% | 1,067.7 | +9.1% | 1,381 | +7.0% |
| b1 | c=64 | 40.51 | -9.0% | 1,312.0 | +5.0% | 2,046 | +53.4% |
| b1 | 32K/256 c=1 | 9.54 | -10.3% | 57.0 | -1.0% | 2,093 | +17.9% |

Transports (server NCCL INFO): b0 `SHM/direct` on every connection, b1 and b2 `P2P/CUMEM`. b2's TP group dispatch
list starts `FLASHINFER_PCIE_IPC` and logged `Initialized FlashInfer PCIe IPC all-reduce for TP8`. Setup's eager
probe: pynccl at batch 1 34.9 us (b0) vs 18.9 us (b1), FlashInfer 13.4 us (b2); at batch 8192 (80 MiB) 4,545 us over
SHM vs 6,288 us over P2P.

- H0 holds: the stack boots and serves at TP8 on sm_120 with the backport, GSM8K 0.96.
- H1 holds: both levers cut c=1 median ITL 21% (10.56 to 8.33 ms), +26% single-stream tok/s, and still 15% at c=32-64.
- H2 holds: `NCCL_P2P_LEVEL=SYS` alone cuts decode ITL 9-12% at TP8, against 2-3% at TP4 (issues/sglang-33422).
- H3 holds: FlashInfer's lead over plain P2P is 12% at c=1 and about 4% at c=32; at c=8 plain P2P was ahead.
- H4 is killed on this host: prefill got slower with P2P, not faster. The 32K TTFT rose 18% in both P2P arms, and
  the probe shows why: an 80 MiB all-reduce takes 6.3 ms over P2P against 4.5 ms over SHM on this dual-socket box,
  while issues/sglang-33422's single-socket 4x box had P2P ahead at 32 MiB (1.3 vs 2.9 ms). Cross-socket P2P wins for
  decode-size messages and loses for prefill-size ones here. TTFT at c=8-64 also includes queueing behind prefill,
  so those deltas mix both effects.
- H5 holds: 48/50 in every arm, no failed request.
- H6 holds: b2 ran FlashInfer.

Not run: b3 (FlashInfer alone, NCCL left on SHM). It is the arm that could keep b2's decode gain without the prefill
loss, since FlashInfer only takes decode-size inputs and NCCL would stay on SHM for prefill. Setup said it fit; the
unattended chain ran only the default three.
