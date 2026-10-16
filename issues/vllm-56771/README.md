# vllm#56771: DeepSeek-V4.1-Flash + DSpark, illegal memory access in sm_120 sparse-MLA prefill

https://github.com/vllm-project/vllm/issues/56771, reporter moshemalawach (LibertAI, which sells DeepSeek V4.1
Flash on its public API). Same reporter as vllm#55757. Covers vllm#58910 (TP4, no spec, 35K prompt) in the same
pod.

## Reporter setup (09-16 and 09-17, copied)

- 4x RTX PRO 6000 Blackwell Server Edition, TP4, PCIe only, sm_120, driver 595.x, CUDA 13.0.
- `deepseek-ai/DeepSeek-V4.1-Flash` (476 GiB, original checkpoint, vision enabled).
- vLLM nightly `69db1c26` + PR #56214 overlay, then nightly `fdcd42350` (09-17), flashinfer-python 0.6.18 /
  0.6.18.post1, plus a private SM120 patch set (topk 1152/640 decode instantiations, prefill dispatch arms, SWA
  block 32 -> 64, divergent page sizes).
- Command:

```
vllm serve /workspace/models/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash \
  --tensor-parallel-size 4 --engram-config.cpu_offload true \
  --max-model-len 262144 --gpu-memory-utilization 0.95 \
  --max-num-seqs 64 --max-num-batched-tokens 4096 \
  --kernel-config '{"enable_flashinfer_autotune": false, "enable_cutedsl_warmup": false, "enable_jit_warmup": false}' \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block"}' \
  --host 127.0.0.1 --port 18000
env: VLLM_DEEP_GEMM_WARMUP=skip NCCL_P2P_LEVEL=PXB NCCL_MIN_NCHANNELS=32
```

- Trigger: one prompt of about 4K tokens or more (12K reproduces 100%); the engine dies with an illegal memory
  access in the first 4092-token prefill chunk, around layers 27-31. `CUDA_LAUNCH_BLOCKING=1` puts it in
  FlashInfer `sparse_mla_sm120_prefill.cu` `launch_prefill_mg_dual`, `<DSV4, FP8, nh=16, TOPK=1152, pbs=64,
  pbs_x=64>`.
- Their 09-17 matrix: graphs (FULL_AND_PIECEWISE) + DSpark crash; `cudagraph_mode: NONE` + DSpark OK at about
  4 tok/s; `--enforce-eager` + DSpark OK at about 5 tok/s; graphs without DSpark OK at 60 tok/s in production.
  Not fixed by exact capture sizes, not Engram. "We're stuck until this is fixed, happy to test any patch."

## What changed upstream since (checked 2026-10-16)

- **FlashInfer 0.7.2** (vLLM main and nightly `49cb2bbf` pin it; v0.31.0 pins 0.7.0.post1) carries FI #5197, the
  SM120 sparse-MLA rewrite with DSv4.1 support and the fix for the `w_head_sc_all` race in
  `sparse_mla_prefill_mg_kernel` (FI#6101). Nobody has rerun this repro on it.
- **#60945 is not this fix.** It guards `_compute_local_residual_mass_kernel` in the rejection sampler, which only
  runs with `enable_adaptive_verification` (default off, not set by the reporter). Closed duplicate PR #59355
  calls #56771 a "different root cause". It is in the nightly anyway; variant a5 covers its path.
- **Stock vLLM still cannot boot DSV4.1 on sm_120** (#59203): block size 32 is unsupported and `--block-size 64`
  finds no common size because the DSV41 sparse backend and indexer only declare 128. Open PR #60762 declares 64
  on sm_120; with `--block-size 64 --kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype":"fp8"}'` it is the
  working recipe on 4x and 8x PRO 6000. `patches/pr60762-dsv41-block64.diff` is its `vllm/` half at head
  `dd0a1ee5`; it applies cleanly to `49cb2bbf`. We do not use the reporter's private patch set.

## Hypotheses

| # | Hypothesis | Confirmed by | Killed by |
|---|---|---|---|
| H1 | Fixed on FlashInfer 0.7.2 + current nightly: FI #5197 rewrote the kernel that faults | a1 serves 12K and 120K with DSpark and graphs, coherent output | a1 crashes in prefill |
| H2 | Still a graph-replay corruption of sparse metadata under DSpark | a1 crashes, a2 (graphs off) serves | a2 also crashes |
| H3 | The SWA prefill width with image tokens is what overruns (he-yufeng, 09-14) | a1 crashes, a3 (`--language-model-only`) serves | a3 crashes the same way |
| H4 | Stock nightly still needs #60762 to boot on sm_120 | g0 fails at block-size negotiation | g0 boots |
| H5 | vllm#58910 (no spec, 35K, TP4) is fixed by FI 0.7.2 | a0 serves 12K/35K/120K | a0 crashes |

## Matrix

Image `vllm/vllm-openai:nightly-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294` by digest (FlashInfer 0.7.2). Every
variant: the reporter's flags and env, `--revision 314113f8`. Prompts are real prose (War and Peace, Project
Gutenberg) cut to 12K / 35K / 120K tokens with the model's tokenizer (`build_prompts.py`), sent one at a time
after the smoke test by `post_smoke` (`long_prompts.py`): first a short arithmetic probe, then the long prompts,
then a 512-token generation at concurrency 1 for decode speed.

| Variant | Change | Prompts | Run when |
|---|---|---|---|
| g0-boot-stock | no patch, no geometry flags | 12K | always (fast fail expected) |
| a1-dspark-graphs | #60762 + geometry + DSpark k=5, graphs default | 12K, 120K | always |
| a0-nospec-graphs | #60762 + geometry, no spec | 12K, 35K, 120K | always |
| a2-dspark-nograph | a1 + `cudagraph_mode: NONE` | 12K | a1 crashes |
| a3-dspark-graphs-lmo | a1 + `--language-model-only` | 12K | a1 crashes |
| a4-dspark-graphs-dump | a1 + `CUDA_LAUNCH_BLOCKING=1`, GPU core dump | 12K | a1 crashes |
| a5-dspark-adaptive | a1 + `enable_adaptive_verification` (#60945's path) | 12K, smoke at 16 | a1 serves |

Sequence: `harness up`, `harness exec vllm-56771 g0-boot-stock a1-dspark-graphs a0-nospec-graphs`, then a2 a3 a4
or a5 by the a1 result, then `harness down`. Stop rule: nothing boots within 60 minutes of setup done -> down.

## Hardware, hours, cost

4x RTX PRO 6000 Server Edition, SECURE, about $9.96/hr (2.49/GPU-hr). Stock was LOW on 2026-10-16 (EUR-IS-1,
US-NE-1, host CUDA 13.2). Setup: image about 3 min, 510.3 GB of weights about 10 min at our measured 0.83 GB/s,
prompts under a minute. Variants: g0 about 5 min, a1 cold boot plus prompts about 25 min, a0 about 15 min, two
follow-ups about 30 min. Expected 2-2.6 h, about $20-26; `planned_hours = 3.5` caps it at $38.35 worst case.
The container's memory limit (the cgroup's, not the host RAM `free` shows) must be at least 256 GiB for the pinned
Engram tables; `setup.sh` stops before the download otherwise, and also after it when a cold read of the largest
shard runs under 500 MB/s (then `harness down`). The weights live on container disk (`container_disk_gb = 600`), not
the FUSE volume.

## Results

### Attempts 1-5 (2026-10-16, no run, $25.43)

- Pods 1 and 2 ended as "proxy-only" while the uncached 9.6 GB nightly image pulled (a harness bug, fixed in
  2f34e1b), $0.52 and $0.53.
- Pod 3 (sm120-vllm-56771-4f142b, EUR-IS-2, host 81.27.69.177) waited 25 min for the image, then sat in
  `setup.sh` for 1h52m: the weights came down at 6-9 MB/s, 60 of 510 GB. Terminated by hand at 15:55Z, $22.78.
- Pod 4 landed on the same host. `setup.sh` now times the first 3 min of the download and stops under 400 MB/s;
  it stopped there, $0.85.
- Pod 5 (no setup, $0.51) measured the host: container disk writes 1.2 GB/s, /workspace (a MooseFS fuse mount)
  130 MB/s, and one HF shard downloads at 15 MB/s. The host's network is the limit, not the disk: 510 GB would
  take over 9 h. The host has 728 GB of RAM.
- The live catalog (count 4, CUDA >= 13.0) lists 4x Server Edition stock only in EUR-IS-2, SECURE; community has
  none. Waiting for stock elsewhere, with the harness told to avoid EUR-IS-2.

### Attempt 6 (2026-10-16, EU-CZ-1, a1 never finished booting, $7.70)

The harness now takes `avoid_data_centers` (0ba8c45); 4x stock showed up in EU-CZ-1 at 17:23Z.

- Pod sm120-vllm-56771-21e503 (EU-CZ-1, host 213.192.2.101): container up in 2 min, 1604 MB/s download, 510 GB in
  313 s, setup done at 17:31Z. RunPod allotted 754 GB of RAM (the container limit RunPod's monitor reports is
  702.2 GiB); `free -g` inside shows the host's 1511 GB, so the setup RAM gate read the wrong number.
- a1 (patch pr60762, DSpark k=5, CUDA graphs) launched 17:32Z. vLLM logged `Filesystem type for checkpoints: FUSE.
  Checkpoint size: 475.25 GiB. Available RAM: 198.03 GiB`, offloaded the Engram table (23.60 GiB per rank to
  pinned host memory), and loaded 48 shards in 7m35s (17:40:52Z). Nothing more reached server.log.
- After that the 4 workers sat in D state at 600-800% CPU each (load average 77), GPUs at 82.7 GB used and 0%
  util. RunPod's system log shows container memory climbing about 9 GiB/min, 538.3 GiB at 17:49Z to 695.1 GiB at
  18:06Z; at 98% RunPod stopped and removed the container (18:06:42Z). The pod went to EXITED and the harness
  polled a dead SSH endpoint until I stopped it; server.log was not pulled.
- What filled the memory is not in the log. The 198 GiB "Available RAM" right after setup suggests the page cache
  from the 510 GB download counted against the container. Not a result about the reporter's bug: the server
  never reached the first prompt.

For the next attempt (`feat/harness-memory-guard`): `setup.sh` gates on the cgroup's limit instead of `free -g`,
keeps the weights on container disk behind `$HF_HOME/hub`, drops them from the page cache after the download (sync,
then posix_fadvise DONTNEED), stops under a 500 MB/s cold read, and installs py-spy. Every variant runs the memory
guard (`harness/README.md`), which logs memory.jsonl every 5 s and, at 93% of the limit, stops the server's process
tree, writes memory_guard.txt, and kills it before RunPod's 98%; `exec` pulls the variant every 5 minutes and stops
waiting on a pod RunPod reports EXITED.

What filled it, from the source at 49cb2bbf (desk research, not yet measured): the DSV4.1 VL wrapper sorts the whole
weight iterator (`vl_model.py:325`), so the 48 "loaded" shards are only lazy mmap views and the real copy runs after
the progress bar, faulting the checkpoint into page cache that stays mapped, so neither fadvise nor easy reclaim can
drop it. On top of that, torch rounds each pinned Engram slice up to a power of two (23.6 GiB pins 33 GiB, about 264
GiB at TP4; vllm#59849). Estimated need: 264 + 468 cache + ~30 processes, about 762 GiB against the 688 GiB stop.
vllm#57469 (4x H20, cgroup OOM while loading) looks like the same thing. `patches/dsv41-stream-load.diff` streams the
language model in checkpoint order, loads the vision tensors after it, and evicts each shard once its last tensor is
released, then every touched file after the load; `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:64` (torch 2.14.0,
`AllocatorConfig.cpp:245-252`) pins exact sizes. Estimated peak with both: about 318 GiB (413 if a rank lags a whole
Engram shard). Every patched variant now carries both; memory.jsonl will show whether the estimate holds.

Total for this issue: $33.13 of $40 by the ledger ($32.61 billed), not enough for another 4x boot. Attempt 7 quote:
4x Server Edition SECURE, $9.96/hr, planned 1.75 h (setup, a1, a0), worst case $17.43.
