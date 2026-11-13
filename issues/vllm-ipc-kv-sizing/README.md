# FlashInfer PCIe IPC all-reduce workspace vs. vLLM's KV cache sizing (vllm#67142)

vllm#67142 (a recipe row for DeepSeek-V4.1-Flash on 4x RTX PRO 6000 Max-Q, opened 2026-11-12) reports that the
FlashInfer PCIe IPC all-reduce (`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`, vllm#53576, first in v0.30.0) allocates its
buffers after vLLM sizes the KV cache, and that with DSpark this leaves too little KV cache for one 1M-token request,
so vLLM does not start. No issue is filed. This kit checks the claim on Qwen3-8B at TP4, IPC off against on, in both of
vLLM's sizing modes, and measures the workspace on its own.

## Reporter setup (vllm#67142, quoted)

> 4x RTX PRO 6000 Blackwell Max-Q, PCIe, no NVLink, 503 GiB host RAM.
> `vllm/vllm-openai:nightly` `sha256:902b49de0568` (0.34.0rc1.dev206+gf5d93296c).
> Checkpoint 1017.

```bash
VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_USE_RUST_FRONTEND=1 \
vllm serve deepseek-ai/DeepSeek-V4.1-Flash-1017 --tokenizer-mode deepseek_v41 --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 --max-num-seqs 128 \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41"}' --engram-config '{"cpu_offload":true}' \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --mm-encoder-tp-mode data
# spec decoding toggle: drops --max-num-seqs 128, appends
#   --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}' --max-num-seqs 32
```

> PCIe IPC all-reduce: its buffers are allocated after KV sizing, and with DSpark they take the KV cache below one
> 1M-token request (2.45 GiB after warmup without it, 2.23 GiB needed).

| Command | Error |
|---|---|
| nightly, DSpark with `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` | `To serve at least one request with the model's max seq len (1048576), (2.23 GiB KV cache is needed, which is larger than the available KV cache memory (1.85 GiB)` |

Without DSpark the same env booted ("The log shows `Initialized FlashInfer PCIe IPC all-reduce for TP4`") and raised one
stream from 111 to 124 tok/s. So with DSpark, IPC took 2.45 - 1.85 = 0.60 GiB of KV cache per GPU.

## Source

Read at vLLM main `ab72c466` (2026-11-13), the reporter's nightly `f5d93296c` (2026-11-11, FlashInfer 0.7.3), and
v0.33.0 `4ba787ec` (FlashInfer 0.7.2.post1). Paths are under `vllm/` unless they name FlashInfer.
`distributed/device_communicators/flashinfer_pcie_ipc_all_reduce.py` is identical at all three.

1. **When the workspace is allocated.** The constructor only checks and clears `disabled`
   (`flashinfer_pcie_ipc_all_reduce.py:44-89`); it runs with the TP communicator (`cuda_communicator.py:140-149`). The
   allocation is in `setup()` (`:95-156`): `max_numel = batches[-1] * self.hidden_dim` (`:126`), then
   `PcieIpcAllReduceWorkspace(...)` (`:127-133`), which `cudaMalloc`s one slab per rank and opens every peer's over CUDA
   IPC (FlashInfer `flashinfer/comm/pcie_ipc_ar.py:297-326`, `flashinfer/comm/cuda_ipc.py:268-269` at 0.7.3 and
   0.7.4), then the tune and prepare (`:139-146`). `setup()` is called from `kernel_warmup`
   (`model_executor/warmup/kernel_warmup.py:383-388` main, `:351-356` f5d93296c, `:258-263` v0.33.0), which
   `compile_or_warm_up_model` calls (`v1/worker/gpu_worker.py:1089` main, `:1078` f5d93296c, `:982` v0.33.0).
2. **Order in the engine** (`v1/engine/core.py`, main / f5d93296c / v0.33.0): `determine_available_memory` profiles
   (`:325` / `:341` / `:310`), `initialize_from_config` allocates the KV cache (`:389` / `:407` / `:372`),
   `compile_or_warm_up_model` then allocates the slab in kernel warmup and captures graphs (`:392` / `:410` / `:389`).
   - v0.33.0 has only estimate-based sizing (no `enable_extensible_kv_cache` in `config/cache.py`): the KV cache is
     final before the slab exists, so the slab is in nobody's budget. This is the reporter's sentence, exactly.
   - main and f5d93296c default to the extensible KV cache on CUDA where the driver has VMM (`config/cache.py:397-410`
     main, `config/vllm.py:3257-3259` f5d93296c). Its profile estimate and reserved capacity still predate the slab, but
     the final size is measured after warmup and capture (`core.py:393-403` main; `gpu_worker.py:1257-1265` main,
     `:1238` f5d93296c), from free device memory (`v1/worker/extensible_kv_cache.py:620-623` main), which does include
     the slab. On the reporter's build the slab is charged to the final KV size, not missed: "2.45 GiB after warmup" is
     that measurement.
   - `--no-enable-extensible-kv-cache` gives the nightly v0.33.0's order (`core.py:365-369` main).
3. **Size.** FlashInfer's `compute_workspace_layout` (`include/flashinfer/comm/pcie_ipc_all_reduce.cuh:2107-2146`) and
   `memop_workspace_bytes` (`csrc/pcie_ipc_all_reduce.cu:41-46`), the same at 0.7.3 and 0.7.4: with payload
   P = max_numel x 2 bytes, two scratch regions of 2 x world x P, a copy-engine ring of 2(N-1)/N x P, and at 4 or 8
   ranks on SM120 a second copy-engine region of about the same size: about 20.5 P per rank at TP4, 9 P at TP2.
   max_numel is the largest CUDA graph capture size times the hidden size; vLLM's default largest capture size on
   sm_120 is min(max_num_seqs x (1 + speculative tokens) x 2, 512) (`config/vllm.py:2422-2429` main).

   | Shape | Largest capture | Hidden | max_numel | Slab per GPU at TP4 |
   |---|---:|---:|---:|---:|
   | Reporter, DSpark k5, 32 seqs | 384 | 5120 | 1,966,080 | 76.9 MiB |
   | Reporter, default command, 128 seqs | 256 | 5120 | 1,310,720 | 51.3 MiB |
   | Kit e arms, Qwen3-8B | 480 | 4096 | 1,966,080 | 76.9 MiB (33.8 at TP2) |
   | Kit m and s arms, Qwen3-8B | 8192 | 4096 | 33,554,432 | 1,312.0 MiB |

   The reporter's 0.60 GiB is about 8x the slab their shape predicts. Either their largest capture size was not 384
   (the `Initialized ... max_tokens=` line in their log settles it) or something besides the slab moved.
4. **Where it engages.** World sizes 2, 4, and 8 (`:29`), one node, bf16 or fp16, at least one capture size, no
   DBO/ubatching (`:60-85`, `:106-122`, `:219-229`). TP2 engages. The kit runs TP4 anyway: at TP2 the slab has no SM120
   copy-engine region and is 9 P, not 20.5 P, so only TP4 lets a small model allocate the reporter's exact slab.
5. **vLLM's own all-reduce buffers, for comparison.** pynccl, symmetric memory, and the custom all-reduce are built in
   `CudaCommunicator.__init__` (`cuda_communicator.py:108, 129, 160` main; `:108, 128, 159` v0.33.0) during
   `init_worker_distributed_environment` (`gpu_worker.py:515-521` main, `:504` v0.33.0), which runs before the memory
   snapshot on purpose: "Initialize the distributed environment BEFORE taking memory snapshot. This ensures NCCL buffers
   are allocated before we measure available memory" (`:511-514` main). The custom all-reduce's IPC buffers are
   allocated there too (`distributed/device_communicators/custom_all_reduce.py:296-306`, 8 MiB `max_size` at `:138`),
   and it turns itself off above 2 PCIe-only GPUs (`:268-274`). Startup then checks free memory against the
   `--gpu-memory-utilization` budget (`v1/worker/utils.py:797-810` main, `:761` v0.33.0). The FlashInfer PCIe IPC
   slab is the only all-reduce buffer allocated after the snapshot, and after the KV cache.
6. **The tune's transient.** The extensible cache keeps free the allocator peak of warmup (`gpu_worker.py:1047` reset,
   `:1107-1115` read, main). On main the IPC setup runs inside that window: `_outside_kv_cache_transient_window`
   (`kernel_warmup.py:66-91`) wraps only FlashInfer's autotune (`:397-398`). At f5d93296c that window does not exist.
   The KV cache then pays any transient increase about 1.75 times once 0.75 x transient is above the 1 GiB margin floor
   (`extensible_kv_cache.py:58-59, 573-577` f5d93296c).
7. **Existing fixes.** None found. `gh search prs --repo vllm-project/vllm` for "flashinfer pcie ipc", "pcie ipc
   memory", "allreduce workspace memory profiling", and "all-reduce kv cache memory" turn up #53576
   (the backend), #54869 (lazy import), and #65192 (open: fall back without peer access). Open and related: #67228 (a 3
   GiB extensible margin floor on CUDA) and #67296 (headroom for allocator fragmentation), either of which would cut a
   tight KV cache like the reporter's further.

## Hypotheses

| # | Hypothesis | Confirmed if | Killed if |
|---|---|---|---|
| H0 | The IPC arms hold the workspace | `Initialized FlashInfer PCIe IPC all-reduce for TP4 ... max_tokens=480` (e1) or `8192` (m1); hooks.sh fails an IPC arm without it | a fallback warning and no Initialized line (the arm is POST_FAIL and does not count) |
| H1 | With estimate sizing (v0.33.0's only mode) the slab is allocated after the KV cache and outside the budget | s1 logs the same `Available KV cache memory` and KV tokens as s0, then fails with out of memory in kernel warmup or capture, or boots with each GPU holding about 1.28 GiB more than s0 | s1's KV cache is about 1.28 GiB (about 37,300 tokens) smaller than s0's |
| H2 | With the extensible cache (the reporter's build) the estimate misses the slab but the final size charges it | e1's `Available KV cache memory` equals e0's and e2's, and e1 holds about 2,200 fewer KV tokens (77 MiB) than both; m1 about 37,300 fewer than m0 | e1 and m1 lose no KV tokens: the slab is then outside the post-warmup measurement too, and H1's failure applies to the default build |
| H3 | Something besides the slab moves with IPC on, which would explain the reporter's 0.60 GiB | e1's KV loss exceeds ws_probe's held bytes for e by more than the e0/e2 spread. The line that moved names it: `kept for transient activations` (H3a, the tune in the peak window, Source 6), `CUDA graph pool memory (actual)` or `Graph capturing ... took` (H3b, graphs), or the same excess in e and m (H3c, a fixed cost: streams, peer mappings) | e1's and m1's losses both match the slab within the spread; the reporter's 0.60 GiB then points at their own shape (a larger max_tokens, or DSV4.1 and DSpark), and the next step is their max_tokens line, not more pod time |
| H4 | The loss scales with the largest capture size as the slab formula does | m1's loss minus e1's within 10% of 1,235 MiB (1,312.0 - 76.9) | it does not scale |

ws_probe.py (setup) gives the slab FlashInfer reports and what building the workspace takes per rank, so H2 and H3
compare the server's loss with both. KV tokens convert at 36,864 bytes per token per GPU (Qwen3-8B at TP4: 36 layers,
K and V, 2 KV heads, 128, bf16).

## Arms

| Arm | KV sizing | Capture sizes | IPC | Role |
|---|---|---|---|---|
| e0-off | extensible (default, util 1.0) | default grid to 480 | off | control; warms the compile caches |
| e1-ipc | extensible | default grid to 480 | on | H0, H2, H3 at the reporter's slab |
| e2-off | extensible | default grid to 480 | off | repeat: the e0/e2 spread is the noise |
| m0-off | extensible | 1, 2, 4, ..., 8192 | off | control for m1 |
| m1-ipc | extensible | 1, 2, 4, ..., 8192 | on | H2, H3, H4 with a 1.28 GiB slab |
| s0-off-est | estimate (`--no-enable-extensible-kv-cache`), util 0.98 | 1, 2, 4, ..., 8192 | off | control for s1 |
| s1-ipc-est | estimate, util 0.98 | 1, 2, 4, ..., 8192 | on | H1 |

Every arm: Qwen3-8B at `b968826d`, `--tensor-parallel-size 4 --max-model-len 32768 --max-num-seqs 128
--max-num-batched-tokens 8192`, the default CUDA graph mode, nothing else changed, and the harness's smoke request.
`post_smoke` (hooks.sh) checks the Initialized line on IPC arms and saves `nvidia-smi` memory per GPU on the live
server. At 0.98 the s arms' budget leaves 1.9 GiB of each card, part of it already taken by the CUDA context and NCCL,
for a 1.28 GiB slab.

`summarize.py` reads each arm's `server.log`, rank 0's line where workers each log one: the profile estimate, the
Initialized line's max_tokens, the CUDA graph pool, the capture delta, `Memory after warmup` (free, transient, KV), the
KV tokens, and the first out-of-memory, KV, extensible fallback, or startup memory error; then each IPC arm against its
control, beside ws_probe's slab and held bytes.

## Box and hours

4x RTX PRO 6000, any edition (all 96 GB; the reporter's is the Max-Q), PCIe only, `require_p2p` (P2P OK on all 12
ordered pairs, since the workspace maps every peer), Vast first, then RunPod SECURE. Image
`vllm/vllm-openai:nightly-f5d93296c8ec82b7e7dabdb308d24d43cb1907bf` by digest (`sha256:902b49de...`, 10.0 GB
compressed): the reporter's build. It has the extensible KV cache their numbers come from, and
`--no-enable-extensible-kv-cache` gives v0.33.0's order on the same build. v0.33.0 would serve Qwen3-8B too (#60762
matters only for DSV4.1), but it cannot show the extensible path.

| Step | Minutes | Running |
|---|---:|---:|
| up: create, image, SSH, PCIe, P2P, and idle gates, kit copy | 10 | 10 |
| setup: checks, 16.4 GB of weights, ws_probe.py (cold JIT build of the IPC module, the e tune) | 8 | 18 |
| e0-off: cold compile and capture | 5 | 23 |
| e1-ipc: cold IPC tune | 5 | 28 |
| e2-off | 3 | 31 |
| m0-off: 14 capture sizes up to 8192 tokens | 4 | 35 |
| m1-ipc: tune of the 8192 ladder | 5 | 40 |
| s0-off-est | 4 | 44 |
| s1-ipc-est: ends in kernel warmup or capture under H1 | 3 | 47 |
| down: pull, destroy | 3 | 50 |

`planned_hours = 1.0`. setup.sh stops a download still running 38 min before the deadline.

## Run

```bash
harness quote vllm-ipc-kv-sizing
harness budget set vllm-ipc-kv-sizing <usd> --note <text>        # nothing rents without it
harness up vllm-ipc-kv-sizing \
  && harness exec vllm-ipc-kv-sizing e0-off e1-ipc e2-off m0-off m1-ipc s0-off-est s1-ipc-est; \
  harness down vllm-ipc-kv-sizing
python3 -I issues/vllm-ipc-kv-sizing/summarize.py results/vllm-ipc-kv-sizing
harness budget show vllm-ipc-kv-sizing
```

The arms that decide H2 and H3 go first, so a deadline cuts the s pair before them. Never leave the names out: `exec`
with none runs every variant in file order until the deadline.

## Stop rules

- A setup check fails (GPU count, an inherited setting, /dev/shm, a late download, ws_probe.py failing or hanging):
  setup exits 1 before any arm; after `up`, run `harness down`.
- e0 BOOT_FAIL, SMOKE_FAIL, or TIMEOUT: the base command does not serve on this box; `harness down`, report the fault.
- An IPC arm without the Initialized line: POST_FAIL by hooks.sh; its numbers do not count.
- s1 BOOT_FAIL on out of memory is H1's outcome, not a fault.
- e0's log has no `Memory after warmup` line: the host could not use the extensible cache and fell back to estimate
  sizing at util 0.92 (`config/cache.py:186, 491-497` f5d93296c), so the e and m pairs test H1 too and H2 cannot be
  read on this box.

## Outputs (`results/vllm-ipc-kv-sizing/`)

| File | What |
|---|---|
| `env.txt`, `p2p-<pod>.txt`, `jobs/setup.log` | the box, its P2P matrix, setup's log |
| `ws_probe/ws_probe.json`, `.log`, `tune_e.json`, `fault.rank<N>.txt` | the workspace alone: slab, held, transient, kept after destroy, per rank |
| `<arm>/server.log`, `gpu_mem_live.csv`, `verdict.json`, `memory.jsonl` | each arm |

## Fix (H1 holds; the fix is not written or tested)

- Set the workspace up in `Worker.determine_available_memory`, before `memory_profiling` (`gpu_worker.py:736` main),
  so profiling's non-KV memory includes the slab and the estimate subtracts it; or keep the allocation where it is
  and subtract FlashInfer's exact `workspace_size(world, max_numel, 2, max_blocks)` from
  `available_kv_cache_memory_bytes` beside the CUDA graph estimate (`gpu_worker.py:795-799` main). What the first needs
  is known by then: hidden size and dtype from the model config, capture sizes from the compilation config, and the
  tune cache path from the model runner, which exists after `init_device`. To check: anything that changes the capture
  sizes after `determine_available_memory` would leave an early slab too small, and `setup()` would have to refuse
  rather than reallocate.
- The run favors the first. The setup costs more than the slab (torch held 0.13 GiB more after it at e and 0.45 GiB at
  m, Results), and an early setup inside the profile should count that too; subtracting `workspace_size` counts the
  slab only.
- The extensible path needs no reordering: it charges the slab (H2). What torch holds after the setup (FlashInfer's
  tune or `prepare`, or vLLM's wrapper) is a separate question, to answer before a PR claims to cover the full cost.
- H3a is killed, so wrapping the IPC setup in `_outside_kv_cache_transient_window` (`kernel_warmup.py:66-91` main) is
  not needed at these shapes.

## Risks that the repro will not show the bug

- On the reporter's build the extensible cache measures free memory after warmup, so the slab is charged, not missed
  (Source 2). At the reporter's shape the slab is 77 MiB, about 0.1% of the KV cache here: e1 against e0 may sit inside
  run-to-run noise, which is what e2 and the m pair are for.
- The reporter's 0.60 GiB may be specific to DSV4.1 and DSpark (drafter graphs, mHC, a larger max_tokens). If e and m
  both match the slab, this kit reproduces the mechanism but not the size; the follow-up is their max_tokens line, or a
  DSV4.1 TP4 DSpark run on issues/vllm-61530's box shape.
- A host whose driver lacks VMM falls back to estimate sizing with one log line; the e and m arms then test H1, not H2.
- FlashInfer's cold tune stalled past 20 min at boot on 4x RTX PRO 4500 (vllm#65192). ws_probe.py runs the e tune in
  setup under a 15 min cap; m1's tune of the 8192 ladder is not probed and runs under the 20 min boot timeout.
- s1 may boot instead of failing if the graph budget still unspent at kernel warmup covers the slab; H1 then reads from
  its KV tokens and memory per GPU.
- Qwen3-8B is a dense bf16 model. The slab depends only on world size, max_numel, and dtype size, but H3's transient
  and graph effects may differ for DSV4.1's MoE with mHC and a DSpark drafter.
- Vast reports P2P per pair only at `up`; a box without P2P on all 12 pairs is refused and destroyed before setup.

## Results

### 2026-11-13, 4x RTX PRO 6000 Server Edition, Vast machine 157204 (Nevada)

Box: 4x RTX PRO 6000 Blackwell Server Edition (97,887 MiB, compute 12.0), driver 615.71.09, 128 CPUs on 2 NUMA nodes,
1.5 TB RAM. GPUs 0-1 and 2-3 each share a PCIe switch (`PIX`), the two pairs are `SYS` across the sockets, and
`topo -p2p r` is OK on all 12 ordered pairs; FlashInfer picked its `pcieswitch-pairs` profile. Image: the reporter's
nightly by digest (vLLM 0.34.0rc1.dev206+gf5d93296c, torch 2.14.0+cu130, FlashInfer 0.7.3, Triton 3.8.0). Weights came
down in 11 s and setup took about a minute, most of it ws_probe.py's 36 s cold tune. The arms ran 09:07-09:18 UTC in
the order e0, e1, e2, m0, m1, s0, s1; 0.26 h on the box. Boots: e0 160 s (cold compile), e1 85 s, e2 45 s, m0 55 s,
m1 110 s, s0 40 s, s1 35 s to its failure. Six arms OK and s1 BOOT_FAIL on out of memory, which is H1's outcome. One
rep per arm; e2 repeats e0.

`summarize.py` (rank 0's lines; KV tokens are the engine's, which is the smallest rank's):

| Arm | Verdict | IPC max_tokens | Profile estimate GiB | Graph pool GiB | Capture GiB | After warmup: free / transient / KV GiB | KV tokens | Used MiB per GPU |
|---|---|---:|---:|---:|---:|---|---:|---|
| e0-off | OK | - | 90.70 | - | 0.56 | 89.11 / 0.72 / 87.68 | 2,553,856 | 95517 95533 95517 95533 |
| e1-ipc | OK | 480 | 90.70 | - | 0.40 | 89.04 / 0.72 / 87.61 | 2,551,808 | 95517 95549 95517 95549 |
| e2-off | OK | - | 90.81 | - | 0.56 | 89.11 / 0.72 / 87.68 | 2,553,856 | 95517 95533 95517 95533 |
| m0-off | OK | - | 90.70 | - | 0.77 | 88.91 / 0.72 / 87.54 | 2,547,712 | 95505 95521 95505 95521 |
| m1-ipc | OK | 8,192 | 90.70 | - | 0.78 | 87.13 / 0.72 / 85.71 | 2,496,512 | 95527 95559 95527 95559 |
| s0-off-est | OK | - | 88.06 | 0.64 | 0.75 | - / - / - | 2,565,056 | 96365 96381 96365 96381 |
| s1-ipc-est | BOOT_FAIL | 8,192 | 88.13 | - | 0.75 | - / - / - | 2,565,056 | - |

The s arms' Capture column is the profile's estimate capture, before the KV cache; s0's real capture is its 0.64 GiB
graph pool.

| IPC arm vs control | KV tokens lost | As GiB per GPU | Transient delta GiB | Used delta MiB (max GPU) | Slab MiB | Workspace held MiB (max rank) |
|---|---:|---:|---:|---:|---:|---:|
| e1-ipc vs e0-off | 2,048 | 0.070 | 0.000 | 16 | 76.9 | 238.0 |
| e1-ipc vs e2-off | 2,048 | 0.070 | 0.000 | 16 | 76.9 | 238.0 |
| m1-ipc vs m0-off | 51,200 | 1.758 | 0.000 | 38 | 1,312.0 | 1,314.0 |
| s1-ipc-est vs s0-off-est | 0 | 0.000 | - | - | 1,312.0 | 1,314.0 |

ws_probe.py, the workspace alone on the same box, identical on all four ranks: `e` (vLLM's wrapper at 480 x 4096,
with its tune and `prepare`) slab 76.9 MiB, held 238.0 MiB, torch transient 89.7 MiB, 160.0 MiB still held after
`destroy()`; `m` (FlashInfer's workspace alone at 8192 x 4096, no tune) slab 1,312.0 MiB, held 1,314.0 MiB, nothing
held after `destroy()`. The 8-token warm-up shape held 2.0 MiB.

**s1, in order** (rank 0 unless noted): `Available KV cache memory: 88.13 GiB` (s0: 88.06), `GPU KV cache size:
2,565,056 tokens` (s0: the same), then in kernel warmup `Initialized FlashInfer PCIe IPC all-reduce for TP4,
hidden_dim=4096, dtype=torch.bfloat16, max_tokens=8192.` at 09:17:54, in the same second warmup started (consistent
with m1's persisted tune being reused; m1's setup took 44 s). Two seconds later every rank failed in `capture_model`
(`gpu_worker.py:1106` at f5d93296c): ranks 0 and 2 on a 32 MiB inductor buffer in the eager pass before a capture,
ranks 1 and 3 on 2 MiB in `flash_attn_varlen_func` inside a piecewise capture. Rank 0: "GPU 0 has a total capacity of
95.06 GiB of which 28.88 MiB is free. Including non-PyTorch memory, this process has 95.03 GiB memory in use. Of the
allocated memory 92.62 GiB is allocated by PyTorch, with 536.00 MiB allocated in private pools (e.g., CUDA Graphs)".
s0, the same flags without the variable, captured all 14 sizes into its 0.64 GiB pool and served with 1,506-1,522 MiB
per GPU unused (nvidia-smi).

**Where m1's loss past the slab sits.** m1 loses 51,200 tokens (1,800 MiB) for a 1,312.0 MiB slab, 488 MiB more,
while e0 and e2 agree to the token. None of the lines H3 named moved: `kept for transient activations` is 0.72 GiB in
all five extensible arms, IPC or not, and m1's capture took 0.78 GiB against m0's 0.77. The line that moved is the
end of `Memory after warmup`, `Torch may reserve up to`: 5.87 GiB in m0, 6.37 GiB in m1 (rank 0; rank 1 5.93 and
6.34). At f5d93296c that cap is torch's `memory_reserved()` after `empty_cache()`, plus what stays free after the KV
extension less a 0.5 GiB non-torch reserve (`v1/worker/extensible_kv_cache.py:597-623`). The second term, free - (KV -
committed) - 0.5 from the same line, is 1.22 GiB in m0 and 1.27 GiB in m1, so torch still held about 0.45 GiB more
after the IPC setup that `empty_cache()` could not return (4.65 against 5.10 GiB, on ranks 0 and 1 alike). Rank 0's
1.78 GiB less free memory is then the 1.28 GiB slab, about 0.45 GiB held by torch, and about 0.05 GiB the two-decimal
log lines do not resolve. The same arithmetic gives +0.13 GiB at e (4.11 to 4.24 GiB), where e1's capture also took
0.16 GiB less than e0's and e2's (0.40 against 0.56 GiB), so e's net loss lands on the slab. The logs do not say what
torch holds, live tensors from the setup or segments it cannot release. ws_probe points the same way: the wrapper's
`e` shape (tune and `prepare`) kept 160.0 MiB after `destroy()`, the bare shapes nothing.

- H0 holds: e1 logged `max_tokens=480`, m1 and s1 `max_tokens=8192`, no log has a fallback warning, and hooks.sh
  passed e1 and m1.
- H1 holds: with estimate sizing the slab is allocated after the KV cache and outside the budget. s1 sized the same
  2,565,056 KV tokens as s0, allocated the 1.28 GiB slab, and ran out of memory in graph capture 2 s later.
- H2 holds: the profile estimate misses the slab (90.70 GiB in e0, e1, m0, and m1; e2's 90.81 is run-to-run), and the
  post-warmup measurement charges it. e1 lost 2,048 tokens (72.0 MiB) for a 76.9 MiB slab, about two 64-token blocks
  under it; m1 lost 51,200 (1,800 MiB) for 1,312.0 MiB. On the reporter's build the slab is charged, not missed.
- H3 is killed at e and holds at m. e1's loss is on the slab and below ws_probe's 238.0 MiB held. m1's is 488 MiB past
  the slab with a zero e0/e2 spread, through none of the lines the table named: H3a (transient reserve) and H3b
  (capture) did not move, and H3c fails because e shows no fixed cost. It is torch memory held after the setup.
- H4 is killed as written: m1's loss minus e1's is 1,728 MiB against the slab's 1,235 MiB growth (1.40x). The loss
  grows with the largest capture size faster than the slab, because the part torch holds grows too (0.13 to 0.45 GiB).

**The reporter's 0.60 GiB.** The e arms allocate the exact slab the reporter's DSpark command should (76.9 MiB at 384
x 5120, Source 3) and lose 0.070 GiB of KV cache, about a ninth of 0.60. So on this evidence their 0.60 GiB comes from
a larger largest capture size, not from the slab at 384. At TP4 the slab is 20.5 x max_numel x 2 bytes: alone it
reaches 0.60 GiB at a max_numel of about 15.7M, a largest capture of about 3,070 tokens at hidden 5120; with m's
torch-held share on top (loss 1.37x the slab), about 2,240. The `max_tokens=` in their `Initialized FlashInfer PCIe IPC
all-reduce` line settles it.

Caveats:

- One rep per arm. e0 and e2 agree to the token after warmup, but their profile estimates differ by 0.11 GiB, so the
  estimate column is noise at that size.
- Qwen3-8B is dense bf16; the reporter runs DSV4.1-Flash with DSpark. The slab depends only on world size, max_numel,
  and dtype, but the part torch holds and the graph effects may differ for an MoE with mHC and a drafter.
- Server Edition here, Max-Q for the reporter: the same 97,887 MiB GB202, so the memory numbers carry over.
- The s arms are v0.33.0's order of sizing and warmup on the f5d93296c build, not v0.33.0 itself, at 0.98 so a
  1.28 GiB slab overflows. At lower utilization the slab comes out of the unbudgeted headroom instead, which this run
  does not measure.
- ws_probe did not tune the m shape, so its 1,314.0 MiB has no tune in it; m1's boot tuned it.
- Ranks differ by up to 0.03 GiB free after warmup; the tables use rank 0's lines.
