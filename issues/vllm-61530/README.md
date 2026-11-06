# DeepSeek-V4.1-Flash-1017 NVFP4 at TP4 on 4x RTX PRO 6000 (vllm#61530 on sm_120)

Can DeepSeek-V4.1-Flash run on half of an 8x RTX PRO 6000 box? The FP8 checkpoint is 475 GiB, more than four 96 GB
cards hold (382 GiB), so the sm_120 deployments we know of run it at TP8 (issues/vllm-59203, vllm#59203). NVIDIA's
NVFP4 checkpoint, `nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`, is about 400 GiB, and its Engram tables can live in
pinned host memory, which leaves the GPUs about 302 GiB of weights. It needs vllm#61530 (MXFP4 Engram tables, open)
and was validated only on four GB300s. This kit serves it at TP4 on four PCIe-only RTX PRO 6000s and measures what
fits: GPU memory per card, KV cache left, host memory the Engram tables pin, quality, and decode speed.

Two TP4 engines per 8x box (one per NUMA half) is the shape helixml/ramjet#264 wants for this model, on a host with
125 GiB of RAM; the host number below decides whether that shape fits.

## What is known

- Checkpoint `nvidia/DeepSeek-V4.1-Flash-1017-NVFP4` at `3ae9c0d3fbbea5370bd06e679d22044db15883f1` (main, 11-04):
  48 shards, 428,986,460,424 B. Routed experts are NVFP4 W4A4 (group 16); the Engram tables (shards 47 and 48,
  104,766,153,352 B, 97.6 GiB) are MXFP4 with UE8M0 scales, 136 B per 256-value row; attention, shared experts,
  vision, and MTP keep their source precision. `config.json`'s `quantization_config` sets `engram_embed_quant:
  mxfp4` (block 32, `ue8m0`, layers [1, 14]), the metadata #61530 reads.
- NVIDIA's command (model card): TP4, `--attention-config
  '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'`,
  `--kv-cache-dtype fp8`, `--max-num-seqs 128`, `--engram-config '{"cpu_offload":true}'`. MTP was not exercised.
- vllm#61530 at c2e1c436 (rebased on main cb95d994, 11-05) changes one source file,
  `vllm/models/deepseek_v41/common/engram.py`; `patches/pr61530-mxfp4-engram.diff` is that file's hunk. Its validation
  is on the earlier release, `nvidia/DeepSeek-V4.1-Flash-NVFP4` (c48dca01, MXFP4 tables, against 3431dde3, FP8
  tables), not this checkpoint: four GB300s, GSM8K 96.74% vs 95.91%, MMLU 91.20% vs 91.38%. For the 1017 checkpoint
  the model card reports NVIDIA's own runs (vLLM v0.31.0 plus the PR's Engram changes) and SGLang GSM8K 96.59%
  without speculation. The 1017 release has the same architecture config as the checkpoint issues/vllm-59203 served
  (deepseek-ai/DeepSeek-V4.1-Flash at 314113f8) but different weights, and NVFP4 W4A4 experts in place of MXFP4, so
  the TP8 rows there compare by shape, not as a precision-only change.
- The image is vLLM main at 4fab5a07 (nightly, 11-06), 42 commits past cb95d994 with no change to `engram.py`; the
  patch applies there cleanly (dry run, 11-06). v0.33.0 is 868 commits behind cb95d994.

## What sm_120 needs instead of NVIDIA's flags (source at 4fab5a07)

- `indexer_sparse_logits` requires an SM100-class GPU (`vllm/config/attention.py:121-127`), so it is left out.
- `FLASHINFER_MLA_SPARSE_DSV41` supports major 12 with an fp8 KV cache and 64-token blocks
  (`models/deepseek_v41/nvidia/flashinfer_sparse.py:66-71, 115-123`); the block-size negotiation fix (vllm#60762)
  is in main, so no block-size flag or patch. The backend is left to vLLM's selection, as on our TP8 runs.
- The indexer cache: fp8 is what our sm_120 DSV4.1 runs used (a0, a1). mxfp4 is refused off sm_10x:
  `dsa_indexer_uses_fp4` raises `indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs (sm_10x, e.g.
  B200/GB200); sm_120 (consumer Blackwell) and earlier architectures are not supported`
  (`vllm/v1/attention/backends/mla/indexer.py:101-106`), and the DSV4.1 indexer calls it in its constructor
  (`models/deepseek_v41/attention.py:1470`), so an mxfp4 arm would end at model construction, before the weights load;
  the kit has none.
- Engram host tables: `cpu_offload` (default on) pins each rank's slice of each of the two tables and logs `Engram
  table offloaded to pinned host memory: ... GiB per rank` once per table (`common/engram.py:1134`, 1184 with the
  patch). The `... GiB exceeds the ... GiB of available host memory` check (`:808`) is `_shm_shortfall`, which only
  the DP-shared /dev/shm path reaches, and there it is a warning and a fallback to per-rank tables
  (`:975-980`); a single TP4 engine never logs it. A host too small for the pinned slices fails in the pinned
  allocation or at the memory guard's 93% line instead. `dp_shared_memory` (`vllm/config/engram.py`) shares one host
  copy between co-located DP replicas; with a single TP4 engine it does not apply.

## Memory, before measuring

| | GiB | Source |
|---|---:|---|
| Checkpoint | 399.5 | Hub tree at the pinned revision |
| Engram tables, host (all ranks) | 97.6 | shards 47 + 48 (the tables alone: 768,022,850 rows x 136 B, 97.3) |
| Routed experts per GPU at TP4 | 71.2 | 284.8 / 4, from the 48 shard headers |
| Rest of the language model, vision | 9.0, 0.8 | attention 4.7, embed + head 2.5, shared experts 1.3, the rest 0.5 |
| Weights per GPU at TP4 | 73.7-81.0 | experts plus the 9.8 split 4 ways, up to all of it replicated; MTP (7.4) loads only with speculation |
| Usable per GPU at 0.92 | 87.4 | vLLM sees 94.97 GiB per card (vllm-59203 b0 server.log: `(0.92, 87.37 GiB)`) |
| Left per GPU for activations, CUDA graphs, KV | 6.4-13.7 | at TP8 graphs took 1.0-1.7 GiB; a 128K request needs about 0.62 GiB of KV per GPU |

The same header count for vllm-59203's checkpoint gives at most 33.6 + 9.0 = 42.6 GiB per GPU at TP8, and vLLM
reported `Model loading took 47.52 GiB`: 4.9 GiB per GPU that the tensors do not explain (load-time buffers or
repacking). If TP4 carries the same, 1.5-8.8 GiB is left, and the low end does not boot at 128K: a0 then ends with
vLLM's KV cache error, after `Model loading took` has given H2's weights number.

Two TP4 engines on one host would pin about 195 GiB of Engram tables between them.

## Hypotheses

- H1: with #61530 the checkpoint loads and serves correctly on sm_120 at TP4 (GSM8K 50 at least 46/50; our TP8 runs
  of the FP8 checkpoint scored 48-49). Killed by a load error, a kernel without an sm_120 path, or GSM8K under 46.
- H2: GPU memory per card after load is 74-81 GiB of weights (the table above), with a KV cache that holds at least
  one 128K request. Killed if vLLM reports a KV cache under `--max-model-len` (it refuses to start) or weights above
  81.
- H3: host memory pinned for Engram is about 97.3 GiB per engine: two log lines per rank of about 12.16 GiB (96.0M
  rows x 136 B), 24.3 per rank. If so, two independent TP4 engines cannot share a 125 GiB host; one DP2 x TP4 server
  with `dp_shared_memory` would hold one copy (not tested here).
- H4: FlashInfer's PCIe IPC all-reduce (a1) cuts c=1 median ITL at TP4 as it did at TP8 (-20%), with 32K TTFT
  unchanged, on a box with P2P on every pair.
- H5: if a0 cannot fit its KV cache (the low end of the table above), dropping the vision encoder and raising the
  memory fraction to 0.95 (f0) frees enough for a 128K request; the gap between a0's and f0's KV cache is the price of
  serving images.

## Arms

| Arm | Memory | All-reduce | Runs |
|---|---|---|---|
| a0-fp8idx | vision on, 0.92 | NCCL defaults | always, alone first |
| a1-fp8idx-fi | vision on, 0.92 | + `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` | after an a0 that booted, when P2P reads OK on all 12 pairs |
| f0-fp8idx-lm | `--language-model-only`, 0.95 | NCCL defaults | only when a0 could not fit its KV cache |
| f1-fp8idx-lm-fi | `--language-model-only`, 0.95 | + FlashInfer | after f0, when P2P reads OK on all pairs and it fits |

Every arm: the fp8 indexer cache, the #61530 patch, `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:64` (exact-size pinned tables), NCCL
INFO to a file per rank, GSM8K 50, `mem_snapshot.sh` (GPU, server-log, host, cgroup, and per-process memory), then
`decode_bench.py`: 1K in / 512 out at c=1, 8, 32 and 32K in / 256 out at c=1, the same rows and seeds as
issues/vllm-59203 minus c=64. The harness's memory guard records the container's memory every 5 s
(`<arm>/memory.jsonl`), which gives the boot's peak.

## Box, hours, cost

4x RTX PRO 6000 (Server Edition first, then Workstation, then Max-Q) on Vast with at least 280 GiB of host RAM,
1.5 Gbps down, and downloads at most $10/TB; then RunPod's 4x Server Edition (SECURE), which charges nothing for
downloads. Setup about 5 min plus the download (25 min at 300 MB/s), a0 about 30 min, every later arm about 25, 6 for
the last exec and down: about 1.5 h for a0 and a1. `planned_hours = 2.0`.

On 2026-11-06 no Vast offer met the floors (the four 4x offers were either $39-41/TB downloads or 569 Mbps), and
`harness quote` gave RunPod US-NE-1 at $9.96/hr, $21.91 worst case. Budget: $22.

## Run

```bash
harness quote vllm-61530
harness budget set vllm-61530 <usd> --note <approval>        # nothing rents without it
harness up vllm-61530 && harness exec vllm-61530 a0-fp8idx
# then the second exec the next: line names, by a0's verdict
harness exec vllm-61530 <arms>; harness down vllm-61530
harness budget show vllm-61530
```

The second exec takes its arms from the `next:` line at the end of `results/vllm-61530/jobs/setup.log`: a1 after an
a0 that booted (`"verdict": "OK"` or `"POST_FAIL"`), f0 and f1 after an a0 that ended on vLLM's KV cache error.
Never leave the names out: `exec` with none runs every variant in file order until the deadline.

## Stop rules

- A setup check fails: setup exits 1 before the download; `watch` destroys the box and avoids the machine.
- The download would end, or is still running, less than 65 min before the deadline: setup exits 1.
- a0 ends on vLLM's `KV cache is needed` or `No available memory for the cache blocks` (H2): run f0, then f1 if
  the `next:` line names it.
- a0 BOOT_FAIL, SMOKE_FAIL, or TIMEOUT for any other reason: no further arm; `harness down`, and report the fault
  (H1). A `memory_guard.txt` (the guard fired at 93%) or a failed pinned allocation is H3 on a host smaller than its
  RAM check assumed.
- A FlashInfer arm without the `Initialized ... for TP4` line: POST_FAIL before its rows, not a slow arm.

## Outputs (`results/vllm-61530/`)

| File | Holds |
|---|---|
| `env_extra.txt`, `p2p-read.txt`, `nvidia-smi-q.txt` | the box: links, clocks, P2P, NUMA, IOMMU, NVIDIA registry settings |
| `<arm>/memory_snapshot.txt`, `<arm>/memory.jsonl` | memory after GSM8K; the container's memory over the run |
| `<arm>/decode_rows.csv`, `<arm>/bench/<row>.json` | one line per row; the client's full result |
| `<arm>/server.log`, `verdict.json`, `gsm8k.txt`, `gsm8k_responses.jsonl` | the harness's per-variant record |
| `<arm>/nccl.log`, `nccl_transport.txt` | NCCL's INFO lines per rank, and condensed |

## Results

Not run yet.
