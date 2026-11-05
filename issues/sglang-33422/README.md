# sglang#33422: DeepSeek-V4-Flash prefill on 4x PRO 6000, sglang 2-7K tok/s vs vLLM ~12.5K

https://github.com/sgl-project/sglang/issues/33422, reporter hassellof (founder, Ombori), open since 08-03. A paid
benchmark: the same prompts on sglang (this folder) and on hermia's stock vLLM recipe (`issues/sglang-33422-vllm/`,
a sibling so it has its own ledger and budget), on the same pod type, so the gap is measured instead of quoted.

## Reporter setup (copied, ASCII-normalized)

- 4x RTX PRO 6000 Blackwell, SM120, 96 GB, TP=4, PCIe gen5, no NVLink. His PR #33407 and his public repo
  (ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000) name the card: **Max-Q Workstation Edition** (300 W).
- `deepseek-ai/DeepSeek-V4-Flash-0731`, sglang v0.5.16 + his #33407 dispatch fix, `dsv4` attention backend,
  `flashinfer_mxfp4` MoE, fp8_e4m3 KV, DSPARK.
- Measurements:
  - Aggregate prefill, 9 concurrent ~90K-token prompts: ~5,540 tok/s (813K prompt tokens / 146.9 s wall; outputs
    negligible at 61 tokens each)
  - Aggregate prefill, 3 concurrent ~650K-token prompts: ~2,805 tok/s (1.95M / 695.5 s)
  - Single ~650K-token prompt: ~2,870 tok/s (226.4 s to first token)
  - Scheduler logs: per-batch input throughput ~6.5K tok/s at low sequence position, ~1.8-2.1K past several
    hundred K tokens
  - `--chunked-prefill-size` 4096 vs 8192: 146.9 vs 149.0 s on the 9x90K test
  - Startup: `Breakable CUDA graph is incompatible with DeepSeek-V4 (heavy capture-pool memory pressure); disabling
    prefill CUDA graph`
  - `--max-prefill-tokens` at default 16384; `--disable-flashinfer-autotune` set
- Config:

```
--tp-size 4 --trust-remote-code
--mem-fraction-static 0.90 --context-length 1048576
--swa-full-tokens-ratio 0.1
--max-running-requests 64 --cuda-graph-max-bs 64
--kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_mxfp4
--speculative-algorithm DSPARK --speculative-attention-mode decode
--disable-custom-all-reduce --chunked-prefill-size 4096
--disable-flashinfer-autotune
```

- Questions: (1) is dsv4 prefill on SM120 known to be unoptimized; (2) would `--ep-size 4` help MoE prefill on
  PCIe, and does it work with `flashinfer_mxfp4`; (3) does the disabled prefill CUDA graph matter, or is it the MoE
  GEMM; (4) which flags to try.
- His production compose file adds `--dp-size 4 --enable-dp-attention --enable-dp-lm-head --ep-size 4` and
  `--speculative-dspark-block-size 7`.
- Thread since: lllllinux pointed at #29927 (08-04); KushagraKanaujia: #29927 ships in v0.5.20, rerun the 9x90K test
  there (09-24); Lei-010 measured ~5,982 tok/s cold prefill (one 4,714-token prompt) on DSV4.1, TP4 EP4, an
  unversioned build (09-27). Nobody has rerun the 9x90K test on a release with #29927.

## The vLLM number

hermia-ai/deepseek-v4-flash-sm120-stock-vllm (commit 9fa3a628, 07-18): vLLM 0.25.1 + flashinfer-python 0.6.14,
`--tensor-parallel-size 4 --enable-expert-parallel --kv-cache-dtype fp8 --max-model-len 262144
--gpu-memory-utilization 0.90 --kernel-config '{"moe_backend":"marlin"}'`, `NCCL_P2P_DISABLE=1`, no speculation, on
`deepseek-ai/DeepSeek-V4-Flash-DSpark`. Its ~12,500 tok/s is `bench/needle.py`: one non-streamed request of about
238K prompt tokens, `usage.prompt_tokens` / wall seconds. Two differences from the reporter's number make them not
comparable: one request vs 9 concurrent, and 238K vs 90K tokens each. Full detail in `issues/sglang-33422-vllm/`.

## What changed upstream since (sglang v0.5.22 source, checked 2026-10-16)

- **v0.5.22 carries #29927** (merged 09-02: SM120 DeepGEMM paged-MQA indexer, page split, the `hc_combine` kernels)
  and pins FlashInfer 0.7.1 (`python/pyproject.toml:42`, `docker/Dockerfile:21`).
- **`--cuda-graph-max-bs` is gone.** v0.5.22 has `--cuda-graph-max-bs-decode` and `--cuda-graph-max-bs-prefill`
  (`arg_groups/fields/exec_.py:567-571`), and `prepare_server_args` builds a default argparse parser
  (`server_args.py:665`), so the old spelling is an ambiguous prefix and the server exits. `base_flags` carries his
  64 as `--cuda-graph-max-bs-decode 64`. Every other flag of his, and every flag the matrix adds, exists at v0.5.22
  under `arg_groups/fields/`: `parallel.py` (`tp_size` :45, `dp_size` :73, `enable_dp_lm_head` :199, `ep_size`
  :269), `model.py` (`trust_remote_code` :100, `context_length` :104, `kv_cache_dtype` :191 with `fp8_e4m3` :206),
  `schedule.py` (`mem_fraction_static` :27, `max_running_requests` :31, `chunked_prefill_size` :52,
  `max_prefill_tokens` :67, `swa_full_tokens_ratio` :146), `exec_.py` (`disable_flashinfer_autotune` :339,
  `disable_custom_all_reduce` :663, `moe_runner_backend` :821, choices in `choices.py:145,150`), `spec.py`
  (`speculative_algorithm` :31, `speculative_attention_mode` :129). `--enable-dp-attention` is a deprecated alias that
  still sets the field (`server_args.py:417-423`, resolved to `--attn-dp-size`).
- **The DSV4 prefill CUDA graph is still disabled by default** (`arg_groups/cuda_graph_hook.py:164-165`), so an
  upgrade alone does not answer question 3.
- **Backend selection is automatic:** DSV4 gets `attention_backend=dsv4` and `page_size=256`
  (`arg_groups/model_overrides/deepseek_v4.py:24,51-61`) and `flashinfer_mxfp4` MoE on SM120 (`:75-88`). Marlin is
  wired for the checkpoint's FP4 experts (`layers/quantization/fp8.py:439`), and the FlashInfer CUTLASS runner passes
  `ep_size`/`ep_rank` (`layers/moe/moe_runner/flashinfer_cutlass.py:311`), so s2 and s3 are supported paths.
- **PR #33407 is not needed on v0.5.22, and would break it.** FlashInfer 0.7.1 rewrote SM120 sparse MLA: DSV4 decode
  takes topk at runtime (`flashinfer/mla/_sparse_mla_sm120/_policy.py:129`, and 192 is one of its calibration points,
  `:191`), and v0.5.22 sends batches over 64 tokens to the prefill kernel (`kernels/ops/attention/flash_mla_sm120.py:212,317`).
  Neither of the errors #33407 fixed (`num_tokens > 64`, `Unsupported sparse-MLA prefill configuration`) exists in
  FlashInfer 0.7.1. The PR's source half applies to v0.5.22 with `patch -p2`, but imports
  `_decode_dsv4_dispatchable`, which FlashInfer 0.7.1 does not export, so every decode-sized call would raise. Not
  used. FlashInfer 0.7.1's own dispatch error (`SM120 sparse-MLA has no decode kernel`, `_api.py:396`) is in
  `fail_pattern`.
- **DSpark draft depth 5 on SM120** corrupted output on v0.5.16 (#33800, closed); #29927's `hc_combine` kernel fixed
  it as a side effect, per his repo, and v0.5.22 has that kernel. The standalone fix #34021 is still open. s0 runs the
  posted config (no block size, so the checkpoint's 5), with GSM8K as the check.

## Hypotheses

| # | Hypothesis | Confirmed by | Killed by |
|---|---|---|---|
| H1 | v0.5.22 (with #29927) closes most of the gap on the reporter's own config | s1 and s0 W1 within 20% of vLLM v0 W1 | s1 W1 under 0.67x v0 W1 |
| H2 | The gap is the MoE GEMM (FlashInfer CUTLASS MXFP4 vs Marlin) | s2 beats both s1 and s1a by more than 15% on W1 and W2 | s2 within 10% of s1a |
| H3 | EP4 speeds MoE prefill on PCIe (question 2) | s3 beats s1 by more than 10% | s3 at or below s1 |
| H4 | DSpark does not change prefill throughput | s0 within 5% of s1 on W1 | s0 more than 10% slower |
| H5 | hermia's 12.5K is specific to one long request | v0 W2 near 12.5K and v0 W1 well below it | v0 W1 at or above v0 W2 |
| H6 | DP attention + EP4 (his production layout) prefills slower than plain TP | s4 below s1 on W1 | s4 at or above s1 |
| H7 | His `--disable-flashinfer-autotune` costs prefill: the MXFP4 MoE runs untuned tactics | s1a beats s1 by more than 10% | s1a within 5% of s1, or s1a fails to boot (the instability he cites) |

The rented cards are Server Edition (600 W), his are Max-Q (300 W), and prefill is compute-bound, so our sglang
number above 5,540 would not show a software gain by itself. The comparison that counts is sglang vs vLLM on the
same pod type.

## Matrix

Image `lmsysorg/sglang:v0.5.22-cu130` by digest. Every variant: `base_flags` (his config minus the MoE backend,
chunk size, autotune flag, and DSpark, which vary), `--revision 7872f01b`. Listed in run order, by value, so the
deadline cuts from the bottom. "no-autotune" is his `--disable-flashinfer-autotune`.

| Variant | Flags beyond `base_flags` | GSM8K | Reps | Question |
|---|---|---|---|---|
| s1-base | `--moe-runner-backend flashinfer_mxfp4 --chunked-prefill-size 4096`, no-autotune | 50 | 2 | His config without DSpark: the baseline |
| s3-ep4 | s1 + `--ep-size 4` | 20 | 2 | His question 2 (H3) |
| s2-marlin | `--moe-runner-backend marlin --chunked-prefill-size 8192 --max-prefill-tokens 16384` | 20 | 2 | vLLM's MoE kernel family on sglang (H2); autotune never runs for Marlin here (`flashinfer_autotune.py:129`) |
| s1a-autotune | s1 without no-autotune, `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1` | 20 | 2 | Tuned MXFP4, the fair baseline for s2 (H7) |
| s0-dspark | s1 + `--speculative-algorithm DSPARK --speculative-attention-mode decode` | 50 | 1 | His config as posted; does DSpark cost prefill (H4); is depth 5 clean |
| s4-dp-attn | `--chunked-prefill-size 16384 --enable-dp-attention --dp-size 4 --enable-dp-lm-head --ep-size 4`, no-autotune | 20 | 1 | His production layout (H6); 4096 per DP rank |

s1a's autotune runs at boot, before graph capture (`base_runner.py:268-274`), so it never lands in a measured rep.
By default it tunes only decode-sized buckets, and DSV4 declines the model's own prefill tuning
(`flashinfer_autotune.py:405`), so the env var adds the pass at the prefill token ceiling (`:415`). GSM8K (50 on
the reporter's two configs, 20 on the rest) keeps a broken MoE or EP path from posting a fast, meaningless number.

## Workloads (`build_prompts.py`, `prefill_bench.py`)

- Prompts: War and Peace (Project Gutenberg 2600, sha256 pinned, so both pods send the same text), instruction
  first, sized by the fully templated count: the checkpoint's `encoding/encoding_dsv4.py` (chat mode, one user
  message) through the `PreTrainedTokenizerFast` both servers load, with one fixed nonce in front, so the files
  are byte-identical on both pods. Built locally: 8,000 / 90,000 / 238,000 tokens, and 20 random nonces of the
  client's shape moved the count by -3 to +5 tokens.
- Why vllm-56771's "12k" prompt was 10,740 tokens: the book has CRLF line ends, its builder counted them, and the
  client's `read_text()` turned them into LF before sending. Checked locally: the LF counts plus 30 template tokens
  are exactly the server's 10,740 / 30,989 / 106,187. The builder now normalizes to LF. tok/s still comes only from
  the server's `usage.prompt_tokens`; the builder's count rides along as `built_tokens`.
- Every prompt starts with `Request <uuid4 hex>.`: W1's nine prompts share one text, and both engines key their
  prefix cache from the first block (sglang's radix cache in 256-token pages here, vLLM by chained block hashes), so
  the nonce leaves nothing to reuse. sglang's `Prefill batch` lines, kept in `prefill_log.txt`, show
  `#cached-token` per chunk.
- Order per variant: a 17 x 23 probe (`correct` in its row), one 8K warmup (it also absorbs any kernel compiled
  lazily on first use), then the workloads in `BENCH_WORKLOADS` order (default `w1,w2`), `BENCH_REPS` times each
  (default 2).
  - W1: 9 concurrent streamed requests of ~90,000 tokens, `max_tokens` 61 (the reporter's workload).
  - W2: one ~238,000-token request, `max_tokens` 40 (hermia's).
- One JSONL row per workload rep in `<variant>/prefill_bench.jsonl`: engine, variant, workload, rep, prompt_tokens
  (server), completion_tokens, built_tokens, wall_s (first request sent to last response received), prefill_tok_s
  (prompt_tokens / wall_s, the reporter's and hermia's method), ttft_wall_s (first request sent to last first token),
  ttft_tok_s (prompt_tokens / ttft_wall_s, without the decode tail), ttft_max_s, ok, error, sample (first 200
  chars of the first reply), and per-request TTFT. The first failed request, a dropped connection included, writes
  its row and ends the run as POST_FAIL.
- Quoting: the best rep's prefill_tok_s, with every rep and its ttft_tok_s listed beside it.
- Timeouts: 1800 s per read; the last W1 request waits behind eight 90K prefills, about 400 s at 2K tok/s.
- Dry run against a fake streaming server (local, 10-16): each rep's 9 requests arrived within 20 ms and were all in
  flight at once, 21 of 21 nonces unique, tok/s recomputed from the fake's usage (not the builder's count), a 500
  and a connection dropped mid-stream each end the run with exit 1 and an error row.

## Before quoting an sglang vs vLLM gap

The two pods are rented separately and may land on different hosts. Compare their `results/<issue>/env.txt`
(`nvidia-smi topo -m`, which must show no NV# links, and the current PCIe gen and width) and the setup log's
maximum link gen, width, and power limit; a gap between hosts that differ there is not an engine gap. hermia's vLLM env sets `NCCL_P2P_DISABLE=1` (all-reduce through
host memory); the reporter's sglang flags only disable the custom all-reduce, but the US-PA-1 host hung NCCL init
with P2P on, so every sglang variant sets it too and the two recipes match there.

## Hardware, hours, cost

4x RTX PRO 6000 Server Edition, SECURE, $9.96/hr. `harness quote` (10-17 01:40Z): stock in EU-CZ-1, EUR-IS-1,
US-NC-2; `planned_hours = 1.75` caps it at $19.17 worst case.

The first boot compiles DeepGEMM kernels for every batch size (`deep_gemm_wrapper/compile_utils.py:168-176`, which
warns "typically 10-20 mins"); later boots on the same pod reuse the cache. Estimates, minutes:

| Step | Boot | GSM8K | Workloads | Total | Running total |
|---|---|---|---|---|---|
| setup (image, 167 GB at the 0.8-1.6 GB/s seen on 10-16, prompts) | - | - | - | 10 | 10 |
| s1-base | 25 (DeepGEMM precompile) | 3 | 8 (2 reps) | 36 | 46 |
| s3-ep4 | 8 | 1.5 | 8 | 18 | 64 |
| s2-marlin | 8 | 1.5 | 8 | 18 | 82 |
| s1a-autotune | 12 (autotune) | 1.5 | 8 | 22 | 104 |
| s0-dspark | 10 (draft graphs) | 3 | 4 (1 rep) | 17 | 121 |
| s4-dp-attn | 8 | 1.5 | 4 (1 rep) | 14 | 135 |

The deadline is at 105 min, so on these estimates s1, s3, s2, and s1a fit and s0 and s4 run only if the boots come
in faster (W1 is about 2.5 min per rep at the reporter's 5.5K tok/s, W2 about 1 min). Expected cost is the whole
1.75 h, $17.43. The vLLM side (`issues/sglang-33422-vllm/`) is 1.0 h, $10.96 worst case, about $7.50 expected.
Both: $30.13 worst case, about $25 expected.

## Stop rules

- s1 fails to boot -> switch `pod.image` to `lmsysorg/sglang:v0.5.21-cu130@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`
  and `want_flashinfer=0.6.18` in `setup.sh`, rerun s1 once; if it fails again, stop.
  v0.5.21 also has `--cuda-graph-max-bs-decode` and `--enable-dp-attention`.
- `setup.sh` stops under 400 MB/s of download after 3 min or a 500 MB/s cold read -> `harness down`.
- s0 BOOT_FAIL is a result, not a reason to patch: #33407 does not fit FlashInfer 0.7.1 (above).
- The deadline ends the pod at 1.75 h whatever is left; the order above means s4 goes first, then s0. `exec` pulls
  each variant as it finishes, so a cut loses only the variant in flight.

Sequence (`harness run` ends with `down`, so the matrix goes through `exec` on a pod kept up):

```
harness budget set sglang-33422 <usd> --note <approval>
harness up sglang-33422
harness exec sglang-33422 s1-base s3-ep4 s2-marlin s1a-autotune s0-dspark s4-dp-attn
harness down sglang-33422
```

## Files

| File | Purpose |
|---|---|
| `harness.toml` | Pod, image, base flags, variants |
| `setup.sh` | Image check, prompts, weights to container disk, page-cache drop, cold-read gate |
| `build_prompts.py` | The warmup, W1, and W2 prompts, sized by the templated count |
| `prefill_bench.py` | The workloads against the live server (post_smoke) |
| `hooks.sh` | post_smoke runs `prefill_bench.py`, or `decode_bench.py` for `BENCH_KIND=decode`; post_stop keeps the scheduler's prefill lines and, with `NCCL_DEBUG=INFO`, `nccl_transport.txt` |
| `decode_bench.py` | The 11-05 decode rows: `sglang.bench_serving` per row, `decode_rows.csv` (post_smoke) |
| `nccl_probe.py` | One allreduce group over all GPUs under one P2P setting, run by setup.sh per decode arm |
| `nccl_transport.sh` | NCCL's transport lines from a log, for the probe and each decode arm |

`build_prompts.py` and `prefill_bench.py` are copied byte for byte into `issues/sglang-33422-vllm/`, since the
harness ships only the issue folder to the pod; change both copies together.

## Results (2026-10-17)

Three pods. The first (ezpsbqqlivnajk, US-NC-1) came up with a GPU at 90,892 MiB used by a process outside the
container: setup downloaded the weights, then s1's idle wait failed after 300 s ($1.49). `harness up` now refuses
any GPU at 2 GiB or more before setup (e649887), and it refused the second pod (z4c5536vfxm5bf, likely the same
host: GPU 0 at 90,892 MiB, GPU 2 at 70,462) in 47 s ($0.13); US-NC-1 is in `avoid_data_centers` since.

The third pod (3l8n2nz2zfqecf, sm120-sglang-33422-7115da) ran the matrix: US-PA-1, 4x RTX PRO 6000 Blackwell Server
Edition, driver 595.91.07, PCIe Gen5 x16 max, 600 W, container memory limit 754 GB. Topology: GPU0 on NUMA 0, GPUs
1-3 on NUMA 3 (GPU0 to the others SYS, GPU1-GPU2 PIX, the rest NODE). Setup 4.5 min (weights in 229 s, cold read
2978 MB/s). s1's first boot hung right after `sglang is using nccl==2.30.7` for 15 min, all four GPUs at 100% with
about 1 GB used: the split-NUMA NCCL P2P hang. Stopped by hand with the pod kept, every variant now sets
`NCCL_P2P_DISABLE=1` (as the vLLM recipe does), and the rerun got past NCCL init in 30 s.

tok/s is server `usage.prompt_tokens` / wall s; TTFT tok/s in brackets. Every rep, every workload `ok`, the 17 x 23
probe correct on every variant, `#cached-token` 0 on every W1/W2 chunk.

| Variant | Boot s | GSM8K | W1 9 x 90K rep 1 / 2 | W2 1 x 238K rep 1 / 2 |
|---|---|---|---|---|
| s1-base | 474 (DeepGEMM precompile) | 0.98 (50) | 8,979 / 8,983 (9,091) | 7,697 / 7,728 (7,836) |
| s3-ep4 | 101 | 0.95 (20) | 9,264 / 9,272 (9,386) | 7,889 / 7,937 (8,051) |
| s2-marlin | 277 | 1.00 (20) | 8,543 / 8,541 (8,630) | 7,374 / 7,397 (7,483) |
| s1a-autotune | BOOT_FAIL after 1,327 | - | - | - |
| s0-dspark | 167 | 0.98 (50), accept 4.30 | 8,897 (9,014) | 7,680 (7,748) |
| s4-dp-attn | 91 | 0.95 (20) | 11,756 (11,998) | 6,188 (6,291) |
| vLLM v0-hermia (US-NE-1, `issues/sglang-33422-vllm/`) | 261 | 1.00 (50) | 8,170 / 8,171 (8,235) | 7,129 / 7,150 (7,227) |

- H1 confirmed: s1 W1 is 1.10x vLLM v0 W1 (W2 1.08x). Same pod type, different hosts; both split NUMA, both
  `NCCL_P2P_DISABLE=1`.
- H2 killed: s2 (Marlin, 8192 chunks) is 4.9% below s1 on W1 and 4.3% on W2. s1a, its planned fair baseline, did not
  boot, so the comparison is against untuned MXFP4, which still wins.
- H3 killed: EP4 is +3.2% (W1) and +2.7% (W2), under the 10% bar. It works with `flashinfer_mxfp4`.
- H4 confirmed: s0 is within 1% of s1 on both shapes (one rep). Depth-5 DSpark is clean on stock v0.5.22, no #33407:
  GSM8K 0.98, accept length 4.30.
- H5 killed: v0 W2 is 7.1K, not 12.5K, and v0 W1 (8.2K) is above v0 W2.
- H6 killed on W1, holds on W2: s4 is +30.9% on W1 (11,756) and -19.9% on W2. W1's nine requests spread over four
  attention DP ranks; W2's one request runs its attention on one rank.
- H7 inconclusive. The default autotune (decode buckets) finished in 11 s. The `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`
  pass ("extra EXTEND pass at 4096 tokens (4096 seqs x 1 tokens)") ran from 07:04:53 to 07:25:10, then
  `RuntimeError: Triton Error [CUDA]: out of memory` in `expand_prefill_causally_triton`
  (`deepseek_v4_backend.py:4110`) from `maybe_flashinfer_autotune_extend` (`flashinfer_autotune.py:467`), with
  8.96 GB free after the KV pool and `--max-running-requests 64`. Only TP0 logged "per-rank caches disagree,
  discarding them and tuning from scratch". Not chased: the env var is ours, not the reporter's.
- Position cost (s1, W2): sglang's per-chunk input throughput went from about 9.8K tok/s at the start of the 238K
  prompt to 7.9K at the middle and 6.7K at the end, on every variant within a few percent. The reporter's ~2K past
  several hundred K (v0.5.16, Max-Q) is beyond what W2 reaches.
- Host memory peak (s1): 178.3 GiB of 754 GB.

Cost: $16.87 of the $20 budget ($1.49 + $0.13 + $15.25, ledger estimates); with the vLLM side, $21.39 for the
issue. Posted on sglang#33422 (issuecomment-6231294981).

## Decode pair: P2P on vs off (planned 2026-11-05)

### Question

The reporter plans four arms on his 8x RTX PRO 6000 Max-Q box (325 W limit, 300 W default, four NUMA domains, every
GPU in production) in a maintenance window, and offered a two-arm subset first (issuecomment-6650523367): GPUs 0-3,
`--tp-size 4 --ep-size 4`, DSpark off, NCCL P2P on vs off (`NCCL_P2P_DISABLE=1`), on `lmsysorg/sglang:dev-dsv41`
with DeepSeek-V4.1-Flash. His method: `NCCL_DEBUG=INFO` with the per-channel transport lines (P2P vs SHM) quoted, so
"P2P on" is evidence; W1 = 9 x 90K and W2 = 1 x 238K at c=1 and c=8; `sglang.bench_serving` with a fixed-length
random dataset, greedy, N >= 8 per row; per row TTFT, median ITL beside mean TPOT, and the per-request decode tok/s
distribution, since at 9 x 90K and c=8 a request that finished prefill waits on the others' prefill chunks, so mean
TPOT carries prefill time. We said we would run the same pair on a rented box whose `nvidia-smi topo -p2p r` reads
OK on every pair and post the rows before his window. The 10-17 run could only run P2P off (the split-NUMA hang
above).

Our rows use this kit's image and checkpoint, DeepSeek-V4-Flash-0731 on `lmsysorg/sglang:v0.5.22-cu130`, not his
V4.1-Flash on `dev-dsv41`. Both sides agreed the rows pair only with each other: P2P on against P2P off on one box,
never against his numbers.

### Arms

Every arm is s3-ep4: `base_flags` (his config, `--disable-custom-all-reduce` included) plus `--moe-runner-backend
flashinfer_mxfp4 --chunked-prefill-size 4096 --disable-flashinfer-autotune --ep-size 4`, DSpark off, GSM8K 20,
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH`, and `BENCH_KIND=decode`. Only NCCL's P2P setting differs.

| Variant | NCCL setting | Runs |
|---|---|---|
| d1-ep4-p2p-on | `NCCL_P2P_DISABLE=0`, which NCCL reads as unset: its defaults | always, first |
| d2-ep4-p2p-off | `NCCL_P2P_DISABLE=1` | always, second |
| d3-ep4-p2p-sys | `NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS` | only by the rule under Hours and cost |

Why d3: the image's NCCL is 2.30.7 (`sglang is using nccl==2.30.7` in the 10-17 logs), and NCCL 2.30.7 uses P2P by
default only up to PXB, between GPUs behind a common PCIe switch (`src/graph/paths.cc:356`; the AMD exception at
:361 needs two GPUs or fewer, and `DEV` counts GPUs). Between GPUs that meet at a CPU root complex (PHB) or across
NUMA domains (SYS) it picks SHM whether `NCCL_P2P_DISABLE` is set or not; `ncclGetLevel` maps 1 to LOC and leaves 0
unset (:244-279). On such a box d1 runs d2's transport, and so would his P2P-on arm if GPUs 0-3 meet at the CPU.
`NCCL_P2P_LEVEL=SYS` allows P2P on every pair whose NVML P2P read and write status is OK (:383-391); at SYS NCCL
logs nothing when write is not (:400), so setup prints `nvidia-smi topo -p2p w` beside the read matrix. NCCL reads
`NCCL_P2P_DISABLE` before the level (:247-253), so d3 sets it to 0 against an inherited 1. sglang sets
`NCCL_CUMEM_ENABLE=0` (`entrypoints/engine.py:1875` at v0.5.22), so P2P lines read `via P2P/IPC`, SHM lines
`via SHM/direct`.

`--disable-custom-all-reduce` stays on every arm: it is his flag, and the pair shares it. sglang's custom allreduce,
which reads peers' buffers through CUDA IPC, is a separate P2P lever this pair does not test.

### Hypotheses

| # | Hypothesis | Confirmed by | Killed by |
|---|---|---|---|
| H8 | NCCL's default follows the PXB rule here: P2P inside a PCIe switch, SHM between GPUs that meet at the CPU | d1's `nccl_transport.txt` pairs match `nvidia-smi topo -m` that way | `via P2P` under defaults on a PHB or SYS pair, or `via SHM` on a PIX or PXB pair |
| H9 | Where NCCL uses P2P it cuts decode ITL at TP4: with TP4 attention and EP4 MoE each of the 43 layers runs at least two allreduces (attention output, MoE combine), so at c=1 a step runs at least 86 of 8 KiB (one token at hidden 4096, bf16), where SHM's copy through host memory costs latency, not bandwidth | the P2P arm (d1 when it reads P2P, else d3) at least 5% under d2 in median ITL at c=1 on W1 and W2 | within 2% of d2 |
| H10 | When d1 reads SHM on every pair, d1 and d2 are the same run | d1 within 2% of d2 on median ITL and TTFT | d1 more than 5% off d2 |
| H11 | At c=8, mean TPOT carries the other requests' prefill, if the scheduler runs prefill chunks ahead of decode (no mixed chunk) | on w1-c8 and w2-c8, mean TPOT above 2x median ITL and a wider per-request decode tok/s spread than at c=1, on every arm | mean TPOT within 20% of median ITL |

Setup's probe (below) times one allreduce at 8 KiB and 64 KiB under each setting, so H9 has a per-call number
beside the per-step one: 86 times the per-call gap between transports is about the least ITL gap to expect (launch
overhead, which the server's CUDA graphs remove, is common to both transports and drops out of the gap).

### Rows (`decode_bench.py`)

- Client: `python3 -m sglang.bench_serving --backend sglang` against the live server, after GSM8K 20 (18 s on s3-ep4;
  a P2P path that corrupts data would run at full speed, so the rows need it).
- Rows, in order: w1-c1 (9 x 90K, c=1), w1-c8 (9 x 90K, c=8), w2-c1 (8 x 238K, c=1), w2-c8 (8 x 238K, c=8). W1 keeps
  his 9 prompts; W2 sends 8 for N >= 8.
- KV: the 10-17 boots allocated 6,873,088 full tokens and 687,104 SWA tokens (`DSV4 pool sizes` in s3-ep4's
  server.log, 96 GB cards, `--mem-fraction-static 0.90`). w2-c8's 8 x 238K is 1.90M tokens, so all 8 run at once and
  nothing is cut down. The Vast cards are 96 GB too; d1's log has its own pool line to check.
- `--dataset-name random` (ShareGPT text repeated to length; setup.sh downloads it) with `--random-range-ratio 1`.
  sglang draws each length from [ratio x target, target] (`benchmark/datasets/common.py:56-64` at v0.5.22), so 1
  fixes the lengths and 0, the ratio that fixes vLLM's, spreads them from 1 token to the target. His "range ratio 0"
  in sglang's client would send random lengths; worth telling him before his window.
- Greedy and `ignore_eos` are the sglang backend's defaults (`benchmark/serving.py:695-697`, `--temperature` default
  0 at :2514), so every request decodes all 512 tokens.
- Output 512: 511 gaps per request (over 4,000 per row) hold the median and p99 ITL steady, and at the ~90 tok/s per
  request the 10-17 decode lines show at 2-7 running requests, a request decodes for about 6 s. 1024 would add about
  2 min per arm and resolve nothing the gaps do not.
- `--seed 1` on every row, so every arm sends the same prompts. `--flush-cache`: the client's one warmup sends the
  row's first prompt (`serving.py:1462-1478`), which would otherwise serve that request's prefill from the radix
  cache. `--cache-report` records cached tokens per request, which should read 0.
- `--tokenizer` is the pinned revision's snapshot, which setup.sh links; the model id alone would load main's.
- Out: `<variant>/decode/<row>.jsonl` (bench_serving's record with `--output-details`: per-request TTFT, inter-token
  gaps, server prompt tokens) and `<row>.txt`; `<variant>/decode_rows.csv`: variant, shape, input_len, concurrency,
  prompts, completed, input_tokens (server), output_tokens, cached_tokens, TTFT mean and median, mean TPOT, median
  and p99 ITL, per-request decode tok/s min, median, and max (a request's gaps over their sum), duration. A row that
  completes fewer requests than it sent writes its line, then ends the variant as POST_FAIL.
- NCCL's lines: NCCL 2.30.7 writes INFO to stdout without flushing (`src/debug.cc:376`) and the ranks end on
  SIGKILL, so in server.log each rank would lose its last lines and the ranks interleave mid-line. `pre_launch`
  sets `NCCL_DEBUG_FILE=<variant>/nccl.%h.%p.log`, which NCCL line-buffers (`debug.cc:240`); post_stop joins the
  per-rank files (kept) into `<variant>/nccl.log` and condenses that into `<variant>/nccl_transport.txt`
  (`nccl_transport.sh`): NCCL version, the NCCL env it read, each communicator's search result (the path type NCCL
  saw), connections per rank pair and transport, and every channel line deduplicated across processes and
  communicators.
- Setup's NCCL probe (`nccl_probe.py`, before the weights): each arm's setting once over all four GPUs with the same
  debug env, `NCCL_CUMEM_ENABLE=0`, and per-rank debug files, printing allreduce times at 8 KiB, 64 KiB, and 32 MiB
  and the transport summary into the setup log; logs in `results/sglang-33422/nccl-probe/`. All three run before any
  verdict, so the log tells the cases apart. Then: any probe that failed -> every GPU must drop under 2 GiB within
  60 s (`harness/pod/wait_gpu_idle.sh`), since a rank killed in a hang can hold its GPU and fail d1's idle wait after
  the download; d1's or d2's probe failed or hung -> setup ends, before the 167 GB download; d3's -> d3 is ruled out.
  On 10-17 P2P on hung a boot for 15 min, and a d1 boot would wait out `boot_timeout_s` (30 min). The base image's
  `NCCL_VERSION=2.28.3-1` env label is not the NCCL torch loads, so setup leaves it out; the probe's own log names
  the version.
- Setup's last lines: per probe arm, the transport each GPU pair used (P2P/IPC or SHM/direct), minutes spent and
  minutes left to `HARNESS_DEADLINE_EPOCH`, and the d3 verdict with the exec command it implies.
- Dry run (local, 11-05): `decode_bench.py` against a fake bench_serving that writes v0.5.22's record shape took the
  argv above, wrote four CSV rows, and on a row one request short wrote its line and exited 1. `nccl_transport.sh`
  against hand-written NCCL 2.30.7 lines (P2P, SHM, env, pattern, disabled-pair) and against a log with none.

### Box

Vast only, 4x RTX PRO 6000: Max-Q first (his card), then Server, then Workstation Edition, each capped at $10/hr
with reliability 0.95, and `require_p2p = true`. RunPod is out of this run: its one 4x host that ran on 10-17 hung
NCCL with P2P on, and RunPod picks the host. `avoid_data_centers` stays for a RunPod rerun of the 10-17 matrix; the
harness accepts it beside Vast-only choices (`spec.rs:619-625`) and no Vast choice reads it. `harness quote`
(11-05 20:34Z): Max-Q on machine 57045 (Lithuania), $5.94/hr, reliability 0.991, 2,885 Mbps down, downloads
$2.67/TB; fallbacks Server Edition at $6.90-7.60 and Workstation at $6.06-7.97. Stock moves within minutes: the
21:00Z quote had no free Max-Q, and its first choice was a Server Edition on machine 153928 (Minnesota), $6.05/hr,
900 Mbps down. The search keeps only hosts whose `cuda_max_good` is at least `min_cuda` 13.0.

On Vast the box has one 280 GB disk (`container_disk_gb` + `volume_gb`): the image, the weights (setup.sh keeps them
in `/root/hf-hub` as before), ShareGPT, the prompts, and logs. `/workspace` is a plain directory there, and nothing
in the kit reads RunPod's env. `needs_hf_token = false`: the checkpoint is not gated (HF API at the pinned revision)
and the 10-17 pods downloaded it without a token. `expected_download_gb = 190`: image 15.6 GB compressed, weights
166.9 GB, ShareGPT 0.67 GB, apt. setup.sh's download floor is 60 MB/s at 3 min (140 on RunPod), and its ceiling 46
min, checked every 30 s: 166.9 GB at 60 MB/s takes 46 min, the most that still leaves d1 and d2 inside the deadline
on a Max-Q, and the harness would otherwise wait on setup until the deadline. `/dev/shm` under 2 GiB (Docker's
default is 64 MB, and the harness's Vast create asks for no size) ends setup before the probe.

### Hours and cost

Minutes on a Max-Q downloading at ~100 MB/s (800 Mbps), the rate this plan assumes for these hosts; the quoted
host lists 2,885 Mbps, which would make the download about 10 min. The image is 15.62 GB in 69 layers, so the pull
and its unpacking take several minutes before SSH answers. Rows assume the Max-Q prefills at 0.65x the 10-17 Server
Edition (W1 6.0K, W2 5.1K tok/s) and decodes at ~80 tok/s, each row with its warmup: w1-c1 3.6, w1-c8 2.9, w2-c1
8.0, w2-c8 7.2, and 30-60 s per row for bench_serving to load ShareGPT (0.67 GB of JSON) and the tokenizer again,
about 25 in all; at the 10-17 rates (Server or Workstation Edition) about 18.

| Step | Minutes | Running total |
|---|---|---|
| up: create, 15.62 GB / 69-layer image pull and unpack, SSH, apt, watchdog, topology and P2P gates, kit copy | 12 | 12 |
| setup: image check, P2P matrices, `/dev/shm`, NCCL probe (3 settings), prompts, ShareGPT | 5 | 17 |
| setup: 166.9 GB at 100 MB/s, cache drop, cold read | 30 | 47 |
| d1: boot with the DeepGEMM precompile (474 s for s1 on 10-17) 10, GSM8K 1, rows 25 | 36 | 83 |
| d2: boot (101 s for s3-ep4 on 10-17) 3, GSM8K 1, rows 25 | 29 | 112 |
| d3, by the rule below | 29 | 141 |

`planned_hours = 2.25` (135 min): d1 and d2 with 23 min to spare. d1, d2, and d3 take about 95 min after setup on a
Max-Q, so d3 fits only when setup ends early: on the quoted host's listed speed setup ends near 27 min, with 108
left. Worst case at the $10 cap: 2.25 h x $10 x 1.10 + 190 GB at $20/TB = $24.75 + $3.80 = $28.55. On the quoted
box: $14.71 + $0.51 = $15.22 worst case; expected about $11.60 for d1 and d2 (1.9 h), and $12.60 with d3 after a
fast setup (122 min). $16.93 of the
$20 approved on 10-16 is spent, so a $46 total covers the worst case at the cap.

The d3 rule: run d3 when d1's probe shows SHM on any GPU pair, d3's probe passed, and setup ends with at least 95 min
left to the deadline; otherwise d1 and d2 only. Setup's last lines (`results/sglang-33422/jobs/setup.log`) give all
three and the verdict. d3 then goes in the same exec, last, so a deadline cut loses only d3.

```
# The pod's pulls merge file by file and would overwrite the 10-17 env.txt, jobs/setup.log, and verdicts.jsonl.
cp -R results/sglang-33422 results/sglang-33422-1017
harness budget set sglang-33422 46 --note <approval>
harness up sglang-33422
# d3 rule met:
harness exec sglang-33422 d1-ep4-p2p-on d2-ep4-p2p-off d3-ep4-p2p-sys
# otherwise:
harness exec sglang-33422 d1-ep4-p2p-on d2-ep4-p2p-off
harness down sglang-33422
```

`harness run sglang-33422 d1-ep4-p2p-on d2-ep4-p2p-off` does up, exec, and down in one, with no room for the d3 call.
Never leave the names out: `run` and `exec` with none run every variant in file order, d1, d2, d3, then the six
10-17 variants, until the 2.25 h deadline ends the box, all of it billed.

### Stop rules

- `/dev/shm` under 2 GiB -> setup stops before the probe; `harness down`, add the machine to `avoid_machines`, rent
  again.
- The probe fails or hangs under d1's or d2's setting, or a failed probe leaves a GPU busy -> setup stops after all
  three probes; the same.
- Download under 60 MB/s at 3 min, or still running at 46 min -> setup stops; `harness down`.
- d1 BOOT_FAIL after a clean probe -> still run d2, and report d1's fault beside the probe's lines.
- d3 only by the rule above.

### Reporting

Per arm, the `decode_rows.csv` rows as one table (shape, c, N, TTFT mean and median, mean TPOT, median and p99 ITL,
per-request decode tok/s min, median, and max, output tokens), with the arm's connections-per-pair block from
`nccl_transport.txt` quoted beside it (the channel lines below it on request), the probe's allreduce times per
setting, and the box: card, power limit and default, `nvidia-smi topo -m` from env.txt, the `topo -p2p r` matrix
from `results/sglang-33422/p2p-<pod>.txt`, driver, NCCL 2.30.7. Labeled as V4-Flash-0731 on v0.5.22, paired within
this box only. The 10-17 variants stay in harness.toml, with P2P off, and are not part of this run; the one shared
change is `boot_timeout_s` 2400 -> 1800, which still covers a cold read plus the DeepGEMM precompile, and s1a needs
2400 back for its 20 min autotune pass.

Not verifiable without the box: the size of `/dev/shm` (the probe exercises SHM under d1's and d2's settings);
whether P2P works past `-p2p r` (the probe again); the decode rates, so every row time above; whether H11's
scheduler behavior holds; image pull and first-boot time on the host's CPU.

### Results (2026-11-05)

Box: 4x RTX PRO 6000 Server Edition, Vast (New York), all four GPUs on NUMA node 1 and connected `NODE` (separate
PCIe root ports, one socket), `topo -p2p r` OK on every pair. DeepSeek-V4-Flash-0731 on `lmsysorg/sglang:v0.5.22-cu130`,
NCCL 2.30.7, TP4 + EP4, flashinfer_mxfp4, DSpark off, `--disable-custom-all-reduce`. Setup 23 min (167 GB at about
120 MB/s), all three arms OK, GSM8K 19/20 on each. $9.03 billed.

Transport, from each arm's `nccl_transport.txt` (8 connections per rank pair, all pairs alike):

- d1, NCCL defaults: `via SHM/direct` on every pair. NCCL's default level is PXB (`paths.cc:356`), and these pairs
  are `NODE`, so "P2P on" is SHM.
- d2, `NCCL_P2P_DISABLE=1`: `via SHM/direct` on every pair.
- d3, `NCCL_P2P_LEVEL=SYS`: `via P2P/IPC` on every pair.

Setup's probe (torch all-reduce over the 4 GPUs, eager): 32 MiB, the size of a 4096-token prefill chunk, 2901 us
under d1, 2910 us under d2, 1318 us under d3.

One rep per arm; N = 9 (W1) or 8 (W2) requests per row, output 512, fixed lengths (`--random-range-ratio 1`):

| Arm | Shape | c | TTFT mean (ms) | Mean TPOT (ms) | Median ITL (ms) | p99 ITL (ms) | Per-request decode tok/s min / median / max |
|---|---|---|---|---|---|---|---|
| d1 P2P "on" (SHM) | W1 9 x 90K | 1 | 9,910 | 10.76 | 10.76 | 10.92 | 92.9 / 92.9 / 93.0 |
| d2 P2P off (SHM) | W1 9 x 90K | 1 | 9,917 | 10.76 | 10.76 | 10.94 | 92.8 / 92.9 / 93.0 |
| d3 P2P SYS (P2P) | W1 9 x 90K | 1 | **7,048** | 10.50 | **10.50** | 10.67 | 95.2 / 95.2 / 95.3 |
| d1 | W1 9 x 90K | 8 | 40,516 | 75.46 | 17.72 | 18.47 | 6.6 / 13.6 / 93.0 |
| d2 | W1 9 x 90K | 8 | 40,359 | 75.33 | 17.71 | 18.34 | 6.7 / 13.6 / 92.9 |
| d3 | W1 9 x 90K | 8 | **28,516** | **57.50** | **17.18** | 17.78 | 9.1 / 17.7 / 95.3 |
| d1 | W2 8 x 238K | 1 | 30,115 | 11.05 | 11.05 | 11.19 | 90.4 / 90.5 / 90.6 |
| d2 | W2 8 x 238K | 1 | 30,083 | 11.06 | 11.06 | 11.18 | 90.4 / 90.4 / 90.5 |
| d3 | W2 8 x 238K | 1 | **22,561** | 10.79 | **10.79** | 10.91 | 92.6 / 92.7 / 92.8 |
| d1 | W2 8 x 238K | 8 | 133,884 | 219.91 | 18.61 | 19.05 | 2.4 / 4.6 / 53.6 |
| d2 | W2 8 x 238K | 8 | 133,706 | 220.16 | 18.63 | 19.10 | 2.4 / 4.6 / 53.5 |
| d3 | W2 8 x 238K | 8 | **99,898** | **167.79** | **18.13** | 18.59 | 3.1 / 6.1 / 55.0 |

- d1 and d2 agree within 0.4% on every column, which is both the noise floor and the evidence that NCCL's "P2P on"
  was the same path as "off" here (H10).
- Real P2P (d3) cuts TTFT 25-30% on every row: prefill all-reduces are large, and the probe's 32 MiB all-reduce is
  2.2x faster over P2P than over SHM.
- Pure decode moves 2-3% at TP4 (median ITL 10.76 -> 10.50 ms at W1 c=1, 18.61 -> 18.13 ms at W2 c=8). At four ranks
  the decode all-reduces are a small share of a 10.5 ms step.
- Mean TPOT at c=8 falls 24% while median ITL falls 3%: the c=8 TPOT carries the other requests' prefill (H11), so
  most of that 24% is the prefill speedup, not decode. Median ITL is the decode number.
