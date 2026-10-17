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
maximum link gen, width, and power limit; a gap between hosts that differ there is not an engine gap. The recipes
also differ in one host-facing setting by design: hermia's vLLM env sets `NCCL_P2P_DISABLE=1` (all-reduce through
host memory), and the reporter's sglang flags only disable the custom all-reduce, so NCCL may still use P2P. State
that next to any gap quoted.

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
| `hooks.sh` | post_smoke runs `prefill_bench.py`; post_stop keeps the scheduler's prefill lines |

`build_prompts.py` and `prefill_bench.py` are copied byte for byte into `issues/sglang-33422-vllm/`, since the
harness ships only the issue folder to the pod; change both copies together.

## Results

None yet.
