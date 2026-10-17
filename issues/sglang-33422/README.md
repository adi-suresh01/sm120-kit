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
| H2 | The gap is the MoE GEMM (FlashInfer CUTLASS MXFP4 vs Marlin) | s2 beats s1 by more than 15% on W1 and W2 | s2 within 10% of s1 |
| H3 | EP4 speeds MoE prefill on PCIe (question 2) | s3 beats s1 by more than 10% | s3 at or below s1 |
| H4 | DSpark does not change prefill throughput | s0 within 5% of s1 on W1 | s0 more than 10% slower |
| H5 | hermia's 12.5K is specific to one long request | v0 W2 near 12.5K and v0 W1 well below it | v0 W1 at or above v0 W2 |
| H6 | DP attention + EP4 (his production layout) prefills slower than plain TP | s4 below s1 on W1 | s4 at or above s1 |

The rented cards are Server Edition (600 W), his are Max-Q (300 W), and prefill is compute-bound, so our sglang
number above 5,540 would not show a software gain by itself. The comparison that counts is sglang vs vLLM on the
same pod type.

## Matrix

Image `lmsysorg/sglang:v0.5.22-cu130` by digest. Every variant: `base_flags` (his config minus the MoE backend,
chunk size, and DSpark, which vary), `--revision 7872f01b`.

| Variant | Flags beyond `base_flags` | Question | Run when |
|---|---|---|---|
| s1-base | `--moe-runner-backend flashinfer_mxfp4 --chunked-prefill-size 4096`; GSM8K 50 | His config without DSpark: the baseline | always, first |
| s0-dspark | s1 + `--speculative-algorithm DSPARK --speculative-attention-mode decode`; GSM8K 50 | His config as posted; does DSpark cost prefill (H4); is depth 5 clean | always |
| s2-marlin | `--moe-runner-backend marlin --chunked-prefill-size 8192 --max-prefill-tokens 16384` | vLLM's MoE kernel family on sglang (H2) | always |
| s3-ep4 | s1 + `--ep-size 4` | His question 2 (H3) | always |
| s4-dp-attn | `--chunked-prefill-size 16384 --enable-dp-attention --dp-size 4 --enable-dp-lm-head --ep-size 4` | His production layout (H6); 4096 per DP rank | only if time is left |

## Workloads (`build_prompts.py`, `prefill_bench.py`)

- Prompts: War and Peace (Project Gutenberg 2600, sha256 pinned, so both pods send the same text), instruction
  first, sized by the fully templated count: the checkpoint's `encoding/encoding_dsv4.py` (chat mode, one user
  message) through the `PreTrainedTokenizerFast` both servers load, with a nonce in front. Built locally: 8,000 /
  90,000 / 238,000 tokens, and 20 fresh nonces each moved the count by -3 to +5 tokens.
- Why vllm-56771's "12k" prompt was 10,740 tokens: the book has CRLF line ends, its builder counted them, and the
  client's `read_text()` turned them into LF before sending. Checked locally: the LF counts plus 30 template tokens
  are exactly the server's 10,740 / 30,989 / 106,187. The builder now normalizes to LF. tok/s still comes only from
  the server's `usage.prompt_tokens`; the builder's count rides along as `built_tokens`.
- Every prompt starts with `Request <uuid4 hex>.`: W1's nine prompts share one text, and both engines key their
  prefix cache from the first block (sglang's radix cache in 256-token pages here, vLLM by chained block hashes), so
  the nonce leaves nothing to reuse. sglang's `Prefill batch` lines, kept in `prefill_log.txt`, show
  `#cached-token` per chunk.
- Order per variant: a 17 x 23 probe (`correct` in its row), one 8K warmup, W1 twice, W2 twice.
  - W1: 9 concurrent streamed requests of ~90,000 tokens, `max_tokens` 61 (the reporter's workload).
  - W2: one ~238,000-token request, `max_tokens` 40 (hermia's).
- One JSONL row per workload rep in `<variant>/prefill_bench.jsonl`: engine, variant, workload, rep, prompt_tokens
  (server), completion_tokens, built_tokens, wall_s (first request sent to last response received), prefill_tok_s,
  ttft_max_s, ok, error, sample (first 200 chars of the first reply), and per-request TTFT. The first failed request
  ends the run as POST_FAIL.
- `BENCH_REPS` and `BENCH_WORKLOADS` in a variant's env trim the run if time is short.
- Timeouts: 1800 s per read; the last W1 request waits behind eight 90K prefills, about 400 s at 2K tok/s.
- Dry run against a fake streaming server (local, 10-16): each rep's 9 requests arrived within 20 ms and were all in
  flight at once, 21 of 21 nonces unique, tok/s recomputed from the fake's usage (not the builder's count), a 500 ends the run
  with exit 1.

## Hardware, hours, cost

4x RTX PRO 6000 Server Edition, SECURE, $9.96/hr. `harness quote` (10-17 01:15Z): stock in EU-CZ-1, EUR-IS-1,
US-NC-2; `planned_hours = 1.5` caps it at $16.43 worst case. Expected: setup about 10 min (image, 167 GB of weights
at the 0.8-1.6 GB/s seen on 10-16), then s1 about 20 min, s0 about 22, s2 and s3 about 15 each: about 85 min, $14.
s4 runs only inside the deadline. The vLLM side (`issues/sglang-33422-vllm/`) is 1.0 h, $10.96 worst case, about
$7.50 expected. Both: $27.39 worst case, about $21 expected.

## Stop rules

- s1 fails to boot -> switch `pod.image` to `lmsysorg/sglang:v0.5.21-cu130@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`
  and `want_flashinfer=0.6.18` in `setup.sh`, rerun s1 once; if it fails again, stop.
  v0.5.21 also has `--cuda-graph-max-bs-decode` and `--enable-dp-attention`.
- `setup.sh` stops under 400 MB/s of download after 3 min or a 500 MB/s cold read -> `harness down`.
- s0 BOOT_FAIL is a result, not a reason to patch: #33407 does not fit FlashInfer 0.7.1 (above).
- The deadline ends the pod at 1.5 h whatever is left; s4 is first to go.

Sequence: approval, `harness budget set`, `harness run sglang-33422 s1-base s0-dspark s2-marlin s3-ep4`, then
`harness exec sglang-33422 s4-dp-attn` only if the pod has 20 minutes left.

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
