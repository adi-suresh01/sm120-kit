# runpod-workers/worker-vllm: the startup OOM relaunch on an RTX 5090

RunPod's official serverless vLLM worker. PR #346 (merged 2026-09-25, `16d63772`) relaunches vLLM once after a
startup OOM with `ENFORCE_EAGER=true` and `MAX_NUM_BATCHED_TOKENS=8192`. The relaunch is logged as a warning and
nothing in a response says it happened. The worker moved to vLLM v0.32.0 on 2026-10-18 (#357). Its Hub listing
offers only `ADA_80_PRO,AMPERE_80`, and its Hub test runs on an A40.

Questions:

1. Which common model and context settings send the worker down that OOM path on a 32 GB RTX 5090, with the
   worker's own defaults, and does the relaunch then serve?
2. What does the silent relaunch cost: eager mode plus 8192 batched tokens, against keeping CUDA graphs and freeing
   the same memory another way (here `MAX_NUM_SEQS=32`)?

The 96 GB RTX PRO 6000 half is `issues/runpod-worker-vllm-l70b` (same design, Llama-3.3-70B FP8).

## What the worker does (runpod-workers/worker-vllm @ `71d503e6`, release v2.30.1, 2026-10-21)

- `src/main.py:94-115` runs `vllm serve --host 127.0.0.1 --port 8000` plus `build_vllm_args()`.
  `src/args_builder.py:314-352` turns every env var named like a flag on its allowlist (`MAX_MODEL_LEN` ->
  `--max-model-len`) into that flag, booleans into `--flag` / `--no-flag`, `MODEL_NAME` into `--model`, then appends
  `VLLM_EXTRA_ARGS`. The code sets no engine defaults of its own.
- The defaults come from the Hub deploy form, `.runpod/hub.json`: `MAX_MODEL_LEN=auto` (line 185),
  `GPU_MEMORY_UTILIZATION=0.95` (580), `MAX_NUM_SEQS=256` (285), `ENFORCE_EAGER=false` (601),
  `ENABLE_PREFIX_CACHING=false` (255), `MAX_CONCURRENCY=30` (680), `PYTORCH_ALLOC_CONF=expandable_segments:True`
  (792), preset model `deepseek-ai/deepseek-r1-distill-llama-8b` (13). `README.md` documents the bare image instead:
  no `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION` 0.9, so vLLM's own defaults apply there. That a Hub deploy starts from
  every form default is an assumption from RunPod's docs: `runpodctl serverless create --hub-id` takes `--env` values
  that "override the Hub release defaults".
- The relaunch, `src/main.py:181-221` and `292-304`: when `startup_errors.memory_shortfall()`
  (`src/startup_errors.py:88-98`) finds `torch.OutOfMemoryError` / `CUDA out of memory`, `No available memory for
  the cache blocks`, or `larger than the available KV cache memory` in the last 400 lines of vLLM's output, the
  worker sets `ENFORCE_EAGER=true`, sets `MAX_NUM_BATCHED_TOKENS=8192` when it is unset, 0, or above 8192, and
  starts vLLM again, once. If that fails too, every job is answered with `classify(..., oom_retried=True)`'s
  message. A failure `classify()` does not recognise ends in `sys.exit(1)` (`main.py:305-311`): the platform
  restarts the worker on the same config.
- `Dockerfile:4-5` builds `FROM vllm/vllm-openai:v0.32.0` and `Dockerfile:25` asserts the vLLM version, so the
  engine here is exactly that image's.

What it depends on in vLLM v0.32.0 (tag `5df63a72`):

- `vllm/utils/argparse_utils.py:76-93`: `--max-model-len auto` is -1. `vllm/v1/core/kv_cache_utils.py:2414-2482`
  then binary-searches the longest context the KV cache holds and logs `Auto-fit max_model_len: reduced from A to
  B`. So the Hub default never fails on context length; it boots with a shorter one. When not even one token fits,
  it raises `Cannot auto-fit max_model_len: not enough GPU memory available` (2453-2457), which
  `memory_shortfall()` does not match. Under `auto` the auto-fit (2635-2636) runs before the capacity check that
  prints `No available memory for the cache blocks` (2638-2647, raised at 1009-1017 for a budget of 0 or less), so
  even a negative KV budget ends in the unmatched message.
- An explicit length that does not fit raises `No available memory for the cache blocks` or `... larger than the
  available KV cache memory ... estimated maximum model length is N` (`kv_cache_utils.py:1004-1040`), which it does.
- `vllm/v1/worker/gpu_worker.py:695-745`: KV budget = utilization x total - (weights + non-torch + peak activation)
  - the CUDA graph estimate.
- `vllm/engine/arg_utils.py:2923-2955`: `vllm serve` defaults to 2048 batched tokens and 256 sequences on cards
  under 70 GiB, 8192 and 1024 at 70 GiB and up (A100 excepted), 16384 only at 160 GiB and up. On a 5090 the
  relaunch's 8192 is 4x vLLM's own default: it frees the graph memory and spends some of it on a bigger prefill
  chunk. The message after a failed relaunch still says the budget was "reduced to 8192 where it was higher or
  unset".
- `vllm/config/vllm.py:2354-2364`: graphs are captured up to min(2 x max_num_seqs, 512) tokens, so
  `MAX_NUM_SEQS=32` captures up to 64. The worker never has more than `MAX_CONCURRENCY=30` jobs in flight.

## Setup

- 1x RTX 5090 (32 GB), secure, community as fallback. A real card, not a ballast on a PRO 6000: vLLM picks the batch
  defaults above from the card's total memory, and that default is what the relaunch's 8192 is measured against.
- `vllm/vllm-openai:v0.32.0@sha256:befccf0e...5b34`, the worker's base image (pushed 2026-10-18).
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` @ `6a6f4aa4`, the Hub listing's preset model: Llama-3.1-8B architecture,
  14.96 GiB of BF16 weights, 131072-token context, 128 KiB of KV cache per token, not gated.
- Server flags: exactly what `args_builder.py` builds from the Hub env defaults and the Dockerfile's ENV (rerun with
  the worker's own `build_vllm_args` for this review), `--model` last as the worker passes it, plus our `--revision`.
  Variant env: the Hub's `PYTORCH_ALLOC_CONF` and `VLLM_USE_DEEP_GEMM`, the Dockerfile's `TOKENIZERS_PARALLELISM`.
  Not mirrored, none of which changes what the engine sizes: the Dockerfile's cache paths under `/runpod-volume`
  (`HF_HOME`, `VLLM_CACHE_ROOT`), `HF_HUB_ENABLE_HF_TRANSFER=0`, the `PYTHONUNBUFFERED=1` main.py adds, and the
  RunPod SDK and `vllm-bnb-plugin` layers.
- `setup.sh` records versions, free memory at start, the data center, PCIe link, power limit, and idle clocks and
  clock-event reasons, fetches the worker's `src/startup_errors.py` at `71d503e6` (sha256 checked), and downloads the
  weights to container disk.
- `hooks.sh`: `pre_launch` starts `<variant>/gpu_clocks.csv` (clocks, power, temperature, PCIe link and clock-event
  reasons every 5 s until the kill) and picks the band context (below). `post_smoke` writes the live server's
  `PYTORCH_ALLOC_CONF`, `VLLM_USE_DEEP_GEMM` and `TOKENIZERS_PARALLELISM` from `/proc` to `server_env.txt`.
  `post_stop` greps the memory lines into `memory_facts.txt`, vLLM's parsed args and engine config line into
  `settings.txt`, and runs `worker_decision.py`, which feeds the last 400 log lines to the worker's own classifier
  with `main.py`'s rules and writes `worker_decision.json` (relaunch, answer jobs with a message, or exit 1; it also
  reports whether the revision or disk relaunch, which main.py checks first, would have matched).
- Benchmarked variants: smoke test, GSM8K 50 (5-shot chat, 2048-token cap: a correctness check on a reasoning model,
  not a score), `vllm bench serve` random 1024 in / 512 out at concurrency 1, 8, and 30 (the worker's
  `MAX_CONCURRENCY`).

## Hypotheses

| # | Hypothesis | Confirmed by | Killed by |
|---|---|---|---|
| H1 | The Hub default (`MAX_MODEL_LEN=auto`) never reaches the relaunch on context length; it boots with a shorter context and says so only in the log | worker-default OK with a `reduced from 131072` line | worker-default BOOT_FAIL with a matched message |
| H2 | On a 5090 the relaunch fits about the same context as the default: the graph memory it frees goes to the 4x prefill chunk | N_f - N_d under 2048 tokens; worker-fallback-band BOOT_FAIL | N_f - N_d of 2048 or more |
| H3 | Keeping graphs with `MAX_NUM_SEQS=32` fits at least what the relaunch fits | N_g >= N_f; graphs-fit-band OK | N_g < N_f |
| H4 | When the weights leave no KV room under `auto`, vLLM's error is one the worker does not recognise: no relaunch, no message, exit 1 | auto-no-room `worker_decision.json` says exits 1 | it says relaunch or answer |

N_d, N_f, N_g: the auto-fit context of worker-default, worker-fallback, graphs-fit.

## Matrix

| Variant | Flags against worker-default | Expected |
|---|---|---|
| worker-default | Hub defaults: graphs, `--max-num-seqs 256`, `--max-model-len auto` | OK; N_d about 105-115K; "max_num_batched_tokens=2048" |
| worker-fallback | `--enforce-eager --max-num-batched-tokens 8192` | OK in eager; N_f within 2K of N_d; slower than worker-default at c=1 |
| graphs-fit | `--max-num-seqs 32` | OK with graphs; N_g a few K above N_d; tok/s at c=1 and 8 close to worker-default |
| auto-no-room | `--gpu-memory-utilization 0.45` | BOOT_FAIL "Cannot auto-fit max_model_len"; worker exits 1 |
| worker-default-band | `--config band.yaml` (explicit length L) | BOOT_FAIL "larger than the available KV cache memory"; worker relaunches |
| worker-fallback-band | relaunch flags at L | BOOT_FAIL if H2 holds (worker answers every job with the error after two boots); OK in eager if not |
| graphs-fit-band | `--max-num-seqs 32` at L | OK with graphs |

Variants run in this order. Band context L (`hooks.sh` `pre_launch`, written to `<variant>/band.txt`): halfway from
N_d to the smaller of N_f and N_g when both are above N_d, else halfway to whichever is, else N_d + 1024. So the
default fails by construction, and the band shows which of the two ways out fits. It needs worker-default,
worker-fallback, and graphs-fit to have run first on the same pod; without worker-default's auto-fit line the first
band variant stops `exec` with an error, which is why auto-no-room runs before the band.

auto-no-room stands in for a model that fills the card at 0.95 (Qwen3-30B-A3B-Instruct-2507-FP8 is 29.03 GiB on a
31.8 GiB card); the 0.45 flag comes after base_flags' 0.95, and vLLM keeps the last. Its budget is negative, where
the real model's may sit just above zero; both end in the same branch as long as not one KV block fits, since under
`auto` the auto-fit raises before the capacity check (above). A full card with room for a few blocks would instead
boot at a tiny context.

## Cost

Quoted before the run (`harness quote runpod-worker-vllm-r1d8b`, 2026-10-23 18:17Z): 1x NVIDIA GeForce RTX 5090,
secure, $1.19/hr, 1.0 h planned, $1.31 worst case. Secure stock moved through the day: EU-RO-1 and EUR-NO-1 at
10:20Z, EUR-NO-1 only at 18:15Z, none at 18:16Z (community then quoted $0.69/hr, $0.76 worst case), back at 18:17Z.
Community is the second choice in `harness.toml`, which is why the kit has no `avoid_data_centers`. Setup about 8 min,
then about 30 min of variants.

## Reading the results

- Per variant: `memory_facts.txt` has `Available KV cache memory`, the auto-fit line, the graph estimate and actual,
  the batched-token budget, and on a successful boot the `Free memory on device ... Actual usage is` breakdown;
  `worker_decision.json` is what the worker would have done; `server.log` has the full error.
- Question 1, from N_d, N_f, N_g (tokens), for a Hub deploy on this card:
  - `MAX_MODEL_LEN` left at `auto`: boots with graphs at N_d, short of the model's 131072, with nothing in responses.
  - An explicit L <= N_d boots with graphs. N_d < L <= N_f: the relaunch path, served in eager with nothing in
    responses. L > N_f: two boots, then every job gets the error. Place 32768, 65536, and 131072 in those ranges.
  - The bare image (README path, `MODEL_NAME` only) asks for the model's 131072 at utilization 0.9, 1.6 GiB below
    the Hub budget; with N_d under 131072 it takes the relaunch path, and the relaunch fails unless N_f >= 131072.
- Question 2: worker-fallback against graphs-fit in `verdicts.jsonl` (`tok_s` per concurrency) and
  `bench_c*.json` (TPOT, TTFT), worker-default as the reference. The band variants are boot-only: same flags,
  so the benchmark of the auto variant stands for them. GSM8K is a check that eager and graphs answer alike, not a
  score: the worker sets no reasoning parser, so the model's thinking is in the reply, the 2048-token cap cuts some
  of it off, and the client's `Question:` stop can end a reply inside its thinking. With greedy decoding, compare
  `gsm8k_responses.jsonl` reply by reply and `hit_max_tokens` across variants rather than the scores.
- `settings.txt` is the check that each variant ran with what the matrix says (`enforce_eager`, `max_num_seqs`,
  `max_num_batched_tokens`, `max_model_len`), and `gpu_clocks.csv` that no variant ran at a lower clock or under a
  power or thermal cap the others did not.
- H2 is about this card class: on PRO 6000 or H100 the 8192 is vLLM's own default, so the relaunch frees only the
  graph estimate (see the l70b kit).
- A 5090 host's free memory at start (in the setup log) moves every N; compare N across variants of one pod only.
- Numbers for an upstream comment come from these runs, not from the expectations above.
