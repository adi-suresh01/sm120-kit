# runpod-workers/worker-vllm: the startup OOM relaunch on an RTX PRO 6000

The 96 GB half of `issues/runpod-worker-vllm-r1d8b`, which has the full reading of the worker source
(runpod-workers/worker-vllm @ `71d503e6`, release v2.30.1) and of the vLLM v0.32.0 code it depends on. In short:
after a startup OOM that `src/startup_errors.py:88-98` recognises, `src/main.py:181-221,292-304` relaunches vLLM
once with `ENFORCE_EAGER=true` and `MAX_NUM_BATCHED_TOKENS=8192`, logs it, and says nothing in responses. The Hub
form (`.runpod/hub.json`) deploys with `MAX_MODEL_LEN=auto`, `GPU_MEMORY_UTILIZATION=0.95`, `MAX_NUM_SEQS=256`, and
lists no Blackwell GPU (`gpuIds` `ADA_80_PRO,AMPERE_80`).

Questions, for a 96 GB RTX PRO 6000:

1. Which context settings send a model that fills the card down the worker's OOM path, and does the relaunch then
   serve?
2. What does the silent relaunch cost, eager plus 8192 batched tokens, against keeping CUDA graphs with
   `MAX_NUM_SEQS=32`?

What differs from the 5090: at 70 GiB and up, vLLM v0.32.0's own `vllm serve` default is 8192 batched tokens
(`vllm/engine/arg_utils.py:2933-2942`), so the relaunch's `MAX_NUM_BATCHED_TOKENS=8192` changes nothing here. The
relaunch frees only the CUDA graph estimate that `vllm/v1/worker/gpu_worker.py:695-745` takes off the KV budget.
The same holds on the H100 80 GB the Hub listing offers.

## Model

`RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` @ `f50dbad2`: 67.68 GiB of FP8 weights, 131072-token context, 320 KiB
of KV cache per token, not gated (Meta's own repo is), 104K downloads last month. The worker's examples are Llama
(`README.md`'s config example and `configs/llama/llama3.1_8b_instruct.yaml` use Llama-3.1-8B-Instruct; the Hub preset
is a Llama-architecture distill), and on one 96 GB card the family's choice is the 70B at FP8 (the BF16 weights are
141 GB). The ~30B models do not reach the OOM path on this card at all: `Qwen/Qwen3-30B-A3B` (the worker README's own
OOM example, 56.87 GiB BF16) leaves about 30 GiB of KV against 3.75 GiB for its 40960 tokens, and
`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (29.03 GiB) about 58 GiB against 24 GiB for its 262144.

Expected size of the effect, from `issues/vllm-61100` (Apertus-70B FP8-dynamic on this card at 0.95, vLLM 0.25.1):
67.82 GiB loaded, 19.66 GiB of KV with graphs, 1.53 GiB of graphs. For this model that is about 64K tokens with the
defaults and about 5K more without the graph estimate, which puts the common `MAX_MODEL_LEN=65536` at the edge.

## Setup

- 1x RTX PRO 6000 Blackwell Server Edition (Workstation Edition fallback), secure. Same 96 GB either way; clocks
  only move tok/s, and every variant runs on the same card.
- `vllm/vllm-openai:v0.32.0@sha256:befccf0e...5b34`, the worker's base image.
- Server flags, env, `setup.sh`, `hooks.sh`, and `worker_decision.py`: as in the r1d8b kit (the Hub env defaults and
  the Dockerfile's ENV run through the worker's own `build_vllm_args`, `--model` last as the worker passes it, our
  `--revision`; `gpu_clocks.csv`, `server_env.txt` and `settings.txt` per variant).
- Benchmarked variants: smoke test, GSM8K 50 (5-shot chat, 1024-token cap), `vllm bench serve` random 1024 in / 512
  out at concurrency 1, 8, and 30 (the worker's `MAX_CONCURRENCY`).

## Hypotheses

| # | Hypothesis | Confirmed by | Killed by |
|---|---|---|---|
| H1 | The Hub default (`MAX_MODEL_LEN=auto`) boots with a context short of 131072 and says so only in the log | worker-default OK with a `reduced from 131072` line | BOOT_FAIL |
| H2 | The relaunch frees only the graph estimate: N_f - N_d is that estimate over 320 KiB per token, about 4-5K tokens | N_f - N_d within 1K of estimate / 320 KiB; worker-fallback-band OK in eager | N_f - N_d under 1K |
| H3 | Keeping graphs with `MAX_NUM_SEQS=32` fits at least what the relaunch fits | N_g >= N_f; graphs-fit-band OK | N_g < N_f |

N_d, N_f, N_g: the auto-fit context of worker-default, worker-fallback, graphs-fit.

## Matrix

| Variant | Flags against worker-default | Expected |
|---|---|---|
| worker-default | Hub defaults: graphs, `--max-num-seqs 256`, `--max-model-len auto` | OK; N_d about 60-65K; "max_num_batched_tokens=8192" |
| worker-fallback | `--enforce-eager --max-num-batched-tokens 8192` | OK in eager; N_f about 4-5K above N_d |
| graphs-fit | `--max-num-seqs 32` | OK with graphs; N_g >= N_f |
| worker-default-band | `--config band.yaml` (explicit length L) | BOOT_FAIL "larger than the available KV cache memory"; worker relaunches |
| worker-fallback-band | relaunch flags at L | OK in eager: the silent relaunch, served |
| graphs-fit-band | `--max-num-seqs 32` at L | OK with graphs |

Band context L, from `hooks.sh` `pre_launch` into `<variant>/band.txt`: halfway from N_d to the smaller of N_f and
N_g when both are above N_d, else halfway to whichever is, else N_d + 1024. The band variants need the three auto
variants to have run first on the same pod.

The no-KV-room case (`Cannot auto-fit`, which the worker does not recognise) is in the r1d8b kit only: it does not
depend on the card.

## Cost

Quoted before the run (`harness quote runpod-worker-vllm-l70b`, 2026-10-23): 1x NVIDIA RTX PRO 6000 Blackwell
Server Edition, secure, $2.49/hr (EU-RO-1 and US-NC-2 at 18:15Z; Workstation Edition $2.59/hr in EU-CZ-1 as
fallback), 1.25 h planned, $3.42 worst case. Setup 8-17 min, then about 40 min of variants.

## Reading the results

- Per variant: `memory_facts.txt` (KV memory, auto-fit line, graph estimate and actual, batched-token budget, the
  `Actual usage is` breakdown on a successful boot), `worker_decision.json` (what the worker would have done),
  `settings.txt` (vLLM's parsed args and engine config: check `enforce_eager`, `max_num_seqs`,
  `max_num_batched_tokens` and `max_model_len` against the matrix), `gpu_clocks.csv` (no variant at a lower clock or
  under a cap the others were not), and `server.log`.
- Question 1 for a Hub deploy on this card: `auto` boots at N_d; an explicit L <= N_d boots with graphs; N_d < L <=
  N_f is the relaunch path, served in eager with nothing in responses; L > N_f is two 70B boots, then every job gets
  the error. Place 32768, 65536, and 131072 in those ranges. The bare image (`MODEL_NAME` only) asks for 131072 at
  utilization 0.9 with vLLM's 1024 sequences: 40 GiB of KV against about 20, so it fails both boots whatever N_f is.
- Question 2: worker-fallback against graphs-fit in `verdicts.jsonl` (`tok_s` per concurrency) and `bench_c*.json`
  (TPOT, TTFT), worker-default as the reference. The band variants are boot-only; the auto variant's benchmark
  stands for them. GSM8K checks that eager and graphs answer alike.
- Each boot of the failing pair costs the user a 68 GB load; worker-default-band's plus worker-fallback-band's
  `boot_s` is what a cold start in the band waits before the first token.
- Numbers for an upstream comment come from these runs, not from the expectations above.

## Results (2026-10-24)

One pod, 1x RTX PRO 6000 Blackwell Server Edition, EUR-NO-1. Cost $1.69 of the $3.42 budget.

| Variant | 1 | 8 | 30 concurrent | GSM8K (n=50) | boot |
|---|---|---|---|---|---|
| worker-default (Hub defaults, graphs) | 19.2 | 140.4 | 412.8 | 49/50 | 210 s |
| worker-fallback (`ENFORCE_EAGER`, 8192 batched tokens) | 19.1 | 137.2 | 399.4 | 50/50 | 45 s |
| graphs-fit (`MAX_NUM_SEQS=32`) | 19.2 | 140.3 | 412.4 | 48/50 | 90 s |

- On a 70B FP8 model the relaunch costs almost nothing (0-3%): decode is bound by the GPU, not by kernel launches.
  The 8B model on the 5090 lost 1.5-2.0x to the same relaunch, so its cost depends on the model, not the worker.
- Band (63,680 tokens): worker-default-band failed with the shortfall the worker matches (relaunch); the fallback and
  graphs-fit both served it.
- graphs-fit again matched the default's throughput and booted in 90 s instead of 210 s.
