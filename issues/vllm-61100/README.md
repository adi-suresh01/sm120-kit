# vllm#61100: Apertus-70B FP8 degenerates on sm_120 in vLLM 0.25.1, fine in 0.22.1

https://github.com/vllm-project/vllm/issues/61100, reporter juniorUniv (opened 10-11). A silent failure: the
server reports healthy, returns HTTP 200, and generates "The,,,,,,,,,, and,,, ..." at temperature 0.
LingYi-Liang could not reproduce with either version (different driver, not the reporter's images).

## Reporter setup (copied)

| Component | Value |
|---|---|
| GPU | RTX PRO 6000 Blackwell Workstation Edition, sm_120 |
| Driver | 595.99.02 |
| Model | `RedHatAI/Apertus-70B-Instruct-2509-FP8-dynamic` (compressed-tensors W8A8 FP8, dynamic per-token activations) |
| KV cache | fp8_e4m3, 65536 context, TP1, V1 engine |
| Kernels | `CutlassFP8ScaledMMLinearKernel`, FLASHINFER attention, FlashInfer top-p/top-k sampler |
| Compile | VLLM_COMPILE (Inductor), FULL_AND_PIECEWISE graphs |
| Working | `vllm/vllm-openai:v0.22.1@sha256:953d3a06...` (FlashInfer 0.6.11.post2, compressed-tensors 0.15.0.1, transformers 5.10.2) |
| Failing | `vllm/vllm-openai:v0.25.1@sha256:e4f88a83...` (FlashInfer 0.6.13, compressed-tensors 0.17.0, transformers 5.13.1) |

Failing request: Slovak system prompt "Odpovedaj vylucne po slovensky, strucne a presne." and user "Kolko je 1+1?
Odpovedz iba cislom." (with diacritics in `probe_requests.py`), temperature 0, 256 tokens, expected "2"; got 256
tokens of commas and "the of and".

## Desk research (2026-10-16, no GPU)

Nothing upstream fixes or duplicates it. Ruled out by reading code and configs: xIELU (unchanged between the
tags), the Apertus weight-loader refactor (#47058, checkpoint beta/eps equal the defaults), transformers 5.13.1
(identical rope config and prompt token ids), compressed-tensors 0.17 (parses the config identically), the q_scale
change (#53754; the checkpoint has no q/k/v scales), and the FlashInfer sampler (unused at temperature 0). Full
notes: the research report in the session scratchpad, summarized here.

## Hypotheses

| # | Hypothesis | Variant that decides it |
|---|---|---|
| H1 | Model Runner V2 became the default for Apertus FP8 in 0.25.0 (#44443, a2f713002); it has a host-dependent silent-garbage mode through stale UVA views of pinned memory (#57224, fixed only by `VLLM_USE_V2_MODEL_RUNNER=0`) | a1 (V1 runner), a8 (0.24.0 forced onto V2), uva_probe in every variant |
| H2 | FlashInfer 0.6.13's FP8-KV attention rewrite (flashinfer#3485, c3c40a7b) mis-computes for this geometry on sm_120 | a2 (BF16 KV), a3 (Triton attention, FP8 KV), a6 (FlashInfer 0.6.12 under vLLM 0.25.1) |
| H3 | FULL decode graph capture under V2 replays stale metadata | a4 (piecewise), a5 (eager) |
| H4 | Host or driver specific (595.x, Workstation Edition) | a0 on a CUDA 13.2 (595.x) host; a clean a0 points here |
| H5 | vLLM's FlashInfer backend restructure (#43232, #47485), not FlashInfer itself | a3 fixes it while a6 does not |

## Matrix

One RTX PRO 6000 on a CUDA 13.2 host, the reporter's v0.25.1 image by digest, the reporter's flags (FP8 KV, 64K
context, default compile and graphs) plus `--gpu-memory-utilization 0.95`. Every variant runs GSM8K-200 over the
chat API (max 1024 tokens per reply), then `probe_requests.py` (the reporter's exact request plus two English
checks, raw output saved), and records versions, the driver, the UVA probe, and the runner and kernels the server
picked. Version swaps use `PYTHONPATH` overlays built in `setup.sh`, so nothing leaks between variants.

| Variant | Change vs a0 |
|---|---|
| a0-stock | none |
| a1-v1-runner | `VLLM_USE_V2_MODEL_RUNNER=0` |
| a2-kv-auto | `--kv-cache-dtype auto --max-model-len 32768` (BF16 KV needs the smaller context to fit) |
| a3-triton-attn | `--attention-backend TRITON_ATTN` |
| a4-piecewise | `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` |
| a5-eager | `--enforce-eager` |
| a6-fi0612 | FlashInfer 0.6.12 (python, cubin, jit-cache cu130) overlay |
| a7-v0240 | vLLM 0.24.0 + FlashInfer 0.6.12 overlay (Apertus on the V1 runner): positive control |
| a8-v0240-mrv2 | a7 + `VLLM_USE_V2_MODEL_RUNNER=1` |

Sequence: `harness up`, `harness exec vllm-61100 a0-stock`. If a0 degenerates, run a1-a8. If a0 is clean, stop:
the bug does not reproduce on this host, and the useful answer is the reporter-side toggles (V1 runner, BF16 KV)
plus the UVA probe for their host.

## Hardware, hours, cost

1x RTX PRO 6000 Server Edition SECURE, $2.49/hr. Setup about 12 min (8.8 GB image, 72.8 GB weights, about 4.4 GB of
overlays), about 8 min per variant. Expected 0.4 h if a0 is clean, about 1.6 h with the full matrix;
`planned_hours = 2.0` caps it at $5.48.

## Results

Not run yet.
