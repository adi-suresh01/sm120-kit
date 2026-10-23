# Puget Systems Team LLM pack v1.1.0, entry 7, on 2x RTX PRO 6000

Follow-up to `issues/puget-team-llm` (entry 4). Puget released the pack as v1.1.0 on 2026-10-22
(Puget-Systems/puget-docker-app-packs @ `fed0505e`, PR #10). The NVIDIA path is unchanged from what entry 4 tested:
`--enforce-eager --no-enable-prefix-caching` on menu entries 1-4 and 6-8, `--disable-custom-all-reduce` in the base
compose, and `v0.20.2` as the default image. `init.sh` writes `NCCL_P2P_DISABLE=1` to `.env`, but neither
`docker-compose.yml` nor `docker-compose.nvidia.yml` passes it into the container (no `env_file:`, and `environment:`
lists only the proxy and HF variables; only `docker-compose.amd-vllm.yml` passes it), so NVIDIA installs run with
NCCL's defaults. The release lists only DGX Spark and Arc B70 as tested.

Question: does entry 7, the other model the menu offers for 80 GB+ boxes, serve correctly with CUDA graphs and prefix
caching on, and what does eager mode cost it? Together with entry 4, that is the evidence for a PR that drops the
eager flags only where they are not needed.

## Setup

- 2x RTX PRO 6000 Blackwell Workstation Edition (Server Edition fallback), PCIe only.
- Menu entry 7: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` @ `ff433f54` (80 GB of safetensors; NVFP4 MoE with
  FP8 and BF16 layers), TP2, the pack's exact flags and environment. No `--max-model-len`: the pack writes none for
  this entry, so vLLM takes the model's own 262144.
- `vllm/vllm-openai:nightly`, the tag the menu writes for this entry, pinned to the same 10-19 build as entry 4
  (`nightly-e68efc68`) by digest.
- Each variant: boot, smoke test, GSM8K 100 (5-shot chat, greedy), then `vllm bench serve` at concurrency 1 / 16 / 64
  (random 1024 in / 512 out).

## Matrix

| Variant | Flags vs shipped | Expected |
|---|---|---|
| shipped | as shipped (NCCL defaults) | serves; eager baseline |
| graphs | CUDA graphs and prefix caching on; all-reduce settings as shipped | serves; every answered GSM8K reply correct, as in shipped; higher tok/s at 1 and 16 users |

Custom all-reduce is left out: the PR keeps the pack's all-reduce settings.

## Cost

`harness quote` on 2026-10-23: 2x Workstation Edition, secure, $5.18/hr, 1.0 h planned, $5.70 worst case. Actual
$2.54 (estimate).

## Results (2026-10-23)

Pod sm120-puget-team-llm-e7-3a6e58: 2x RTX PRO 6000 Blackwell Workstation Edition, secure, CUDA peer access 0<->1
true. Image `nightly-e68efc68`: vLLM 0.32.1rc1.dev36+ge68efc685, FlashInfer 0.7.2, torch 2.14.0+cu130. Both variants
ran with NCCL's defaults, as the pack installs; vLLM chose PYNCCL for all-reduce (custom all-reduce off).

Output tok/s from `vllm bench serve` (random 1024 in / 512 out), mean TPOT and TTFT in ms:

| Variant | Boot | c=1 | c=16 | c=64 | TPOT c=1 / 16 / 64 | TTFT c=1 / 16 / 64 | GSM8K, answered replies correct |
|---|---|---|---|---|---|---|---|
| shipped | 175 s | 17.4 | 262.6 | 964.8 | 57.5 / 59.5 / 63.0 | 101 / 779 / 1,724 | 99 / 100 |
| graphs | 181 s | 125.0 (7.2x) | 860.5 (3.3x) | 1,654.9 (1.7x) | 7.8 / 17.2 / 35.6 | 100 / 732 / 1,601 | 98 / 100 |

- Both variants served with 0 failed requests. The ERROR lines at the end of shipped's server.log are the harness's
  shutdown (SIGTERM at 07:39:27, after quality_done), not a fault.
- Every GSM8K reply reached an answer in both variants (0 empty, 0 at the token cap), so here the scores are a real
  correctness check: 99 vs 98 of 100 is within noise.
- Prefix caching was on in graphs: the log shows `enable_prefix_caching=True` and Mamba cache mode 'align'. The
  benchmark's random prompts do not reuse prefixes, so its gain on repeated prompts is not in these numbers.
- With no `--max-model-len`, vLLM took 262,144 and reported 60.6x maximum concurrency at that length, so the missing
  `MAX_CONTEXT` does not stop this entry from booting on 2x 96 GB.
- Eager mode costs more here than on entry 4: 57.5 ms per token for one user versus 7.8 ms with CUDA graphs.

## Reading the results

- GSM8K is a correctness check, not a score: `harness/pod/gsm8k_chat.py` stops generation at "Question:", which cut
  30-42 of entry 4's 100 replies inside the model's reasoning, and Nemotron Super also reasons by default. Use entry
  4's bar: every reply that reached an answer is correct in both variants, and the `empty_content` counts are close.
- Prefix caching is real on this hybrid model only if the server log shows the "Mamba cache mode ... 'align'" line.
- If shipped fails to boot, record the error as the finding and stop; the checkpoint's config code was written for an
  older transformers than the image's 5.19.0.
