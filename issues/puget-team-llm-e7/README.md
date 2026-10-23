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

Quoted before the run; see `harness budget show puget-team-llm-e7` for the actual cost.

## Reading the results

- GSM8K is a correctness check, not a score: `harness/pod/gsm8k_chat.py` stops generation at "Question:", which cut
  30-42 of entry 4's 100 replies inside the model's reasoning, and Nemotron Super also reasons by default. Use entry
  4's bar: every reply that reached an answer is correct in both variants, and the `empty_content` counts are close.
- Prefix caching is real on this hybrid model only if the server log shows the "Mamba cache mode ... 'align'" line.
- If shipped fails to boot, record the error as the finding and stop; the checkpoint's config code was written for an
  older transformers than the image's 5.19.0.
