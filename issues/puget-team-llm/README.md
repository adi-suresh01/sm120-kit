# Puget Systems Team LLM pack on 2x RTX PRO 6000

[Puget-Systems/puget-docker-app-pack](https://github.com/Puget-Systems/puget-docker-app-pack) ships a vLLM-based
"Team LLM" server for Puget workstations. Read at `fix/installer-docker-restart-intel-check` @ `ef329d2` (open PR
#7, 2026-10-14). Its NVIDIA path, for every model:

- `--disable-custom-all-reduce` in `packs/team_llm/docker-compose.yml`, and `NCCL_P2P_DISABLE=1` written to `.env`
  by `packs/team_llm/init.sh`, so tensor-parallel all-reduce on a multi-GPU box never uses GPU peer-to-peer.
- `--enforce-eager --no-enable-prefix-caching` ("Common flags for MoE/nightly models that need eager mode") on menu
  entries 1-4 and 6-8, so those run without CUDA graphs and without prefix caching, on a server meant for many
  users at once.
- Floating image tags (`nightly`, `latest`, default `v0.20.2`), so the stack a customer gets changes with the day
  they install.

Question: on the hardware Puget sells, are those flags still needed with the vLLM a customer gets today, and what do
they cost?

## Setup

- 2x RTX PRO 6000 Blackwell Workstation Edition (Server Edition fallback), PCIe only.
- Menu entry 4, the menu's recommended model for 80 GB+: `cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit` @ `be51bea8`
  (80 GB, compressed-tensors W4A16), TP2, 128K context, the pack's exact flags.
- `vllm/vllm-openai:nightly`, the tag the menu writes for this entry, pinned to its 10-19 build
  `nightly-e68efc68` by digest.
- Each variant: boot, smoke test, GSM8K 100 (5-shot chat, greedy), then `vllm bench serve` at concurrency 1 / 16 / 64
  (random 1024 in / 512 out).

## Matrix

| Variant | Flags vs shipped | Expected |
|---|---|---|
| shipped | as shipped | serves; baseline |
| graphs | CUDA graphs and prefix caching on | serves with the same GSM8K; faster decode, most at low concurrency |
| graphs-p2p | graphs + custom all-reduce + NCCL P2P | faster again at TP2 if the host allows P2P; vLLM falls back to NCCL on its own if not |

The GSM8K bar is "same score as shipped within noise" (n=100, about +/-4 points), not an absolute number. Prefix
caching's gain needs repeated prompts, which the random benchmark does not have, so these numbers do not show it.

## Cost

`harness quote` on 2026-10-19: 2x Workstation Edition, secure, $5.18/hr, 1.5 h planned, $8.55 worst case.

## Results (2026-10-19)

Pod 9f0esq6n050mgo: 2x RTX PRO 6000 Blackwell Workstation Edition, secure, $5.18/hr, CUDA peer access 0<->1 true.
Image `nightly-e68efc68`: vLLM 0.32.1rc1.dev36+ge68efc685, FlashInfer 0.7.2, torch 2.14.0+cu130. Cost $3.18 (estimate)
of $8.55.

Output tok/s from `vllm bench serve` (random 1024 in / 512 out), mean TTFT / TPOT in ms:

| Variant | Boot | c=1 | c=16 | c=64 | TPOT c=1 / 16 / 64 | GSM8K, answered replies correct |
|---|---|---|---|---|---|---|
| shipped | 95 s | 29.2 | 385.2 | 1,428.2 | 34.2 / 40.1 / 41.5 | 56 / 56 |
| graphs | 106 s | 122.8 (4.2x) | 870.1 (2.3x) | 1,716.1 (1.2x) | 8.0 / 17.1 / 34.6 | 58 / 59 |
| graphs-p2p | 80 s | 139.1 (4.8x) | 944.2 (2.5x) | 1,778.4 (1.25x) | 7.0 / 15.7 / 33.3 | 47 / 47 |

- Every variant served; no fault in any log.
- graphs-p2p ran with `disable_custom_all_reduce=False` and vLLM logged no custom all-reduce fallback. vLLM guards
  this itself: it turns custom all-reduce off on more than two PCIe-only GPUs and when the P2P check fails
  (`custom_all_reduce.py` on main, lines 270 and 284), so dropping the flag is safe on 4-GPU boxes too.
- GSM8K scores (0.56 / 0.58 / 0.47) are not model quality. The harness stops generation on `Question:`, and this
  thinking model writes it inside its reasoning, so 33 / 30 / 42 replies ended mid-thought with no answer; 11 per
  variant hit the 4096-token cap. Every reply that reached an answer was correct in all three variants. The harness
  needs reasoning-aware stops before GSM8K means anything on thinking models.
- Not covered: the other menu entries, 4-GPU and Max-Q boxes, the `latest` (v0.32.0) tag, and prefix caching's gain
  on repeated prompts.
