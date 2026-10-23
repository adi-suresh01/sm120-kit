# Puget Systems Team LLM pack on 2x RTX PRO 6000

[Puget-Systems/puget-docker-app-pack](https://github.com/Puget-Systems/puget-docker-app-pack) ships a vLLM-based
"Team LLM" server for Puget workstations. Read at `fix/installer-docker-restart-intel-check` @ `ef329d2` (open PR
#7, 2026-10-14). Its NVIDIA path, for every model:

- `--disable-custom-all-reduce` in `packs/team_llm/docker-compose.yml`. `packs/team_llm/init.sh` also writes
  `NCCL_P2P_DISABLE=1` to `.env`, but the NVIDIA compose never passes it into the container (corrected 2026-10-23,
  see below), so installs keep NCCL's peer-to-peer default.
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

## Correction (2026-10-23)

The 10-19 shipped and graphs variants ran with `NCCL_P2P_DISABLE=1`, read as part of the pack's config. It is not:
`init.sh` writes it to `.env`, but `docker-compose.yml` has no `env_file:` and its `environment:` lists only the proxy
and HF variables, and `docker-compose.nvidia.yml` adds none (only `docker-compose.amd-vllm.yml` passes it). So on
NVIDIA the pack runs with NCCL's defaults, P2P allowed. What that changes:

- shipped was not the exact shipped config, so 29.2 tok/s is not "as installed" and 4.2x is eager vs graphs with P2P
  off on both sides, not shipped vs tuned.
- graphs-p2p's +13% / +9% mixes two changes, custom all-reduce and NCCL P2P; the pack already gets the second.

The `installed` and `graphs-installed` variants rerun both with NCCL's defaults.

### Recheck results (2026-10-23): inconclusive, the host differs

Pod sm120-puget-team-llm-ebb837: 2x RTX PRO 6000 Blackwell Workstation Edition, secure, same image. Unlike the
entry 7 pod run the same hour (PCIe Gen 5, GPUs on different NUMA nodes, `SYS`), this host reports PCIe Gen 4 x16 as
the link maximum, both GPUs behind one PCIe switch (`PIX`), and cgroup v1. vLLM chose PYNCCL for all-reduce in both
variants. Cost $3.54 (estimate).

| Variant | Boot | c=1 | c=16 | c=64 | TPOT c=1 / 16 / 64 | TTFT c=1 / 16 / 64 | GSM8K, answered replies correct |
|---|---|---|---|---|---|---|---|
| installed | 146 s | 26.7 | 355.8 | 620.3 | 37.1 / 41.1 / 93.9 | 224 / 2,019 / 4,799 | 48 / 48 |
| graphs-installed | 135 s | 73.4 (2.7x) | 372.7 (1.05x) | 651.1 (1.05x) | 13.2 / 39.3 / 90.8 | 228 / 1,877 / 3,899 | 46 / 46 |

- Both served with 0 failed requests; the ERROR lines at the end of each server.log are the harness's shutdown.
- Against 10-19, eager at one user is about the same (37.1 vs 34.2 ms per token), but everything that leans on
  all-reduce is about 2x slower: graphs at one user 13.2 vs 8.0 ms, and both variants at 64 users 91-94 vs 35-42 ms.
  That fits a slow GPU-to-GPU path on this host with NCCL free to use P2P across the switch, but nothing here
  measured the path, and the 10-19 pod's topology was not recorded, so the cause is not established.
- Conclusion for now: eager costs single-user throughput on every host tested (2.7x here, 4.2x on 10-19 with P2P
  off, 7.2x for entry 7), but the size depends on the host, and the "as installed" multi-user numbers for entry 4
  need a run that measures P2P on and off on one host, with an all-reduce bandwidth check, before they are quoted.

### P2P on vs off on one host (2026-10-23)

All four variants on one pod, so the host is the same for every comparison: installed and graphs-installed (NCCL
defaults, P2P allowed, as the pack installs) against shipped and graphs (`NCCL_P2P_DISABLE=1`, what the pack's
`.env` asks for). setup.sh first timed an NCCL all-reduce both ways (`allreduce_bw.py`).

Pod sm120-puget-team-llm-99bd70 landed on the same physical host as the recheck (213.173.111.97): PCIe Gen 4 x16
link maximum, both GPUs behind one switch (`PIX`), CUDA peer access true, cgroup v1. Same image. Cost $6.36
(estimate).

NCCL all-reduce, 2 ranks, bf16:

| Message | P2P allowed (transport P2P) | P2P disabled (transport SHM) |
|---|---|---|
| 8 KiB | 11.1 us | 11.1 us |
| 1 MiB | 66.9 us, 15.7 GB/s bus | 153.9 us, 6.8 GB/s bus |
| 64 MiB | 3.64 ms, 18.5 GB/s bus | 7.54 ms, 8.9 GB/s bus |

Serving, output tok/s (random 1024 in / 512 out), mean TPOT in ms; vLLM used PYNCCL in every variant:

| Variant | P2P | Mode | c=1 | c=16 | c=64 | TPOT c=1 / 16 / 64 | GSM8K, correct / replies with a number |
|---|---|---|---|---|---|---|---|
| installed | allowed | eager | 26.6 | 352.6 | 616.9 | 37.2 / 41.5 / 94.5 | 56 / 57 |
| shipped | disabled | eager | 26.7 | 348.2 | 585.9 | 37.1 / 41.6 / 98.9 | 52 / 53 |
| graphs-installed | allowed | graphs | 73.2 | 373.5 | 644.6 | 13.2 / 39.4 / 91.7 | 57 / 57 |
| graphs | disabled | graphs | 69.6 | 362.7 | 618.6 | 13.9 / 40.1 / 94.7 | 56 / 58 |

- P2P is not why this host is slow. Allowing it doubles all-reduce bandwidth for large messages and leaves the
  8 KiB latency unchanged, and in serving it is worth 0-5%: 4% at 64 users in both modes, 5% at 1 user with graphs.
  The slowdown against 10-19 is the host itself: the recheck's numbers reproduce here within 1% (installed c=1 26.6
  vs 26.7, graphs-installed 73.2 vs 73.4), and both P2P settings are equally slow. GPU clocks and throttle reasons
  were not recorded, so the cause on the GPU side is not established.
- On this host, as installed, the eager flags cost 2.75x output tok/s at 1 user (26.6 vs 73.2) and 4-6% at 16 and 64
  users. On the 10-19 host (P2P off on both sides) the same comparison was 4.2x, 2.3x and 1.2x. The single-user cost
  holds on both hosts; the multi-user cost depends on the host.
- For the pack: `NCCL_P2P_DISABLE=1` never applied on NVIDIA, and on this host P2P allowed is equal or slightly
  faster, so the unapplied setting can be dropped rather than wired through. The 10-17 split-NUMA NCCL hang
  (sglang-33422) is the reason to keep it a per-board check rather than a blanket default either way.
- GSM8K stays a weak check on this thinking model: 42-47 replies per variant ended without a number, as on 10-19.

The whole recheck cost $3.56 + $6.36 on top of the 10-19 run's $3.20.
