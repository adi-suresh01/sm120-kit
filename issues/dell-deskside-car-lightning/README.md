# dell/deskside-agentic-ai: custom all-reduce at TP=2, the recipe's model (stage 2 of 2)

Stage 2 of `issues/dell-deskside-car`, which holds the recipe lines, the vLLM code reading, the hypotheses, and the
stop rules. Run only if stage 1 confirms H1 (the recipe's `expandable_segments:True` causes the
`custom_all_reduce.cuh` crash) and kills H2 (missing P2P).

## Setup

- `recipes/precision/vllm/docker-nemotron-3.5-lightning-nvfp4.sh` at b4e09e6c on a 2-GPU host:
  `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` at `bee75962` (21.6 GB), TP=2 (the recipe's default on two
  GPUs), the container env at :117-119, and the serve flags at :127-141, without `--served-model-name`.
- The recipe's image pin, `vllm/vllm-openai@sha256:cb9bc64f...` (vLLM 3f1bcfd8).
- 2x RTX PRO 6000 Blackwell, PCIe only, P2P gate on; Max-Q first, then the Workstation and Server Editions.
- `setup.sh` records peer access and vLLM's IPC probe on this box, then downloads the weights to container disk. As
  in stage 1, it ends the box if the container sets `PYTORCH_ALLOC_CONF`, which torch reads first.
- At 3f1bcfd8 the recipe's `VLLM_NVFP4_GEMM_BACKEND=marlin` is no longer read (`--linear-backend` replaced it; vLLM
  logs it as an unknown variable), so the NVFP4 linear layers take vLLM's automatic choice, as they do in the recipe's
  own container on this image. `VLLM_WSL2_ENABLE_PIN_MEMORY` is read on WSL only.

## Matrix

In run order (es-off before car-off, so a deadline that cuts the run costs the arm the recipe already ships):

| Variant | `PYTORCH_CUDA_ALLOC_CONF` | All-reduce flag | Expected if H1 |
|---|---|---|---|
| default | `expandable_segments:True` | none | BOOT_FAIL at `custom_all_reduce.cuh:164` |
| es-off | `expandable_segments:False` | none | serves on `CUSTOM`; GSM8K matches car-off; lower c=1 TPOT |
| car-off | `expandable_segments:True` | `--disable-custom-all-reduce` | serves on `PYNCCL` (the recipe's TP=2 config) |

Each arm that boots: smoke test, GSM8K 100 (5-shot chat, greedy), then `vllm bench serve` with random prompts, 1024
tokens in and 512 out, at c=1 (8 prompts), c=16 (64), and c=64 (256). Lightning reasons by default, so read GSM8K as
in `issues/puget-team-llm-e7`: every reply that reached an answer should be right in both arms, with similar
`empty_content` counts.

Hours: about 35 min with the image pull and the 21.6 GB download (`planned_hours = 0.75`).

## Results (2026-11-13)

H1 holds on the recipe's own launch: `default` aborts at `custom_all_reduce.cuh:164`, and `es-off`, one env value away
from the recipe, serves with custom all-reduce on, 5.5% faster at c=1 than the recipe's TP=2 advice (`car-off`).

Box: Vast pod `sm120-dell-deskside-car-lightning-a17d72`, 2x RTX PRO 6000 Blackwell Server Edition (600 W, driver
595.71.05), `topo -m` PIX, PCIe Gen5 x16 at most. The Workstation Edition offer the harness picked first was gone at
create time (Vast `no_such_ask`), so this is a different box and edition from stage 1. Same image and versions as
stage 1 (`env.txt`: vLLM 0.32.0rc1.dev190+g3f1bcfd82, torch 2.14.0+cu130, CUDA 13.0). The weights downloaded in 31 s.

From `jobs/setup.log` (`topo -p2p w` there and `topo -p2p r` in `p2p-sm120-dell-deskside-car-lightning-a17d72.txt`
read OK both ways). This setup does not run `vmm_ipc_probe.py`; the allocator probe is stage 1's.

```
PYTORCH_CUDA_ALLOC_CONF in the image env: None
can_device_access_peer 0->1: True 1->0: True
vllm IPC probe 0->1: True 1->0: True
```

Output tok/s from `vllm bench serve` (random 1024 in / 512 out; every request completed), mean TPOT in ms, the `tp:0`
all-reduce backends from `server.log` (`ep:0` is `['PYNCCL']` in every arm), and rank 0's KV cache at boot:

| Variant | Allocator | Backends | Boot | c=1 | c=16 | c=64 | TPOT c=1 / 16 / 64 | GSM8K 100 | KV cache |
|---|---|---|---|---|---|---|---|---|---|
| default | `True` | `['CUSTOM', 'PYNCCL']` | BOOT_FAIL, 105 s | - | - | - | - | - | - |
| es-off | `False` | `['CUSTOM', 'PYNCCL']` | 75 s | 352.1 | 2,235.6 | 4,307.8 | 2.74 / 6.46 / 13.22 | 93 | 74.13 GiB |
| car-off | `True` | `['PYNCCL']` | 101 s | 333.8 | 2,174.8 | 4,118.2 | 2.89 / 6.57 / 13.90 | 94 | 73.68 GiB |

- default: both workers print `Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:164 'invalid argument'`
  (`server.log:153`, and `:147` inside a progress-bar line) after the first capture pass, as in stage 1.
- es-off against car-off: +5.5% at c=1, +2.8% at c=16, +4.6% at c=64 output tok/s; mean TPOT -5.2%, -1.7%, -4.9%.
- KV cache at the 200K context: `False` reports 74.13 GiB (41,363,333 tokens, 206.82x at 200K) against 73.68 GiB
  (41,110,000 tokens, 205.55x) under `True`, so turning expandable_segments off costs no headroom at boot.
- Every arm logs `Unknown vLLM environment variable detected: VLLM_NVFP4_GEMM_BACKEND`, `linear_backend='auto'` in the
  engine config, and `Using MarlinNvFp4LinearKernel for NVFP4 GEMM`: the recipe's variable is no longer read, and the
  automatic choice is Marlin anyway on this box.
- GSM8K, read as planned. es-off's 7 misses: 3 empty replies (2 at the 4,096-token cap), 2 replies that repeat the
  few-shot `**Question:**` header before a correct `#### 60` and `#### 8` (the scorer keeps only the text before the
  first `Question:`), question 12 (answers 12, reference 13), and question 93 (answers 40/1.1 = 36.36 where the
  reference takes 10% off the time, 36). car-off's 6: 5 empty replies (2 at the cap) and the same question 12. Both
  arms make the same slip on question 12, and the empty counts are 3 against 5. The arms share 1 of 100 reply texts,
  as expected with 64 requests in flight.

Hypotheses:

- H1 confirmed on the recipe's model, flags, and container env.
- H2 killed on a second box and edition (Server Edition, PIX).
- H3 killed: es-off served all 328 benchmark requests and GSM8K 100 with no fault; 93 against 94.

Caveats, besides stage 1's (cloud boxes with P2P, not Dell's Max-Q towers; boot-time KV cache only):

- One rep per arm. The c=16 gain (2.8%) is smaller than at c=1 and c=64, which may be noise.
- `--ipc=host`, `--shm-size=8g`, the ulimits, `--restart`, and `--served-model-name` were not carried over, as planned.
