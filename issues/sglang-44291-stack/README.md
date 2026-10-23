# sglang#44291 rerun on the merged #44480 stack

The EP2 + NEXTN crash from sglang#44291 (stage 2: `issues/sglang-44291/stage2.md`) was fixed by
sgl-project/sglang#44480. A broader refactor was built on top of it as a stack: #44480 -> #44991 -> #44993 -> #44994,
and all four merged into main on 2026-10-20. #44993's description says the full #44291 reproduction was not run on the
stack. This folder runs it on a stock nightly, with no patches, on 2x PCIe RTX PRO 6000 (sm_120), the reporter's
hardware.

## Setup

- Model: `nvidia/Qwen3.8-Flash-Next-NVFP4` @ `fc694b54` (132.7 GB), whose fp8 MTP draft experts fall back to triton
  under `--moe-runner-backend flashinfer_cutlass`.
- Flags: stage 2's base flags, `--tp-size 2 --ep-size 2`, NEXTN 3/1/4, custom all-reduce off, default allocator.
  `NCCL_P2P_DISABLE=1` on every variant (split-NUMA hosts can hang at NCCL init).
- Image: `lmsysorg/sglang:nightly-dev-cu13-20261021-c1c287a4`, built from main at `c1c287a4`, which contains #44480
  (`e2e1a97fc3`, 13 commits earlier), #44993 (`4f8f06faf3`) and #44994 (`299a75ac5d`).

## Matrix

| Variant | Code | Expected |
|---|---|---|
| nightly-nextn | nightly at c1c287a4, no patches | serves; GSM8K 200 and accept length near stage 2's A6 (0.975, 2.94) |
| nightly-nospec | same, no speculation | serves; GSM8K 50 |

The "before" is stage 2's A1 on main without #44480: BOOT_FAIL, `CUDA error: an illegal memory access` in draft
prefill graph capture.

## Cost

Quoted before the run; see `harness budget show sglang-44291-stack` for the actual cost.

## Reading the results

- The run shows whether the stock nightly serves the reporter's config. It cannot credit GSM8K, accept-length or tok/s
  differences from stage 2's A6 to the stack: c1c287a4 is 438 commits after A6's image (a6a8c560), including
  #45289 (NVFP4 MoE scales for ModelOpt checkpoints like this one), #37958 (prefill CUDA graph), #43927 (FlashInfer
  startup tuning) and FlashInfer 0.7.2, and this pod sets `NCCL_P2P_DISABLE=1`.
- GSM8K at n=200 has a standard error of about 1.1 points.
- TP2 still needs #42566 and custom all-reduce with `expandable_segments` still needs #44448; neither is in this run.
