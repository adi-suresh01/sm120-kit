# sglang#44291 rerun on the #44480 stack

The EP2 + NEXTN crash from sglang#44291 (stage 2: `issues/sglang-44291/stage2.md`) was fixed by
sgl-project/sglang#44480. A broader refactor now sits on top of it as a stack: #44480 -> #44991 -> #44993 -> #44994.
#44993's description says the full #44291 reproduction was not run on the stack. This folder runs it on 2x PCIe
RTX PRO 6000 (sm_120), the reporter's hardware.

## Setup

- Model: `nvidia/Qwen3.8-Flash-Next-NVFP4` @ `fc694b54` (132.7 GB), whose fp8 MTP draft experts fall back to triton
  under `--moe-runner-backend flashinfer_cutlass`.
- Flags: stage 2's base flags, `--tp-size 2 --ep-size 2`, NEXTN 3/1/4, custom all-reduce off, default allocator.
  `NCCL_P2P_DISABLE=1` on every variant (split-NUMA hosts can hang at NCCL init); the control checks the crash
  still reproduces with it.
- Image: `lmsysorg/sglang:nightly-dev-cu13-20261019-3c560089`, two commits past the stack's base `6708426728`, same
  torch and sglang-kernel pins.
- Patches: `git diff 6708426728 <head> -- python/sglang`, applied to the image's editable install with `-p2`.

## Matrix

| Variant | Code | Expected |
|---|---|---|
| main-nextn | main at 3c560089 | BOOT_FAIL, `CUDA error: an illegal memory access` in draft prefill graph capture (stage 2's A1; A2 named `fused_moe_kernel` with graphs off) |
| stack-nextn | #44994 head `54ff1e4943` (carries #44480 at `478dd700`) | serves; GSM8K 200 and accept length near stage 2's A6 (0.975, 2.94) |
| pr44480-nextn | #44480 head `478dd7000f` | serves; same-host baseline for the stack's numbers |
| stack-nospec | #44994 head, no speculation | serves; GSM8K 50 |
| pr44993-nextn | #44993 head `3c2aa6c8a1` | only run if stack-nextn fails, to tell which PR it is |

Run the first four by name (`harness run sglang-44291-stack main-nextn stack-nextn pr44480-nextn stack-nospec`);
file order would also run pr44993-nextn.

## Cost

`harness quote` on 2026-10-18: 2x RTX PRO 6000 Server Edition, secure, $4.98/hr, 1.25 h planned, $6.85 worst case
($7.12 on the Workstation Edition fallback at $5.18/hr). Stage 2's A6 took about 10 min per serving variant, so the
four named variants need about 60 of the 75 min.

## Reading the results

- Compare tok/s only between stack-nextn and pr44480-nextn: this pod differs from stage 2 in `NCCL_P2P_DISABLE=1`,
  FlashInfer 0.7.2 (was 0.7.1) and sglang-kernel 0.4.10 (was 0.4.9).
- Each patched tree is that PR head plus the image's two later scheduler commits (`schedule_batch.py`,
  `schedule_policy.py`, `encoder_swa_replay.py`), which no patch touches.
- GSM8K at n=200 has a standard error of about 1.1 points.
- Re-fetch the heads before `up`; regenerate the patches if any moved.
