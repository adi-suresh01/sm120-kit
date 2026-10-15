# sglang#44291: custom all-reduce IPC failure on 2x PCIe sm_120 (stage 1 of 2)

- Issue: https://github.com/sgl-project/sglang/issues/44291 (opened 2026-10-14, 0 comments at planning time)
- Reporter: sorenmat (Trubit, deployment consultant).
- Related: vLLM #47266 / PR #47751 (same failure in vLLM, blanket sm_12x disable, unreviewed since Aug),
  sglang PR #34528 (merged 09-29, opt-in FlashInfer PCIe-IPC all-reduce for switch-free hosts),
  sglang PR #36787 (open, sm_120 support for Qwen3.8-Flash-Next; relevant to stage 2).

The issue is three bugs stacked: (1) custom all-reduce IPC handshake fails, (2) EP2 illegal memory access in the
MoE path, (3) TP2 blocked by a 320 % 128 shard. **Stage 1 (this document) covers bug 1 only**, because it
reproduces with a small model, affects every 2-GPU PCIe sglang user, and has no sglang PR. Bugs 2 and 3 get their
own design after stage 1 lands (see Stage 2 handoff).

## Reporter setup (verbatim, ASCII-normalized)

```
GPUs       : 2x NVIDIA RTX PRO 6000 Blackwell Server Edition (97887 MiB), driver 580.178.04
Images     : lmsysorg/sglang:dev-qwen38-next-local (qwen4-main-squashed, build 4ccff141)
             lmsysorg/sglang:dev-cu13 (mainline nightly, 2026-10-02)
Checkpoint : nvidia/Qwen3.8-Flash-Next-NVFP4 @ fc694b54
Flags      : --tp-size 2 --ep-size 2 --moe-runner-backend flashinfer_cutlass --ple-offload-embedding
             --json-model-override-args '{"language_model_only":true}' + NEXTN 3/1/4 speculative decoding
```

| Attempt | Flags | Result |
|---|---|---|
| 1 | vanilla EP2 | `get_graph_buffer_ipc_meta` -> `RuntimeError: invalid argument` (custom all-reduce CUDA-IPC handshake) |
| 2 | `+ --disable-custom-all-reduce` | `cudaErrorIllegalAddress` during draft CUDA-graph capture (`eagle_draft_cuda_graph_runner.py:274`) |
| 3 | `+ --disable-cuda-graph` | `Triton Error [CUDA]: an illegal memory access was encountered` in the EP2 path |

NCCL initializes cleanly in every attempt. TP2: `ValueError: The output_size of gate's and up's weight = 320 is
not divisible by weight quantization block_n = 128`. TP1 on one GPU works.

## How sglang main picks the all-reduce (read 2026-10-15)

- `custom_all_reduce.py`: v2 (`CustomAllReduceV2`, default via `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=True`) is used
  only if `can_use_custom_all_reduce_v2()` passes, and on a single node that requires full NVLink. PCIe fails it,
  so sglang falls back to legacy v1 `CustomAllreduce`.
- v1 eligibility (`can_use_custom_all_reduce_with_nvlink`) disables only for `world_size > 2 and not full_nvlink`,
  so 2x PCIe passes, then a P2P capability check.
- v1 `get_graph_buffer_ipc_meta` (`custom_all_reduce.cuh:602`) takes the allocation base via
  `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_RANGE_START_ADDR)` and calls `cudaIpcGetMemHandle` on it.
- sglang's own v2 code says the cudaIpc path "fails on VMM pointers (expandable_segments)" and has a separate VMM
  path. v1 has no such check.
- The Qwen3.8-Flash-Next cookbook recommends `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for 200k+
  contexts.

## Hypotheses

| # | Hypothesis | Confirmed by | Killed by |
|---|---|---|---|
| H1 | VMM-backed memory (`expandable_segments`) breaks v1's IPC handshake on graph buffers | S2 fails with `invalid argument` in `get_graph_buffer_ipc_meta`, S1 serves | S2 serves |
| H2 | Legacy CUDA IPC does not work between these two GPUs at all (IOMMU/ACS, container, driver) | `ar_repro.py` fails on plain `cudaMalloc` memory | plain memory works |
| H3 | Handshake succeeds but the v1 kernel faults on sm_120 (vLLM #47751's assumption) | S1 passes the handshake and fails at replay with a different error | S1 serves coherently |

## Experiment matrix (2x RTX PRO 6000, PCIe, `Qwen/Qwen3-0.6B`, TP2, CUDA graphs on)

Record which all-reduce backend each run selected (the `[AR] Using ...` line is debug level, so run with
`--log-level debug` or probe it in-process).

| Run | Change from defaults | Expected if H1 |
|---|---|---|
| S1 | none | serves, on v1 |
| S2 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | `invalid argument` in `get_graph_buffer_ipc_meta` (reporter's attempt 1) |
| S3 | S2 + `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1` | serves (PCIe-IPC copies inputs into its own workspace) |
| S4 | `--disable-custom-all-reduce` | serves on NCCL |

`ar_repro.py` (no server, torchrun with 2 ranks) captures 8 all-reduces in one CUDA graph and toggles: v1 as
shipped (graph inputs IPC-registered after capture), v1 staged (graph inputs copied into v1's pre-registered
IPC buffer, the path `SGLANG_MEMORY_SAVER_CUDA_GRAPH` already uses), and NCCL; each with the allocator VMM-backed
or not. It checks the result against eager NCCL and times us per all-reduce at 16 KB to 8 MB (bf16). PCIe-IPC
(#34528) is tied into sglang's group setup and tuning, so it is covered by server run S3 only.

If S1 does not select v1 (sglang's P2P check fails on the rented host), this pod cannot reproduce the bug:
record `nvidia-smi topo -p2p r` and the P2P check log, stop, and terminate.

If S1 and S2 both serve, H1 is dead in our environment. Then try the reporter's remaining differences in order:
EP2 group layout (`--tp-size 2 --ep-size 2`, which needs an MoE model, so switch to `allenai/OLMoE-1B-7B-0924`,
~14 GB, ungated), then `--ple-offload-embedding` does not apply to that model and is left to stage 2. If nothing
reproduces within the first hour, stop and ask the reporter in-thread for his `PYTORCH_CUDA_ALLOC_CONF` and exact
command.

## Fix paths

- **H1:** v1 `CustomAllreduce` already has a staging path for VMM graph memory: under
  `SGLANG_MEMORY_SAVER_CUDA_GRAPH` it calls `_all_reduce_impl(..., registered=False)` during capture, which
  copies each input into the pre-registered `cudaMalloc` buffer, so only that buffer reaches
  `cudaIpcGetMemHandle`. The fix turns staging on when the caching allocator is VMM-backed (probe with
  `is_vmm_pointer`, new helper `is_caching_allocator_vmm_backed` in `cuda_vmm_utils.py`). Custom all-reduce keeps
  working under `expandable_segments` at the cost of one device-to-device copy per call. Fallback, only if
  `ar_repro.py` shows staged v1 slower than NCCL at <= 1 MB: disable v1 on a VMM-backed allocator with a warning.
  PCIe-IPC as a default stays out of the code (#34528 made it opt-in on purpose).
- **H2:** make the eligibility check exercise the real IPC path (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`
  across ranks), not only P2P. If it fails, disable v1 with a warning and fall through to NCCL.
- **H3:** sm_12x PCIe guard like vLLM #47751, backed by our repro and latency numbers.

Tests and evidence for the PR: a GPU test that, with a VMM-backed allocator, v1 stays enabled, stages graph
inputs (`tms_cudagraph` is set), and its captured all-reduce matches NCCL exactly (for the fallback, the test
asserts v1 is disabled instead); server S2 with the fix boots and serves coherently; the `ar_repro.py` latency
table in the PR body.
The patch is dry-run checked against sglang main before the pod. PR goes from `adi-suresh01/sglang`; Aditya
approves the PR body and the #44291 comment before anything is opened or posted.

## Pod

- 2x RTX PRO 6000 (Server or Workstation Edition), PCIe. Catalog read 2026-10-14: $1.69/GPU-hr community,
  $2.49 secure, so $3.38 or $4.98/hr for two. Re-read stock and price before asking for approval.
  Not 2x RTX 5090: GeForce drivers disable P2P, so sglang's P2P check would turn v1 off before the bug.
- Before anything else: `nvidia-smi topo -m` must show PHB/NODE/SYS between the GPUs, not NV#. Record
  `nvidia-smi topo -p2p r` and the host driver (reporter: 580.178.04).
- Image `lmsysorg/sglang:dev-cu13` (record the digest), entrypoint overridden to start sshd as in vllm-49476.
- Volume disk 40 GB at `/workspace`, `HF_HOME=/workspace/hf` (room for the OLMoE fallback). No network volume.

## Files

| File | Purpose |
|---|---|
| `README.md` | This spec, then results |
| `setup.sh` | Version check (sglang, torch, flashinfer, driver) and model download |
| `run.sh` | Server variants S1-S4, reusing `scripts/serve_and_smoke.sh` (sglang serves `/health` and `/v1/chat/completions`) |
| `ar_repro.py` | Minimal `CustomAllreduce` + graph-capture repro and latency table; becomes the PR's regression test |

## Cost estimate and stop rules

Expected 1.5-2 h: setup 0.25 h, matrix S1-S4 at ~5 min each, `ar_repro.py` iterations, one fix check.
Hard cap 3 h: ~$10 community, ~$15 secure. Stop if not reproduced in the first hour. Terminate the pod the moment
the matrix and fix check are done; write code locally between runs, not on paid time.

## Stage 2 handoff (EP2 crash and TP2), not in scope here

- TP2 `block_n = 128` error likely comes from the FP8 block-scaled MTP experts (cookbook: the NVIDIA export is
  "NVFP4 experts, FP8 N-gram table, FP8 block-scaled MTP experts"). Test TP2 without NEXTN first. PR #36787 claims
  NVFP4 TP2/EP1 serves on its branch.
- EP2 illegal memory access candidates: the 47.7 GiB FP8 N-gram table in pinned host memory under NUMA pressure
  (#36787 hit Xid 31 MMU faults from that and fixed it with NUMA-interleaved pinning), or invalid expert ids
  reaching an EP kernel (same class as vllm#49476's `-1` sentinel).
- Needs the 133 GB checkpoint and 2x PRO 6000 for several hours; separate design and approval.

## Pod log

| Date | Pod id | GPU | Start | Stop | Cost |
|---|---|---|---|---|---|

## Results

| Run | Backend | Verdict | Fault | Notes |
|---|---|---|---|---|

## Outcome

Not started.
