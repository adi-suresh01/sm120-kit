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
| H1 | VMM-backed memory (`expandable_segments`) breaks v1's IPC handshake on graph buffers | S2 fails in `get_graph_buffer_ipc_meta` (`invalid argument`, or `failed to get pointer attr` one step earlier), S1 serves | S2 serves |
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
  `cudaIpcGetMemHandle`. The fix stages any graph input that is VMM memory, checked per input with the existing
  `is_vmm_pointer` at capture time, so it also covers the memory saver and symmetric-memory pools, and costs
  nothing at replay beyond one device-to-device copy per staged call. Fallback, only if `ar_repro.py` shows
  staged v1 more than 5% slower than pynccl at 2-16 KB in both runs: `should_custom_ar` declines VMM graph
  inputs so pynccl takes them.
  PCIe-IPC as a default stays out of the code (#34528 made it opt-in on purpose).
- **H2:** make the eligibility check exercise the real IPC path (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`
  across ranks), not only P2P. If it fails, disable v1 with a warning and fall through to NCCL.
- **H3:** sm_12x PCIe guard like vLLM #47751, backed by our repro and latency numbers.

Tests and evidence for the PR: a GPU test that, with a VMM-backed allocator, v1 stays enabled, stages VMM
graph inputs, and its captured all-reduce matches NCCL exactly (for the fallback, the test asserts the input
is declined instead); server S2 with the fix boots and serves coherently; the `ar_repro.py` latency
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
| `run.sh` | Server variants S1-S4, reusing `harness/pod/serve_and_smoke.sh` (sglang serves `/health` and `/v1/chat/completions`) |
| `ar_repro.py` | Minimal `CustomAllreduce` + graph-capture repro and latency table; becomes the PR's regression test |
| `ar_matrix.sh` | Runs `ar_repro.py` over modes x allocators x repeats with timeouts, keeping the full log |

## Cost estimate and stop rules

Expected 1.5-2 h: setup 0.25 h, matrix S1-S4 at ~5 min each, `ar_repro.py` iterations, one fix check.
Hard cap 3 h: ~$10 community, ~$15 secure. Stop if not reproduced in the first hour. Terminate the pod the moment
the matrix and fix check are done; write code locally between runs, not on paid time.

## Stage 2 handoff (EP2 crash and TP2), not in scope here

Stage 2 spec and results: `stage2.md`. The notes below are the original handoff.

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
| 2026-10-15 | wrbkikt79p44es | 2x RTX PRO 6000 Server, secure, EUR-IS-2, host CUDA 13.2, $4.98/hr (no community or WK stock) | 08:15:43Z | 08:42:38Z (terminated) | ~$2.23 (26.9 min x $4.98/hr) |

## Environment (pod wrbkikt79p44es)

```
GPUs     : 2x RTX PRO 6000 Blackwell Server Edition, 97887 MiB, cc 12.0, driver 595.91.07, host CUDA 13.2
Topology : GPU0-GPU1 SYS (PCIe across the CPU interconnect, no NVLink); nvidia-smi topo -p2p r: OK both ways
Image    : lmsysorg/sglang:dev-cu13, sglang 0.0.0.dev1+ga6a8c5601 (editable, /sgl-workspace/sglang),
           torch 2.14.1+cu130, flashinfer-python 0.7.1, triton 3.8.0, transformers 5.19.0
Allocator: no PYTORCH_*ALLOC_CONF preset in the image
```

The image runs Python from a venv (`/opt/sglang`) set through the image's ENV. With the entrypoint overridden for
sshd, SSH sessions do not inherit it, so every command sourced `/proc/1/environ` exported to
`/workspace/container_env.sh`.

## Results (2026-10-15)

Server runs, `Qwen/Qwen3-0.6B`, TP2, CUDA graphs on:

| Run | Backend | Verdict | Fault | Notes |
|---|---|---|---|---|
| s1 | v1 (registers 2850 + 2052 graph addresses) | OK, boot 81 s | - | default allocator |
| s2 | v1 | BOOT_FAIL, 35 s | `RuntimeError: invalid argument` in `get_graph_buffer_ipc_meta` | `expandable_segments`; reporter's attempt 1 |
| s3 | v1 + PCIe-IPC | BOOT_FAIL, 40 s | same | PCIe-IPC all-reduce enabled and tuned, v1 still captures and fails: not a workaround |
| s4 | NCCL | OK, 41 s | - | `--disable-custom-all-reduce` |
| s2-fix | v1 (2850 + 2052, staged) | OK, 41 s | - | `expandable_segments` + fix; same text as s1-fix at temperature 0 |
| s1-fix | v1 (2850 + 2052) | OK, 41 s | - | default allocator + fix |

`ar_repro.py`, 8 captured all-reduces per graph, median of 2 runs, rank 0. The dispatcher picked v1
`CustomAllreduce` in every run.

| KB | registered, default | staged, default | staged, VMM | pynccl, default | nccl, default |
|---|---|---|---|---|---|
| 2 | 4.58 us | 5.38 us | 5.38 us | 12.39 us | 12.37 us |
| 16 | 5.25 us | 6.17 us | 6.13 us | 14.39 us | 14.34 us |
| 256 | 14.07 us | 14.95 us | 15.01 us | 22.63 us | 22.66 us |
| 1024 | 42.01 us | 43.08 us | 43.08 us | 47.67 us | 47.66 us |
| 8192 | 309.28 us | 318.30 us | 321.34 us | 335.47 us | 334.73 us |

- registered + VMM: fails at the first size, `RuntimeError: invalid argument` at `custom_all_reduce.cuh:614`
  (`cudaIpcGetMemHandle`). Registered + default and staged under both allocators: zero error at every size.
- Staging costs about 0.8-0.9 us at decode sizes and about 1-4% from 1 MB up; it stays 2.3-2.7x faster than
  pynccl at 2-16 KB, so the staging fix was kept (Task 12 gate).
- pynccl under VMM returned wrong sums (max abs error 3329) at most sizes in both runs. A server-level check
  (`--disable-custom-all-reduce`, default vs `expandable_segments`, 8 concurrent requests) gave 8/8 identical
  responses, so this is an artifact of how `ar_repro.py` drives pynccl, not a server bug. Not reported.

Regression test (`test_custom_all_reduce_v1_vmm.py`, fork commit 7706bbf):
- without the fix: `1 failed, 1 passed`; `test_allocator_is_vmm` passes, `test_graph_all_reduce_matches_nccl[16-False]`
  fails with `RuntimeError: invalid argument`.
- with the fix: `9 passed`.
- default-path check with the fix (registered mode, default allocator): zero error at every size, 4.54 us at 2 KB
  (4.58 us before the fix), so the new in-capture `is_vmm_pointer` call is harmless for default users.

Decode A/B, `sglang.benchmark.one_batch_server`, input 256 / output 128, steady-state generation tok/s, one run each:

| Batch | before fix, default | after fix, default | after fix, `expandable_segments` |
|---|---|---|---|
| 1 | 704 | 705 | 681 |
| 8 | 3988 | 3995 | 3883 |
| 32 | 12261 | 12310 | 12030 |

Default users see no change; `expandable_segments` now boots and decodes 2-3% slower than the default allocator.

## Outcome

H1 confirmed. On 2x PCIe RTX PRO 6000, sglang falls back to v1 `CustomAllreduce`, and with
`expandable_segments` (which the Qwen3.8 cookbook recommends) v1's CUDA-graph registration calls
`cudaIpcGetMemHandle` on VMM memory and fails with `invalid argument`. The fix (fork branch
`fix/custom-ar-vmm-graph-buffers`, commit ad19c6a) stages VMM graph inputs through v1's pre-registered IPC
buffer. It is verified red/green on the pod, leaves default users unchanged, and costs 2-3% decode for
`expandable_segments` users, who could not boot before.

Workarounds today: drop `expandable_segments`, or `--disable-custom-all-reduce`. `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1`
does not help (s3).

Cost: ~$2.23 of the ~$10-15 estimate. Next: PR from the fork and a comment on #44291.
Stage 2 (EP2 crash, TP2 alignment) stays open per the handoff above.
