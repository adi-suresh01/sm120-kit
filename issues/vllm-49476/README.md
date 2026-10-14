# vllm#49476: b12x MoE workspace OOM, then CUDA-graph Xid 31 on sm_120

- Issue: https://github.com/vllm-project/vllm/issues/49476
- Draft fix: https://github.com/vllm-project/vllm/pull/60069 (head 5a80ca1d, vendored in `patches/`)
- Reporter: jhsmith409 (Jim Smith, Joshua8.AI). Company reporter, still blocked as of 2026-10-05.
- Related: FlashInfer #6434, #4317, PR #4827 (b12x MoE workspace/memory cluster).

## Reporter setup (verbatim)

```
vLLM       : 0.31.0 (vllm/vllm-openai:v0.31.0)
FlashInfer : 0.7.0.post1, PyTorch 2.13.0+cu130
GPU        : 1x RTX 5090 (31.45 GiB), TP=1/PP=1
Model      : nvidia/Qwen3.6-35B-A3B-NVFP4, --moe-backend=flashinfer_b12x, --max-model-len=262144,
             --gpu-memory-utilization=0.92, --kv-cache-dtype=fp8, --max-num-seqs=16
```

- Unpatched: OOM in `profile_run`. `FlashInferB12xExperts._ensure_wrapper()` builds one `B12xMoEWrapper` per
  MoE layer, each calling `allocate_sm120_static_workspace`, so workspaces scale with layer count.
- `--moe-backend marlin`: boots and serves correctly. `flashinfer_cutlass`: rejected at startup.
- v0.25.1 runs the same model with a different MoE backend; only `flashinfer_b12x` fails.
- With PR #60069 (one workspace shared across layers): `--enforce-eager` boots with 6.31 GiB KV cache and
  correct output. With CUDA graphs: illegal memory access (Xid 31) during graph-capture profiling at batch 2.
- Reporter's open question: does unpatched v0.31 + b12x + CUDA graphs also crash on a 96 GB PRO 6000
  (where the OOM does not happen)? That separates "PR introduced the fault" from "pre-existing fault".

## Hypotheses

| # | Hypothesis | Confirmed by | Killed by |
|---|---|---|---|
| H1 | Xid 31 is pre-existing in b12x + CUDA graphs, unrelated to the PR | `baseline-graphs` crashes on PRO 6000 | `baseline-graphs` serves |
| H2 | Shared workspace is sized for the first caller's shape; a larger batch or different layer overruns it during capture | `patched-graphs` crashes, `baseline-graphs` serves; `patched-min` faults at size 2 | Fault persists with per-batch sizing |
| H3 | Captured graphs hold workspace pointers that the shared allocator later frees or reallocates | Fault only after a reallocation event in the log | Workspace allocated once and never resized |
| H4 | b12x kernels are not graph-capture safe on sm_120 (host sync or dynamic alloc inside the kernel path) | `baseline-graphs` crashes at capture regardless of memory | Any graphs variant serves |

## Experiment matrix (1x RTX PRO 6000, 96 GB)

| Variant | Patch | Graphs | Expected if H1 | Expected if H2/H3 |
|---|---|---|---|---|
| marlin | no | yes | OK | OK |
| baseline-eager | no | no | OK | OK |
| baseline-graphs | no | yes | crash | OK |
| patched-eager | yes | no | OK | OK |
| patched-graphs | yes | yes | crash | crash |
| patched-min | yes | [1,2] + LAUNCH_BLOCKING | crash, localized | crash, localized |

Then repeat `baseline-eager` and `patched-eager` with `MEM_BUDGET_GIB=31.45` to confirm the OOM and the PR's fix
under a 5090-sized budget without renting a 5090.

## Run

```
export HF_HOME=/workspace/hf        # persistent volume
mkdir -p results && bash scripts/env_report.sh | tee results/env.txt
bash issues/vllm-49476/setup.sh
bash issues/vllm-49476/run.sh all
MEM_BUDGET_GIB=31.45 bash issues/vllm-49476/run.sh baseline-eager
MEM_BUDGET_GIB=31.45 bash issues/vllm-49476/run.sh patched-eager
bash issues/vllm-49476/run.sh patched-min      # only if a graphs variant crashed
```

## Cost estimate

1x RTX PRO 6000 at ~$1.10/hr. Install + model download ~1 h, matrix ~1.5-2 h (each boot with 256K context
and graph capture is several minutes), debug ~2-4 h. Total ~5-7 h, ~$6-8. Budget $15 with buffer.
The 10 h / $22 estimate assumed a separate 5090 pod; `MEM_BUDGET_GIB` removes that.

## Pod log

| Date | Pod id | GPU | Start | Stop | Cost |
|---|---|---|---|---|---|

## Results

| Variant | Verdict | Boot s | Fault | Notes |
|---|---|---|---|---|

## Outcome

Not started.
