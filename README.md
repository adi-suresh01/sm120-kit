# sm120-kit

Reproduce, root-cause, and fix LLM serving bugs on NVIDIA sm_120 GPUs (RTX PRO 6000 Blackwell, RTX 5090) and on
PCIe-only multi-GPU servers, where vLLM, SGLang and FlashInfer get less testing than on datacenter parts.

Each folder under `issues/` is one upstream issue or deployment question: the reporter's setup, hypotheses, an
experiment matrix, and a `harness.toml` that runs the matrix on a rented GPU inside a fixed budget. Results go in
the folder's README with the cost of every pod.

## Results

| Kit | Question | Outcome | GPU cost |
|---|---|---|---|
| [vllm-49476](issues/vllm-49476) | b12x MoE workspace OOM, then CUDA-graph Xid 31 | Reproduced. The graph crash is a `-1` padding id in `topk_ids` indexed by FlashInfer 0.7.0.post1's b12x kernels; fixed by FlashInfer 0.7.1. The OOM is fixed by vllm#60069 | $1.09 |
| [vllm-47990](issues/vllm-47990) | SM120 CUTLASS blockwise FP8 GEMM rejects N % 128 != 0 | Reproduced on v0.31.0. vllm#48588 rebased and verified; vllm#47988's guard falls back to Marlin, measured against padded CUTLASS | $0.94 |
| [vllm-56771](issues/vllm-56771) | DeepSeek-V4.1-Flash + DSpark illegal memory access in sparse-MLA prefill | Does not reproduce on FlashInfer 0.7.2 + vllm#60762 (10.7K and 106K prompts, graphs on). Loading at TP4 exposed a host-memory problem: data for vllm#58857 (merged) and our vllm#62110 | $43.46 |
| [vllm-61100](issues/vllm-61100) | Apertus-70B FP8 degenerates on sm_120 in 0.25.1 | Does not reproduce on this host; reporter-side splits posted | $1.11 |
| [sglang-44291](issues/sglang-44291) | Crashes serving Qwen3.8-Flash-Next NVFP4 on 2x PCIe PRO 6000 | Three bugs. Custom all-reduce with expandable segments: sglang#44448. EP2 + NEXTN expert ids: sglang#44480 (merged). TP2 block alignment: sglang#42566 validated | $16.5 |
| [sglang-33422](issues/sglang-33422) | DeepSeek-V4-Flash prefill 2-7K tok/s vs vLLM ~12.5K on 4x PRO 6000 | No gap on v0.5.22: 8,983 tok/s with the reporter's flags vs 8,171 for the vLLM recipe on 9 x 90K prompts; DP attention + EP4 reaches 11,756 | $21.47 |
| [puget-team-llm](issues/puget-team-llm) | Puget Systems Team LLM pack flags on 2x PRO 6000 Workstation | `--enforce-eager` with prefix caching off costs 4.2x output tok/s at 1 user and 2.3x at 16; custom all-reduce with P2P adds 13% and 9%. Same answers | $3.18 |
| [vllm-54331](issues/vllm-54331) | Hybrid-GDN NVFP4 dies under saturation with CUDA graphs on (0.26-0.28) | Not reproduced. 25-min soaks at concurrency 70 on 0.28.0 and the nightly (text, bf16 KV), then on the report's own checkpoint with fp8 KV and xqa decode: 25 min of video and 15 of text, 0 failed requests | $6.72 |
| [sglang-44291-stack](issues/sglang-44291-stack) | The #44291 repro on the stack built over sglang#44480 | Kit ready, not run | - |

## Upstream PRs

| PR | What | State |
|---|---|---|
| [sglang#44480](https://github.com/sgl-project/sglang/pull/44480) | Map expert ids for MoE layers whose runner needs local ids under EP | Merged 2026-10-20 ([e2e1a97](https://github.com/sgl-project/sglang/commit/e2e1a97fc3)) |
| [sglang#44448](https://github.com/sgl-project/sglang/pull/44448) | Stage custom all-reduce inputs through the registered buffer when expandable segments are on | Open |
| [vllm#62110](https://github.com/vllm-project/vllm/pull/62110) | Evict loaded checkpoint shards from the page cache when the checkpoint exceeds RAM | Open |

## Layout

```
harness/            Rust CLI that rents, runs and tears down pods (src/), and the bash it runs on the pod (pod/)
issues/<kit>/       README (plan + results), harness.toml, optional setup.sh, hooks.sh, patches/, harness-ledger.json
docs/runpod.md      RunPod setup and spend rules
```

## Running a kit

```
cargo install --path harness --locked
harness quote vllm-47990                 # GPU, $/hr, deadline, worst-case cost
harness budget set vllm-47990 6 --note "approved"
harness run vllm-47990                   # up, setup.sh, every variant, pull results, terminate
harness budget show vllm-47990
```

The harness needs `RUNPOD_API_KEY` in the environment and an SSH key registered with RunPod; see
[harness/README.md](harness/README.md) and [docs/runpod.md](docs/runpod.md). Every pod carries an on-pod watchdog
that terminates it at the deadline, and nothing rents without a budget.
