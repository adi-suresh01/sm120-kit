# vllm#54331: hybrid-GDN NVFP4 dies under sustained load with CUDA graphs on (sm_120)

Reporter (Miracle Mobile, [vllm#54331](https://github.com/vllm-project/vllm/issues/54331), 2026-08-29, 0 replies):
on 1x RTX PRO 6000 Workstation Edition, `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` and `nvidia/Qwen3.6-35B-A3B-NVFP4`
(Qwen3.5-MoE family, 30 of 40 layers GDN linear attention, modelopt mixed FP8/NVFP4, FP8 KV) die after 2-8 min of
saturated serving (up to 70 concurrent multi-turn requests over a shared prefix) whenever CUDA graphs are on:
SIGSEGV in `cuGraphLaunch -> at::cuda::CUDAGraph::replay()`, or an asynchronous `illegal instruction` /
`unspecified launch failure`. Clean on 0.24.0; fails on 0.26.0, 0.27.0, 0.27.1, 0.28.0. `PIECEWISE`, `TRITON_ATTN`,
`expandable_segments` and `max_num_seqs=128` all still crash; `enforce_eager` and a fresh engine per round survive;
a 100-min soak at ~65% utilization did not crash; Gemma4-26B (MoE, no GDN) is clean.

Why it matters here: issues/puget-team-llm measured 4.2x from dropping `--enforce-eager` on a Qwen3.5-family hybrid
model, with short benchmarks only. If this crash still exists, that advice needs a saturation soak first.

## Hypotheses

1. Still present on the nightly (0.32.1rc1): soak-graphs dies within 25 min, soak-eager survives.
2. Fixed between 0.28.0 and the nightly: soak-graphs survives. Only meaningful if the same workload kills 0.28.0
   (a second pod on `vllm/vllm-openai:v0.28.0`), otherwise the workload may just not trigger it.
3. Prefix reuse matters: the report's workload shares long prefixes, so the soak reuses 16 prefixes of 4096 tokens.

## Matrix

| Variant | Flags | Expected if H1 |
|---|---|---|
| soak-graphs | defaults (FULL_AND_PIECEWISE graphs, prefix caching) | POST_FAIL with one of the three surfaces |
| soak-eager | `--enforce-eager` | survives 25 min; run only if soak-graphs dies |

Workload: `vllm bench serve --dataset-name prefix_repetition`, prefix 4096, suffix 1024, 16 prefixes, output 1024
with `--ignore-eos`, concurrency 70, rounds of 600 prompts for 25 minutes.

## Cost

`harness quote` on 2026-10-19: 1x RTX PRO 6000 Server Edition (Workstation out of stock), secure, $2.49/hr, 1.25 h
planned, $3.42 worst case.
