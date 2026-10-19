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

## Results (2026-10-19)

Pod gizonyehl8edm8: 1x RTX PRO 6000 Blackwell Workstation Edition (the quote had priced the Server Edition, then the
Workstation Edition came into stock), secure, $2.59/hr, driver 610.57.04.
Image `nightly-e68efc68`: vLLM 0.32.1rc1.dev36. Cost $1.53 (estimate) of $3.42.

soak-graphs: **survived 25 minutes**, 7 rounds x 592 requests = 4,144 requests at concurrency 70, 0 failed, output
throughput 2,863-2,908 tok/s every round, mean TPOT 22.6-22.9 ms, no drift. The only error in server.log is the
`EngineDeadError` the API server logs when the harness stops it at 09:27:52, after round 7 finished at 09:27:45.
soak-eager was not needed and was stopped before it booted.

### Control on 0.28.0 (issues/vllm-54331-v028)

Pod 4lkd6rwrx2rnu0, same card and driver, `vllm/vllm-openai:v0.28.0`, same model, flags and workload. Cost $1.50.
**Also survived 25 minutes**: 4,144 requests, 0 failed, 2,760-2,790 output tok/s, mean TPOT 23.3-23.6 ms. The only
`EngineDeadError` is the harness stopping the server at 10:05:55, after round 7 ended at 10:05:49.

Verdict: **does not reproduce** with this workload on either version, so the nightly's survival says nothing about a
fix. Two differences from the report are large enough to matter:

- **KV cache dtype.** Both runs here stored KV in **bf16**: the server logs show `kv_cache_dtype=auto` and an
  attention block of 1056 tokens (0.28.0) / 1088 (nightly). The checkpoint's `hf_quant_config.json` declares
  `kv_cache_quant_algo: FP8`, but `auto` does not turn that on. The report ran `float8_e4m3fn` with a 2096-token
  block and FlashInfer `xqa` decode. (An earlier version of this README and of harness.toml said FP8 KV; that was
  wrong.)
- **Vision path.** The report is offline batch *video* inference, and both runs here pass `--language-model-only`,
  so the vision encoder never runs (the report's 0.25 runs also hit a separate ViT illegal memory access).

Smaller ones: offline `LLM` engine vs the OpenAI server, driver 610.57.04 vs 610.43.02, and `ornith-ai/Ornith-1.5`
for most of the report's arms.

What it does show: text-only serving of this model with a bf16 KV cache, saturated at concurrency 70 over shared
prefixes with CUDA graphs on, ran 25 minutes clean on 0.28.0 and on the 10-19 nightly (2.9K output tok/s on the
nightly, 3% above 0.28.0).

## Follow-up: the reporter's 0.28.0 setup (issues/vllm-54331-ornith)

Every 0.27.1 and 0.28.0 crash in the report was on `ornith-ai/Ornith-1.5-35B-A3B-NVFP4`, whose `quantization_config`
carries an fp8 `kv_cache_scheme`. On 0.28.0, `--kv-cache-dtype auto` reads only that field, so Ornith gets fp8 KV by
default and Qwen3.6 (FP8 declared only in `hf_quant_config.json`) gets bf16, which is likely what happened in the
report's Qwen3.6 runs too. On sm_120, fp8 KV also moves attention from FLASH_ATTN to FlashInfer with xqa decode.
The kit runs Ornith-1.5 @ 94e431d on `vllm/vllm-openai:v0.28.0` with default flags, and the hook stops before the soak
unless the server log shows `decode_backend=xqa` with `float8_e4m3fn`, as the report's does.

| Variant | Workload | If it dies |
|---|---|---|
| video (runs first) | vision encoder on; `soak_video.py` | the report reproduces on its own setup |
| text | issues/vllm-54331's text soak, `--language-model-only` | the trigger does not need the multimodal path |

Video workload (`soak_video.py`): 200 synthetic 96-frame 640x480 clips, of which vLLM's loader samples 32 (a clip of
exactly 32 frames would be resampled to 8 by the Qwen3-VL processor). Request n is window n % 8 of clip
(n // 8) % 200, each window one more turn over the same system prompt and clip, 512 output tokens with `ignore_eos`,
concurrency 70 for 25 minutes. A new clip starts every few seconds, so the encoder runs for the whole soak while
later windows reuse cached blocks. A probe first checks that the clip adds prompt tokens.

If either variant dies, the next run is the same variant on the nightly (it has #50729, the mamba state-copy race fix
that 0.28.0 lacks), to tell the reporter whether upgrading is enough.

### Results (2026-10-19)

`harness quote`: 1x RTX PRO 6000 Server Edition, secure, $2.49/hr, 1.5 h, $4.11; budget raised to $5.36 after the
first pod. Spent $3.69.

- Pod c8q968e99wzny0 (US-NE-1) was still pulling the image 30 min after create; the harness terminated it ($1.25).
  US-NE-1 is now in `avoid_data_centers`.
- Pod m9jmgtbnf21xgu (EUR-IS-1), driver 595.91.07, vLLM 0.28.0, FlashInfer 0.6.16.post3, torch 2.13.0+cu130, no
  `flashinfer-cubin` (the report had to uninstall it too). Weights in 60 s. $2.45.

Both variants logged the report's attention setup: `Setting attention block size to 2096 tokens`, `decode_backend=xqa,
kv_cache_dtype=torch.float8_e4m3fn, arch=sm120`. KV pool 59.19 GiB (4,408,785 tokens), `max_num_batched_tokens=8192`,
`max_cudagraph_capture_size` 512, prefix caching on.

| Variant | Ran | Requests | Failed | Output tok/s | Notes |
|---|---|---|---|---|---|
| video | 25.1 min | 8,401 | 0 | 2,856 (2,950-2,990 per minute at steady state) | one clip adds 1,234 prompt tokens; mean prompt 2,818; 63% of prompt tokens from the prefix cache |
| text | 15.4 min | 3 rounds x 592, then 350 of round 4 | 0 | 2,542-2,554 per round | stopped by the container, see below |

video: verdict OK. The only `ERROR` lines are the harness stopping the server at 22:52:30, after the last `200 OK`.

text: every log ends at 23:11:52, in the same second: `server.log` (70 requests running, 3,233 generation tok/s, every
response 200), round 4's progress, and `memory.jsonl` (36 of 188 GB), which a separate process writes. No crash
signature, no `EngineDeadError`. The harness then found no SSH endpoint ("job variant-text died without an exit code
(killed, or the container restarted)") and terminated the pod. The container went away from outside the processes we
ran; the cause is not visible from inside. So text counts as 15.4 clean minutes, not 25.

Verdict: **does not reproduce** on the report's checkpoint, version, KV dtype, attention backend and a video workload,
for 25 minutes of video and 15 of text. The report died at 2.5-7.75 min. Remaining differences: offline `LLM` vs the
OpenAI server, real video vs synthetic clips (frames per window and resolution unknown), driver 610.43.02 vs
595.91.07, Workstation vs Server Edition.
