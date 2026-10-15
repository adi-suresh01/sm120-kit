# Five LLM serving bugs on PCIe RTX PRO 6000, for $18 of rented GPU time

I'm Aditya, an ML engineer who most recently served models on PCIe RTX PRO 6000 clusters at Kos AI, now founder of
Provedex. The RTX PRO 6000 and the RTX 5090 are sm_120 Blackwell, and a lot of vLLM and SGLang is tuned for H100 and
B200 with NVLink, so things that work there crash on these cards. This week I took two open issues where someone
was still blocked on these GPUs, reproduced them on rented RunPod GPUs, and found the root causes.

- Issues: [vLLM #49476](https://github.com/vllm-project/vllm/issues/49476) and [SGLang #44291](https://github.com/sgl-project/sglang/issues/44291)
- Bugs root-caused: 5
- PRs I opened: 2 on SGLang ([#44448](https://github.com/sgl-project/sglang/pull/44448), [#44480](https://github.com/sgl-project/sglang/pull/44480)), not merged yet
- Pod time: 3.8 hours across 3 pods, about $18

## 1. vLLM #49476: Qwen3.6-35B-A3B NVFP4 on one RTX 5090

The reporter ran vLLM 0.31.0 with `--moe-backend flashinfer_b12x` at 262k context. Startup ran out of memory. With
the open fix for that ([#60069](https://github.com/vllm-project/vllm/pull/60069)) it booted in eager mode, but CUDA
graphs crashed with an illegal memory access (Xid 31). The open question was whether the fix caused the crash.

I ran it on one 96 GB PRO 6000 and emulated the 5090 with a ballast process that holds memory so only 31.45 GiB is
free when vLLM starts. It was two bugs, and the first one hid the second:

- **OOM:** vLLM builds one b12x workspace per MoE layer. On the 96 GB card b12x reserved 16.6 GiB more than marlin,
  which a 32 GB card can't fit. Sharing the workspace across layers (#60069) got the 17.2 GiB back.
- **Crash:** not caused by the fix. It reproduced with no PR on the 96 GB card. vLLM writes expert id `-1` into the
  padding rows of CUDA-graph batches, and FlashInfer 0.7.0.post1's b12x kernels index with that `-1` and write out
  of bounds. The first padded capture size is 2, which is exactly where the reporter's run died.

What serves today under the 5090 ceiling with CUDA graphs: #60069 plus any one of FlashInfer 0.7.1,
[vLLM #57946](https://github.com/vllm-project/vllm/pull/57946), or `VLLM_MOE_SKIP_PADDING=0`. All three served the
reporter's config with padded concurrent batches; with #57946, 13 of 13 concurrent requests came back coherent with
550k tokens of KV cache. Full table in
[the issue comment](https://github.com/vllm-project/vllm/issues/49476#issuecomment-6186410252).

26 minutes on 1 GPU, $1.09.

## 2. SGLang #44291: Qwen3.8-Flash-Next NVFP4 on 2x PCIe RTX PRO 6000

The reporter had two PRO 6000 Server Edition cards with no NVLink and tried three configs:

1. EP2 as is: `RuntimeError: invalid argument` in custom all-reduce setup.
2. EP2 with custom all-reduce off: illegal memory access while capturing the speculative decoding draft's CUDA graphs.
3. TP2 instead: `320 is not divisible by weight quantization block_n = 128`.

That looks like one bug getting worse. It was three separate bugs.

### Bug 1: custom all-reduce with expandable_segments

Without NVLink, SGLang falls back to its v1 custom all-reduce. During CUDA graph capture, v1 shares the graph
buffers between the two GPUs with `cudaIpcGetMemHandle`. With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
which the model's cookbook recommends for long context, those buffers are VMM memory, and `cudaIpcGetMemHandle`
rejects VMM memory. It reproduces with Qwen3-0.6B, so this took a 27 minute pod, not the 133 GB checkpoint.

My fix ([#44448](https://github.com/sgl-project/sglang/pull/44448)) copies VMM graph inputs into v1's own
pre-registered buffer, a path SGLang already had for its memory saver. It adds about 0.8-0.9 us per all-reduce at
decode sizes and stays about 2.3x faster than NCCL at 2-16 KB. Users on the default allocator see no change.

### Bug 2: illegal memory access at EP2

The NVIDIA export keeps the experts of the MTP draft head in FP8 block-scaled format. With
`--moe-runner-backend flashinfer_cutlass`, SGLang runs those experts on Triton instead. Under EP2 each GPU
holds 256 of the 512 experts, so Triton needs local expert ids. The dispatcher only looked at the global backend
(cutlass, which takes global ids) and skipped the translation, so ids 256-511 read past the local weights. TP1 and
TP2 never translate ids, which is why only EP crashed.

Two open PRs change the same dispatcher code ([#40172](https://github.com/sgl-project/sglang/pull/40172),
[#43330](https://github.com/sgl-project/sglang/pull/43330)). I ran both, and neither changes this crash.

My fix ([#44480](https://github.com/sgl-project/sglang/pull/44480)) makes the dispatcher also check the runner the
layer actually built, and that check can only turn the translation on, so layers on other backends behave exactly
as before. A review before opening the PR caught that my first version would have broken ROCm's AITER MoE under
EP, which is why it ended up that narrow.

### Bug 3: TP2 block size

The FP8 MTP experts have a 640-wide dimension. Split over 2 GPUs that's 320, which isn't a multiple of the
128-wide quantization block. [#42566](https://github.com/sgl-project/sglang/pull/42566) (not mine) refines the
scale blocks to fit. I confirmed it on this exact checkpoint.

### Workaround with no patches

RadixArk publishes the same MTP head in BF16. Pointing the draft at it with
`--speculative-draft-model-path <dir> --speculative-draft-model-quantization modelopt_fp4` serves at both EP2 and
TP2 on stock SGLang. The head only lives in 3 of the shards (14.7 GB of 135 GB), so I downloaded just those and
trimmed the index to them. Command and script are in
[the issue comment](https://github.com/sgl-project/sglang/issues/44291#issuecomment-6192214621).

### Numbers (2x RTX PRO 6000, PCIe, NEXTN 3/1/4)

GSM8K is 200 questions, 5-shot, thinking on. Accept length is the mean over the GSM8K run. Throughput is
`bench_serving` random 1024 in / 512 out, output tok/s.

| Config | GSM8K | Accept length | c=1 | c=16 | c=64 |
|---|---|---|---|---|---|
| EP2 + #44480 | 0.975 | 2.94 | 181 | 950 | 1363 |
| EP2, RadixArk draft | 0.970 | 2.94 | 183 | 960 | 1359 |
| TP2, RadixArk draft | 0.985 | 2.93 | 187 | 935 | 1286 |
| TP2 + #42566 | 0.975 | 2.94 | 182 | 953 | 1281 |

An open PR with sm_120 kernels for this model ([#36787](https://github.com/sgl-project/sglang/pull/36787)) reports
216 tok/s at c=1 on its own branch, so there is still headroom on mainline.

### What I couldn't test

The second pod's GPUs failed SGLang's own P2P check (`gpu_p2p_access_check`, false 3 out of 3 times) while
`nvidia-smi` and PyTorch said P2P was fine, and custom all-reduce hung there for every model, including Qwen3-0.6B.
So #44448 still has no result on the real checkpoint. If you rent these cards, check P2P with the engine's own test,
not with `nvidia-smi`.

Stage 1 took 27 minutes on 2 GPUs ($2.23), stage 2 took 2.9 hours ($14.30).

## Takeaways

- Search PRs by function and kernel names, not by issue number. vLLM #49476's OOM had two more open fixes besides the
  one linked in the thread, and neither cited the issue.
- A report with several crashes is often several bugs: #44291 was three, #49476 was two. Each got its own repro.
- Both issues had the same class of bug: expert ids that are fine for one kernel backend reaching another one that
  reads them differently (`-1` padding in vLLM, global vs local ids in SGLang). Mixed-backend MoE layers are where I
  would look first on the next crash.
- Reproduce on the smallest thing that still fails. A 0.6B model was enough for the all-reduce bug, and one 96 GB
  card with a memory ballast stood in for a 5090.

## Status (2026-10-15)

| Bug | Root-caused | Fix | Merged |
|---|---|---|---|
| vLLM b12x OOM | yes | #60069 (not mine) | no |
| vLLM `-1` padding crash | yes | FlashInfer 0.7.1, already on vLLM main | yes |
| SGLang custom all-reduce + VMM | yes | #44448 (mine) | no |
| SGLang EP2 expert ids | yes | #44480 (mine) | no |
| SGLang TP2 block size | yes | #42566 (not mine) | no |

If you're serving on PRO 6000 or 5090 boxes and hit a crash or a number that looks low, I'd love to hear about it.
