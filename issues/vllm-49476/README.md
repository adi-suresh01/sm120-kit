# vllm#49476: b12x MoE workspace OOM, then CUDA-graph Xid 31 on sm_120

- Issue: https://github.com/vllm-project/vllm/issues/49476
- Draft fix: https://github.com/vllm-project/vllm/pull/60069 (head 5a80ca1d, vendored in `patches/`)
- Reporter: jhsmith409 (Jim Smith, Joshua8.AI). Company reporter, still blocked as of 2026-10-05.
- Related: FlashInfer #6434, #4317, PR #4827 (b12x MoE workspace/memory cluster).
- Re-checked 2026-10-14: issue open, no comments since 10-05. PR #60069 still draft at 5a80ca1d, no reviews.
  Nobody has answered the PRO 6000 question yet.

## Other PRs on the same code (keyword search, 2026-10-14)

None of these cite #49476, which is why the first sweep missed them.

- vLLM #53081 (lucifer1004, opened 08-20, unreviewed): same approach as #60069, sharing static workspace,
  dynamic workspace, and output across layers via the FI#4603 parameters, with nearly the same key.
- vLLM #48698 (dumko2001, opened 07-15, draft): one shared wrapper sized `max_num_tokens * dp_size`, for #47982
  (DP all-gather overflow). A third party validated it on 2x RTX 5060 Ti with FlashInfer 0.6.14.
- FlashInfer #4603 (merged 08-25): adds the `shared_*` wrapper parameters all three rely on.
- FlashInfer #4827 (ormandj, open, unreviewed, head 0ab76fd6): a larger prefill after decode-graph capture can
  free a workspace or prepared weights (`_WEIGHT_CACHE`, padded weights) that captured graphs still point to,
  and the next replay hits an illegal memory access. That is H3. vLLM's wrapper path calls
  `_get_weight_views`, which uses `_WEIGHT_CACHE`, so it may apply here. Vendored as
  `patches/fi4827-graph-pinning-on-0.7.0.post1.diff` (rebased: one hunk needed two context comment lines
  dropped; apply and revert verified against the 0.7.0.post1 wheel; pyflakes clean).

What this means: the workspace-sharing fix exists three times and has zero reviews. Our contribution is the
PRO 6000 + 5090-ceiling A/B showing which combination serves with CUDA graphs, posted on #49476 and linked from
the three PRs and FI#4827. A PR from us only makes sense if a gap remains after that.

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

## Prior 96 GB data point (jahnclawdmonet, 2026-08-26)

- v0.25.1 + FlashInfer 0.6.13, same model, b12x, 180K context, `--max-num-seqs=8`,
  `--max-num-batched-tokens=8192`, no `--enforce-eager`: boots and serves on 1x PRO 6000.
- Available KV cache: 52.89 GiB with b12x vs 64.36 GiB with marlin, so b12x reserves ~11.47 GiB more.
- Transient peak during b12x profiling ~91.8 GiB, above the 0.92 target.

What this changes:
- b12x with CUDA graphs served on 0.25.1. A graphs crash on unpatched 0.31 would be a regression between
  0.25.1 and 0.31, which weakens H4 as a property of b12x on sm_120 in general.
- `--gpu-memory-utilization` does not bound the profiling peak, so scaling it down on a 96 GB card cannot
  reproduce the 5090 OOM. `MEM_BUDGET_GIB` now also starts `harness/pod/vram_ballast.py`, which holds memory so
  only 31.45 GiB is free before vLLM starts.

## Hypotheses

| # | Hypothesis | Confirmed by | Killed by |
|---|---|---|---|
| H1 | Xid 31 is pre-existing in b12x + CUDA graphs, unrelated to the PR | `baseline-graphs` crashes on PRO 6000 | `baseline-graphs` serves |
| H2 | Shared workspace is sized for the first caller's shape; a larger batch or different layer overruns it during capture | `patched-graphs` crashes, `baseline-graphs` serves; `patched-min` faults at size 2 | Fault persists with per-batch sizing |
| H3 | Captured graphs hold workspace pointers that the shared allocator later frees or reallocates | Fault only after a reallocation event in the log, or only under the 31.45 GiB ballast | Workspace allocated once and never resized |
| H4 | b12x kernels are not graph-capture safe on sm_120 (host sync or dynamic alloc inside the kernel path) | `baseline-graphs` crashes at capture regardless of memory | Any graphs variant serves |

## Experiment matrix (1x RTX PRO 6000, 96 GB)

Full card (`run.sh all`):

| Variant | Patch | Graphs | Expected if H1 | Expected if H2/H3 |
|---|---|---|---|---|
| marlin | no | yes | OK | OK |
| baseline-eager | no | no | OK | OK |
| baseline-graphs | no | yes | crash | OK |
| patched-eager | yes | no | OK | OK |
| patched-graphs | yes | yes | crash | crash |

5090 ceiling (`MEM_BUDGET_GIB=31.45`, ballast + util scaled to the same 28.93 GiB budget):

| Variant | Reporter saw on 5090 | If we see something else |
|---|---|---|
| baseline-eager | OOM in `profile_run` | No OOM: emulation is off (SM count 188 vs 170, driver reservation); a 1x 5090 run is needed |
| patched-eager | boots, 6.31 GiB KV, correct output | Compare KV GiB to 6.31 to calibrate the emulation |
| patched-graphs | Xid 31 at capture batch 2 | Crash only here, not on the full card: memory pressure matters, H3 leads |

Debug, only if a graphs variant crashes:
1. `fi4827-graphs` (PR #60069 + FI#4827), on the full card and under `MEM_BUDGET_GIB=31.45`. If it serves where
   `patched-graphs` crashed, FI#4827 is the missing half and the A/B goes on #49476 and FI#4827.
2. `patched-min`: capture sizes [1,2] + `CUDA_LAUNCH_BLOCKING=1` to localize the faulting kernel.
3. Buffer bisect: on the pod, edit the patched `_ensure_wrapper` so only one of `shared_static_workspace`,
   `shared_dynamic_workspace`, `shared_output` is passed; one `patched-min` run each. The buffer that
   reproduces the fault is the fix target.
4. `compute-sanitizer --tool memcheck` on the smallest failing case if 1-3 do not localize it.

## Pod

RunPod catalog reads, 2026-10-14 (1 GPU, host CUDA >= 13.0):

| GPU | Cloud | $/hr | Stock |
|---|---|---|---|
| RTX PRO 6000 WK | community | 1.69 | LOW |
| RTX PRO 6000 WK | secure | 2.59 | not probed |
| RTX PRO 6000 Server | community | 1.69 | none |
| RTX PRO 6000 Server | secure | 2.49 | HIGH, 12 data centers |
| RTX 5090 | community | 0.69 | none |
| RTX 5090 | secure | 1.19 | not probed |

Spec:
- GPU: 1x `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`, community. Fallback: Server Edition, secure.
- Image: `vllm/vllm-openai:v0.31.0`, the reporter's image, so vLLM, FlashInfer, and torch match without
  installs. The image does not read `PUBLIC_KEY`, so the entrypoint is overridden to start sshd:
  `["/bin/bash", "-c"]` + `apt-get update && apt-get install -y openssh-server tmux && mkdir -p /run/sshd ~/.ssh && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && /usr/sbin/sshd && sleep infinity`.
  If SSH is not up within ~5 min of the pod running, terminate and fall back to a RunPod PyTorch CUDA 13
  image, where `setup.sh` installs the pins.
- `startSsh: true`, ports `22/tcp` (scp needs direct SSH), `minCudaVersion: 13.0`.
- Container disk 40 GB, volume disk 80 GB at `/workspace`, `HF_HOME=/workspace/hf`.
- No network volume: weights are ~20 GB and re-download in minutes, and a network volume pins later pods to
  one data center while PRO 6000 stock is LOW.

## Run

Copy the working tree without `.git`, `results/`, or `docs/private/` (private repo, rented box):

```
tar -czf - --exclude .git --exclude results --exclude docs/private . \
  | ssh -p <port> root@<ip> 'mkdir -p /workspace/sm120-kit && tar -xzf - -C /workspace/sm120-kit'
```

On the pod, inside tmux:

```
cd /workspace/sm120-kit
export HF_HOME=/workspace/hf
mkdir -p results && bash harness/pod/env_report.sh | tee results/env.txt
bash issues/vllm-49476/setup.sh
bash issues/vllm-49476/run.sh all
for v in baseline-eager patched-eager patched-graphs; do
    MEM_BUDGET_GIB=31.45 bash issues/vllm-49476/run.sh "$v"
done
# only if a graphs variant crashed:
bash issues/vllm-49476/run.sh fi4827-graphs
MEM_BUDGET_GIB=31.45 bash issues/vllm-49476/run.sh fi4827-graphs
bash issues/vllm-49476/run.sh patched-min
```

Pull back: `scp -P <port> -r root@<ip>:/workspace/sm120-kit/results/ ./results/`, then terminate the pod.

## Cost estimate

Setup ~0.5 h (image pull, sshd, model download). Matrix 8 boots at ~8-10 min each, ~1.5 h. Debug ~1 h if a
graphs variant crashes. Expected 3-4 h, hard cap 6 h.

- PRO 6000 WK community at $1.69/hr: ~$5-7, cap ~$10.
- PRO 6000 Server secure at $2.49/hr: ~$7.50-10, cap ~$15.

Volume disk is billed on top (not in the catalog read); terminating the pod ends it. The earlier ~$1.10/hr
figure was stale.

## Pod log

| Date | Pod id | GPU | Start | Stop | Cost |
|---|---|---|---|---|---|
| 2026-10-15 | qkt9wl9ihcqstz | 1x RTX PRO 6000 Server, secure, US-NC-1, host CUDA 13.2, $2.49/hr (WK community had no stock) | 06:24:43Z | 06:51Z (terminated) | ~$1.09 (26.3 min x $2.49/hr; billing read had no records yet at 06:51Z) |

## Environment (pod qkt9wl9ihcqstz, `harness/pod/env_report.sh`)

```
GPU        : 1x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, cc 12.0, driver 595.91.07
Host CUDA  : 13.2 (RunPod host), kernel 6.8.0-106-generic, 128 host CPUs, 1.5 TB RAM
Image      : vllm/vllm-openai:v0.31.0 @ sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
             (same digest prefix the reporter cites)
Packages   : vllm 0.31.0, torch 2.13.0+cu130, flashinfer-python 0.7.0.post1, triton 3.7.1, transformers 5.17.0,
             nvidia-nccl-cu13 2.30.7 (image default; a setup.sh reinstall had swapped it to 2.29.7, restored)
```

## Results (2026-10-15, pod qkt9wl9ihcqstz)

Every run: `nvidia/Qwen3.6-35B-A3B-NVFP4`, `--max-model-len=262144 --kv-cache-dtype=fp8 --max-num-seqs=16`,
TP=1. "31.45g" = ballast leaves 31.45 GiB free and `--gpu-memory-utilization=0.303` (5090 emulation).
"cN" = N concurrent smoke requests. Capture sizes are [1, 2, 4, 8, 16, 24, 32], captured largest first.

| Variant | Verdict | Boot s | Fault | Notes |
|---|---|---|---|---|
| marlin | OK | 120 | - | KV 64.14 GiB |
| baseline-eager | OK | 61 | - | KV 47.55 GiB: b12x reserves 16.6 GiB more than marlin |
| baseline-graphs | BOOT_FAIL | 90 | illegal memory access in `profile_cudagraph_memory` | No PR, 96 GB card. Dies after 5/7 captures, i.e. at size 2 |
| patched-eager | OK | 50 | - | KV 64.74 GiB: PR #60069 recovers 17.2 GiB |
| patched-graphs | BOOT_FAIL | 60 | same, after 5/7 captures | |
| baseline-eager-31.45g | BOOT_FAIL | 55 | `torch.OutOfMemoryError` in `profile_run` -> `_dummy_run` | Reporter's 5090 OOM, reproduced |
| patched-eager-31.45g | OK | 50 | - | KV 6.14 GiB, 609,683 tokens (reporter: 6.31 GiB, ~625K) |
| patched-graphs-31.45g | BOOT_FAIL | 60 | same illegal memory access, after 5/7 captures | Reporter's crash, same capture step. Reporter saw Xid 31 in dmesg; the container cannot read dmesg |
| noskip-graphs | OK | 90 | - | No PR, `VLLM_MOE_SKIP_PADDING=0`; KV 46.88 GiB |
| guard-graphs | OK | 70 | - | PR #57946 only; KV 46.88 GiB |
| fixed-graphs | OK | 65 | - | PR #60069 + #57946; KV 64.14 GiB |
| fixed-graphs-31.45g | OK | 65 | - | KV 5.54 GiB, 550,105 tokens, 2.10x at 256K |
| fixed-graphs-31.45g-c3 | OK | 65 | - | 3/3 requests coherent |
| fixed-graphs-31.45g-c13 | OK | 65 | - | 13/13 requests coherent, 64 tokens each |
| patched-noskip-graphs-31.45g-c3 | OK | 65 | - | PR #60069 + `VLLM_MOE_SKIP_PADDING=0`; KV 5.55 GiB, 3/3 coherent |
| baseline-graphs-c3-fi071 | OK | 85 | - | No PR, FlashInfer 0.7.1 (python, cubin, jit-cache, jit-cache-sm120f); KV 46.88 GiB, 3/3 coherent |
| patched-graphs-31.45g-c3-fi071 | OK | 65 | - | PR #60069 + FlashInfer 0.7.1; KV 5.55 GiB, 550,105 tokens, 3/3 coherent |

All graphs variants that served finished both capture passes ("Graph capturing finished", 0.58 + 0.07 GiB) with
zero illegal-memory lines in the log. Output is the model's thinking trace, cut at the 64-token cap.

## Root cause

Two independent bugs. The OOM hid the second one on 32 GB cards.

1. **OOM (what the issue title says):** one b12x workspace per MoE layer. b12x reserves 16.6 GiB more than
   marlin on a 96 GB card, which a 5090 cannot fit. Fixed by sharing workspaces across layers: PR #60069
   (also #53081, #48698).
2. **Illegal memory access under CUDA graphs (the reporter's Xid 31):** `VLLM_MOE_SKIP_PADDING` (default on) makes the
   top-k routers write expert id `-1` into the padding rows of graph batches. FlashInfer 0.7.0.post1's b12x
   kernels index per-expert state with that id and write out of bounds. The first padded capture is size 2,
   which is exactly where the reporter's run died. It is pre-existing (H1): it reproduces with no PR on a 96 GB
   card, and PR #60069 does not change it.

Fixes for (2): FlashInfer #5451 (merged 09-30, ships in 0.7.1) makes the kernels skip negative ids; vLLM #57946
(open) guards the sentinel in `FlashInferB12xExperts`. Both confirmed here on SM120: unpatched v0.31.0 serves with
graphs on FlashInfer 0.7.1, and with #57946 on 0.7.0.post1. vLLM `main` already pins `flashinfer-python==0.7.1`
(v0.31.0 pins 0.7.0.post1), so the crash is gone on `main` without #57946. #57946's author only had GB10
(sm_121) numbers.

Hypotheses: H1 confirmed. H2 and H3 killed (workspace sharing is irrelevant to the crash). H4 killed (b12x
captures and replays fine once the `-1` sentinel never reaches it). FI#4827 was not needed and not run.

## Outcome

For the reporter on v0.31.0 + 5090 today: PR #60069 (or #53081) for the OOM, plus one of
`VLLM_MOE_SKIP_PADDING=0`, PR #57946, or FlashInfer 0.7.1 for the crash. All three combinations served the
reporter's exact config with CUDA graphs under the 5090 ceiling, with padded concurrent batches.

Upstream, only the OOM is left on `main`: one of the three workspace-sharing PRs (#60069, #53081, #48698) has to
land. #60069 on FlashInfer 0.7.1 (the closest stand-in for `main` we ran) serves under the 5090 ceiling.
No new PR from us is needed for this issue; the contribution is the A/B and the root cause, posted to the threads.

Next: draft comments for #49476 (answers the reporter's PRO 6000 question, full table, workarounds), #57946
(SM120 numbers its author asked for), and the workspace PRs (they duplicate each other; evidence #60069 works).

Cost: ~$1.09 of the ~$7.50-10 estimate. The FlashInfer 0.7.1 wheels for SM120 live on
`https://flashinfer.ai/whl` (cubin) and `https://flashinfer.ai/whl/cu130` (jit-cache), not PyPI.
