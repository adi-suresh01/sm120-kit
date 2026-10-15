# sglang#44291 stage 2: EP2 illegal memory access and TP2 block_n (spec, 2026-10-15)

Stage 1 (custom all-reduce, PR sgl-project/sglang#44448) is in `README.md`, which also holds the reporter's setup
verbatim. This file covers bugs 2 and 3 of the issue.

Goal: get sorenmat (Trubit) serving `nvidia/Qwen3.8-Flash-Next-NVFP4` @ fc694b54 on 2x PCIe RTX PRO 6000 with
NEXTN, and know which upstream PRs that takes. Our own PR only where the matrix shows a gap the open PRs miss.
The most likely merged PR of ours is still #44448; run C below gives its reviewers real-model evidence.

## Upstream state (keyword PR search, 2026-10-15, sglang main 71a17db)

Both bugs already have open PRs from other contributors, none reviewed:

| Bug | PR (head) | What it does |
|---|---|---|
| 2, EP2 | #40172 (5df150c7b), fixes #40156 | Draft batches take their token width from `spec_info` when `input_ids` is None, and `StandardDispatcher` keeps the `-1` drop sentinel through `local_expert_mapping` (clamp, then restore) |
| 2, EP2 | #43330 (f45d5f008), fixes #43324 | The clamp/restore part only |
| 3, TP2 | #42566 (a900a02a6, carries #43556 547bbf429) | Refines block-FP8 MoE scale blocks so a 640 / 2 = 320 shard loads; cites this checkpoint's MTP experts by name |

Regressed by #39574 (merged 09-15), so both reporter images carry it. #36787 (Dev-Jahn, sm120 support for this
model) has not changed since 09-24; it serves NVFP4 at TP2/EP1 with NEXTN and is used here only as a speed
reference and as a source of candidate causes (NUMA-interleaved PLE pinning, PLE prefetch buffer chosen from
the driver's capture state).

Code facts on main that shape the hypotheses:

- `enable_num_token_non_padded()` returns True whenever `moe_ep_size > 1`, so EP2 masks padded `topk_ids` rows
  to `-1` and TP2/EP1 does not. This is why #36787's TP2/EP1 + NEXTN serves while the reporter's EP2 + NEXTN dies.
- `ForwardBatch.init_new` sets the token count to 0 when `input_ids` is None, which is the case for draft
  batches, so every draft row is masked to `-1`.
- `StandardDispatcher` skips `local_expert_mapping` for `flashinfer_cutlass`, but the cookbook says the FP8
  block-scaled MTP experts run on Triton under the `flashinfer_cutlass` pin (#38121). Triton does go through
  the mapping, where `-1` wraps to the table's last entry, a real expert.
- Cookbook: the NVIDIA export's FP8 block-scaled MTP experts "cannot be TP-sharded (640/2 = 320 is not a
  multiple of the 128 block) and fault on the triton fp8 path under EP". Its two-node cell reads the MTP draft
  from `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ 7b719225 (the same trained head kept in BF16) with
  `--speculative-draft-model-path <RadixArk> --speculative-draft-model-quantization modelopt_fp4`. Only three
  RadixArk shards hold `mtp.*` tensors (`model-bf16-00010/00011/00012`, ~14.7 GB of 135 GB).

## Hypotheses (EP2 crash)

| # | Hypothesis | Confirmed by | Killed by |
|---|---|---|---|
| H1 | Draft count 0 at EP > 1 masks all draft ids to `-1`; the Triton MTP MoE aliases them through `local_expert_mapping` and faults | A0 serves, A2 faults in a Triton MoE kernel, A4 serves | A0 crashes, or A4 still faults in eager mode |
| H2 | The draft CUDA-graph capture crash has a separate cause (#40156 says capture never applies the mask; candidate: PLE prefetch buffer picked from a runner flag the draft runner never sets) | A4 serves with `--disable-cuda-graph` but still dies in draft capture | A4 serves with graphs |
| H3 | The target model's EP path (`flashinfer_cutlass` EP or sharded PLE) faults without speculation | A0 crashes | A0 serves |
| H4 | Pinned host memory (NUMA pressure, memlock) faults the PLE table | Fault during load, tracks `numactl -H` / `ulimit -l` | Load completes in every run |

Bug 3 needs no hypothesis table: #42566 names the cause (FP8 MTP experts, 128-wide blocks, 320-wide shard).
B1 confirms it on our hardware.

## Experiment matrix (2x RTX PRO 6000, PCIe)

Base flags are the reporter's: `--model-path nvidia/Qwen3.8-Flash-Next-NVFP4 --moe-runner-backend
flashinfer_cutlass --ple-offload-embedding --json-model-override-args '{"language_model_only":true}'`, default
allocator, `--disable-custom-all-reduce`. NEXTN is `--speculative-algorithm NEXTN --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4`. Cookbook memory levers
(`--mem-fraction-static`, `--mamba-radix-cache-strategy extra_buffer_lazy`) are added only if a boot runs out
of memory, and the run records which.

| Run | Config | Expected if H1 |
|---|---|---|
| A0 | `--tp-size 2 --ep-size 2`, no spec, graphs on | serves |
| A1 | A0 + NEXTN | `cudaErrorIllegalAddress` in draft graph capture (reporter attempt 2) |
| A2 | A1 + `--disable-cuda-graph`, `CUDA_LAUNCH_BLOCKING=1` | Triton illegal memory access, faulting kernel named (attempt 3) |
| A3 | A1 + #43330 | no crash, accept length near 1 (draft MoE gets no experts) |
| A4 | A1 + #40172 | serves, accept length near 3 |
| A5 | A1 with the RadixArk BF16 draft, no patches | if the draft MoE also goes through the Triton mapping: same fault as A2; if it serves, accept length shows whether the count-0 bug silently empties the draft |
| B0 | `--tp-size 2`, no spec, graphs on | serves (NVFP4 experts use 16-wide groups, so 320 splits) |
| B1 | B0 + NEXTN | `ValueError: ... = 320 is not divisible by weight quantization block_n = 128` |
| B2 | B1 + #42566 | serves, accept length near 3 |
| B3 | B1 with `--speculative-draft-model-path` on the RadixArk BF16 MTP head, no patches | serves (cookbook workaround) |
| C | the A4 / A5 / B2 / B3 config that passes quality with the highest c=16 tok/s, its patches + #44448, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, custom all-reduce on | serves; the stack the reporter would run |

Run order: A0, A1, A2, A4, A3, A5, B0, B1, B3, B2, C. A0 first so that an H3 failure surfaces before any spec run.

Branches:
- A0 crashes (H3): one `compute-sanitizer --tool memcheck` run on the smallest case (`--max-running-requests 1`,
  short prompt), capped at 30 min, then stop.
- A4 serves eager but dies in capture (H2): keep the log, stop the pod, find the cause offline. A second
  session needs a new approval.
- Load faults (H4): record `ulimit -l`, `free -g`, `numactl -H` next to the error and stop.

## Quality and speed (every NEXTN run that serves: A4, A5, B2, B3; A3 gets accept length only)

- GSM8K: `python -m sglang.test.run_eval --eval-name gsm8k --num-examples 200 --max-tokens 16384` (the
  cookbook's protocol: 5-shot, greedy, thinking on). Reference: cookbook NVIDIA export at TP1, 97.41% on 1319.
  Pass at >= 94%, about three standard errors below the reference at n = 200.
- Accept length: from the server log over the GSM8K run. Reference 3.16-3.33 (cookbook, TP1) and 2.94-3.02
  (#36787, TP2/EP1). Pass at >= 2.5. A3 below 1.5 is the silent-degradation signal.
- Speed: `python -m sglang.bench_serving --backend sglang --dataset-name random --random-input-len 1024
  --random-output-len 512` at `--max-concurrency` 1 / 16 / 64 with `--num-prompts` 8 / 64 / 256, the #36787
  protocol. Reference, not a pass bar: #36787 NVFP4 TP2/EP1 NEXTN at 216.0 / 1069.2 / 1703.1 output tok/s
  (that branch has sm120 kernels mainline lacks). Record the scheduler's `max_running_requests` cap if it
  lands below 64.
- A config counts as "sorenmat unblocked" when it boots with CUDA graphs and NEXTN, passes GSM8K and accept
  length, and has a speed row.
- Base flags also carry `--reasoning-parser qwen3` (the cookbook's cells use it) so GSM8K scores the answer, not
  the thinking text. bench_serving uses `--random-range-ratio 1` (fixed lengths); #36787 does not state its ratio.

## Pod

- 2x RTX PRO 6000, PCIe, secure tier (stage 1: $4.98/hr at EUR-IS-2, no community stock). Re-read live stock
  and price before asking for approval. Host RAM >= 128 GB: the pinned PLE table is 47.7 GiB (23.8 per rank)
  plus page cache for the checkpoint load.
- 320 GB volume disk at `/workspace`, `HF_HOME=/workspace/hf`: the 133 GB checkpoint plus the three RadixArk MTP
  shards, with room to fetch the full 135 GB RadixArk export if the draft does not load from the partial
  directory. No network volume; a second session re-downloads.
- Image `lmsysorg/sglang:dev-cu13@sha256:0a1ba33aa4e032b8c5bcebdc69c45664074c032074add4f7db01629ec1fcc381`
  (pushed 2026-10-15 00:53Z, the stage 1 image, sglang 0.0.0.dev1+ga6a8c5601). Entrypoint overridden for sshd;
  it also writes `export -p` to `/root/container_env.sh`, which every SSH command sources (venv image).
  `ssh-add --apple-use-keychain ~/.ssh/id_rsa` first.
- PR patches are vendored with only their `python/sglang/` hunks and checked against a6a8c5601 alone and in the
  combinations C needs (#40172 and #43330 conflict and are never combined). Every run restores a pristine copy
  of the package before applying its set.
- First five minutes: `nvidia-smi topo -m` shows SYS/PHB/NODE (not NV#), `scripts/env_report.sh`, then
  `ulimit -l`, `free -g`, `numactl -H`. The cookbook says the pinned table needs `--ulimit memlock=-1` under
  Docker; if memlock is limited, record it and try, since the first pinned load shows whether it matters.

## Local prep before any spend

- Fetch the four PR diffs at the heads above into `patches/`, dry-run each set (#40172, #43330, #42566, #44448)
  against the image's sglang commit, and note which combinations conflict.
- Resolve the RadixArk MTP shard list and the exact draft-path flags from the safetensors index and the
  cookbook's two-node cell.
- Write the scripts below; reuse `scripts/serve_and_smoke.sh`, `scripts/wait_gpu_idle.sh`, `scripts/patch_pkg.sh`.

## Time, cost, stop rules

Setup and download ~25 min, eleven boots at ~6-8 min each ~80 min, quality and speed in the same boots on up to
four configs ~60 min: ~3 h, ~$15 at $4.98/hr. Hard cap 4 h (~$20).

- Neither A1 nor A2 reproduces within the first hour: stop; the fallback is the reporter's exact images
  (`dev-cu13` 10-02 digest, `dev-qwen38-next-local` 4ccff141), which needs a new approval.
- Terminate the moment the matrix is done. Code is written locally between sessions, never on paid time.

## Deliverables

- Results appended to this file; raw logs under
  `results/sglang-44291-stage2/<run>/`.
- For sorenmat: the exact working 2-GPU command with GSM8K, accept length, and tok/s, and which of EP2 / TP2 is
  faster if both serve. Posted on #44291.
- Evidence comments: #40172 (A2 -> A4),
  #43330 (A3, stated as data: the clamp alone stops the crash but draft rows stay empty while the count is 0),
  #42566 (B1 -> B2 on the checkpoint it cites), #44448 (run C on the real model).
- Gap found (H2, H3, H4): our PR from the fork with a red/green test. If the fix already exists as a commit in #36787, comment there with the repro and
  suggest splitting it out instead of re-submitting Dev-Jahn's code.
- Wrap-up: this is the third run (vllm#49476, stage 1, stage 2); list the repeated steps and propose the harness.

## Files

| File | Purpose |
|---|---|
| `stage2.md` | This spec, then results |
| `stage2_setup.sh` | Pod checks (topology, memlock, RAM, NUMA), checkpoint and RadixArk MTP shard download |
| `stage2_run.sh` | Runs A0-A5, B0-B3, C with their patch sets through `serve_and_smoke.sh`, one log dir per run |
| `stage2_quality.sh` | GSM8K 200, accept length, bench_serving at 1 / 16 / 64 against a running server |
| `patches/pr<N>.diff` | PR diffs pinned to the heads above |
