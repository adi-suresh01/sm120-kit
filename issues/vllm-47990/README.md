# vllm#47990: SM120 CUTLASS blockwise FP8 GEMM rejects N % 128 != 0

- Issue: https://github.com/vllm-project/vllm/issues/47990 (waynehacking8, @ailabstw). Open, marked stale 10-13,
  auto-closes about 11-12 without activity.
- Fix under test: https://github.com/vllm-project/vllm/pull/48588 (jszzr). Pads N to the 128 scale block at weight
  load and slices the output. Python only. Approved by the reporter, no maintainer review, merge conflicts since 08-06.
- Competing guard: https://github.com/vllm-project/vllm/pull/47988 (waynehacking8). `can_implement` declines
  N % 128 != 0 on SM12x so the layer falls through to another kernel; also fixes E8M0 scale decoding (orthogonal).
  Mergeable, no review, updated 10-15.
- Related: #47818 (SM120 block-FP8 crash with ue8m0 scales, same code path), #56659 (SM120 blockwise leading strides,
  open), #48587 (guard-only, closed). Both #52775 (SM120 misaligned-M routing) and #55180 (SM120 blockwise CTA
  swizzle) are in v0.31.0 and came after the reporter's f7fc0ca99, so H1 is a real question on this build.
- Re-checked 2026-10-15: issue state unchanged; #48588 head 1b0e863d is one commit on 0ecc2847 (08-24).

## Reporter setup (verbatim)

```
RTX PRO 6000 Blackwell (SM120), CUDA 13.0, torch 2.11.0+cu130, current main (f7fc0ca99) with the precompiled
nightly kernels.
```

> The SM120 CUTLASS c3x blockwise FP8 GEMM (`scaled_mm_blockwise_sm120_fp8.cu`) rejects every problem whose weight
> output dim N is not a multiple of 128, failing with `cutlass_gemm_caller, .../cutlass_gemm_caller.cuh:52, Invalid
> status` (CUTLASS `can_implement` fails). All three tile configs in `scaled_mm_blockwise_sm120_fp8_dispatch.cuh`
> (default 128x128x128, pingpong 64x128x128, swap-ab 128x32x128) reject these shapes, so it is not M-range dependent.
>
> | N | result |
> |---|---|
> | 512, 1536, 7168 | OK at every M/K (max rel err 2-3e-3 vs fp32 dequant reference = bf16 rounding) |
> | 576, 2112 | `Invalid status` at every M/K |

Data already on the threads, all from other machines:
- jszzr, RTX PRO 5000: `test_block_fp8.py -k cutlass` 10 passed with #48588. Padded CUTLASS including the slice copy
  vs default-config Triton: 2.68x / 2.31x / 2.87x / 1.93x at M=1/16/64/256 on N=576 K=7168.
- waynehacking8, RTX PRO 6000 Max-Q (co-tenanted, "take the ratios loosely"): main fails, #48588 passes all 10;
  ~1.4-2.3x over Triton at decode-sized M.
- F21HGG, RTX PRO 5000: for a skinny N=32 K=1024 projection, a tuned Triton config was 1.70-4.39x faster than padded
  CUTLASS. Argues against "always keep unaligned N on CUTLASS" as a general rule.

Nobody has posted a whole-model run: every number is a GEMM in isolation, and every comparison is against Triton.

## What we add

1. The rebase: #48588 conflicts with main only in `test_block_fp8.py` (main parametrized the same test over M for
   SM100, #57377). Resolved by keeping both: M in {1, 32, 65, 66, 67, 256, 1026} x the PR's (N, K) pairs. The source
   files apply cleanly to main and to v0.31.0. Kept in `upstream/pr48588-rebased-on-main.patch` (author jszzr).
2. A clean single-tenant PRO 6000 run of the kernel tests and the sweep, stock vs #48588.
3. The guard's real fallback. Both PR threads assume #47988 sends kv_a_proj to Triton. On v0.31.0 the next kernel
   after CUTLASS that accepts N=576 is Marlin (weight-only FP8, BF16 activations), because B12x also needs
   N % 128 == 0 and Marlin's `can_implement` accepts everything. So the comparison that decides between the PRs is
   padded CUTLASS vs Marlin, which nobody has measured.
4. A whole-model A/B on a real block-FP8 MLA checkpoint: stock crashes or not, #48588 (pad) vs #47988 (guard) on
   GSM8K and throughput.

## Test model

`gaunernst/DeepSeek-V2-Lite-Chat-FP8` at `c6f13916`: DeepseekV2ForCausalLM, 16B total (2.4B active), 16.1 GB,
`quant_method: fp8`, `weight_block_size: [128, 128]`, dynamic activations, 33k downloads. `kv_lora_rank` 512 +
`qk_rope_head_dim` 64 makes `kv_a_proj_with_mqa` N=576 in all 27 layers, with `weight_scale_inv` in the checkpoint
(block-FP8, not left in BF16). It is the only block-FP8 layer with N % 128 != 0 (q_proj 3072, kv_b 4096, o_proj
2048, dense MLP 11008, shared experts 2816), so a CUTLASS rejection in this model can only come from kv_a_proj.
`q_lora_rank` is null, so there is no fused 2112 layer; the sweep and bench cover 2112 and GLM-4.7-Flash's 1344.

Rejected: DeepSeek-V3/R1 (too big for one card), GLM-4.7-Flash FP8 block (only a 19-download community quant),
LongCat-Flash-Lite (larger, less common), per-channel or compressed-tensors FP8 checkpoints (a different code path).

## Kernel selection on v0.31.0 (read from source)

`_POSSIBLE_FP8_BLOCK_KERNELS` for CUDA, in order: FlashInferFp8DeepGEMMDynamic, DeepGemm, Cutlass, B12x, Marlin,
Humming, Triton, Torch.
- DeepGEMM accepts the SM120 family when present, and it ships with vLLM (`vllm.third_party.deep_gemm`), so on a
  default install it takes the 576 x 2048 layer before CUTLASS is tried. The FlashInfer dynamic kernel needs
  FlashInfer's block-scale GEMM, which is SM90-only.
- CUTLASS gets kv_a_proj when DeepGEMM is unavailable or turned off (#47436, #47169 describe SM12x setups where it
  is). The matrix forces that with `VLLM_DISABLED_KERNELS`.
- The server logs `Selected <kernel> for Fp8LinearMethod` once per kernel class, not per layer. `route.py` prints
  the kernel for kv_a_proj and for an aligned layer directly, under each variant's env and patch.

## Hypotheses

- **H1**: on stock v0.31.0 the SM120 CUTLASS blockwise kernel still rejects N % 128 != 0. Confirm: sweep shows
  `Invalid status` for N in {64, 192, 576, 1344, 2112} on both paths, and `test_w8a8_block_fp8_cutlass_matmul` fails
  the unaligned cases. Kill: they pass (then find the commit that fixed it). The PR's layer test also fails on stock,
  but at its 640-row shape assert, so it is not evidence for H1.
- **H2**: a real block-FP8 MLA model crashes at startup on stock when kv_a_proj routes to CUTLASS. Confirm: variant
  `cutlass` ends BOOT_FAIL with `Invalid status`, and its route.py names CutlassFp8BlockScaledMMKernel for
  kv_a_proj. Kill: it boots.
- **H3**: the default v0.31.0 image never reaches CUTLASS for kv_a_proj, because DeepGEMM takes it. Confirm or kill:
  variant `auto` boots, and its `Selected ... for Fp8LinearMethod` lines name DeepGEMM. This only describes the
  Docker image; it says nothing about other installs.
- **H4**: #48588 fixes it without hurting accuracy. Confirm: `pad-48588` boots and smokes OK; the sweep's `wrapper`
  path is OK everywhere with rel err ~2-3e-3 while `raw` still rejects (the kernel gap remains, the integration
  covers it); all `-k cutlass` tests pass; GSM8K within noise of `guard-47988`. `auto` is not the accuracy baseline:
  DeepGEMM requantizes every layer's scales to E8M0, so it differs everywhere, not only in kv_a_proj.
- **H5**: padded CUTLASS beats the guard's real fallback (Marlin) where it matters. Confirm: bench layer level,
  N=576, CUTLASS faster than Marlin at decode M; whole-model decode tok/s of `pad-48588` >= `guard-47988`. kv_a_proj
  is one small GEMM per layer, so the whole-model difference may sit inside run-to-run noise; that is a valid result.
- **H6**: the reviewer's skinny-shape concern holds on PRO 6000 at default Triton config. Bench row N=32 K=1024:
  Triton faster than padded CUTLASS. Either way it tells the maintainers whether the pad needs a size floor.

## Experiment matrix

All variants on the same pod, stock image `vllm/vllm-openai:v0.31.0` (digest-pinned), model as above,
`--max-model-len 4096`. `DISABLED` = `VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
DeepGemmFp8BlockScaledMMKernel`. Checks run in `hooks.sh` pre_launch, after the patch is applied, before the boot,
each under its own timeout (route 5 min, sweep 5, pytest 10, bench 10, eval 20).

| Order | Variant | Patch | Routing | Checks before boot | Expected |
|---|---|---|---|---|---|
| 1 | auto | none | vLLM default | sweep, pytest, GSM8K + throughput | boots on DeepGEMM (H3); sweep/pytest fail unaligned (H1) |
| 2 | cutlass | none | DISABLED | route | route: kv_a_proj on CUTLASS; BOOT_FAIL `Invalid status` (H2) |
| 3 | guard-47988 | `pr47988-guard` | DISABLED | route, GSM8K + throughput | route: kv_a_proj on Marlin; OK |
| 4 | pad-48588 | `pr48588-pad-n` | DISABLED | route, sweep, pytest, GEMM bench, GSM8K + throughput | route: CUTLASS; OK; all pass (H4-H6) |

Guard runs before pad so that a slow pad variant cannot cost the comparison the run exists for.

- `patches/pr48588-pad-n.diff`: the `vllm/` half of #48588 rebased onto v0.31.0 (applies cleanly).
- `patches/pr47988-guard.diff`: only the `can_implement` hunk of #47988, without its E8M0 change, which this
  checkpoint (fp32 scales) never reaches.
- `pr48588-tests.diff`: the `tests/` half, applied by `setup.sh` to a sparse v0.31.0 tests checkout, so stock and
  patched run the same tests.
- `route.py`: the kernel vLLM picks for kv_a_proj (576 x 2048) and q_proj (3072 x 2048) under the current env.
- `sweep.py`: the issue's sweep, N {64, 192, 512, 576, 1344, 1536, 2112, 7168} x K {1024, 2048, 7168} x M {1..1024},
  through the module wrapper and the raw op, vs an fp32 dequant reference.
- `bench_gemm.py`: CUDA-graph timings. GEMM level: CUTLASS vs Triton (the PR author's method). Layer level:
  `apply_weights` of CUTLASS, Triton, and Marlin, including each one's activation handling. Default Triton config:
  there is no tuned one for this GPU.
- `eval_offline.py`: GSM8K 5-shot greedy over all 1319 questions, then output tok/s at batch 1 (512 in / 512 out),
  64 and 256 (1024 in / 256 out), median of 3, prefix caching off. Same routing env and patch as the server.

Measurement notes for the write-up:
- The bench's layer level quantizes activations with the CUDA op; the compiled server may run the inductor version.
  Both CUTLASS and Triton rows use the same one, so their ratio holds.
- The sweep and bench pass `use_ue8m0=False`, as the CUTLASS kernel class does.

## Hours and cost

| Step | Est. |
|---|---|
| Pod create, image pull, SSH, setup (pip, tests checkout, 16 GB model, GSM8K) | 10-12 min |
| auto: sweep + pytest ~4 min, eval ~7 min (cold compile, DeepGEMM warmup), boot + smoke ~3 min | 14 min |
| cutlass: route, then a cold compile before the crash | 4 min |
| guard-47988: route, eval ~7 min, boot ~3 min | 11 min |
| pad-48588: route, sweep + pytest ~4, bench ~4, eval ~7, boot ~3 | 19 min |
| Pulls, idle waits | 3 min |
| Total | ~60-65 min |

`planned_hours = 2.0` (the deadline). `harness quote vllm-47990` (2026-10-15): 1x RTX PRO 6000 Blackwell Server
Edition, SECURE, $2.49/hr; worst case $5.48 with the 10% margin. Fallbacks: Workstation Edition SECURE $2.59/hr,
Workstation Edition COMMUNITY $1.69/hr. Expected spend ~$2.70. Budget ask: $6.

Run: `harness run vllm-47990` (all four variants in file order, then down).

## After the run

- Results table here with real numbers only, plus `env.txt` highlights.
- If H1, H2, H4 hold: a comment on #47990 (stops the stale close) and one on #48588 with the PRO 6000 numbers and the
  rebase. The rebased patch keeps jszzr as author; offer it to them rather than opening a competing PR.
- On #47988: the Marlin fallback finding and the pad-vs-guard numbers, whichever way they go.

## Results (2026-10-16 05:05-05:27 UTC, pod sm120-vllm-47990-f12efe)

Environment (`results/vllm-47990/env.txt`): 1x RTX PRO 6000 Blackwell Server Edition (cc 12.0, 97887 MiB, driver
580.173.02), SECURE. vllm 0.31.0, torch 2.13.0+cu130, flashinfer-python 0.7.0.post1, triton 3.7.1.
Spend: $0.93 (estimate, `harness budget show`), 22.5 min from `run` to terminate. The plan guessed 60-65 min: setup
took 90 s (model download 27 s) and each eval 2.5-5 min.

### Verdicts and routing

| Variant | kv_a_proj (576 x 2048) | Aligned layers | Verdict | Boot |
|---|---|---|---|---|
| auto | DeepGEMM (server log: only `DeepGemmFp8BlockScaledMMKernel`) | DeepGEMM | OK | 86 s |
| cutlass | `CutlassFp8BlockScaledMMKernel` (route.py) | CUTLASS | BOOT_FAIL | 56 s |
| guard-47988 | `MarlinFP8ScaledMMLinearKernel` (route.py; server log names both) | CUTLASS | OK | 80 s |
| pad-48588 | `CutlassFp8BlockScaledMMKernel`, padded to 640 (route.py) | CUTLASS | OK | 75 s |

`cutlass` dies in the memory profiling pass: `profile_run` -> `_dummy_run` -> `deepseek_v2.py forward` ->
`RuntimeError: cutlass_gemm_caller, .../c3x/cutlass_gemm_caller.cuh:52, Invalid status`. kv_a_proj is the model's
only block-FP8 layer with N % 128 != 0.

- H1 holds: the bug is still in v0.31.0, after #52775 and #55180.
- H2 holds: a real block-FP8 MLA model crashes at startup once kv_a_proj routes to CUTLASS.
- H3 holds for the Docker image: DeepGEMM takes every block-FP8 layer, so the default config never reaches CUTLASS.
- #47988's guard falls through to Marlin W8A16 on SM120, not Triton.

### Kernel tests and sweep

| | stock (auto) | pad-48588 |
|---|---|---|
| `test_block_fp8.py -k cutlass` (rebased tests, 22 selected) | 7 passed, 15 failed | 22 passed |
| sweep, wrapper path, N in {64, 192, 576, 1344, 2112} | `Invalid status` at all 27 M x K each | OK at all 27, max rel err 1.4-1.6e-3 |
| sweep, raw op, same N | `Invalid status` | `Invalid status` (kernel unchanged, as expected) |
| sweep, aligned N {512, 1536, 7168} | OK, max rel err 1.4-1.5e-3 | same |

The 14 failing stock matmul cases fail on `Invalid status`; the 15th is the PR's layer test (640-row assert). Errors
are vs an fp32 dequant reference; the padded rows match the aligned ones.

### GEMM and layer timings (median us, CUDA graphs, default Triton config)

N=576 K=2048 (this model's kv_a_proj), layer level (`apply_weights`, activation handling included):

| M | Triton | CUTLASS padded | Marlin (guard's fallback) |
|---|---|---|---|
| 1 | 20.5 | 8.3 | 16.0 |
| 4 | 20.9 | 11.2 | 16.1 |
| 16 | 21.1 | 11.4 | 19.5 |
| 64 | 21.3 | 11.7 | 34.3 |
| 256 | 21.8 | 17.2 | 14.0 |
| 1024 | 23.1 | 31.8 | 18.4 |
| 4096 | 33.1 | 59.2 | 50.5 |

N=576 K=7168 (DeepSeek-V3 kv_a_proj):

| M | GEMM Triton | GEMM CUTLASS padded | Layer Triton | Layer CUTLASS | Layer Marlin |
|---|---|---|---|---|---|
| 1 | 63.9 | 18.0 | 65.4 | 19.2 | 20.1 |
| 16 | 65.1 | 19.9 | 66.7 | 22.6 | 24.1 |
| 64 | 65.2 | 20.0 | 67.0 | 24.2 | 42.6 |
| 256 | 65.9 | 32.1 | 68.3 | 42.3 | 23.2 |
| 1024 | 66.5 | 57.3 | 72.2 | 91.3 | 39.3 |
| 4096 | 96.2 | 63.1 | 135.8 | 184.4 | 136.4 |

- vs Triton (what both threads compared): padded CUTLASS GEMM is 3.3-3.6x faster at M <= 64 on N=576 K=7168, 2.1x
  at 256, 1.2-1.5x at 1024-4096. Same direction as jszzr's PRO 5000 numbers, larger at small M.
- vs Marlin (what the guard actually gives): padded CUTLASS is 1.4-2.9x faster at M <= 64 on the model's shape; on
  DeepSeek-V3's it is within 7% at M <= 16 and 1.8x faster at 64. Marlin is faster at M = 256-1024 (1.2-2.3x). Marlin's times are not
  monotonic in M (34.3 us at M=64, 14.0 at 256), so its own M-dependent tile choice is visible here.
- The CUTLASS layer row carries a large activation-quant overhead at big M that the GEMM row does not: 91.3 vs 57.3 us
  at M=1024, 184.4 vs 63.1 at 4096, and the aligned 512 x 7168 control shows the same (175.6 vs 55.0 at 4096), so it
  is not the padding. It is the eager CustomOp quant path; the compiled server may differ. Worth a separate look.
- H6 (N=32 K=1024, default Triton config): padded CUTLASS GEMM 4.6-8.4 us vs Triton 9.8-10.1 at M <= 256 (1.2-2.1x
  faster), 12.3-12.5 vs 10.2-10.5 at M >= 1024 (Triton 1.2x faster). F21HGG's 1.7-4.4x for Triton was with a tuned
  config, which we did not run, so this neither confirms nor refutes it.
- Max rel diff vs Triton: CUTLASS 5e-7 (same quantized math), Marlin 2.8e-2 (BF16 activations, different numerics).

### Whole model (offline engine, prefix caching off, median of 3)

| Variant | kv_a_proj kernel | GSM8K 5-shot (1319) | tok/s b1 (512/512) | b64 (1024/256) | b256 (1024/256) |
|---|---|---|---|---|---|
| auto | DeepGEMM (all layers) | 0.626 | 245.4 | 3808 | 6925 |
| guard-47988 | Marlin | 0.646 | 224.0 | 3646 | 6854 |
| pad-48588 | CUTLASS padded | 0.635 | 234.1 | 3606 | 6910 |

- GSM8K: pad vs guard differ by 1.1 points; one standard error at p=0.64, n=1319 is 1.3 points. No accuracy
  difference shows. (H4 holds.)
- Throughput, pad vs guard: +4.5% at batch 1, -1.1% at batch 64, +0.8% at batch 256. The batch-1 gain matches the
  decode-M win above; batch 64 includes 64k prefill tokens, likely where Marlin's large-M edge shows (not profiled). kv_a_proj is one
  small GEMM per layer, so the whole-model effect is a few percent either way. (H5: holds at decode, a wash at load.)
- `auto` is not an accuracy baseline (DeepGEMM requantizes every layer to E8M0 scales).

### What this means for the two PRs

Both fix the crash on this model. #48588 keeps kv_a_proj on W8A8 CUTLASS and is faster at decode batch sizes, both
per kernel (1.4-2.9x over Marlin at M <= 64) and end to end at batch 1 (+4.5%). #47988's guard is a correct safety
net, but its fallback on SM120 is Marlin, not Triton, which both threads assumed. Whole-model accuracy is the same
within noise.
