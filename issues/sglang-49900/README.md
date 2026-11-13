# sglang#49900: DeepSeek-V4.1-Flash single-stream decode on RTX PRO 6000, stage 1: the block-FP8 Triton GEMM

On 4x RTX PRO 6000 Server Edition (PCIe only, TP4 + EP4), a decode step with one running request takes about 40 ms
and one with two takes 28.6 ms. The reporter's profile puts 1820 of 2026 ms over 50 steps in one Triton kernel,
`_w8a8_block_fp8_matmul`, at 173 us per call with one request against 104 us with two, on the default config.
Stage 1 is a kernel benchmark on one GPU with synthetic weights at the per-rank TP4 shapes, no checkpoint: why M=1 is
slower than M=2 under the same config, and how much of the roughly 36 ms per step a tuned config, the published
tables, or the MXFP8 route takes back. Stage 2 (described at the end, not planned in hours yet) runs the model on 4x.

## Reporter setup (copied, ASCII-normalized)

sgl-project/sglang#49900, drunkcoding, 2026-11-11:

> DeepSeek-V4.1-Flash, TP4 + EP4, no speculative decoding, default decode CUDA graphs (captured batch sizes 1, 2, 4,
> 8): a decode step with **one** running request takes about 40 ms, while a step with **two** running requests takes
> about 28.6 ms. Single-stream decoding is therefore about 24.8 tok/s where the 2-request step time would give about
> 35 tok/s. It is independent of the context length (1k / 9k / 32k-token prompts give the same numbers) and of
> FlashInfer autotune (on or off).
>
> Step time (ms) = 1000 x running requests / `gen throughput (token/s)` from the server's `Decode batch` lines:
>
> | running requests | default graphs (1, 2, 4, 8) | `--cuda-graph-bs-decode 2 4 8` | `--disable-cuda-graph` |
> |--:|--:|--:|--:|
> | 1 | 40.4 | **27.2** | 79.2 |
> | 2 | 28.6 | 28.6 | 82.6 |
> | 3 | 25.0 | 25.1 | 83.5 |
> | 4 | 26.1 | 26.3 | 80.5 |
> | 8 | 27.9 | 27.7 | 83.1 |
>
> **Where the time goes** (`/start_profile`, 50 decode steps, GPU activities, rank 0):
>
> | | 1 running | 2 running |
> |---|--:|--:|
> | 50 steps | 2026 ms (40.5 ms/step) | 1447 ms (28.9 ms/step) |
> | `_w8a8_block_fp8_matmul` total | 1820 ms | 1097 ms |
> | calls | 10,500 | 10,500 |
> | mean per call | 173 us | 104 us |
>
> The extra about 14.5 ms per step at batch size 1 is almost entirely this one Triton kernel (per call 154 vs 83 us on
> the most common shape, 240 vs 125 us on the next, 1569 vs 959 us on the largest). [...] The log prints `Using
> default W8A8 Block FP8 kernel config` warnings for these shapes, i.e. no tuned config is used.
>
> Not established: why the M=1 call is slower than M=2 [...] we did not test whether a different block configuration
> or a specialised M=1 path changes it.
>
> **Workaround:** `--cuda-graph-bs-decode 2 4 8`.

```
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4.1-Flash --trust-remote-code \
  --tp-size 4 --ep-size 4 --mem-fraction-static 0.9 --max-running-requests 8 \
  --chunked-prefill-size 2048 --context-length 65536 --disable-flashinfer-autotune --decode-log-interval 4

GPU 0,1,2,3: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0, driver 590.48.01
PyTorch 2.14.1+cu130, sglang 0.5.23, sglang-kernel 0.4.10, flashinfer_python 0.7.2, triton 3.8.0, NVCC 13.1
4 GPUs, PCIe only (no NVLink). v0.5.23 plus the two compute-capability-12.x gate fixes for this model (#48399,
#49182); nothing else is modified.
```

Their own reading: with the bs-1 graph removed, a lone request runs in the bs-2 graph with one padded row at 27.2 ms
(24.5 -> 35.8 tok/s), and greedy output is identical. Eager decode is flat across batch sizes.

## Status upstream (checked 2026-11-12, rechecked 2026-11-13)

- **#49900**: open, no comments, no labels.
- **#39872** (nguyenhoangthuan99, "tune DeepSeek-V4.1 block FP8 decode on RTX PRO 6000"): open; last push 09-17
  (`7025bba`), last activity 09-22, CI never ran (the `run-ci` label was never added). Six device-exact tables for
  `NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition` at TP8 shapes, entries with `SPLIT_K` up to 16 and `SWAP_AB`, plus a
  dispatch change that lets SM120 take the split-K kernel when a config names `SPLIT_K`. Its tree had the kernel and
  its `configs/` in `kernels/ops/quantization/`; main has both in `kernels/ops/gemm/`, so it no longer applies as is.
  Reported at TP8: C1 36.7 -> 87.9 tok/s, dense-FP8 kernel time 826.92 -> 96.43 ms over 40 steps. MarkSight
  (09-22, RTX 6000D, tables renamed, not retuned): 51.0 -> 15.0 us per call.
- **Merged after v0.5.23** (moria97, 2026-11-10): #45263 adds `[32, 32]` tables for RTX PRO 5000 (1792x5120 at M=1:
  90.5 -> 19.4 us; bs1 TPOT 57.6 -> 23.2 ms on 8x PRO 5000, DP8 attention + EP8). #45262 serves block-FP8 weights as
  MXFP8 on FlashInfer's b12x kernel under `--fp8-gemm-backend flashinfer_cutlass` (1792x5120 at M=1: CUTLASS 33.5 us,
  b12x 7.6 us; bs1 TPOT 17.4-18.0 ms); waynehacking8 measured the b12x kernel faster on a PRO 6000 Max-Q at 108 of 112
  points (kernel only). Neither changes what an RTX PRO 6000 runs by default: table names are device exact, and the
  MXFP8 route needs the flag.
- **Open, same family**: #45391 (FlashInfer b12x MXFP8 dense backend for SM120), #45086 (`do_not_specialize` on the
  row count for the mHC split-K kernels, the Triton mechanism of H2, there seen as recompiles).
- **Not fixed on main** (`c2578f2`, 2026-11-13): no table carries any RTX PRO 6000 name, and the kernel, its default
  config, and the SM90-only split-K gate are unchanged from v0.5.23 on CUDA (v0.5.23 differs only in a HIP-only
  unrolled kernel).
- Searched: `gh search prs` for `w8a8_block_fp8`, `block fp8 sm120`, `RTX PRO 6000 fp8 config`, `fp8 matmul tuning
  120`, `b12x mxfp8`, `do_not_specialize`, `split-k block fp8`, and `49900`.
- **New by 2026-11-13** (main `5792cf6`: still no RTX PRO 6000 table, kernel and gate unchanged): #49705 (xulunfan,
  open, lint failing, no review yet) routes `SPLIT_K`/`SWAP_AB` entries to the split-K kernel on any CUDA device
  (`split_k_tuned = _is_cuda and (...)`), the rule #39872 carried, and adds split-K entries to the RTX PRO 5000
  `[32, 32]` tables, kept to M keys up to 256 because eager prefill allocates the `(SPLIT_K, M, N)` partials at a new
  size on every call. #49789 and its fix #49810 (open) keep K % 128 != 0 layers on Triton under the MXFP8 view instead
  of CUTLASS MXFP8. #50057 reports the shipped tuner's empty `[32, 32]` search space (T0 below) and its timing with
  the weight resident in L2; #50095 (open) fixes the search space only.

## What the code does (sglang main `c2578f2`; v0.5.23 `297e54f` in brackets)

Paths are under `python/sglang/`.

1. The checkpoint is block FP8 with `weight_block_size [32, 32]`, `scale_fmt ue8m0`, FP4 routed experts (HF
   `config.json`). For a `[32, 32]` weight, `dispatch_w8a8_block_fp8_linear` returns `triton_w8a8_block_fp8_linear`
   because `block[1] != 128` (`srt/layers/quantization/fp8_utils.py:753-760` [732-739]); the DeepGEMM block-32 path
   is SM90, or SM100 on request (:739-751 [718-730]). `auto` becomes `cutlass` on SM120 (:1358-1359 [1212-1213]),
   and block-FP8 is served as MXFP8 only under an explicit `flashinfer_cutlass` or `flashinfer_cutedsl` (:944-962
   [900-918]).
2. `triton_w8a8_block_fp8_linear` quantizes the activation per 32 columns with power-of-two fp32 scales (:2114-2118
   [1803-1807]) and calls `w8a8_block_fp8_matmul_triton` (:2140 [1829]).
3. `w8a8_block_fp8_matmul_triton` (`kernels/ops/gemm/fp8_kernel.py:574` [776]) loads
   `configs/N={N},K={K},device_name={torch name, spaces to _},dtype=fp8_w8a8,block_shape=[32, 32].json` (:392-398
   [572-578]) and takes the entry with the nearest M (:607). main ships `[32, 32]` tables for H20, H200, and RTX PRO
   5000 only, so on any RTX PRO 6000 every shape logs the reporter's warning (:428-434 [608-614]) and runs
   `BLOCK_SIZE_M 64, BLOCK_SIZE_N 32, BLOCK_SIZE_K 32, GROUP_SIZE_M 32, num_warps 4, num_stages 3` (:627-634
   [813-820]): one M tile, N/32 programs, each walking K 32 columns at a time.
4. Split-K and swap-AB exist in `_w8a8_block_fp8_matmul_hopper` (:144 [147], plain `tl.dot`, no TMA or WGMMA), but
   only SM90 may take it (:638-640 [824-826]). A table entry naming `SPLIT_K` on SM120 would reach the base kernel,
   which has no such parameter; #39872's dispatch change is what makes such tables usable there.
5. The base kernel takes `M` as a plain int (:60 [63]). Triton 3.8 compiles an int argument equal to 1 as a constant
   (`python/src/specialize.cc:274` at v3.8.0), so one running request runs a different binary from two, under the
   same config. That is H2.
6. The shipped tuner, `benchmark/kernels/quantization/tuning_block_wise_kernel.py`, searches `BLOCK_SIZE_K` 64 and
   128 on CUDA (:164 [179]) and keeps only sizes dividing `block_k` (:338 [353]), so for `[32, 32]` its search space
   is empty and `tune` stops at `assert best_config is not None` (:278 [293]). It also tunes the base kernel only (no
   split-K) and times single eager launches. T0 runs it once to confirm.

## Shapes

Per-rank `(N, K)` of every block-FP8 linear in one decode step at `--tp-size 4`, from the checkpoint's safetensors
headers (revision `314113f`) and the model's parallel layout. They add up to 210 calls, the reporter's 10,500 over 50
steps. `wo_a` is block FP8 too but runs its own batched kernel (`srt/models/deepseek_v4.py:1158-1167`), and the MTP
layers' `main_proj` (5120x15360) runs only with speculation.

| (N, K) | Module (checkpoint shape) | At TP4 | Calls per step | MB | Default: programs x K steps |
|---|---|---|--:|--:|---|
| 1792, 5120 | `attn.wqkv_a`: `wq_a` 1280x5120 + `wkv` 512x5120, fused (`SGLANG_OPT_FUSE_WQA_WKV` on, `srt/environ.py:2125`) | replicated (`srt/models/deepseek_v4.py:1105`) | 40 | 9.2 | 56 x 160 |
| 8192, 1280 | `attn.wq_b` 32768x1280 | column, attn TP (`:1128`) | 40 | 10.5 | 256 x 40 |
| 5120, 2048 | `attn.wo_b` 5120x8192 | row, attn TP (`:1168`) | 40 | 10.5 | 160 x 64 |
| 1152, 5120 | `shared_experts` gate_up, 2 x 2304x5120 | column, TP (`srt/models/deepseek_v2.py:880-904`) | 40 | 5.9 | 36 x 160 |
| 5120, 576 | `shared_experts` down 5120x2304 | row, TP | 40 | 2.9 | 160 x 18 |
| 4096, 1280 | `indexer.wq_b`, 8 index layers | replicated (`srt/layers/attention/dsv4/dsv41_sparse.py:230`) | 8 | 5.2 | 128 x 40 |
| 25600, 6144 | `engram.wkv`, layers 1 and 14 | replicated (`srt/layers/engram.py:1539`) | 2 | 157.3 | 800 x 192 |

Shared experts stay TP-sharded unless an all-to-all MoE backend or `SGLANG_SHARED_EXPERT_TP1` is set, and the
reporter sets neither.

A step reads 1.9 GB of these weights, under 2 ms at the card's bandwidth, against the 36.4 ms (M=1) and 21.9 ms (M=2)
the profile shows. #39872's six tables are TP8 shapes: its 1792x5120 and 25600x6144 are replicated layers and match
TP4, its 4096x1280 (`wq_b` at TP8) happens to be the indexer's shape, and the other three are the TP8 halves of TP4
rows. Of the reporter's three quoted shapes, only the largest per call (1569 vs 959 us) is identifiable: 25600x6144.

## Hypotheses

| # | Hypothesis | Confirmed if | Killed if |
|---|---|---|---|
| H0 | The bench stands for the server: synthetic operands in the server's layout, weights cycled past L2, CUDA graph | `shipped` summed over the shapes lands within about 25% of the profile's 36.4 ms (M=1) and 21.9 ms (M=2) per step | far off; then only ratios between arms are quoted, not absolute times |
| H1 | The default config is the problem at every M: N/32 programs, each a serial walk of K in 32-column steps, latency bound | the best tuned table cuts the per-step estimate at M=1 and M=2 to under half of `shipped` | within 20% of `shipped` |
| H2 | M=1 is slower than M=2 under the same config because Triton compiles M == 1 as a constant | `default` M1/M2 at 1.3 or more, `default-nospec` M1/M2 at 1.1 or less with its M=1 near `default`'s M=2, and the two M=1 PTX files differ | `default` M1/M2 at 1.1 or less in the bench (go to H3), or `default-nospec` as slow at M=1 |
| H3 | The M=1 penalty needs the server's graph: operand layout there, a pre-quantized input, or overlap with other kernels | no M=1 penalty in the bench, graph or eager, while the server shows 1.7x | H2 confirmed |
| H4 | Serving these weights as MXFP8 on FlashInfer (CUTLASS on v0.5.23, b12x on main under `--fp8-gemm-backend flashinfer_cutlass`) beats the best Triton config, activation quant included | `mxfp8-b12x` under `linear-tuned` on the shapes that carry most of the step | slower, failing, or over tolerance on this stack |
| H5 | The published tables transfer partly: #39872's (Server Edition, TP8 shapes) and #45263's (PRO 5000) help on a PRO 6000 at TP4 but leave time against a TP4 tune | `pr39872` and `pro5000` between `shipped` and `tuned-*` | at or under `tuned-*`, or no better than `shipped` |

Stage 2 is worth planning if an arm with no row over tolerance cuts the per-step estimate at M=1 by 10 ms or more (a
quarter of the reporter's 40 ms step), or if H2 holds, since its fix is one decorator argument to prove end to end.

## Arms (`compare.py`)

Every arm runs on every shape at M = 1, 2, 4, 8, in a CUDA graph that cycles through copies of the weight adding up
to twice the L2 (as 210 different weights stream through a decode step), replayed for about 100 ms. Kernel arms get
the activation already quantized by sglang's own function; linear arms include that quantization. Each output is
checked against the float64 product of the dequantized operands (kernel arms, tolerance 1e-2) or of the bf16
activation and dequantized weight (linear arms, 5e-2).

| Arm | What runs |
|---|---|
| `shipped` | the image's `w8a8_block_fp8_matmul_triton`, which finds no table for this card (env.json lists the ones it has) and runs the default config; also eager |
| `default` | `run_config` with the default config: the same launch, to show the mirror matches the wrapper; also eager |
| `default-nospec` | the default config on the same kernel source re-jitted with `do_not_specialize=["M"]`; also eager, and at M = 16, 128, 2048 for prefill-sized M |
| `pr39872` | #39872's tables (head `7025bba`), the exact shape where it has one, else the TP8 table of the same module; its dispatch rule |
| `pro5000` | #45263's RTX PRO 5000 tables, same lookup |
| `tuned-stock` | `tune.py`'s base-kernel table: stock sglang loads it with no code change |
| `tuned-stock-nospec` | the same configs on the `do_not_specialize` kernel |
| `tuned-splitk` | `tune.py`'s best overall table, split-K and swap-AB included; needs #39872's dispatch rule |
| `linear-shipped`, `linear-tuned` | activation quant plus `shipped`, or plus `tuned-splitk` |
| `mxfp8-cutlass`, `mxfp8-b12x` | sglang's `flashinfer_mxfp8_blockscaled_linear` on the weight's MXFP8 view, as `--fp8-gemm-backend flashinfer_cutlass` serves it; b12x only where K % 128 == 0, as main does |

`tune.py` sweeps 97 base-kernel configs (BLOCK_SIZE_M 16 or 64, BLOCK_SIZE_N 32 to 256, 2 to 8 warps, 2 to 5 stages,
plus the default) and 270 split-K configs (BLOCK_SIZE_N 32 to 128, 2 to 8 warps, 2 to 4 stages, SPLIT_K 1 to 16,
swap-AB on and off) per shape and M. BLOCK_SIZE_K stays 32: above it the kernel's scale step divides by zero. Every
config is compiled first for the two M classes and two scale-stride classes Triton keys on, on up to 16 processes (one
per CPU and per 4 GiB of the container's memory), within half the sweep's budget; a config not compiled by then
leaves the sweep, base-kernel configs first in line. Each is then screened in a 10 ms graph, and the 5 fastest per
family are timed for 100 ms and checked. Shapes go heaviest first, so a short budget still tables what the step
spends most on. Each table gets the default config at M=16 so nearest-M lookup hands prefill back to it. T0 runs the
shipped tuner on 1792x5120.

## Box and time

- `lmsysorg/sglang:v0.5.23-cu130` pinned by its manifest-list digest (Docker Hub, pushed 2026-10-30), the reporter's
  release: torch 2.14.1+cu130, Triton 3.8.0, FlashInfer 0.7.2. Its CUDA kernel path is main's (above), so the stage 1
  numbers hold for both.
- 1x RTX PRO 6000 Blackwell Server Edition (the reporter's card), falling back to the Workstation Edition (same
  sm_120 die and SM count). Each rank of a TP4 server runs exactly these per-rank GEMMs, so one GPU covers the kernel
  question; all-reduce and the server's own graph do not enter it.

| Step | Estimate | Limit in `setup.sh` |
|---|---|---|
| Pod create and image pull, 16.1 GB compressed | 5-10 min | |
| Harness gates: SSH, watchdog, idle GPU, upload | 3-5 min | |
| `env_probe.py`, `import fp8bench` (exits 1 if it fails), `fetch_tables.py` | 1 min | 3 + 5 + 2 min |
| `baseline`: 5 arms x 7 shapes x 4 M, 3 of them also eager | 2-4 min | 8 min |
| `mxfp8`: `linear-shipped` and 2 MXFP8 arms, FlashInfer may build kernels on first use | 1-8 min | 10 min |
| `tune`: precompile about 1,500 binaries, then about 10,000 graph screens | 6-12 min | what is left after `tuned`'s 7 min, at most 20 |
| `tuned`: 4 arms | 1-3 min | 7 min |
| `nospec_large_m`, `stock_tuner` | 2-3 min | 5 + 5 min |
| Pull and `down` | 2-3 min | |
| **Total** | **about 25-50 min** | |

`planned_hours = 1.0`. `setup.sh` starts no step whose limit would end within 8 minutes of the deadline, in the order
above, so a slow pull costs the last arms, never the pod or the baseline. MXFP8 runs before the sweep, and the sweep's
limit is cut to leave the tuned arms theirs (under 5 min left, it is skipped).

## Run

```bash
harness quote sglang-49900
harness up sglang-49900; harness down sglang-49900
harness budget show sglang-49900
```

`up` rents, runs `setup.sh`, and pulls the results; `down` pulls again and ends the pod. `;` rather than `&&`, so
`down` runs even when `up` exits non-zero.

## Outputs (`results/sglang-49900/`)

| File | Holds |
|---|---|
| `summary.txt` | per-call time per shape, arm, and M with the M1/M2 ratio; dense FP8 time per decode step per arm next to the reporter's; H2's stock vs nospec times with registers, spills, and shared memory; the tuned configs; rows over tolerance; T0's last line; steps that did not end 0 |
| `compare.csv`, `compare.jsonl` | every measurement: arm, shape, M, graph or eager, us, rel_err, config, table shape |
| `tune.jsonl` | every screen and final timing of the sweep, and the precompile's failed configs |
| `tables/tuned-stock/`, `tables/tuned-splitk/` | the tuned tables under sglang's file names for this card, ready for `kernels/ops/gemm/configs/` |
| `tables/pr39872/`, `tables/pro5000/` | the published tables as fetched |
| `ptx/default-m{1,2}.ptx`, `ptx/default-nospec-m{1,2}.ptx` | the M=1 and M=2 binaries of the default config on 1792x5120 |
| `env.json`, `env_extra.txt`, `env.txt` | versions, SMs, L2, which tables the image has for this card, clocks and power |
| `runs.tsv`, `<step>.log` | each step's exit code (124 or 137 is a timeout, `skipped` is the deadline guard), seconds, log; `import_check.log` is the import gate's output |

`summarize.py` reads only these files, so it reruns on the laptop: `python3 issues/sglang-49900/summarize.py
results/sglang-49900`.

## Stage 2 (described, not planned in hours)

The reporter's server on 4x RTX PRO 6000 Server Edition, PCIe only (`require_pcie`), with real weights, to show the
stage 1 winner end to end.

- Image: an sglang nightly from 2026-11-10 or later, pinned by digest, which has #48399 and #49182 (the reporter's two
  gate fixes) and #45262 and #45263; or v0.5.23 with those two patches, as the reporter ran.
- Weights: `deepseek-ai/DeepSeek-V4.1-Flash` at `314113f`, about 510 GB of safetensors (the engram tables included),
  on container disk, with the download floor and cold-read check from `issues/sglang-33422/setup.sh`.
- Variants, the reporter's flags in all: the baseline; `--cuda-graph-bs-decode 2 4 8` (the workaround); the
  `do_not_specialize` patch, if H2 holds; the tuned tables copied into `configs/` by a `pre_launch` hook, with
  #39872's dispatch rule as a patch if the split-K table won; `--fp8-gemm-backend flashinfer_cutlass`, if H4 holds.
- Per variant: step time at 1, 2, 4, and 8 running requests from the `Decode batch` lines, the reporter's method; a
  50-step `/start_profile` at 1 and 2 running requests for the per-kernel split; GSM8K on 50 questions.
- Hours and box are planned after stage 1, from its winner.

## Files

| File | Does |
|---|---|
| `harness.toml` | 1x RTX PRO 6000, the pinned image, no variants |
| `setup.sh` | stage 1, every step under its own limit and the deadline guard |
| `fp8bench.py` | shapes, synthetic operands, the kernel launch, graph and eager timing |
| `compare.py` | the arms |
| `tune.py` | the sweep and the tuned tables |
| `fetch_tables.py` | #39872's and #45263's tables at pinned commits |
| `env_probe.py` | `env.json` |
| `summarize.py` | `summary.txt` and `compare.csv` |

## Results (stage 1)

### 2026-11-13, 1x RTX PRO 6000 Blackwell Server Edition, RunPod secure cloud, pod `sm120-sglang-49900-c69a74`

Box: 188 SMs, 128 MB L2, 95 GiB, 600 W limit, max SM clock 2430 MHz, driver 595.91.07 (the reporter's is 590.48.01).
Image `lmsysorg/sglang:v0.5.23-cu130` by digest: sglang 0.5.23, torch 2.14.1+cu130, Triton 3.8.0, FlashInfer 0.7.2. It
ships no `[32, 32]` table for this device (`env.json`: `device_tables: []`), so every shape logged the reporter's
`Using default W8A8 Block FP8 kernel config` warning. Pod create to terminate took 13 min: 5.5 min to ready (image
pull), 7 min of `setup.sh`, every step exiting 0 except T0 (exit 1, as predicted). The sweep compiled all 367 configs
in 31 s on 16 workers and ran 10,276 graph screens. No row has an error or is over tolerance: kernel arms are at
1.6e-3 to 1.7e-3 relative error against the float64 reference on every arm, split-K included (tolerance 1e-2), linear
arms at 2.6e-2 to 2.8e-2, which is the activation quant (tolerance 5e-2).

Dense FP8 GEMM per decode step (ms): calls per step x per-call time over the 7 shapes (210 calls), CUDA graph, weights
cycled past L2. Kernel arms get the activation already quantized, as the profiled kernel does:

| Arm | M=1 | M=2 | M=4 | M=8 | M1/M2 | `shipped` / arm at M=1 |
|---|--:|--:|--:|--:|--:|--:|
| Reporter's profile, rank 0 | 36.4 | 21.9 | | | 1.66 | |
| `shipped` (default config) | 28.90 | 17.34 | 12.49 | 10.19 | 1.67 | 1.0x |
| `default-nospec` | 28.74 | 17.34 | 12.49 | 10.18 | 1.66 | 1.0x |
| `pro5000` (#45263's tables) | 5.37 | 5.38 | 5.53 | 5.78 | 1.00 | 5.4x |
| `tuned-stock-nospec` | 4.67 | 4.36 | 4.42 | 4.56 | 1.07 | 6.2x |
| `tuned-stock` | 4.23 | 4.36 | 4.42 | 4.55 | 0.97 | 6.8x |
| `pr39872` (#39872's tables) | 2.54 | 2.46 | 2.68 | 2.74 | 1.03 | 11.4x |
| `tuned-splitk` | 2.11 | 2.09 | 2.12 | 2.17 | 1.01 | 13.7x |

Linear arms, activation quant included, which is what MXFP8 has to beat:

| Arm | M=1 | M=2 | M=4 | M=8 | M1/M2 |
|---|--:|--:|--:|--:|--:|
| `linear-shipped` | 29.12 | 18.35 | 12.70 | 9.86 | 1.59 |
| `mxfp8-cutlass` | 8.04 | 8.09 | 8.02 | 7.80 | 0.99 |
| `mxfp8-b12x` (5120x576 on CUTLASS, as main serves it) | 2.66 | 2.88 | 2.90 | 2.89 | 0.92 |
| `linear-tuned` | 2.40 | 2.38 | 2.40 | 2.45 | 1.01 |

Per call (us), M=1 / M=2:

| Shape | Calls | `shipped` | `tuned-stock` | `tuned-splitk` | `pr39872` | `pro5000` | `linear-tuned` | `mxfp8-b12x` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1792x5120 | 40 | 110.9 / 79.0 | 31.5 / 32.8 | 9.5 / 9.5 | 11.4 / 10.7 | 36.9 / 35.2 | 10.9 / 10.9 | 11.9 / 14.4 |
| 8192x1280 | 40 | 185.9 / 100.2 | 11.7 / 11.5 | 9.8 / 9.8 | 10.1 / 9.8 | 16.9 / 15.2 | 11.2 / 11.1 | 9.5 / 9.7 |
| 5120x2048 | 40 | 183.6 / 96.9 | 15.0 / 15.9 | 9.8 / 9.8 | 10.3 / 9.9 | 25.4 / 19.4 | 11.2 / 11.1 | 9.5 / 10.0 |
| 1152x5120 | 40 | 83.5 / 62.4 | 30.0 / 31.0 | 7.3 / 7.3 | 9.3 / 9.1 | 29.9 / 38.4 | 8.6 / 8.6 | 11.2 / 13.8 |
| 5120x576 | 40 | 73.2 / 41.6 | 5.5 / 5.9 | 4.8 / 4.8 | 5.2 / 4.9 | 10.7 / 13.0 | 6.1 / 6.1 | CUTLASS 16.9 / 16.8 |
| 4096x1280 | 8 | 107.1 / 53.1 | 9.2 / 9.5 | 6.6 / 6.4 | 7.0 / 6.7 | 15.0 / 11.8 | 7.9 / 7.7 | 6.2 / 6.6 |
| 25600x6144 | 2 | 1277.6 / 854.7 | 204.8 / 197.7 | 205.0 / 197.7 | 312.5 / 314.1 | 225.6 / 218.3 | 207.6 / 200.5 | 123.9 / 120.9 |

- Half of `shipped`'s M=1 step is `wq_b` and `wo_b` (8192x1280 and 5120x2048, 7.4 ms each), and those two carry 6.9 of
  the 11.6 ms it spends over M=2. Its M1/M2 per shape runs from 1.34 (1152x5120) to 2.02 (4096x1280). On 25600x6144,
  the one shape the profile identifies, it is 1277.6 vs 854.7 us (1.49); the profile has 1569 vs 959 (1.64).
- The K=5120 shapes with few output tiles need split-K: 1792x5120 and 1152x5120 launch 56 and 36 programs at
  BLOCK_SIZE_N 32 on 188 SMs and stay at 31.5 and 30.0 us under the best base-kernel config; SPLIT_K 16 takes them to
  9.5 and 7.3.
- 25600x6144 (800 programs already) takes no split: the base kernel moves its 157 MB in about 200 us (0.77-0.80 TB/s),
  b12x in 121-124 us (1.27-1.30 TB/s), the one clear MXFP8 win. `tuned-splitk` reads a step's 1.9 GB in 2.11 ms
  (0.91 TB/s).
- `pr39872` gives back 0.43 ms at M=1 against `tuned-splitk`, half of it on 25600x6144 (its SPLIT_K 4 entry, 312.5 vs
  205.0 us). `pro5000` gives back 1.14 ms against `tuned-stock`, 0.42 of it on 5120x2048.

Hypotheses:

- H0 holds: `shipped` is 21% under the profile at both M (28.90 vs 36.4 ms, 17.34 vs 21.9), inside the 25% bar, and
  the ratio matches (1.67 vs 1.66). Mean per call is 137.6 vs 82.6 us against the profile's 173 vs 104. What makes up
  the 21% (one GPU alone against one rank of a TP4 server, profiler timing, driver) is not separated by this run.
- H1 holds: the default config is slow at every M. `tuned-stock` cuts the step 6.8x at M=1 (28.90 to 4.23 ms) and 4.0x
  at M=2; `tuned-splitk` 13.7x and 8.3x.
- H2 dies. Triton does compile M == 1 as a constant (`ptx/default-m1.ptx` differs, 64 registers against 80), but
  `default-nospec` at M=1 runs the same PTX the stock kernel runs at M=2 (`default-m2`, `default-nospec-m1`, and
  `default-nospec-m2` share one md5) and is just as slow: 28.74 against 28.90 ms per step, M1/M2 1.66. Under the tuned
  base-kernel configs the specialization helps (`tuned-stock` 4.23 ms at M=1 against `tuned-stock-nospec` 4.67), so
  `do_not_specialize` on M is not a fix here.
- H3 dies: the penalty shows on one GPU with no server, in a CUDA graph and eager alike (eager `shipped` 28.90 vs
  17.37 ms), on a pre-quantized row-major activation. Why one binary under BLOCK_SIZE_M 64 runs 1.3-2.0x slower per
  call at M=1 than at M=2 is not established. Every tuned arm drops it: M1/M2 per step 0.97 to 1.03.
- H4 dies, narrowly: `mxfp8-b12x` is 2.66 and 2.88 ms at M=1 and M=2 against `linear-tuned`'s 2.40 and 2.38. It wins
  on 8192x1280, 5120x2048, 4096x1280, and 25600x6144, loses on 1792x5120 and 1152x5120 (11.9 vs 10.9 and 11.2 vs 8.6
  us at M=1, wider at M=2), and loses most on 5120x576, which main sends to CUTLASS MXFP8 (K % 128 != 0): 16.9 against
  6.1 us, 0.43 ms per step. Composed from the measured rows, not run as one arm: b12x with 5120x576 on the tuned
  Triton table would be 2.23 ms at M=1, and on the default config (#49810 on a card with no table) 4.97.
- H5 holds, like for like: #39872's split-K TP8 tables give 2.54 ms, between `shipped` and `tuned-splitk`'s 2.11;
  #45263's PRO 5000 base-kernel tables give 5.37, between `shipped` and `tuned-stock`'s 4.23. Both use the exact shape
  where the table has it (1792x5120, 4096x1280, 25600x6144) and the TP8 table of the same module otherwise.
- T0: the shipped tuner on 1792x5120 with `--block-n 32 --block-k 32` stops at `assert best_config is not None`
  (`tuning_block_wise_kernel.py:293` in the image) and exits 1, as predicted.

Tuned configs (`tables/tuned-stock/`, `tables/tuned-splitk/`, 7 files each, named for
`NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition`; copies are committed in this folder under `tables/`):

- `tuned-stock`: every shape and M takes BLOCK_SIZE_M 16, BLOCK_SIZE_N 32, GROUP_SIZE_M 1, 5 stages; 4 warps at M=1
  on five shapes, 2 warps everywhere else. Stock sglang loads these with no code change.
- `tuned-splitk`: SPLIT_K 16 or 8 on every entry at M 1 to 8 except 25600x6144, which keeps the `tuned-stock` config;
  BLOCK_SIZE_N 32 to 128, 2 to 8 warps, 2 to 4 stages, swap-AB on 10 of the 24 split entries. No entry names SPLIT_K
  1 without swap-AB, so #49705's rule routes every entry as the kit's #39872 rule did. On main as it is, the split
  entries reach the base kernel, which has no such parameter, so this table needs #49705 (or #39872's dispatch
  change) first.
- Both carry the default config at M=16, so with nearest-M lookup M up to 12 takes a tuned entry and M of 13 and up
  (prefill) runs exactly what it runs today. Split entries exist only at M up to 8.

Caveats:

- Synthetic weights and activations in the checkpoint's layout (`[32, 32]` blocks, ue8m0 scales); no checkpoint, no
  server.
- Per-call kernel time summed over one step's calls, not end-to-end ITL. The reporter's step (40.4 vs 28.6 ms) holds
  about 4.1 ms at one request and 7.0 ms at two outside this kernel by their profile (40.5 - 36.4 and 28.9 - 21.9 ms
  per step), which this does not move, and all-reduce, MoE, and attention do not enter the bench. The end-to-end gain
  needs stage 2 on 4x with real weights.
- One run, one measurement per arm, shape, and M. Arms run in different steps disagree on some rows by more than the
  activation quant explains: `linear-shipped` is under `shipped` on 25600x6144 at M=2 (726.0 vs 854.7 us) and over it
  on 5120x576 at M=2 (68.6 vs 41.6), while their step totals agree within 6%. Treat single rows as one sample; the
  conclusions rest on step totals and on gaps of 4x or more.
- Server Edition only. The Workstation Edition and the 300 W Max-Q read tables under their own device names and were
  not measured.
- M 1, 2, 4, 8 only, the reporter's `--max-running-requests 8`.
- `mxfp8-b12x` runs FlashInfer 0.7.2's b12x through the image's `flashinfer_mxfp8_blockscaled_linear`, as main's
  #45262 dispatch does; the v0.5.23 server itself has no route to it.

Next, stage 2 (below). By the bar above it is worth planning: `tuned-stock` cuts the M=1 estimate by 24.7 ms and
`tuned-splitk` by 26.8 ms with no row over tolerance. Its variants after this run: the baseline;
`--cuda-graph-bs-decode 2 4 8`; `tuned-stock` copied into `configs/` by a `pre_launch` hook (no code change);
`tuned-splitk` with #49705's routing line as a patch. H2 died, so no `do_not_specialize` variant. H4 died by 0.26 ms
at M=1, so `--fp8-gemm-backend flashinfer_cutlass` is optional, and only worth running on a nightly with #49810 and the
tuned tables together (without a table, #49810 moves 5120x576 to the default config).
