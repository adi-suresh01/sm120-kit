# sglang#40320: FlashInfer autotune cache dropped on every boot under TP/EP > 1

- Issue: https://github.com/sgl-project/sglang/issues/40320 (open; opened 2026-09-19)
- Fix PR: https://github.com/sgl-project/sglang/pull/40420 (open since 2026-09-20, no reviews, GitHub reports it
  conflicting with main)
- Status: run 2026-11-13 on 2x RTX PRO 6000 (see Results). Stock main with FlashInfer 0.7.4 does not reproduce on the
  sm_120 `flashinfer_cutlass` path; with FlashInfer's pre-#6006 rank keys put back it does, and #40420 fixes that.
  The reporter's GB10 `flashinfer_mxfp4` setup is not tested.

## Reporter setup (quoted from the issue, non-ASCII characters replaced)

The reporter (2026-09-19) is not on sm_120:

> - 4x GB10 (DGX Spark), SGLang TP4 / EP2, switchless RoCE ring
> - MoE runner backend `flashinfer_mxfp4` (MXFP4 experts), DeepSeek-V4.1-Flash
> - `--chunked-prefill-size 4096`, `--mem-fraction-static 0.90`
> - FlashInfer 0.6.18, CUDA 13.0, cuDNN 92000
>
> ```
> Running FlashInfer autotune with cache: /root/.cache/sglang/flashinfer/autotune/0.6.18/sm121/<key>/rank_tp3_pp0_dp0.json
> [AutoTuner]: Tuning trtllm::fused_moe::gemm1:   0%|          | 0/8 ...
> ```
>
> Tuning count is **26-27 profiles per boot** regardless of whether a complete cache file is already present at that
> path.

Impact they measured: two boots of one config gave 3169 vs 2431 tok/s single-stream prefill at 8K (the tactic is
re-drawn each boot). Their workaround keeps the files and skips the `unlink`. Their caveat: if some ranks hit the
cache and others miss, the timing reduce desyncs and startup hangs (also flashinfer-ai/flashinfer#6082).

The sm_120 reports are in the comments:

- 2026-10-07: "8x RTX PRO 6000 (SM120, flashinfer_cutlass MoE): cold-start autotune with
  `FLASHINFER_AUTOTUNE_INDEPENDENT=1` completes in ~5 min instead of dying at 1800s gloo timeout."
- 2026-10-08 (8x H20-3e, TP8/EP8, SGLang 0.5.21, FlashInfer 0.6.18): every rank tunes identical shapes, and the
  files still differ because the `MoERunner` key carries the rank; after normalizing only `ep_rank` the 8 files match.
- 2026-10-27: "8x RTX PRO 6000, TP4/EP2, SGLang `main` at `62cee1b15`, FlashInfer `f64b4ba`. Every start logs
  `per-rank caches disagree`, because the `MoERunner` key extras include `tp_rank` and `ep_rank`." Their workaround
  (hash the stamp plus keys without extras) took compile plus autotune on restart from 18.6 min to 0.1 min.

## Root cause (main at c2578f28, 2026-11-13)

All in `python/sglang/srt/model_executor/runner/flashinfer_autotune.py`:

- `:209` each rank keeps its own file, `rank_tp{tp_rank}_pp{pp_rank}_dp{dp_rank}.json`, in a directory keyed on model,
  quantization, MoE backend, `tp_size`, `moe_ep_size` and more (`:179-199`).
- `:225-238` the tactic sync group is the TP group's CPU (gloo) group whenever TP > 1; EP is not part of the test.
- `:306-311` with the cache on (`SGLANG_FLASHINFER_AUTOTUNE_CACHE`, default 1) and a sync group, every autotune round
  calls `_drop_diverged_autotune_cache` before loading.
- `:257-273` `_autotune_cache_digest` hashes the whole file, `{"file": configs, "env": env}` (`:272`).
- `:283-294` the digests are all-gathered; unless every rank's is equal (`:287`), every rank unlinks its file
  (`:294`) and tunes from scratch.

Why the files differ: FlashInfer writes each entry under `str((op, runner, profile, extras))`
(`flashinfer/autotuner/autotuner.py`, `ProfilingCacheKey.file_key`). Up to FlashInfer main at f64b4ba (2026-10-19),
the CUTLASS `MoERunner.get_cache_key_extras` returned `tp_rank`, `ep_rank` and `cluster_rank`
(`flashinfer/fused_moe/core.py:751-780` there), so for any TP > 1, with or without EP, no two ranks could write the
same file. The gate came in with #35343 (first released in v0.5.19, per the 10-08 comment).

What changed underneath: flashinfer-ai/flashinfer#6006 (merged 2026-10-28) dropped the ranks from that key, and the
v0.7.2 tag, 0.7.3 and 0.7.4 all carry rank-free extras (`core.py:1008-1040` at v0.7.4). FlashInfer 0.7.x also
all-reduces the timings before it records them, so two ranks that profile the same shapes should now write
identical files. The gate still compares bytes, so it still drops caches wherever the ranks' files legitimately
differ: runners still keyed on rank-local values (#6006's notes name the unified and CuTe-DSL runners'
`local_expert_offset`; #6139 fixed CuTe-DSL), or ranks that profile different shapes, which the reporter describes
on the GB10 `flashinfer_mxfp4` path. Whether the sm_120 CUTLASS path is still affected on main is H1 below.

The gate's stated purpose (`:279-281`: "A cache hit skips a profile, so caches that disagree desync the
reduction") is also covered inside FlashInfer now: before a cached entry is used, every rank in the tune group
reports hit or miss, and any miss sends all ranks into that profile (`_all_ranks_hit_cache`, `autotuner.py:1369` at
v0.7.4; since flashinfer-ai/flashinfer#6894, merged 10-23, the ranks must also agree on the cached tactic). It is at
f64b4ba and in 0.7.3 and 0.7.4, and absent from the v0.7.2 tag that sglang v0.5.23 pins. H6 tests it.

## Upstream state (checked 2026-11-13)

- #40420 digests `{"loadable": True, "stamp": _metadata, "env": env}` instead of the file. Its source hunk applies to
  main and to the pinned image with a 47-line offset (`patches/pr40420.diff` is that hunk re-cut at main's lines).
  Its test-file hunk no longer applies (`test_flashinfer_autotune_sync.py` grew on main), so the PR needs a rebase of
  the test file only. Its CI runs are red; no maintainer has commented.
- No fix on main: the gate's lines are unchanged. Overlapping work: #48568 (open) edits the same file but only the
  sync group for attention-DP drafts; #42108 (closed unmerged) moved persistence to `autotune_v2`'s managed store;
  flashinfer-ai/flashinfer#6006 and #6139 (merged) and #6161 (closed unmerged, independent per-rank profiling).
- `python/sglang/srt/utils/log_utils.py:273` sets FlashInfer's logger to ERROR, so FlashInfer's own "Loaded" and
  "Saved" lines never reach the server log. `hooks.sh` copies the cache files before and after each boot instead.

## Hypotheses

| | Hypothesis | Confirms | Kills |
|---|---|---|---|
| H1 | On main with FlashInfer 0.7.4, EP2 ranks write identical files, so stock main keeps the cache on the sm_120 CUTLASS path | `ep2-warm`: no gate drop, no tuning bars, 0 entries changed; rank files equal | a `per-rank caches disagree` line in `ep2-warm` |
| H2 | With rank fields in the key (FlashInfer before #6006), stock main drops the cache on every boot: the 10-27 report | `ep2-rankkey-warm`: gate drop, tuning bars, every entry re-measured | `ep2-rankkey-warm` reuses the cache |
| H3 | #40420 keeps per-rank caches that differ in content but load under the same stamp | `ep2-rankkey-pr40420-warm`: no gate drop, no tuning bars, 0 entries changed | any of those present |
| H4 | #40420 still sends every rank into tuning when one rank has no file (the reporter's hang case) | `ep2-rankkey-pr40420-partial`: gate drop, both ranks tune, boot OK | a hang or `Autotune group` error |
| H5 | The gate is a TP-group check, not an EP one: TP2 without EP drops too once ranks are in the key | `tp2ep1-rankkey-warm` drops | it reuses |
| H6 | #40420 passes two loadable files that cover different profiles, and FlashInfer 0.7.4's per-profile hit agreement makes both ranks profile the gap together | `ep2-rankkey-pr40420-skew`: no gate drop, boot OK, one op tuned, 1 entry changed per rank | a hang to the 900 s timeout, or an `Autotune group` error |

H6 is the case a reviewer would ask about: on FlashInfer without the hit agreement (the v0.7.2 tag), #40420 would let
one rank profile alone. On 0.7.4 it should not matter, which would also mean the gate could go entirely there.

## Matrix

Model `nvidia/Qwen3-30B-A3B-NVFP4` at `2538ded2` (18.1 GB, ModelOpt NVFP4, 128 experts, so 64 per rank at EP2),
`--moe-runner-backend flashinfer_cutlass` (on sm_120 this calls FlashInfer's `cutlass_fused_moe`; sglang's own sm_120
NVFP4 kernel only takes non-gated relu2 at TP1/EP1), `--max-running-requests 32 --cuda-graph-max-bs-decode 32`, custom
all-reduce off, `NCCL_P2P_DISABLE=1`, HF offline. Image `lmsysorg/sglang:nightly-dev-cu13-20261113-7e31007a`
(main at 7e31007a, FlashInfer 0.7.4).

`patches/fi-rank-keys.diff` puts `tp_rank`, `ep_rank` and `cluster_rank` back into FlashInfer 0.7.4's CUTLASS
`MoERunner` key, as f64b4ba had them. It recreates the per-rank files the sm_120 reports were writing; it is not
a FlashInfer the reporters ran.

A cold variant deletes `$SGLANG_CACHE_DIR/flashinfer/autotune` (Triton, inductor and FlashInfer JIT caches stay); a
warm one starts from what the variant before it left. The skew arm runs last, since a hang there holds the pod for the
900 s boot timeout; the cold controls before it wipe the tree, so it starts from the copy of the cache the partial
arm's boot left (`ep2-rankkey-pr40420-partial/cache_after`).

| # | Variant | Parallel | Code | Cache at start | Bug present | H1 holds |
|---|---|---|---|---|---|---|
| 1 | ep2-cold | TP2 EP2 | stock | none | tunes | tunes |
| 2 | ep2-warm | TP2 EP2 | stock | from 1 | drop, tunes | reuses |
| 3 | ep2-pr40420-warm | TP2 EP2 | #40420 | from 2 | reuses | reuses |
| 4 | ep2-rankkey-cold | TP2 EP2 | FI rank keys | none | tunes | tunes |
| 5 | ep2-rankkey-warm | TP2 EP2 | FI rank keys | from 4 | drop, tunes | drop, tunes |
| 6 | ep2-rankkey-pr40420-warm | TP2 EP2 | FI rank keys + #40420 | from 5 | reuses | reuses |
| 7 | ep2-rankkey-pr40420-partial | TP2 EP2 | FI rank keys + #40420 | from 6, rank 1's file deleted | drop, tunes | drop, tunes |
| 8 | tp2ep1-rankkey-cold | TP2 | FI rank keys | none | tunes | tunes |
| 9 | tp2ep1-rankkey-warm | TP2 | FI rank keys | from 8 | drop, tunes | drop, tunes |
| 10 | tp2ep1-cold | TP2 | stock | none | tunes | tunes |
| 11 | tp2ep1-warm | TP2 | stock | from 10 | drop, tunes | reuses |
| 12 | tp1-cold | TP1 | stock | none | tunes | tunes |
| 13 | tp1-warm | TP1 | stock | from 12 | reuses (no gate) | reuses |
| 14 | ep2-rankkey-pr40420-skew | TP2 EP2 | FI rank keys + #40420 | 7's after its boot, one entry gone from rank 1's file | one op tunes on both ranks | same |

TP2 without EP is only a control under H1: before #6006 its keys carried `tp_rank`, so it dropped too (H5). TP1 is
the one setup the gate never touches.

Per variant, `results/sglang-40320/<variant>/autotune.txt` (from `autotune_report.py`) gives:

- `gate_drops`: `per-rank caches disagree` lines (rank 0 logs one per dropping round).
- `autotune TPn: rounds, seconds`: from `Running FlashInfer autotune` to `FlashInfer autotune completed.` per rank.
- `tuned_ops`: ops with a `[AutoTuner]: Tuning` bar; `none` means no rank profiled anything.
- Per rank file: entries before and after the boot, and `changed`, the entries whose value differs. Every profiled
  entry records its measured time, so a reused cache shows 0 and a re-tuned one shows nearly all.
- Rank 0 vs rank 1 after the boot: keys only one rank has, and one example of each, which shows whether the ranks
  differ by rank fields or by shapes.

Boot seconds come from each verdict's `boot_s`. The first boot also JIT-compiles whatever the image's FlashInfer
cache lacks, so compare autotune seconds within a cold/warm pair, not across the first pair.

## Box and time

2x RTX PRO 6000 Blackwell (Server Edition first, Workstation Edition as fallback), secure cloud, PCIe only
(`require_pcie` defaults on for 2 GPUs), `planned_hours = 1.25`: about 10 min of image pull and download, then 14
boots at 2-3 min each. The variants run in order of value, so a slow host loses the skew arm and then the TP1 and TP2
controls first; a hang in the skew arm (H6 false) costs up to 15 min but comes after everything else.
`harness/pod/pkg_snapshot.sh` restores sglang and flashinfer before every variant; setup.sh points
`PKG_SNAPSHOT_DIR` at container disk, since from the FUSE `/workspace` the sglang copy alone took 81 s per variant
on 10-23 and flashinfer is about twice its file count.
`harness quote sglang-40320` on 2026-11-13 found Server Edition stock in EU-RO-1, US-CO-1 and US-PA-1. US-PA-1 is
where a split-NUMA host hung NCCL init on 10-17; every variant here sets `NCCL_P2P_DISABLE=1` for that reason.

## Results (2026-11-13)

H1 holds: on main with FlashInfer 0.7.4 the two EP2 ranks write byte-identical cache files, and the warm boot reuses
them with no gate drop. The gate drops the cache on every warm boot only once the rank fields are back in FlashInfer's
CUTLASS `MoERunner` key (`fi-rank-keys.diff`), at TP2 with or without EP. #40420 keeps that cache, and with a skewed
pair of files both ranks profile the missing entry together.

Box: RunPod SECURE US-NC-2, pod `sm120-sglang-40320-dfd1f9`, 2x RTX PRO 6000 Blackwell Server Edition (compute 12.0,
driver 595.91.07, 97887 MiB), `topo -m` SYS (GPU0 on NUMA 0, GPU1 on NUMA 1), 128 CPUs, kernel 6.8.0-147. Image:

```
lmsysorg/sglang:nightly-dev-cu13-20261113-7e31007a@sha256:7d2f322fbe8eda16ee65f969f8c9c2be602ecb0a29bf3c8b219945432f3992b2
```

`env.txt`: sglang 0.0.0.dev1+g7e31007a8, flashinfer-python 0.7.4, torch 2.14.1+cu130, Triton 3.8.0, CUDA 13.0.

Run notes:

- The first pod (`sm120-sglang-40320-a8a9f2`) landed on a proxy-only host: RunPod gave no direct SSH endpoint within
  3 min of its container starting, so the harness ended it before setup.
- On the second pod the first pass failed in 5 s per variant (exit code 2). This nightly split `--cuda-graph-max-bs`
  into `--cuda-graph-max-bs-decode` and `--cuda-graph-max-bs-prefill`, so argparse rejects the old flag as ambiguous.
  ep2-cold, ep2-warm and ep2-pr40420-warm failed that way before the run was stopped; their three BOOT_FAIL lines head
  `verdicts.jsonl`. The flag became `--cuda-graph-max-bs-decode 32`, and all 14 variants then ran in order on the same
  pod (09:18 to 09:36 UTC). Every one booted and served its smoke request.

Per variant, from `results/sglang-40320/<variant>/autotune.txt` and `verdicts.jsonl`. Key directories sit under
`flashinfer/autotune/0.7.4/sm120/`. Entries changed is per rank file, rank 0 / rank 1; `-` means no file before the
boot. "All three" is `fp4_gemm`, `trtllm::fused_moe::gemm1` and `trtllm::fused_moe::gemm2`.

| # | Variant | Key dir | Gate drops | Autotune s (TP0 / TP1) | Tuned ops | Entries changed | Rank files after | Boot s |
|---|---|---|---|---|---|---|---|---|
| 1 | ep2-cold | `ea6fc1d5658cbdfb` | 0 | 31 / 31 | all three | - / - | 54 each, identical (`565378999de6`) | 91 |
| 2 | ep2-warm | `ea6fc1d5658cbdfb` | 0 | 2 / 2 | none | 0 / 0 | identical (`565378999de6`) | 51 |
| 3 | ep2-pr40420-warm | `ea6fc1d5658cbdfb` | 0 | 3 / 3 | none | 0 / 0 | identical (`565378999de6`) | 51 |
| 4 | ep2-rankkey-cold | `ea6fc1d5658cbdfb` | 0 | 8 / 8 | all three | - / - | 54 each, 42 rank-only keys each | 57 |
| 5 | ep2-rankkey-warm | `ea6fc1d5658cbdfb` | 1 | 8 / 8 | all three | 51 / 51 | 42 rank-only keys each | 56 |
| 6 | ep2-rankkey-pr40420-warm | `ea6fc1d5658cbdfb` | 0 | 2 / 2 | none | 0 / 0 | 42 rank-only keys each | 52 |
| 7 | ep2-rankkey-pr40420-partial | `ea6fc1d5658cbdfb` | 1 | 8 / 8 | all three | 51 / - (54 written) | 42 rank-only keys each | 56 |
| 8 | tp2ep1-rankkey-cold | `f1c177c251f83273` | 0 | 8 / 8 | all three | - / - | 42 rank-only keys each | 56 |
| 9 | tp2ep1-rankkey-warm | `f1c177c251f83273` | 1 | 8 / 8 | all three | 54 / 54 | 42 rank-only keys each | 56 |
| 10 | tp2ep1-cold | `f1c177c251f83273` | 0 | 8 / 8 | all three | - / - | identical (`e93caa6a287d`) | 56 |
| 11 | tp2ep1-warm | `f1c177c251f83273` | 0 | 2 / 2 | none | 0 / 0 | identical (`e93caa6a287d`) | 51 |
| 12 | tp1-cold | `c16e3f62c9ba8b5b` | 0 | 12 | all three | - | one file, 54 | 56 |
| 13 | tp1-warm | `c16e3f62c9ba8b5b` | 0 | 1 | none | 0 | one file, 54 | 46 |
| 14 | ep2-rankkey-pr40420-skew | `ea6fc1d5658cbdfb` | 0 | 2 / 2 | `fp4_gemm` | 1 / 1 (rank 1: 53 -> 54) | 42 rank-only keys each | 51 |

- Every file holds 54 entries: 12 `fp4_gemm`, 21 `gemm1`, 21 `gemm2`. In the rank-keys arms the 42 MoE keys differ
  between ranks only in the rank fields (`ep_rank` at EP2, `tp_rank` at TP2). With `tp_rank`, `ep_rank` and
  `cluster_rank` taken out of the keys, the two files match key for key and value for value (0 differing entries in
  arms 4, 5, 8 and 14): the ranks profile the same shapes and record the same tactics and times, as the 10-08 H20
  comment found.
- The gate deletes the files before it tunes, so a dropped boot rewrites every entry. At EP2, 51 of 54 changed; the
  other 3 (two `fp4_gemm` M buckets and one `gemm2` profile) were measured at the same time again. At TP2 all 54
  changed.
- On this model the drop is cheap: a re-tune takes 8 s per rank against 1 to 3 s for a reuse, and a warm boot 56 s
  against 51 to 52 s. ep2-cold's 31 s is the pod's first boot, which also JIT-compiles what the image's FlashInfer
  cache lacks; the later cold arms tune in 8 s (TP2) and 12 s (TP1).
- Skew arm: `hooks.sh` removed rank 1's first key in sort order, `('fp4_gemm', 'CutlassFp4GemmRunner', ((1, 1024),
  (1024, 2048), ...)` (the M=1 bucket). Under #40420 the gate passed both files; both ranks profiled that one entry and nothing else
  (rank 0 replaced its copy, rank 1 added it back), in 2 s, with no `Autotune group` error and no hang.
- No arm logged an `Autotune group` error or a traceback.

Hypotheses:

- H1 confirmed (arm 2): 0 drops, no tuning, 0 entries changed, rank files byte-identical. TP2 without EP is the same
  (arm 11).
- H2 confirmed (arm 5): one gate drop, all three ops tuned, 51 of 54 entries re-measured per rank.
- H3 confirmed (arm 6): 0 drops, no tuning, 0 entries changed, though the two files differ in 42 keys each.
- H4 confirmed (arm 7): with rank 1's file gone the gate still drops, both ranks tune, and the boot finishes (56 s).
- H5 confirmed (arm 9): TP2 without EP drops and re-tunes all 54 entries once the ranks are in the key. The gate trips
  on any TP > 1, not on EP.
- H6 confirmed (arm 14): 0 drops, only `fp4_gemm` tuned, 1 entry changed per rank, boot 51 s.

Conclusion:

- Stock main (7e31007a; `flashinfer_autotune.py` unchanged on main through 5792cf6) with FlashInfer 0.7.4 no longer
  reproduces #40320 for `flashinfer_cutlass` on sm_120. The change is in FlashInfer: #6006 took the ranks out of the
  CUTLASS `MoERunner` key. By tag, that key carries `tp_rank`, `ep_rank` and `cluster_rank` in v0.6.18 (the
  reporter's), v0.7.0 and v0.7.1, and none from v0.7.2 on. Main pins 0.7.4; v0.5.23 pins 0.7.2.
- #40420 fixes the pre-#6006 key format: with rank-keyed files it keeps the cache (arm 6) where stock main re-tunes on
  every warm boot (arm 5). On 0.7.4 it is safe under skew (arm 14), since FlashInfer's per-profile hit agreement
  (`_all_ranks_hit_cache`, in the v0.7.3 and v0.7.4 tags, absent through v0.7.2) sends both ranks into the missing
  profile. On a FlashInfer with rank keys and no hit agreement (v0.7.1 and earlier), #40420 passes a skewed pair with
  nothing to hold the ranks together; not tested.
- The reporter's GB10 `flashinfer_mxfp4` setup is untested here. On main, `flashinfer_mxfp4` on any sm_12x GPU
  (`get_platform().is_sm120` maps to `is_sm120_supported`, major 12, so GB10 too) calls the same `cutlass_fused_moe`
  with MXFP8 activations (`mxfp4.py` kernel `cutlass_sm120`, `_fused_experts_flashinfer_mxfp4_cutlass` in
  `moe_runner/flashinfer_cutlass.py`), so it is keyed by this same `MoERunner`. At v0.7.4 those extras hold no ranks
  (`core.py:1008-1038`, with `use_mxfp8_act_scaling` among them), and the reporter's log tunes
  `trtllm::fused_moe::gemm1`, that runner's op. On 0.7.4 a GB10 drop would need ranks that profile different shapes,
  which this run cannot show.

Caveats:

- One rep per arm.
- sm_120 only: one 2-GPU box and one small model (54 entries per file). No sm_121, no TP4 or wider, no multi-node.
- The rank-keys arms are a reconstruction: `fi-rank-keys.diff` on 0.7.4, not a FlashInfer the reporters ran.
  Everything else in them is 0.7.4, including the timing reduce and the hit agreement that v0.7.1 and earlier lack.
- The skew arm removed a dense `fp4_gemm` entry, not a MoE one.
- Runners still keyed on rank-local values (see Root cause) were not run.

## Risks

- H1 may hold, in which case stock main does not show the bug on this path and only the FI-rank-keys arms do. That is
  a finding for the thread (the sm_120 CUTLASS reports predate FlashInfer #6006), but it is not a reproduction of the
  GB10 `flashinfer_mxfp4` report, which needs sm_121 hardware or a runner whose ranks profile different shapes.
- The fi-rank-keys patch is ours. Any comment upstream has to name it as a reconstruction of FlashInfer before #6006,
  next to the stock arms.
- Only rank 0 logs the gate line, and the tqdm bars from both ranks share one log, so `tuned_ops` says whether any
  rank profiled, not which; the per-rank `changed` counts say which.
- A brand-new nightly may fail to boot this model under EP2; ep2-cold is first, so that shows within minutes.
