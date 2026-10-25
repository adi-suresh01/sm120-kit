# Puget Systems Team LLM pack v1.1.0, entry 6, on 2x RTX PRO 6000

Follow-up to `issues/puget-team-llm` (entry 4) and `issues/puget-team-llm-e7` (entry 7). As installed, eager mode cost
those two entries 2.75x and 7.2x single-user output tok/s, and PR #12 (merged 2026-10-23, main `2c83929a`) dropped
`--enforce-eager --no-enable-prefix-caching` for them. The pack (Puget-Systems/puget-docker-app-packs v1.1.0 @
`fed0505e`) ships the same two flags on entries 1, 2, 3, 6 and 8, unchanged at `2c83929a`. This kit covers entry 6,
Nemotron 3 Nano 30B-A3B NVFP4, the small sibling of entry 7 and one of the entries the menu offers with no VRAM gate.

The maintainer's approval of PR #12 listed follow-ups before a release with graphs on more entries. Two of them need
measurements, so the next PR (graphs for entries 1, 2, 3, 6, 8) needs these kits to answer them:

- "Eager mode was our hedge against nightly regressions, and vllm#54331 reports a hybrid model crashing under
  sustained load with CUDA graphs on."
- "Both entries are also offered on a single 96 GB GPU and on 4-GPU / Max-Q builds, and CUDA graph capture adds memory
  use and startup time there."

Questions:

1. Does entry 6 serve correctly with CUDA graphs and prefix caching on, and what does eager mode cost it as installed?
2. Does it stay up under 25 minutes of saturated load with graphs on?
3. On a single 96 GB GPU, which the menu also offers it on, what do graphs cost in startup time and memory, and what do
   they buy?

With those measured, the PR can drop the flags for this entry too, or keep them for a stated reason.

## Setup

- 2x RTX PRO 6000 Blackwell Workstation Edition (Server Edition fallback), PCIe only.
- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` @ `6efb4a2a` (19.3 GB of safetensors, last modified
  2026-08-24). `NemotronHForCausalLM`: 23 Mamba-2, 23 MoE and 6 attention layers; ModelOpt NVFP4 with an FP8 KV cache
  in `hf_quant_config.json` (on entry 7's checkpoint of the same kind, vLLM picked `fp8_e4m3` KV from it with no flag).
- Image: `vllm/vllm-openai:nightly`, the tag the menu writes (`vllm/vllm-openai:${NIGHTLY_PREFIX}`, and
  `scripts/lib/gpu_detect.sh` line 17 sets `NIGHTLY_PREFIX="nightly"`), pinned by digest to the 10-19 build
  `nightly-e68efc68` that entries 4 and 7 ran. Puget's PR #14 (open on 10-23, head `b315df7d`) pins every NVIDIA
  nightly entry to `ghcr.io/puget-systems/vllm-openai:nightly-e68efc68...@sha256:1ff6716f...`, a mirror of this same
  digest, so the kit runs the build PR #14 pins.
- What the pack launches on this box (we add `--revision 6efb4a2a1c1fa277ce7b3df7a1416255011b1c99`):

  ```
  python3 -m vllm.entrypoints.openai.api_server --model=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
    --tensor-parallel-size=2 --gpu-memory-utilization=0.90 --max-num-seqs=256 --dtype=auto
    --trust-remote-code --disable-custom-all-reduce --reasoning-parser qwen3
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --enforce-eager --no-enable-prefix-caching
  ```

- Where each piece comes from, at `fed0505e`:
  - `packs/team_llm/docker-compose.yml` lines 63-78: the command, run through `bash -c`. `--trust-remote-code` and
    `--disable-custom-all-reduce` are fixed there; `--max-num-seqs=${MAX_NUM_SEQS:-256}` is 256 because neither
    `init.sh` nor `install.sh` writes `MAX_NUM_SEQS`; `${MAX_CONTEXT:+...}` adds nothing when `MAX_CONTEXT` is unset.
  - `scripts/lib/vllm_menu_nvidia.sh` lines 238-244 (entry 6): model, parsers, the eager flags in `EXTRA_ARGS`, the
    nightly image. It sets no dtype, so the default `auto` from line 172 (the checkpoint's bfloat16), and no
    `VLLM_MAX_CTX`, so no `--max-model-len` and vLLM takes the model's own 262144. `install.sh` lines 868-882 would
    cap the context only with under 30 GB left after weights; here 190 x 0.90 - 25 = 146, so it leaves it unset too
    (`init.sh` has no cap).
  - `scripts/lib/gpu_detect.sh` lines 86-99: nvidia-smi reports 97887 MiB per GPU, so `VRAM_GB=95`, `GPU_COUNT=2`,
    `TOTAL_VRAM=190`; `GPU_COUNT` becomes the tensor parallel size (`vllm_menu_nvidia.sh` line 163 copies it).
  - `vllm_menu_nvidia.sh` lines 566-582: the menu's 25 GB is 13% of 190 GB, under the 70% band, so 0.90.
  - `packs/team_llm/init.sh` lines 121-135 (`install.sh` lines 883-897 write the same keys): `THINKING_ARGS`,
    `VLLM_ENABLE_MTP` and `VLLM_EXTRA_PIP` are empty for this entry, so no chat template kwargs, no speculative
    config, no pip install. `NCCL_P2P_DISABLE=1` goes to `.env`, but neither `docker-compose.yml` nor
    `docker-compose.nvidia.yml` passes it into the container, so NCCL keeps its defaults.
- Each quality variant (shipped, graphs, tp1-shipped, tp1-graphs): boot, smoke test, GSM8K 100 (5-shot chat, greedy),
  then `vllm bench serve` at concurrency 1 / 16 / 64 (random 1024 in / 512 out). The soak variants run the soak
  instead (below).

Observation for the PR, not changed here: the pack ships `--reasoning-parser qwen3` and `--tool-call-parser
qwen3_coder` for this entry (`vllm_menu_nvidia.sh` lines 240-241 at `fed0505e`, the same pair as entry 7). The model
card at `6efb4a2a` serves it with the same tool parser but its own `nano_v3` reasoning parser plugin
(`nano_v3_reasoning_parser.py`) and `--kv-cache-dtype fp8`; the image also registers a built-in `nemotron_v3`
reasoning parser. The kit runs the pack's `qwen3` exactly, since "as shipped" is the baseline and the question is
eager mode, not the parser; the KV cache dtype needs no flag, as noted above.

## One 96 GB GPU

At `2c83929a`, one card reporting 97887 MiB gives `GPU_COUNT=1`, `VRAM_GB=95`, `TOTAL_VRAM=95` (`gpu_detect.sh` lines
86-99; 96 would change nothing below). The menu offers entry 6 there and writes the same flags as on two GPUs except the
tensor parallel size:

- Offered: `show_vllm_model_menu` lists entry 6 with no VRAM gate (`vllm_menu_nvidia.sh` line 70), and its branch in
  `select_vllm_model` (lines 240-246 at `2c83929a`) has none either.
- `--tensor-parallel-size=1`: line 163 copies `GPU_COUNT`, and compose line 67 passes it.
- No `--max-model-len`: `install.sh` lines 874-882 leave the context unset with 30 GB or more left after weights, and
  95 x 0.90 - 25 = 60.
- `--gpu-memory-utilization=0.90`: 25 GB is 26% of 95 (lines 569-585), under the 70% band.
- Everything else as on two GPUs, `--disable-custom-all-reduce` included (compose line 73; nothing to disable at TP1).

tp1-shipped and tp1-graphs run exactly that, with the same quality runs as the 2-GPU pair, on GPU 0 of this pod
(`CUDA_VISIBLE_DEVICES=0`), so the pair is compared on one host and GPU 1 sits idle. The 4-GPU and Max-Q builds the
review also names are not covered here.

## Sustained load

Entry 6 is a hybrid with recurrent state, as vllm#54331's models are, though its linear layers are Mamba-2 rather than
the report's Gated DeltaNet. Its attention may also match the report's crashing runs: issues/vllm-54331's follow-up
found those ran FP8 KV, which on sm_120 moves attention to FlashInfer with xqa decode, and this checkpoint asks for FP8
KV, so its 6 attention layers may take the same path (the server log says).

soak-graphs boots graphs' exact configuration (TP2) and, after the smoke test, runs `run_soak` from hooks.sh:
issues/vllm-54331's workload at concurrency 64 (`vllm bench serve --dataset-name prefix_repetition`, 16 shared
4096-token prefixes, 1024-token suffixes, 1024 output tokens with `--ignore-eos`) in rounds of 256 requests until 25
minutes have passed. The round in progress then finishes, so the soak runs 25 minutes plus at most one round of a few
minutes. The shared prefixes keep prefix caching, and with it the hybrid cache's 'align' path, in use the whole time.

- Duration: 25 minutes, the length issues/vllm-54331 ran; the report's crashes came 2.5-7.75 minutes into saturation,
  so this is three times the latest. That kit's soaks (a Qwen 3.5-family GDN hybrid, 1 GPU) ran clean on this image
  and on 0.28.0, so they did not reproduce the report; a clean soak here is evidence for this entry's configuration,
  not a sign that vllm#54331 is fixed.
- Failure: any request a round does not complete. The soak stops after that round, keeps the server log's last 300
  lines in `soak-graphs/soak_failure_tail.txt`, and the verdict is POST_FAIL. The harness's watchdog also stops it if
  the server exits, goes 15 minutes without a log line, or logs a `fail_pattern` line, which now includes vllm#54331's
  `Segfault encountered`, `EngineDeadError` and `EngineCore encountered a fatal error`; `post_stop` then keeps the same
  tail, which ends with the harness's own shutdown of the server.
- soak-shipped is the control: the shipped flags under the same soak, run only if soak-graphs fails (see Run). If it
  survives where soak-graphs died, eager mode is a real hedge for this entry.
- Not soaked: the TP1 configuration.

## Matrix

| Variant | GPUs | Flags vs shipped | Workload | Expected |
|---|---|---|---|---|
| shipped | 2 (TP2) | as shipped (NCCL defaults) | GSM8K, bench | serves; eager baseline |
| graphs | 2 (TP2) | CUDA graphs and prefix caching on; all-reduce settings as shipped | GSM8K, bench | serves; every answered GSM8K reply correct, as in shipped; higher tok/s at 1 and 16 users |
| tp1-shipped | 1 (TP1) | as the menu writes it for one 96 GB GPU | GSM8K, bench | serves; 1-GPU eager baseline |
| tp1-graphs | 1 (TP1) | tp1-shipped with graphs and prefix caching on | GSM8K, bench | serves; longer boot and fewer KV tokens than tp1-shipped, both recorded; higher tok/s at 1 and 16 users |
| soak-graphs | 2 (TP2) | as graphs | 25 min soak at concurrency 64 | every request completes, engine stays up |
| soak-shipped | 2 (TP2) | as shipped | the same soak | opt-in control, only if soak-graphs fails |

Custom all-reduce is left out: the PR keeps the pack's all-reduce settings.

## Run

Every comparison runs on this one pod, the control included, and `harness run` with no names would run soak-shipped
as well, so name the default list:

```
harness up puget-team-llm-e6
harness exec puget-team-llm-e6 shipped graphs tp1-shipped tp1-graphs soak-graphs
harness exec puget-team-llm-e6 soak-shipped     # only if soak-graphs is not OK
harness down puget-team-llm-e6
```

soak-shipped takes about 35 minutes (restore, boot, 25 of soak, one round). planned_hours leaves that when soak-graphs
dies early, as the report's crashes did. If less than 35 minutes remain before the deadline `up` printed, run `harness
down` instead, and rerun soak-graphs and soak-shipped together on one new pod under a new budget.

## Cost

Quoted before the run. `harness quote` on 2026-10-23 15:07 PDT: 2x RTX PRO 6000 Blackwell Workstation Edition,
secure, $5.18/hr, 2.0 h planned, $11.40 worst case (Server Edition fallback at $4.98/hr, $10.96 worst case). The
two-variant kit was quoted at 1.0 h, $5.70; the soak and the 1-GPU pair add the rest.

## Reading the results

- GSM8K is a correctness check, not a score: `harness/pod/gsm8k_chat.py` stops generation at "Question:", and a
  thinking model can write that inside its reasoning. Nemotron 3 reasons by default (the chat template opens
  `<think>` in the prompt); on entry 7, with the same `qwen3` parser, every reply reached an answer. Report it as
  correct / answered (replies with a number), with `empty_content`, `no_number` and `hit_max_tokens` from
  `gsm8k_counts` listed separately, never as one score. Use entry 4's bar: every reply that reached an answer is
  correct in each pair, and the `empty_content` counts are close. A high `empty_content` in every variant points at
  the parser, not at eager mode.
- Compare only within the pod: only the ratios from this one pod mean anything. Entries 4 and 7 ran on Workstation
  Edition pods; if this run lands on the Server Edition fallback (a different board: setup.log has its power limit and
  clocks), its absolute tok/s, TPOT and TTFT are not comparable to theirs, and even two 10-23 Workstation hosts
  differed by 2x in everything that leans on all-reduce. env.txt names the edition; setup.log has the data center,
  PCIe link maximum, power limit and idle clocks; `<variant>/gpu_clocks.csv` samples clocks, power, memory used, PCIe
  link and clock-event reasons for both GPUs every 5 s under load (hooks.sh), so a slow or throttled host can be named.
- `<variant>/server_env.txt` lists the NCCL, VLLM and CUDA variables each server process started with.
  `NCCL_P2P_DISABLE` must be absent in every variant, as on an NVIDIA install, and tp1-* must show
  `CUDA_VISIBLE_DEVICES=0`; in their gpu_clocks.csv, GPU 1's memory used stays near zero.
- In each server.log, check which all-reduce backend ran (`Using ['PYNCCL'] all-reduce backends ... for group
  'tp:0'` on entries 4 and 7), the KV cache dtype and attention backend (`fp8_e4m3`, and whether decode is xqa, as in
  vllm#54331's crashing runs), and in graphs that prefix caching really turned on: `enable_prefix_caching=True` and
  "Mamba cache mode is set to 'align' for NemotronHForCausalLM", which the Mamba layers need. The benchmark's random
  prompts share no prefixes, so prefix caching's gain is not in these numbers.
- One GPU, startup: `boot_s` in each verdict is launch to `/health` answering; `<variant>/boot_summary.txt` splits it
  into weight loading, torch.compile, graph capture and engine init. A real install compiles once and loads the cache
  on later starts; the compile lines say which happened here.
- One GPU, memory: vLLM profiles CUDA graph memory and takes it out of the KV cache at a fixed
  `--gpu-memory-utilization`, so graphs do not raise memory use; they shrink the KV cache. Report "Available KV cache
  memory" and "GPU KV cache size" (tokens, and the maximum concurrency at 262144) for tp1-shipped and tp1-graphs, and
  the graph pool from "Graph capturing finished ... took" and "CUDA graph pool memory". On this hybrid model prefix
  caching changes the cache layout too, so the KV difference is graphs plus prefix caching, and the pool line is
  graphs alone. The same lines for shipped and graphs give the 2-GPU figures.
- Soak: `soak-graphs/soak.txt` has one line per round (requests completed, output tok/s, mean TPOT) and the totals;
  per-round tok/s and TPOT that drift across 25 minutes would point at a slow degradation even with no failure.
  `memory.jsonl` tracks container memory. A POST_FAIL has `soak_failure_tail.txt` and the verdict's `fault`.
- If shipped fails to boot, record the error as the finding and stop; like entry 7's, this checkpoint ships its own
  config code (`configuration_nemotron_h.py`), written for transformers 4.53 against the image's 5.19.

## Results (2026-10-25)

One pod, 2x RTX PRO 6000 Blackwell Workstation Edition, EU-CZ-1, driver 595.91.07, both GPUs on one NUMA node
(`NODE`), PCIe Gen5 x16 under load, CUDA peer access both ways. `NCCL_P2P_DISABLE` absent from every server process,
as installed; tp1-* ran with `CUDA_VISIBLE_DEVICES=0`. SM clocks held 2.77-2.82 GHz median under load in every variant.
All-reduce ran on PYNCCL; graphs' server.log shows `enable_prefix_caching=True` and "Mamba cache mode is set to
'align' for NemotronHForCausalLM", shipped's does not. The checkpoint's own config code loaded on the image's
transformers in every variant. Cost $4.44 of the $11.40 budget.

Output tok/s, random 1024-token prompts, 512 output tokens (mean TPOT in ms in brackets):

| Variant | 1 user | 16 users | 64 users | boot |
|---|---|---|---|---|
| shipped (TP2, eager, no prefix caching) | 35.6 (28.0) | 556.3 (28.4) | 2,108.0 (29.5) | 106 s |
| graphs (TP2) | 324.3 (3.0) | 2,492.1 (6.0) | 5,437.9 (10.9) | 106 s |
| gain | 9.1x | 4.5x | 2.6x | |
| tp1-shipped (1 GPU) | 38.1 (26.2) | 595.1 (26.6) | 2,141.0 (29.1) | 45 s |
| tp1-graphs (1 GPU) | 306.3 (3.2) | 1,912.4 (8.0) | 3,886.1 (15.9) | 90 s |
| gain | 8.0x | 3.2x | 1.8x | |

- GSM8K: 100, 99, 99 and 98 of 100 correct (shipped, graphs, tp1-shipped, tp1-graphs), every reply answered, none
  cut at the token cap.
- Sustained load: soak-graphs ran 22 rounds in 25 min at concurrency 64, 5,632/5,632 requests completed, output
  4,355-4,417 tok/s and mean TPOT 13.5-13.8 ms in every round, with no drift. Container anonymous memory was 9.4 GB at
  mid-soak and 8.5 GB at the end. soak-shipped was not needed. This is a Mamba2 hybrid with prefix caching's 'align'
  path in use for the whole soak, the model class vllm#54331 reports.
- One GPU, memory: graphs plus prefix caching cost 1.80 GiB of graph pool (1.03 + 0.77) and 12.7% of the KV cache in
  tokens: 18,814,948 vs 21,542,759 (71.8x vs 82.2x concurrency at 262144). Available KV memory fell only 1.5 GiB
  (64.32 vs 65.79 GiB), so most of the token drop comes with prefix caching's cache layout for the Mamba layers, not
  with graph memory. On two GPUs: 42,750,443 vs 48,887,899 tokens (-12.6%), 1.84 GiB of graph pool per GPU
  (1.31 + 0.53). Either way the cache still holds over 70 full-length contexts per GPU.
- One GPU, startup: 90 s vs 45 s to `/health`; engine init 63 s vs 21 s, of which torch.compile 8.0 s on a cold cache.
  soak-graphs booted the 2-GPU configuration a second time on this pod with the caches warm (torch.compile 0.2 s, 126
  FlashInfer autotune configs loaded from cache): engine init 19 s instead of 75 s, 50 s to `/health` instead of 106 s.
- Largest gain of the five entries: 9.1x at one user on two GPUs, 8.0x on one.
