# Puget Systems Team LLM pack v1.1.0, entry 2, on 2x RTX PRO 6000

Follow-up to `issues/puget-team-llm` (entry 4) and `issues/puget-team-llm-e7` (entry 7). As installed, eager mode cost
those two entries 2.75x and 7.2x single-user output tok/s, and PR #12 (merged 2026-10-23, main `2c83929a`) dropped
`--enforce-eager --no-enable-prefix-caching` for them. The pack (Puget-Systems/puget-docker-app-packs v1.1.0 @
`fed0505e`) ships the same two flags on entries 1, 2, 3, 6 and 8, unchanged at `2c83929a`. This kit covers entry 2,
Qwen 3.6 27B dense AWQ, the only dense model of the five.

The maintainer's approval of PR #12 listed follow-ups before a release with graphs on more entries. Two of them need
measurements, so the next PR (graphs for entries 1, 2, 3, 6, 8) needs these kits to answer them:

- "Eager mode was our hedge against nightly regressions, and vllm#54331 reports a hybrid model crashing under
  sustained load with CUDA graphs on."
- "Both entries are also offered on a single 96 GB GPU and on 4-GPU / Max-Q builds, and CUDA graph capture adds memory
  use and startup time there."

Questions:

1. Does entry 2 serve correctly with CUDA graphs and prefix caching on, and what does eager mode cost it as installed?
2. Does it stay up under 25 minutes of saturated load with graphs on?
3. On a single 96 GB GPU, which the menu also offers it on, what do graphs cost in startup time and memory, and what do
   they buy?

With those measured, the PR can drop the flags for this entry too, or keep them for a stated reason.

## Setup

- 2x RTX PRO 6000 Blackwell Workstation Edition (Server Edition fallback), PCIe only.
- Model: `cyankiwi/Qwen3.6-27B-AWQ-INT4` @ `46d3c4ac` (20.4 GB of safetensors, last modified 2026-10-19).
  `Qwen3_5ForConditionalGeneration`, dense, 64 layers of which 48 are linear attention (Gated DeltaNet);
  compressed-tensors W4A16, checkpoint dtype float16.
- Image: `vllm/vllm-openai:nightly`, the tag the menu writes (`vllm/vllm-openai:${NIGHTLY_PREFIX}`, and
  `scripts/lib/gpu_detect.sh` line 17 sets `NIGHTLY_PREFIX="nightly"`), pinned by digest to the 10-19 build
  `nightly-e68efc68` that entries 4 and 7 ran. Puget's PR #14 (open on 10-23, head `b315df7d`) pins every NVIDIA
  nightly entry to `ghcr.io/puget-systems/vllm-openai:nightly-e68efc68...@sha256:1ff6716f...`, a mirror of this same
  digest, so the kit runs the build PR #14 pins.
- What the pack launches on this box (we add `--revision 46d3c4aca0cfa475e99283abe9fb1254df1eef03`):

  ```
  python3 -m vllm.entrypoints.openai.api_server --model=cyankiwi/Qwen3.6-27B-AWQ-INT4
    --tensor-parallel-size=2 --gpu-memory-utilization=0.90 --max-num-seqs=256 --dtype=float16
    --trust-remote-code --disable-custom-all-reduce --reasoning-parser qwen3
    --enable-auto-tool-choice --tool-call-parser qwen3_coder
    --language-model-only --enforce-eager --no-enable-prefix-caching
  ```

- Where each piece comes from, at `fed0505e`:
  - `packs/team_llm/docker-compose.yml` lines 63-78: the command, run through `bash -c`. `--trust-remote-code` and
    `--disable-custom-all-reduce` are fixed there; `--max-num-seqs=${MAX_NUM_SEQS:-256}` is 256 because neither
    `init.sh` nor `install.sh` writes `MAX_NUM_SEQS`; `${MAX_CONTEXT:+...}` adds nothing when `MAX_CONTEXT` is unset.
  - `scripts/lib/vllm_menu_nvidia.sh` lines 197-208 (entry 2): model, parsers, `--language-model-only` plus the eager
    flags in `EXTRA_ARGS`, `--dtype=float16`, the nightly image. No `VLLM_MAX_CTX`, so no `--max-model-len` and vLLM
    takes the model's own 262144. `install.sh` lines 868-882 would cap the context only with under 30 GB left after
    weights; here 190 x 0.90 - 18 = 153, so it leaves it unset too (`init.sh` has no cap).
  - `scripts/lib/gpu_detect.sh` lines 86-99: nvidia-smi reports 97887 MiB per GPU, so `VRAM_GB=95`, `GPU_COUNT=2`,
    `TOTAL_VRAM=190`; `GPU_COUNT` becomes the tensor parallel size (`vllm_menu_nvidia.sh` line 163 copies it).
  - `vllm_menu_nvidia.sh` lines 566-582: the menu's 18 GB is 9% of 190 GB, under the 70% band, so 0.90.
  - `packs/team_llm/init.sh` lines 121-135 (`install.sh` lines 883-897 write the same keys): `THINKING_ARGS`,
    `VLLM_ENABLE_MTP` and `VLLM_EXTRA_PIP` are empty for this entry, so no chat template kwargs, no speculative
    config, no pip install. `NCCL_P2P_DISABLE=1` goes to `.env`, but neither `docker-compose.yml` nor
    `docker-compose.nvidia.yml` passes it into the container, so NCCL keeps its defaults.
- Each quality variant (shipped, graphs, tp1-shipped, tp1-graphs): boot, smoke test, GSM8K 100 (5-shot chat, greedy),
  then `vllm bench serve` at concurrency 1 / 16 / 64 (random 1024 in / 512 out). The soak variants run the soak
  instead (below).

## One 96 GB GPU

At `2c83929a`, one card reporting 97887 MiB gives `GPU_COUNT=1`, `VRAM_GB=95`, `TOTAL_VRAM=95` (`gpu_detect.sh` lines
86-99; 96 would change nothing below). The menu offers entry 2 there and writes the same flags as on two GPUs except the
tensor parallel size:

- Offered: `show_vllm_model_menu` lists entry 2 from 18 GB (`vllm_menu_nvidia.sh` line 46), and `select_vllm_model`
  refuses it only under 18 (lines 198-201).
- `--tensor-parallel-size=1`: line 163 copies `GPU_COUNT`, and compose line 67 passes it.
- No `--max-model-len`: `install.sh` lines 874-882 leave the context unset with 30 GB or more left after weights, and
  95 x 0.90 - 18 = 67.
- `--gpu-memory-utilization=0.90`: 18 GB is 18% of 95 (lines 569-585), under the 70% band.
- Everything else as on two GPUs, `--disable-custom-all-reduce` included (compose line 73; nothing to disable at TP1).

tp1-shipped and tp1-graphs run exactly that, with the same quality runs as the 2-GPU pair, on GPU 0 of this pod
(`CUDA_VISIBLE_DEVICES=0`), so the pair is compared on one host and GPU 1 sits idle. The 4-GPU and Max-Q builds the
review also names are not covered here.

## Sustained load

Entry 2 is the kind of model vllm#54331 is about: a Gated DeltaNet hybrid of the same Qwen 3.5 family as the report's
crashing checkpoints (those are MoE; this one is dense, with 48 of 64 layers linear attention). soak-graphs boots
graphs' exact configuration (TP2) and, after the smoke test, runs `run_soak` from hooks.sh: issues/vllm-54331's
workload at concurrency 64 (`vllm bench serve --dataset-name prefix_repetition`, 16 shared 4096-token prefixes,
1024-token suffixes, 1024 output tokens with `--ignore-eos`) in rounds of 256 requests until 25 minutes have passed.
The round in progress then finishes, so the soak runs 25 minutes plus at most one round of a few minutes. The shared
prefixes keep prefix caching, and with it the hybrid cache's 'align' path, in use the whole time.

- Duration: 25 minutes, the length issues/vllm-54331 ran; the report's crashes came 2.5-7.75 minutes into saturation,
  so this is three times the latest. That kit's soak of `nvidia/Qwen3.6-35B-A3B-NVFP4` ran 25 minutes clean on this
  same image (1 GPU, text only, concurrency 70), and so did 0.28.0: it did not reproduce the report, so a clean soak
  here is evidence for this entry's configuration, not a sign that vllm#54331 is fixed.
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
harness up puget-team-llm-e2
harness exec puget-team-llm-e2 shipped graphs tp1-shipped tp1-graphs soak-graphs
harness exec puget-team-llm-e2 soak-shipped     # only if soak-graphs is not OK
harness down puget-team-llm-e2
```

soak-shipped takes about 40 minutes here (restore, boot, 25 of soak, one long round of a dense model at eager speed).
planned_hours leaves that when soak-graphs dies early, as the report's crashes did. If less than 40 minutes remain
before the deadline `up` printed, run `harness down` instead, and rerun soak-graphs and soak-shipped together on one new
pod under a new budget.

## Cost

Quoted before the run. `harness quote` on 2026-10-23 15:07 PDT: 2x RTX PRO 6000 Blackwell Workstation Edition,
secure, $5.18/hr, 2.25 h planned, $12.82 worst case (Server Edition fallback at $4.98/hr, $12.33 worst case). The
two-variant kit was quoted at 1.0 h, $5.70; the soak and the 1-GPU pair add the rest.

## Reading the results

- GSM8K is a correctness check, not a score: `harness/pod/gsm8k_chat.py` stops generation at "Question:", and a
  thinking model writes that inside its reasoning, which cut 30-42 of entry 4's 100 replies short. Report it as
  correct / answered (replies with a number), with `empty_content`, `no_number` and `hit_max_tokens` from
  `gsm8k_counts` listed separately, never as one score. Use entry 4's bar: every reply that reached an answer is
  correct in each pair, and the `empty_content` counts are close.
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
  'tp:0'` on entries 4 and 7), and in graphs that prefix caching really turned on: `enable_prefix_caching=True` and
  "Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration", which the linear-attention layers need.
  The benchmark's random prompts share no prefixes, so prefix caching's gain is not in these numbers.
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
- A dense 27B decodes every weight per token, so expect lower tok/s than the 3B-active MoE entries at every
  concurrency, and longer soak rounds; the comparisons that matter are within each pair on this pod.

## Results (2026-10-25)

One pod, 2x RTX PRO 6000 Blackwell Workstation Edition, EU-CZ-1, driver 595.91.07, both GPUs on one NUMA node
(`NODE`), PCIe Gen5 x16 under load, CUDA peer access both ways. `NCCL_P2P_DISABLE` absent from every server process,
as installed; tp1-* ran with `CUDA_VISIBLE_DEVICES=0`. SM clocks held 2.75-2.82 GHz median under load in every variant.
All-reduce ran on PYNCCL; graphs' server.log shows `enable_prefix_caching=True` and the Mamba 'align' line, shipped's
does not. Cost $6.01 of the $12.82 budget.

Output tok/s, random 1024-token prompts, 512 output tokens (mean TPOT in ms in brackets):

| Variant | 1 user | 16 users | 64 users | boot |
|---|---|---|---|---|
| shipped (TP2, eager, no prefix caching) | 32.4 (30.7) | 445.1 (33.1) | 1,294.5 (42.5) | 80 s |
| graphs (TP2) | 112.2 (8.7) | 1,112.6 (12.0) | 1,964.0 (28.0) | 100 s |
| gain | 3.5x | 2.5x | 1.5x | |
| tp1-shipped (1 GPU) | 35.2 (28.1) | 451.7 (30.5) | 1,111.3 (47.9) | 40 s |
| tp1-graphs (1 GPU) | 75.9 (12.8) | 783.0 (16.2) | 1,346.4 (39.8) | 95 s |
| gain | 2.2x | 1.7x | 1.2x | |

- GSM8K: 98 of 99 answered replies correct in all four variants, with the same counts in each (1 reply hit the
  4096-token cap with empty content).
- Sustained load: soak-graphs ran 9 rounds in 26 min at concurrency 64, 2,304/2,304 requests completed, output
  1,545-1,554 tok/s and mean TPOT 36.0-36.6 ms in every round, with no drift. Container anonymous memory was 10.8 GB
  at mid-soak and 9.9 GB at the end. soak-shipped was not needed.
- One GPU, memory: graphs plus prefix caching cost 0.92 GiB of graph pool (0.62 + 0.30) and 4.9% of the KV cache:
  1,010,665 tokens vs 1,062,536 (3.86x vs 4.05x concurrency at 262144). On two GPUs: 2,356,069 vs 2,434,526 tokens
  (-3.2%), 1.3 GiB of graph pool per GPU (0.99 + 0.31).
- One GPU, startup: 95 s vs 40 s to `/health`; engine init 70 s vs 15 s, of which torch.compile 22.5 s on a cold
  cache. soak-graphs booted the 2-GPU configuration a second time on this pod with the cache warm: torch.compile 0.4
  s, engine init 39 s instead of 69 s, 70 s to `/health` instead of 100 s. That is the restart a real install sees.
- The dense model gains least of the five: eager costs it 3.5x at one user on two GPUs, and at 64 users the GPUs are
  closer to the limit than the CPU launch path, so graphs add 1.5x. On one GPU the gain is 2.2x / 1.7x / 1.2x.
