# Puget Systems Team LLM pack v1.1.0, entry 3, on 2x RTX PRO 6000

Follow-up to `issues/puget-team-llm` (entry 4) and `issues/puget-team-llm-e7` (entry 7). As installed, eager mode cost
those two entries 2.75x and 7.2x single-user output tok/s, and PR #12 (merged 2026-10-23, main `2c83929a`) dropped
`--enforce-eager --no-enable-prefix-caching` for them. The pack (Puget-Systems/puget-docker-app-packs v1.1.0 @
`fed0505e`) ships the same two flags on entries 1, 2, 3, 6 and 8, unchanged at `2c83929a`. This kit covers entry 3,
Qwen 3.5 35B-A3B AWQ: the same family, quantizer and flags as entry 4's 122B, at a third of the size.

The maintainer's approval of PR #12 listed follow-ups before a release with graphs on more entries. Two of them need
measurements, so the next PR (graphs for entries 1, 2, 3, 6, 8) needs these kits to answer them:

- "Eager mode was our hedge against nightly regressions, and vllm#54331 reports a hybrid model crashing under
  sustained load with CUDA graphs on."
- "Both entries are also offered on a single 96 GB GPU and on 4-GPU / Max-Q builds, and CUDA graph capture adds memory
  use and startup time there."

Questions:

1. Does entry 3 serve correctly with CUDA graphs and prefix caching on, and what does eager mode cost it as installed?
2. Does it stay up under 25 minutes of saturated load with graphs on?
3. On a single 96 GB GPU, which the menu also offers it on, what do graphs cost in startup time and memory, and what do
   they buy?

With those measured, the PR can drop the flags for this entry too, or keep them for a stated reason.

## Setup

- 2x RTX PRO 6000 Blackwell Workstation Edition (Server Edition fallback), PCIe only.
- Model: `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` @ `96bde6d5` (24.5 GB of safetensors, last modified 2026-10-19).
  `Qwen3_5MoeForConditionalGeneration`, 40 layers of which 30 are linear attention (Gated DeltaNet), 256 experts with
  8 active; compressed-tensors W4A16.
- Image: `vllm/vllm-openai:nightly`, the tag the menu writes (`vllm/vllm-openai:${NIGHTLY_PREFIX}`, and
  `scripts/lib/gpu_detect.sh` line 17 sets `NIGHTLY_PREFIX="nightly"`), pinned by digest to the 10-19 build
  `nightly-e68efc68` that entries 4 and 7 ran. Puget's PR #14 (open on 10-23, head `b315df7d`) pins every NVIDIA
  nightly entry to `ghcr.io/puget-systems/vllm-openai:nightly-e68efc68...@sha256:1ff6716f...`, a mirror of this same
  digest, so the kit runs the build PR #14 pins.
- What the pack launches on this box (we add `--revision 96bde6d5fb87e572cd5e9843fda6af22811ee23f`):

  ```
  python3 -m vllm.entrypoints.openai.api_server --model=cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit
    --tensor-parallel-size=2 --gpu-memory-utilization=0.90 --max-num-seqs=256 --dtype=float16
    --trust-remote-code --disable-custom-all-reduce --reasoning-parser qwen3
    --enable-auto-tool-choice --tool-call-parser qwen3_coder
    --language-model-only --enforce-eager --no-enable-prefix-caching
  ```

- Where each piece comes from, at `fed0505e`:
  - `packs/team_llm/docker-compose.yml` lines 63-78: the command, run through `bash -c`. `--trust-remote-code` and
    `--disable-custom-all-reduce` are fixed there; `--max-num-seqs=${MAX_NUM_SEQS:-256}` is 256 because neither
    `init.sh` nor `install.sh` writes `MAX_NUM_SEQS`; `${MAX_CONTEXT:+...}` adds nothing when `MAX_CONTEXT` is unset.
  - `scripts/lib/vllm_menu_nvidia.sh` lines 209-216 (entry 3): model, parsers, `--language-model-only` plus the eager
    flags in `EXTRA_ARGS`, `--dtype=float16`, the nightly image. No `VLLM_MAX_CTX` (entry 4 sets 131072), so no
    `--max-model-len` and vLLM takes the model's own 262144. `install.sh` lines 868-882 would cap the context only
    with under 30 GB left after weights; here 190 x 0.90 - 22 = 149, so it leaves it unset too (`init.sh` has no cap).
  - `scripts/lib/gpu_detect.sh` lines 86-99: nvidia-smi reports 97887 MiB per GPU, so `VRAM_GB=95`, `GPU_COUNT=2`,
    `TOTAL_VRAM=190`; `GPU_COUNT` becomes the tensor parallel size (`vllm_menu_nvidia.sh` line 163 copies it).
  - `vllm_menu_nvidia.sh` lines 566-582: the menu's 22 GB is 11% of 190 GB, under the 70% band, so 0.90.
  - `packs/team_llm/init.sh` lines 121-135 (`install.sh` lines 883-897 write the same keys): `THINKING_ARGS`,
    `VLLM_ENABLE_MTP` and `VLLM_EXTRA_PIP` are empty for this entry, so no chat template kwargs, no speculative
    config, no pip install. `NCCL_P2P_DISABLE=1` goes to `.env`, but neither `docker-compose.yml` nor
    `docker-compose.nvidia.yml` passes it into the container, so NCCL keeps its defaults.
- Each quality variant (shipped, graphs, tp1-shipped, tp1-graphs): boot, smoke test, GSM8K 100 (5-shot chat, greedy),
  then `vllm bench serve` at concurrency 1 / 16 / 64 (random 1024 in / 512 out). The soak variants run the soak
  instead (below).

## One 96 GB GPU

At `2c83929a`, one card reporting 97887 MiB gives `GPU_COUNT=1`, `VRAM_GB=95`, `TOTAL_VRAM=95` (`gpu_detect.sh` lines
86-99; 96 would change nothing below). The menu offers entry 3 there and writes the same flags as on two GPUs except the
tensor parallel size:

- Offered: `show_vllm_model_menu` lists entry 3 from 22 GB (`vllm_menu_nvidia.sh` line 52), and its branch in
  `select_vllm_model` (lines 209-216) has no gate at all.
- `--tensor-parallel-size=1`: line 163 copies `GPU_COUNT`, and compose line 67 passes it.
- No `--max-model-len`: `install.sh` lines 874-882 leave the context unset with 30 GB or more left after weights, and
  95 x 0.90 - 22 = 63.
- `--gpu-memory-utilization=0.90`: 22 GB is 23% of 95 (lines 569-585), under the 70% band.
- Everything else as on two GPUs, `--disable-custom-all-reduce` included (compose line 73; nothing to disable at TP1).

tp1-shipped and tp1-graphs run exactly that, with the same quality runs as the 2-GPU pair, on GPU 0 of this pod
(`CUDA_VISIBLE_DEVICES=0`), so the pair is compared on one host and GPU 1 sits idle. The 4-GPU and Max-Q builds the
review also names are not covered here.

## Sustained load

Entry 3 is the kind of model vllm#54331 is about: the report's crashing checkpoints are this architecture
(`Qwen3_5MoeForConditionalGeneration`, 30 of 40 layers Gated DeltaNet, 3B active), quantized differently. soak-graphs
boots graphs' exact configuration (TP2) and, after the smoke test, runs `run_soak` from hooks.sh: issues/vllm-54331's
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
- Not soaked: the TP1 configuration. vllm-54331's clean 25 minutes were at TP1, on the same architecture in NVFP4.

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
harness up puget-team-llm-e3
harness exec puget-team-llm-e3 shipped graphs tp1-shipped tp1-graphs soak-graphs
harness exec puget-team-llm-e3 soak-shipped     # only if soak-graphs is not OK
harness down puget-team-llm-e3
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
  thinking model writes that inside its reasoning, which cut 30-42 of entry 4's 100 replies short. This is entry 4's
  model family, so expect the same. Report it as correct / answered (replies with a number), with `empty_content`,
  `no_number` and `hit_max_tokens` from `gsm8k_counts` listed separately, never as one score. Use entry 4's bar:
  every reply that reached an answer is correct in each pair, and the `empty_content` counts are close.
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
  "Mamba cache mode is set to 'align' for Qwen3_5MoeForConditionalGeneration", which the linear-attention layers need.
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
