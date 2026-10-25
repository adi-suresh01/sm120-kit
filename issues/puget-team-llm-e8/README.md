# Puget Systems Team LLM pack v1.1.0, entry 8, on 2x RTX PRO 6000

Follow-up to `issues/puget-team-llm` (entry 4) and `issues/puget-team-llm-e7` (entry 7). As installed, eager mode cost
those two entries 2.75x and 7.2x single-user output tok/s, and PR #12 (merged 2026-10-23, main `2c83929a`) dropped
`--enforce-eager --no-enable-prefix-caching` for them. The pack (Puget-Systems/puget-docker-app-packs v1.1.0 @
`fed0505e`) ships the same two flags on entries 1, 2, 3, 6 and 8, unchanged at `2c83929a`. This kit covers entry 8,
Gemma 4 26B-A4B AWQ, the only non-Qwen, non-Nemotron model of the five.

The maintainer's approval of PR #12 listed follow-ups before a release with graphs on more entries. Two of them need
measurements, so the next PR (graphs for entries 1, 2, 3, 6, 8) needs these kits to answer them:

- "Eager mode was our hedge against nightly regressions, and vllm#54331 reports a hybrid model crashing under
  sustained load with CUDA graphs on."
- "Both entries are also offered on a single 96 GB GPU and on 4-GPU / Max-Q builds, and CUDA graph capture adds memory
  use and startup time there."

Questions:

1. Does entry 8 serve correctly with CUDA graphs and prefix caching on, and what does eager mode cost it as installed?
2. Does it stay up under 10 minutes of saturated load with graphs on?
3. On a single 96 GB GPU, which the menu also offers it on, what do graphs cost in startup time and memory, and what do
   they buy?

With those measured, the PR can drop the flags for this entry too, or keep them for a stated reason.

## Setup

- 2x RTX PRO 6000 Blackwell Workstation Edition (Server Edition fallback), PCIe only.
- Model: `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` @ `5c5db8a7` (17.2 GB of safetensors, last modified 2026-10-19).
  `Gemma4ForConditionalGeneration` with a vision tower; 30 layers, 25 sliding-window (1024) and 5 full attention,
  128 experts; compressed-tensors W4A16, checkpoint dtype bfloat16.
- Image: `vllm/vllm-openai:nightly`, the tag the menu writes (`vllm/vllm-openai:${NIGHTLY_PREFIX}`, and
  `scripts/lib/gpu_detect.sh` line 17 sets `NIGHTLY_PREFIX="nightly"`), pinned by digest to the 10-19 build
  `nightly-e68efc68` that entries 4 and 7 ran. Puget's PR #14 (open on 10-23, head `b315df7d`) pins every NVIDIA
  nightly entry to `ghcr.io/puget-systems/vllm-openai:nightly-e68efc68...@sha256:1ff6716f...`, a mirror of this same
  digest, so the kit runs the build PR #14 pins.
- What the pack launches on this box (we add `--revision 5c5db8a7aeabb70b27923ef56536aa5343ecbb5b`):

  ```
  python3 -m vllm.entrypoints.openai.api_server --model=cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit
    --tensor-parallel-size=2 --gpu-memory-utilization=0.90 --max-num-seqs=256 --dtype=float16
    --trust-remote-code --disable-custom-all-reduce --reasoning-parser gemma4
    --enable-auto-tool-choice --tool-call-parser gemma4 --enforce-eager --no-enable-prefix-caching
  ```

- Where each piece comes from, at `fed0505e`:
  - `packs/team_llm/docker-compose.yml` lines 63-78: the command, run through `bash -c`. `--trust-remote-code` and
    `--disable-custom-all-reduce` are fixed there; `--max-num-seqs=${MAX_NUM_SEQS:-256}` is 256 because neither
    `init.sh` nor `install.sh` writes `MAX_NUM_SEQS`; `${MAX_CONTEXT:+...}` adds nothing when `MAX_CONTEXT` is unset.
  - `scripts/lib/vllm_menu_nvidia.sh` lines 256-267 (entry 8): model, `gemma4` reasoning and tool parsers, the eager
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

Two observations for the PR, recorded but not changed here:

- `packs/team_llm/README.md` lines 147-152 say Gemma 4 needs a custom image: run `docker compose build inference`,
  which bakes transformers `>=5.5.4,<5.6.0` in from `Dockerfile.gemma4`. Nothing on the NVIDIA install path builds
  it. `inference` has no `build:` block (the compose comment at lines 10-14 forbids one), so that command builds
  nothing for it; `install.sh` runs `smart_build` (`docker compose build`, which builds only the `brain` service) and
  builds one Dockerfile by hand, `Dockerfile.xpu`, for Intel (line 911); no script names `Dockerfile.gemma4`. The menu
  writes `VLLM_IMAGE=vllm/vllm-openai:nightly` for this entry (`vllm_menu_nvidia.sh` line 266), so an NVIDIA install
  runs the stock nightly unmodified, and that is what this kit runs. Built by hand as its own header says (`docker
  build -f Dockerfile.gemma4`, then `VLLM_IMAGE=puget-vllm-gemma4:latest`), it would be a different stack, `FROM
  vllm/vllm-openai:v0.20.2` by default with transformers moved to 5.5.x. Its one purpose is a transformers new enough
  for the `gemma4` architecture (added in 5.5.0, per its own comment); the pinned nightly already ships 5.19.0 (env.txt
  on entry 7's pod, same digest; this run's env.txt records it again).
- `--dtype=float16` (`vllm_menu_nvidia.sh` line 265) on a checkpoint whose `config.json` says bfloat16, top level and
  `text_config`. vLLM refuses float16 for Gemma 2 and 3 ("Numerical instability", `_FLOAT16_NOT_SUPPORTED_MODELS` in
  `vllm/config/model.py` line 2384 at the image's commit), but `gemma4` is not on that list, so it boots. The kit keeps
  float16 because it ships; every variant runs it, so this matrix does not say what float16 costs Gemma 4.

## One 96 GB GPU

At `2c83929a`, one card reporting 97887 MiB gives `GPU_COUNT=1`, `VRAM_GB=95`, `TOTAL_VRAM=95` (`gpu_detect.sh` lines
86-99; 96 would change nothing below). The menu offers entry 8 there and writes the same flags as on two GPUs except the
tensor parallel size:

- Offered: `show_vllm_model_menu` lists entry 8 from 20 GB (`vllm_menu_nvidia.sh` line 77), and `select_vllm_model`
  refuses it only under 20 (lines 260-263 at `2c83929a`).
- `--tensor-parallel-size=1`: line 163 copies `GPU_COUNT`, and compose line 67 passes it.
- No `--max-model-len`: `install.sh` lines 874-882 leave the context unset with 30 GB or more left after weights, and
  95 x 0.90 - 18 = 67.
- `--gpu-memory-utilization=0.90`: 18 GB is 18% of 95 (lines 569-585), under the 70% band.
- Everything else as on two GPUs, `--disable-custom-all-reduce` included (compose line 73; nothing to disable at TP1).

tp1-shipped and tp1-graphs run exactly that, with the same quality runs as the 2-GPU pair, on GPU 0 of this pod
(`CUDA_VISIBLE_DEVICES=0`), so the pair is compared on one host and GPU 1 sits idle. The 4-GPU and Max-Q builds the
review also names are not covered here.

## Sustained load

soak-graphs boots graphs' exact configuration (TP2) and, after the smoke test, runs `run_soak` from hooks.sh:
issues/vllm-54331's workload at concurrency 64 (`vllm bench serve --dataset-name prefix_repetition`, 16 shared
4096-token prefixes, 1024-token suffixes, 1024 output tokens with `--ignore-eos`) in rounds of 256 requests until 10
minutes have passed. The round in progress then finishes, so the soak runs 10 minutes plus at most one round of a few
minutes. The shared prefixes keep prefix caching in use the whole time.

- Duration: 10 minutes, not the 25 of the hybrid entries. vllm#54331 is about hybrid models with linear-attention
  state; this model has none (sliding-window and full attention only), and the report itself lists Gemma 4 26B as
  clean under the load that crashed its hybrids. The review's concern is broader, eager mode as a hedge against any
  nightly regression, so the soak stays, sized to cover the window in which every reported crash came (2.5-7.75
  minutes into saturation). It saves about 15 minutes on this pod.
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
| graphs | 2 (TP2) | CUDA graphs and prefix caching on; all-reduce settings as shipped | GSM8K, bench | serves; GSM8K within noise of shipped; higher tok/s at 1 and 16 users |
| tp1-shipped | 1 (TP1) | as the menu writes it for one 96 GB GPU | GSM8K, bench | serves; 1-GPU eager baseline |
| tp1-graphs | 1 (TP1) | tp1-shipped with graphs and prefix caching on | GSM8K, bench | serves; longer boot and fewer KV tokens than tp1-shipped, both recorded; higher tok/s at 1 and 16 users |
| soak-graphs | 2 (TP2) | as graphs | 10 min soak at concurrency 64 | every request completes, engine stays up |
| soak-shipped | 2 (TP2) | as shipped | the same soak | opt-in control, only if soak-graphs fails |

Custom all-reduce is left out: the PR keeps the pack's all-reduce settings.

## Run

Every comparison runs on this one pod, the control included, and `harness run` with no names would run soak-shipped
as well, so name the default list:

```
harness up puget-team-llm-e8
harness exec puget-team-llm-e8 shipped graphs tp1-shipped tp1-graphs soak-graphs
harness exec puget-team-llm-e8 soak-shipped     # only if soak-graphs is not OK
harness down puget-team-llm-e8
```

soak-shipped takes about 20 minutes (restore, boot, 10 of soak, one round), which planned_hours leaves. If less than
20 minutes remain before the deadline `up` printed, run `harness down` instead, and rerun soak-graphs and soak-shipped
together on one new pod under a new budget.

## Cost

Quoted before the run. `harness quote` on 2026-10-23 15:07 PDT: 2x RTX PRO 6000 Blackwell Workstation Edition,
secure, $5.18/hr, 1.5 h planned, $8.55 worst case (Server Edition fallback at $4.98/hr, $8.22 worst case). The
two-variant kit was quoted at 0.75 h, $4.27; the soak and the 1-GPU pair add the rest.

## Reading the results

- Gemma 4's chat template defaults to `enable_thinking` false, and the pack passes no kwargs, so the prompt ends in a
  closed, empty thought channel (`<|channel>thought\n<channel|>`). vLLM's `gemma4` parser starts a reply in reasoning
  only when the prompt ends inside an open `<|channel>` block, so answers land in `content`, and the "Question:" stop
  in `harness/pod/gsm8k_chat.py` should not cut any short. Here GSM8K is closer to a score than on the thinking
  entries, still with about +/-4 points of noise at n=100. Report it as correct / answered (replies with a number),
  with `empty_content`, `no_number` and `hit_max_tokens` from `gsm8k_counts` listed separately. A high `empty_content`
  in every variant would point at the parser; garbage in every variant (`hit_max_tokens`, `no_number`) at float16;
  neither at eager mode.
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
  'tp:0'` on entries 4 and 7), and in graphs that prefix caching really turned on (`enable_prefix_caching=True`).
  The benchmark's random prompts share no prefixes, so prefix caching's gain is not in these numbers.
- One GPU, startup: `boot_s` in each verdict is launch to `/health` answering; `<variant>/boot_summary.txt` splits it
  into weight loading, torch.compile, graph capture and engine init. A real install compiles once and loads the cache
  on later starts; the compile lines say which happened here.
- One GPU, memory: vLLM profiles CUDA graph memory and takes it out of the KV cache at a fixed
  `--gpu-memory-utilization`, so graphs do not raise memory use; they shrink the KV cache. Report "Available KV cache
  memory" and "GPU KV cache size" (tokens, and the maximum concurrency at 262144) for tp1-shipped and tp1-graphs, and
  the graph pool from "Graph capturing finished ... took" and "CUDA graph pool memory". With no hybrid state, the KV
  difference here is graph capture alone. The same lines for shipped and graphs give the 2-GPU figures.
- Soak: `soak-graphs/soak.txt` has one line per round (requests completed, output tok/s, mean TPOT) and the totals;
  per-round tok/s and TPOT that drift across 10 minutes would point at a slow degradation even with no failure.
  `memory.jsonl` tracks container memory. A POST_FAIL has `soak_failure_tail.txt` and the verdict's `fault`.

## Results (2026-10-25)

Lean run under a $5.00 budget: shipped and graphs only. The 1-GPU pair and the soak were not run.

One pod, 2x RTX PRO 6000 Blackwell Workstation Edition, EU-CZ-1, driver 595.91.07, GPUs on different NUMA nodes
(`SYS`), CUDA peer access both ways. Under load GPU 0 ran at PCIe Gen5 x8 (x16 maximum) and GPU 1 at x16; both
variants ran on this same host, so the ratios hold. `NCCL_P2P_DISABLE` absent from both server processes. SM clocks
held 2.82-2.84 GHz median under load. All-reduce ran on PYNCCL; `enable_prefix_caching=True` in graphs' server.log,
`False` in shipped's. Cost $1.00 of the $5.00 budget.

Output tok/s, random 1024-token prompts, 512 output tokens (mean TPOT in ms in brackets):

| Variant | 1 user | 16 users | 64 users | boot |
|---|---|---|---|---|
| shipped (TP2, eager, no prefix caching) | 47.6 (20.9) | 705.1 (21.8) | 2,452.1 (24.0) | 70 s |
| graphs (TP2) | 227.3 (4.3) | 1,923.1 (7.5) | 3,495.0 (16.8) | 100 s |
| gain | 4.8x | 2.7x | 1.4x | |

- GSM8K: 98 of 100 correct in both variants, every reply answered, none cut at the token cap.
- Memory: graphs plus prefix caching cost 0.83 GiB of graph pool per GPU (0.58 + 0.25) and 0.9% of the KV cache:
  4,639,931 tokens vs 4,680,895 (17.7x vs 17.9x concurrency at 262144). With no hybrid state, that is graph capture
  alone.
- Startup: 100 s vs 70 s to `/health`; engine init 56 s vs 22 s, of which torch.compile 16.9 s on a cold cache.
