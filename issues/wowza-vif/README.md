# Wowza Video Intelligence Framework: the bundled VLM sidecar's eager mode on 1x RTX PRO 6000

Wowza's Video Intelligence Framework
([WowzaMediaSystems/wowza-video-intelligence-framework](https://github.com/WowzaMediaSystems/wowza-video-intelligence-framework),
main @ `cc057543`, 2026-10-16) ships an optional vLLM sidecar for its vision-language analyzers. Each model has a
profile in `vlm-env/`, and the profiles are tuned to boot on the smallest card each model fits on. Two of them,
`nemotron` and `cosmos-nano`, pass `--enforce-eager`. Since 2026-08-14 (`38ac106b`) the profiles' comments and
`docs/VLM_GUIDE.md` say to drop it on a 40 GB+ card, but nothing detects the card, and the guide's model table (line
140) still says Nemotron "needs `trust-remote-code` + eager mode".

Question: on a 96 GB RTX PRO 6000, does the `nemotron` profile serve correctly with CUDA graphs on, and what does the
shipped eager mode cost it? That is the evidence for a helpful issue or PR to the project (for example, the entrypoint
choosing eager mode from the card's memory the way it already chooses the KV cache dtype).

## What the project ships, at `cc057543`

- `docker-compose.yaml` lines 190-226: service `vlm`, image `vllm/vllm-openai:v0.26.0` (line 192), entrypoint
  `vlm-entrypoint.sh` (line 218), model profile `./vlm-env/${VLM_CONF:-qwen}.env` (line 197), `ipc: host`, all GPUs.
  `docker-compose.vlm-multi.yaml` adds a second sidecar, `vlm-2`, on the same image with `${VLM_2_CONF:-nemotron}`
  (line 30) on GPU 1. `.env.example` lines 64 and 72 name the same two defaults.
- `vlm-entrypoint.sh` (revision 2026-07-24) builds the command: fixed `--no-enable-prefix-caching` and
  `--mm-processor-cache-gb=0` ("frames never repeat", lines 12-16), `--kv-cache-dtype=fp8` when every GPU it uses has
  compute capability 8.9 or more (lines 122-131), `--limit-mm-per-prompt={"image": 8, "video": 0}`, defaults of 8192
  batched tokens and TP 1, then the profile's `VLM_EXTRA_ARGS` split on spaces, and `exec vllm serve` (line 182).
- `vlm-env/nemotron.env`: `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8`, `VLM_MAX_MODEL_LEN=8192`,
  `VLM_GPU_MEMORY_UTILIZATION=0.80`, and line 14: `--quantization modelopt --trust-remote-code --enforce-eager
  --max-cudagraph-capture-size=64 --structured-outputs-config {"backend":"xgrammar","disable_any_whitespace":true}`.
  Lines 7-11: "SMALL-GPU TUNING -- drop --enforce-eager on a 40 GB+ card."
- `vlm-env/cosmos-nano.env` is the other profile with `--enforce-eager` (`nvidia/Cosmos3-Nano`, about 32 GB of BF16
  weights; its comment says to drop it on an 80 GB card). `qwen.env`, `gemma.env` and `cosmos-edge.env` keep CUDA
  graphs and only cap capture at batch 64.
- `docs/VLM_GUIDE.md` lines 149-156: "The shipped model files are tuned for small GPUs -- undo that on a big card";
  for `--enforce-eager`: "Remove it. This costs per-request latency on every request." Line 155 says to remove
  `--max-cudagraph-capture-size=64` as well, since batches above 64 then run eager. The README's compute baseline
  for the whole framework is 1x L4 (24 GB) recommended, 1x T4 minimum (lines 304-310).
- Open PR #74 (2026-10-01, updated 10-07, no reviews) replaces the sidecar and `vlm-env/` with a pool of engines that
  the service image configures from a model catalog with per-VRAM tuning tiers. It keeps `v0.26.0`. The catalog and
  its tiers live in the service image, not in this repo, so whether a 96 GB card still gets eager mode after #74 is not
  visible here; the measurement applies to `main` as shipped.

Why Nemotron and not the default model: `VLM_CONF` defaults to `qwen` (`Qwen/Qwen3-VL-4B-Instruct-FP8`), whose
profile has no `--enforce-eager`; its only small-GPU flag is the capture cap at 64, which costs nothing below batch 64.
A shipped-vs-graphs run on it would compare a config with itself. `nemotron` is the default of the second sidecar and
the model the guide singles out as needing eager mode. `cosmos-nano` is not covered.

## Setup

- 1x RTX PRO 6000 Blackwell (Workstation Edition, Server Edition fallback), 96 GB. Wowza's docs address "40 GB+"
  cards; the stack's own baseline is a 24 GB L4, where eager mode may well be needed and is not tested here.
- Model: `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8` @ `e9550fde` (15.4 GB of safetensors, last modified
  2026-09-02). `NemotronH_Nano_VL_V2`: hybrid Mamba-2 / attention language model with a RADIO vision tower; modelopt FP8
  weights, KV cache not quantized in the checkpoint. Its chat template defaults to `enable_thinking` false.
- Image: `vllm/vllm-openai:v0.26.0` pinned by index digest (`ffb2d59b`), CUDA 13.0.2, `TORCH_CUDA_ARCH_LIST` includes
  12.0.
- What the entrypoint execs for this profile on one sm_120 GPU (we add `--revision`; the harness adds `--host` and
  `--port`):

  ```
  vllm serve nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8 --max-model-len=8192 --gpu-memory-utilization=0.80
    --max-num-batched-tokens=8192 --tensor-parallel-size=1 --kv-cache-dtype=fp8 --no-enable-prefix-caching
    --mm-processor-cache-gb=0 --limit-mm-per-prompt={"image": 8, "video": 0}
    --quantization modelopt --trust-remote-code --enforce-eager --max-cudagraph-capture-size=64
    --structured-outputs-config {"backend":"xgrammar","disable_any_whitespace":true}
  ```

- Not mirrored: the compose mounts `vlm-patches/cosmos3_edge.py` over vLLM's Cosmos3-Edge model file (line 202); the
  file only loads for that architecture, and the compose comment calls it inert for other models. `VLLM_API_KEY`,
  `HF_TOKEN` and `HF_HUB_OFFLINE` are empty by default. The compile cache volume (`./vis/vlm-cache`) is the container's
  own `~/.cache/vllm` here.
- Each variant: boot, smoke test, GSM8K 100 (5-shot chat, greedy), then `vllm bench serve` at concurrency 1 / 16 / 64
  (random 1024 in / 512 out).

## What this does not measure

The harness's quality block is text only: GSM8K goes through `/v1/chat/completions` with text, and `vllm bench serve`
sends random text prompts to `/v1/completions`. The VIS workload is different, and none of this is covered:

- Image inputs. VIS sends one request per analysis window carrying up to 8 frames (`inference_fps x duration`, capped
  at 8), so each request is a vision-encoder pass plus a long image-token prefill, which eager mode slows far less
  than decode.
- Structured output. VIS asks for guided JSON through xgrammar; the server here has the same
  `--structured-outputs-config`, but no request uses it.
- Reply length. VIS replies are short JSON, so per-token decode cost, which is where eager mode hurts most, is a
  smaller share of each request than in a 512-token benchmark reply.
- Sharing the GPU with VIS's detection models (the 0.80 utilization leaves room for them); here the sidecar is alone.

So the benchmark gives the decode-side cost of eager mode, an upper bound for VIS's requests, not their latency. A
follow-up could add a `post_smoke` hook running `vllm bench serve --backend openai-chat --dataset-name random-mm` with
a few frames per request and short outputs, once the text result shows a gap worth chasing.

## Matrix

| Variant | Flags vs shipped | Expected |
|---|---|---|
| shipped | as shipped | serves; eager baseline |
| graphs | `--enforce-eager` dropped (the guide's line 156); the capture cap stays at 64, which no batch here exceeds | serves; GSM8K within noise of shipped; higher tok/s at 1 and 16 users |

If graphs fails to boot, or its GSM8K replies turn to garbage while shipped's do not, the guide's "needs eager mode"
is right for this model on this image, and that is the finding to report instead.

## Cost

Quoted before the run. `harness quote wowza-vif` on 2026-10-23 21:37Z: 1x RTX PRO 6000 Blackwell Workstation
Edition, secure, $2.59/hr (EU-CZ-1, EU-RO-1, EUR-IS-1), 0.75 h planned, $2.14 worst case; the Server Edition fallback
is $2.49/hr ($2.05).

## Reading the results

- GSM8K is a correctness check, not a score: about +/-4 points of noise at n=100. The chat template answers without
  reasoning by default, so replies should be short and none should hit `max_tokens` (4096 here); `hit_max_tokens` or
  `no_number` in graphs but not in shipped would point at CUDA graphs on the hybrid Mamba layers.
- `<variant>/settings.txt` (hooks.sh `post_stop`, from server.log) is what each server says it ran with: check that the
  KV cache really is fp8 (`kv_cache_dtype=fp8`), that graphs captured CUDA graphs (the capture sizes stop at 64) and
  shipped did not (`enforce_eager=True`), and the "Maximum concurrency" figure both print.
- Compare only within the pod. setup.sh records the data center, PCIe link maximum, power limit, and idle clocks and
  clock-event reasons; `<variant>/gpu_clocks.csv` samples clocks, power, temperature, PCIe link and clock-event reasons
  every 5 s from launch to kill. A variant whose SM clock sags or whose reasons show a power or thermal cap under load
  is a host effect, not eager mode.
- Prefix caching is off in both variants, as the entrypoint fixes it; the benchmark's random prompts share no
  prefixes anyway.
