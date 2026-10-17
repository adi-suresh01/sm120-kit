# sglang#33422, vLLM side: hermia's stock vLLM 0.25.1 recipe on the same prompts

The comparison half of the sglang#33422 benchmark (`issues/sglang-33422/README.md` has the issue, the hypotheses,
and the workloads). The reporter quotes ~12.5K tok/s prefill from hermia-ai/deepseek-v4-flash-sm120-stock-vllm; this
folder reruns that recipe with the exact prompts the sglang pod sends. Its own folder so it has its own ledger and
budget (the harness takes one package and one image per issue).

## hermia's setup (copied, commit 9fa3a628, 07-18)

`scripts/setup-venv.sh`: stock `vllm==0.25.1`, then `flashinfer-python==0.6.14` ("vLLM 0.25.1 pins
flashinfer-python==0.6.13, but the sm_120 DeepSeek-V4 sparse-MLA decode call site uses a 0.6.14-only API"). No
flashinfer-cubin 0.6.14 exists, hence the version-check bypass. `scripts/serve.sh`:

```
export FLASHINFER_DISABLE_VERSION_CHECK=1     # flashinfer-python 0.6.14 vs cubin 0.6.13 (no 0.6.14 cubin on PyPI)
export NCCL_P2P_DISABLE=1                     # Blackwell PCIe P2P allreduce deadlocks without this
export CUDA_HOME=/usr/local/cuda; export PATH="$CUDA_HOME/bin:$PATH"   # nvcc for the MHC-prenorm TileLang JIT

"$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name deepseek-v4-flash \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --kernel-config '{"moe_backend":"marlin"}' \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  --trust-remote-code
```

- Model: `deepseek-ai/DeepSeek-V4-Flash-DSpark`, 4x RTX PRO 6000 (96 GB, sm_120, PCIe, no NVLink; edition not
  stated). No speculation: MTP and DSpark both die at init on stock 0.25.1 (`num_tokens > 64`, their docs/BUGS.md).
- Prefill ~12,500 tok/s "@ 256K", from `bench/needle.py`: one non-streamed request, `usage.prompt_tokens` / wall
  seconds. `needle.py 256000` builds 237,747 templated tokens with this tokenizer (counted locally; they report the
  needle passing at ~238K and ~119K, and `needle.py 128000` is 118,888), hence W2's 238,000. A single prefill past
  ~256K kills the engine with `CUDA_ERROR_LAUNCH_FAILED` (719) in DeepGEMM.

## What this folder changes, and why

- **Checkpoint: `DeepSeek-V4-Flash-0731` at `7872f01b`, not `-DSpark`.** Both are `DeepseekV4ForCausalLM` with
  byte-identical `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, and
  `model.safetensors.index.json` (48 shards, 166.89 GB), but all 48 shard hashes differ (0731 is the later release).
  The sglang pod serves 0731, so vLLM does too; the comparison is engine against engine on the same weights.
- `--revision 7872f01b`, no `--served-model-name` (the harness sends the checkpoint id), host and port from the
  harness. Everything else verbatim, env included.
- FlashInfer 0.6.14 goes into the v0.25.1 image (which ships 0.6.13 python, cubin, and jit-cache; vllm-61100
  `versions.txt`), with its dependencies as hermia installed it, plus `flashinfer-jit-cache==0.6.14+cu130` from
  flashinfer.ai so the boot does not compile every kernel (hermia's venv compiled them; same source). A jit-cache left
  at 0.6.13 would hand 0.6.14 stale prebuilt kernels.

Checked at v0.25.1: `DeepseekV4ForCausalLM` is registered (`vllm/model_executor/models/registry.py:94`), with the
`deepseek_v4` tokenizer (`vllm/tokenizers/registry.py:42`, chat mode unless the request asks for thinking), tool and
reasoning parsers, and `moe_backend: marlin` in `--kernel-config` (`vllm/config/kernel.py:132,181`). The image
installs `cuda-nvcc` (`docker/Dockerfile:713`) on a CUDA 13.0.2 base and FlashInfer 0.6.13 (`:818`). `setup.sh`
checks vLLM 0.25.1, torch 2.11.0 (the FlashInfer install must not move it), FlashInfer 0.6.14, and nvcc on PATH.

## Variant

| Variant | Flags | Env | Quality |
|---|---|---|---|
| v0-hermia | hermia's `serve.sh` as above | `NCCL_P2P_DISABLE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_HOME=/usr/local/cuda` | GSM8K 50 (`gsm8k_chat.py`) |

Then `prefill_bench.py` (post_smoke): probe, 8K warmup, W1 (9 x ~90K concurrent, 61 out) twice, W2 (one ~238K, 40
out) twice, as on the sglang pod. W2 plus its 40 output tokens stays under `--max-model-len 262144`. vLLM's own
`Avg prompt throughput` and `Prefix cache hit rate` lines go to `prefill_log.txt`.

Note the asymmetry the comparison inherits: hermia disables NCCL P2P (all-reduce through host memory), the reporter's
sglang config only disables the custom all-reduce (NCCL may still use P2P). Both are the published recipes, run as
published.

## Hardware, hours, cost

4x RTX PRO 6000 Server Edition, SECURE, $9.96/hr, the same GPU type as the sglang pod. `harness quote` (10-17
01:15Z): stock in EU-CZ-1, EUR-IS-1, US-NC-2; `planned_hours = 1.0` caps it at $10.96 worst case. Expected: setup
about 10 min, boot about 10-15 min (load, DeepGEMM and TileLang JIT, graph capture), GSM8K 2 min, workloads about
8 min: about 45 min, $7.50. The plan's 0.75 h left no room for a slow first JIT.

## Stop rules

- `setup.sh` fails (version asserts, nvcc, download under 400 MB/s after 3 min, cold read under 500 MB/s) ->
  `harness down`, no run.
- v0 BOOT_FAIL -> record the fault, stop; the recipe is published as working on stock 0.25.1, so a failure here is a
  finding about the recipe or the pod, not something to tune around on paid time.

Sequence: approval, `harness budget set sglang-33422-vllm`, `harness run sglang-33422-vllm v0-hermia`.

## Files

| File | Purpose |
|---|---|
| `harness.toml` | Pod, image, hermia's flags, v0-hermia |
| `setup.sh` | FlashInfer 0.6.14, version and nvcc checks, prompts, weights to container disk, page-cache drop, cold-read gate |
| `build_prompts.py`, `prefill_bench.py` | Byte-for-byte copies of `issues/sglang-33422/`'s; change both copies together |
| `hooks.sh` | post_smoke runs `prefill_bench.py`; post_stop keeps vLLM's throughput lines |

## Results

None yet.
