# AWS g7e sample: Qwen3-32B in BF16 on vLLM 0.15.1, 1x RTX PRO 6000 Server Edition

AWS's SageMaker hosting examples
([aws-samples/sagemaker-genai-hosting-examples](https://github.com/aws-samples/sagemaker-genai-hosting-examples) @
`63f05821`, 2026-10-21) include `03-features/instances/g7e/g7e_inference_benchmark.ipynb`, added on 2026-04-16 in
`59c4be3e` ("adding g7e inference notebook for blog") and unchanged since. It deploys Qwen3-32B in BF16 on
`ml.g7e.2xlarge` (one RTX PRO 6000 Blackwell Server Edition, 96 GB, $4.2039/hr in the notebook) with the AWS vLLM
container `vllm:0.15.1-gpu-py312-cu129-ubuntu22.04-sagemaker`, and compares it with `ml.g6e.12xlarge` (4x L40S, TP4).

Question: what does that stack cost on its own GPU against the current vLLM release with the same flags and weights,
and against Qwen's FP8 checkpoint on the current release? All three on one pod. That is the evidence for a helpful
issue or PR to the sample (for example, a newer container tag, or an FP8 cell for g7e).

## What the sample ships, at `63f05821`

- Cell 2: `VLLM_IMAGE = 763104351884.dkr.ecr.{REGION}.amazonaws.com/vllm:0.15.1-gpu-py312-cu129-ubuntu22.04-sagemaker`.
- Cell 4, config `g7e`: `SM_VLLM_MODEL=Qwen/Qwen3-32B`, `SM_VLLM_TENSOR_PARALLEL_SIZE=1`, `SM_VLLM_DTYPE=bfloat16`,
  `SM_VLLM_MAX_MODEL_LEN=4096`, `SM_VLLM_ENABLE_PREFIX_CACHING=""`, `SM_VLLM_ENABLE_CHUNKED_PREFILL=""`,
  `SM_VLLM_MAX_NUM_SEQS=64`, `SM_VLLM_GPU_MEMORY_UTILIZATION=0.9`, `SM_VLLM_MAX_NUM_BATCHED_TOKENS=8192`.
- The container's entrypoint (`scripts/vllm/sagemaker_entrypoint.sh` in aws/deep-learning-containers @ `a226b9bf`,
  "Update vLLM from 0.15.0 to 0.15.1", 2026-02-06, the code this image was built from: EFA 1.46.0 in both)
  lowercases each `SM_VLLM_*` name into a flag, adds a value only when it is non-empty, and runs
  `python3 -m vllm.entrypoints.openai.api_server --port 8080` with them. So the server is:

  ```
  python3 -m vllm.entrypoints.openai.api_server --port 8080 --model Qwen/Qwen3-32B --tensor-parallel-size 1
    --dtype bfloat16 --max-model-len 4096 --enable-prefix-caching --enable-chunked-prefill --max-num-seqs 64
    --gpu-memory-utilization 0.9 --max-num-batched-tokens 8192
  ```

- Cells 9 and 11, the workload: about 1,000 input tokens per request, a system prompt shared by every request
  (roughly half of them, estimated from its text) plus a dictation, `max_tokens` 600, temperature 0.7, top_p 0.8,
  top_k 20, `chat_template_kwargs: {"enable_thinking": false}`, 50 requests at each of 1, 2, 4, 8, 16 and 32
  concurrent clients. The notebook's outputs are stripped, so it carries no
  numbers to compare with.
- `07-benchmark/sagemaker-inference-benchmark-suite/recipes/qwen3-32b-g7e-vanilla.yaml` (2026-05-20) runs the same
  model, container version and flags on the same instance, with prefix caching off; not covered here.

## Can the 0.15.1 container run on sm_120

Yes on paper, and the notebook was written for a g7e blog post, so AWS ran it there. From the image's config and
layer history (public mirror, digest `4b29ac76`, built 2026-02-06): it is `FROM vllm/vllm-openai:v0.15.1`
(aws/deep-learning-containers `docker/vllm/Dockerfile` @ `a226b9bf`), CUDA 12.9.1,
`TORCH_CUDA_ARCH_LIST=7.0 7.5 8.0 8.9 9.0 10.0 12.0`, FlashInfer 0.6.1 with its cu129 cubin and JIT cache, torch
2.9.1. CUDA 12.9 needs a 575 or newer driver, hence `min_cuda = "12.9"`.

One trap sits in the upstream base: `vllm/vllm-openai:v0.15.1` writes `/etc/ld.so.conf.d/00-cuda-compat.conf`,
pointing the loader at CUDA 12.9's forward-compat `libcuda` (vLLM PR #30784), and that fails to load on newer drivers,
for example 580 / CUDA 13.0 (vLLM issue #32373). AWS's Dockerfile deletes the file (lines 69-71, `RUN rm
/etc/ld.so.conf.d/00-cuda-compat.conf`, commented "issue introduced by" #30784), so the container in the notebook
should not hit it; setup.sh prints `/etc/ld.so.conf.d/` and the `libcuda.so.1` the loader finds. Someone running the
upstream image on a 580 host would, and that is not tested here. If shipped fails to boot anyway, its server.log is
the finding, and setup.sh still installs and downloads what current and fp8 need.

## Setup

- 1x RTX PRO 6000 Blackwell Server Edition, the card `ml.g7e` carries. RunPod's power cap and clocks may differ from
  AWS's; setup.sh records this host's.
- Weights: `Qwen/Qwen3-32B` @ `9216db57` (65.5 GB BF16 safetensors) and `Qwen/Qwen3-32B-FP8` @ `aa55da1e` (34.3 GB,
  block-wise FP8), both last modified 2025-07-26.
- Pod image: the notebook's tag from AWS's public mirror,
  `public.ecr.aws/deep-learning-containers/vllm:0.15.1-gpu-py312-cu129-ubuntu22.04-sagemaker@sha256:4b29ac76...`. The
  docs entry for 0.15.1 (`docs/src/data/vllm/0.15.1-gpu-sagemaker.yml`) marks it `public_registry: true`; the digest
  of the private regional copy the notebook pulls was not checked against it.
- vLLM 0.32.0 (released 2026-10-18), for current and fp8: the harness runs one image per pod, so setup.sh installs the
  release's CUDA 12.9 wheel (`vllm-0.32.0+cu129`, sha256 `9918be80...`, the release asset's digest) into
  `/root/vllm-0.32.0` with `uv`, resolving torch 2.13.0 and its CUDA libraries from PyTorch's cu129 index under
  `unsafe-best-match`, the way `vllm/vllm-openai:v0.32.0-cu129` does (its Dockerfile sets `UV_INDEX_STRATEGY`, and so
  does the AWS image). The default first-index strategy would fail here: the cu129 index carries requests 2.28.1 and
  setuptools 78.1.0, under vLLM's floors of 2.33.0 and 78.1.1. setup.sh then checks that torch is the CUDA 12.9 build,
  that `vllm._C` loads, and that CUDA starts, and stops before the weights download if not. Then `flashinfer
  download-kernels` (FlashInfer 0.7.0.post1: `flashinfer-cubin` and the cu129 `flashinfer-jit-cache`, sm120f
  provider included), as that image's Dockerfile does. Both stacks share the image's CUDA 12.9 userspace (nvcc,
  cudart, cublas on `LD_LIBRARY_PATH`).
- `api_server.sh` runs `python -m vllm.entrypoints.openai.api_server` (deprecated but present in 0.32.0) under the
  image's Python, or under the venv when a variant sets `VLLM_PYTHON`.
- Ours, on top of the notebook's flags: `--revision` for each checkpoint; `--default-chat-template-kwargs
  '{"enable_thinking": false}'`, because the notebook disables thinking per request and the harness's GSM8K client
  cannot (it only changes chat requests; the benchmark uses `/v1/completions`); `--served-model-name Qwen/Qwen3-32B`
  in fp8, so every client addresses it by the same name; `--host 127.0.0.1` from the harness.
- Each variant: boot, smoke test, GSM8K 100 (5-shot chat, greedy, `max_tokens` 2048 from `max_model_len` 4096), then
  `vllm bench serve` at concurrency 1 / 8 / 32 (random 1024 in / 512 out, close to the notebook's 1,000 / 600). The
  benchmark client is the image's 0.15.1 `vllm bench serve` for all three variants.

## Matrix

| Variant | vLLM | Weights | Expected |
|---|---|---|---|
| shipped | 0.15.1, the image's own | BF16 | serves; the baseline |
| current | 0.32.0 (cu129 venv) | BF16 | serves; GSM8K within noise of shipped; tok/s close to shipped's: BF16 decode at these batch sizes is bound by reading 65 GB of weights per step, so a newer vLLM can gain only on scheduling and kernels |
| fp8 | 0.32.0 (cu129 venv) | `Qwen3-32B-FP8` | serves; GSM8K within noise of BF16; about 1.6-1.9x BF16's output tok/s at 1 user (half the weight bytes per step), and more than twice the KV room |

## Cost

Quoted before the run. `harness quote aws-g7e-sample` on 2026-10-23 21:37Z, at 1.25 h: 1x RTX PRO 6000 Blackwell
Server Edition, secure, $2.49/hr (EU-CZ-1, EU-NL-1, EU-RO-1, EUR-IS-1, US-MO-2, US-NC-2, US-PA-1), $3.42 worst case.
planned_hours is now 1.5, so $4.11 worst case: setup can take 32 min on a host at the download floor, and three
variants about 35, which left 1.25 h under 5 min of margin. Expected spend is about an hour, $2.50-2.75.

## Reading the results

- Compare only within the pod. The notebook's g7e numbers are not public in the repo, and RunPod's Server Edition may
  run at a different power cap than AWS's; read setup.sh's power limit and clocks before comparing with anything AWS
  publishes.
- current is a venv on the old image, not `vllm/vllm-openai:v0.32.0-cu129` itself: same vLLM wheel build, same torch
  and CUDA 12.9 wheels, same FlashInfer kernel packages if `flashinfer download-kernels` succeeded (setup.sh's second
  versions line lists `flashinfer-cubin` and `flashinfer-jit-cache`; "absent" means 0.32.0 compiled FlashInfer kernels
  while booting or serving, which inflates its boot time and may slow its first benchmark point). It is also not the
  default CUDA 13.0 build or AWS's own 0.32.0 container (cu130, Ubuntu 24.04).
- Prefix caching is on in all three, as shipped. The notebook's requests share a system prompt of about 400 tokens,
  which prefix caching serves from cache; the benchmark's random prompts share nothing, so that gain is not in these
  numbers.
- GSM8K is a correctness check with about +/-4 points of noise at n=100. With thinking off, replies should be short:
  `hit_max_tokens` or `empty_content` above zero means the chat template kwargs did not apply on that version.
- Which stack ran, from the process, not harness.toml: `<variant>/server_version.json` (the live server's `/version`),
  `server_env.txt` (`VLLM_PYTHON` set for current and fp8 only), and `settings.txt` (hooks.sh `post_stop`, from
  server.log: the engine config line with its version, dtype, quantization, prefix caching and capture sizes, the KV
  cache size and "Maximum concurrency" line; fp8 should show more than twice BF16's KV tokens). server.log also names
  the attention backend and, for fp8, the FP8 linear kernel each version chose on sm_120.
- Host effects: setup.sh records the data center, power limit, and idle clocks and clock-event reasons;
  `<variant>/gpu_clocks.csv` samples clocks, power, temperature, PCIe link and clock-event reasons every 5 s from
  launch to kill.
- If the venv fails (install, the CUDA 12.9 torch check, `vllm._C`, or CUDA start), setup stops before the weights
  download and `harness run` takes the pod down with only the image's facts recorded. The image pull from public ECR
  has not been timed on RunPod.
