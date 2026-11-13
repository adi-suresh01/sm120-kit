# dell/deskside-agentic-ai: custom all-reduce at TP=2 on PCIe RTX PRO 6000 (stage 1 of 2)

The vLLM recipes for Dell Pro Precision towers (PCIe-only RTX PRO 6000 Blackwell) say vLLM's custom all-reduce
aborts with "custom_all_reduce.cuh invalid argument" because PCIe GPU pairs have "no P2P fabric", and tell users to
turn custom all-reduce off for TP>1. The same scripts start the container with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. In the vLLM build the recipe pins, that allocator setting is
enough on its own to make custom_all_reduce.cuh fail with `invalid argument` at the end of CUDA graph capture
(below). This kit tests which of the two explanations holds, and what custom all-reduce is worth at TP=2 when it
runs. Run on 2026-11-13: see Results.

- Recipes: https://github.com/dell/deskside-agentic-ai at b4e09e6c (2026-10-27), read 2026-11-12.
- Upstream: vllm#42609 (open since 05-14: custom all-reduce fails in `cudaIpcGetMemHandle` with
  `expandable_segments:True`), reported again as vllm#49101 and vllm#56180. Fix PRs: vllm#43923 (open since 05-28,
  turns expandable_segments off around capture), vllm#51903, #55135, #56193 (closed unmerged). vllm#47266 (issue 9a)
  hit the same `custom_all_reduce.cuh 'invalid argument'` on 2x RTX PRO 6000 Workstation with
  `expandable_segments:True` exported in its launch script and attributed it to sm_120; its fix, vllm#47751 (disable
  custom all-reduce on sm_12x), closed unmerged on 10-27.
- Third-party isolation: syv-ai/HyperQwen#163 (2x RTX 3090 over NVLink, vLLM 0.28.0, torch 2.13.0) hit
  `custom_all_reduce.cuh:164 'invalid argument'`; `NCCL_P2P_LEVEL=SYS` left it unchanged, and either
  `--enforce-eager` or `expandable_segments:False` cleared it. With `False`, the reporter measured custom all-reduce
  over NCCL at +6.4% median c=1 decode.
- Same mechanism in SGLang's v1 custom all-reduce, run by us: `issues/sglang-44291` (2x RTX PRO 6000 Server, P2P OK
  both ways). There the staged v1 all-reduce was 2.3-2.7x faster than pynccl at 2-16 KB.

## Recipe setup (verbatim, ASCII-normalized)

Hardware (`docs/stack-composition.md:17-19`, `docs/glossary.md:13`): T2 = 1x RTX PRO 6000 Workstation Edition,
T4 = 2x RTX PRO 6000 Max-Q, T6 = 5x Max-Q; Max-Q is the 300 W edition. All GPUs are on PCIe.

`recipes/precision/vllm/docker-nemotron-3.5-lightning-nvfp4.sh`:

```
36   MODEL="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
38   IMAGE="${NEMOTRON_LIGHTNING_IMAGE:-$(ds_image lightning)}"
...
98   # PCIe-only GPU pairs (e.g. RTX PRO 6000 on T4/T6-class hosts) have no P2P
99   # fabric: vLLM's custom all-reduce kernel aborts there with
100  # "custom_all_reduce.cuh invalid argument". Set
101  # NEMOTRON_LIGHTNING_DISABLE_CUSTOM_AR=1 for TP>1 on such boxes; NVLink
102  # systems (GB200/GB300) keep the faster default.
103  AR_FLAGS=()
104  if [[ "${NEMOTRON_LIGHTNING_DISABLE_CUSTOM_AR:-0}" == "1" ]]; then
105    AR_FLAGS=(--disable-custom-all-reduce)
106  fi
...
112  ds_docker run -d --restart unless-stopped \
113    --gpus "${GPUS}" \
114    --ipc=host \
115    --shm-size=8g \
116    "${ULIMITS[@]}" \
117    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
118    -e VLLM_NVFP4_GEMM_BACKEND=marlin \
119    -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
...
125    --entrypoint vllm \
126    "${IMAGE}" \
127    serve "${MODEL}" \
128      --served-model-name nvidia/nemotron-3.5-lightning nemotron-lightning lightning auto \
129      --host 0.0.0.0 --port 8000 \
130      --trust-remote-code \
131      --tensor-parallel-size "${TP}" \
132      --moe-backend "${MOE_BACKEND}" \
133      --kv-cache-dtype fp8 \
134      --enable-expert-parallel \
135      --enable-auto-tool-choice \
136      --tool-call-parser qwen3_coder \
137      --reasoning-parser nemotron_v3 \
138      --gpu-memory-utilization "${GPU_MEM_UTIL}" \
139      --max-model-len "${MAX_MODEL_LEN}" \
140      --max-num-seqs "${MAX_NUM_SEQS}" \
141      --max-cudagraph-capture-size "${MAX_NUM_SEQS}" \
142      "${AR_FLAGS[@]}"
```

Defaults: `MOE_BACKEND=marlin` (:44), `GPU_MEM_UTIL=0.9` (:45), `MAX_MODEL_LEN=200000` (:53), `MAX_NUM_SEQS=512`
(:54). With `TP` unset, it snaps to the largest power of two at or under the host's GPU count (:68-76), so a T4 or
any 2-GPU host runs TP=2. Out of the box (`NEMOTRON_LIGHTNING_DISABLE_CUSTOM_AR` unset), that launch has custom
all-reduce on and expandable_segments on.

The same reading elsewhere in the repo:

- `recipes/precision/vllm/health-nemotron-3.5-lightning-nvfp4.sh:15-20`: TP>1 without `--disable-custom-all-reduce`
  is a WARN, "PCIe-only RTX PRO towers crash here: set NEMOTRON_LIGHTNING_DISABLE_CUSTOM_AR=1 and re-run" (:19).
- `recipes/lib/health.sh:122` and `recipes/lib/common.sh:164-165` map any `custom_all_reduce.cuh` log line to the same
  advice.
- `recipes/precision/vllm/docker-nemotron-3.5-super-vl-nvfp4.sh` (the T4 default model): the same advice (:22), the
  same flag switch (:63), and the same allocator setting (:78). `docs/topology.md:17-18` and
  `solutions/agentic-coding/it-admin-guide.md:116` repeat the advice.
- `recipes/versions.lock:9` pins Lightning to `vllm/vllm-openai@sha256:cb9bc64f...` (release v0.30.0 listed, not
  validated); `:13` records that digest as what `nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2` resolved to when
  Super VL was validated at TP=2 on a T6 on 2026-10-19.
- The repo's TP=2 measurements ran with custom all-reduce off:
  `tests/data/20261019_27689d3/t6-gpu01-tp2_*/MANIFEST.json` lists
  `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition x2 (TP=2)`, and its `container_args` end in
  `--disable-custom-all-reduce`. On RTX PRO 6000, Lightning has TP=1 data only (`tests/data/20261019_422dbe6`, one
  Max-Q).

The pinned image is a multi-arch index; its amd64 image was built 2026-10-15 from vLLM `3f1bcfd8` (build label),
with CUDA 13.0.2, NCCL 2.30.7, and torch 2.14.0 (`requirements/cuda.txt` at that commit), and presets no
`PYTORCH_CUDA_ALLOC_CONF`. The tag no longer resolves on Docker Hub (404 on 11-12); the digest does.

## What vLLM 3f1bcfd8 does with these settings

1. **Gate.** On one node, custom all-reduce is refused only for more than two PCIe-only GPUs
   (`custom_all_reduce.py:270-276`) or when `_can_p2p` fails (`:277-292`). With `VLLM_SKIP_P2P_CHECK=1`, the default
   (`envs.py:1197`), `_can_p2p` is `torch.cuda.can_device_access_peer` (`:86-100`). A pair without P2P gets a warning
   and NCCL, not a crash. The init buffers are `cudaMalloc`'d and exported in `csrc/custom_all_reduce.cu:173-186`, so
   the allocator setting never touches them; the peer opens them at `:193`.
2. **Capture.** During graph capture the inputs are used in place (`registered=self._capture_registered`,
   `custom_all_reduce.py:554-556`) and their addresses collected (`custom_all_reduce.cuh:252-255`). When the capture
   context ends, `register_graph_buffers` (`custom_all_reduce.py:467-497`) calls `get_graph_buffer_ipc_meta`, which
   takes each buffer's allocation base and calls `cudaIpcGetMemHandle` on it (`custom_all_reduce.cuh:151-170`, the call
   at `:164`). On failure `CUDACHECK` prints `Failed: Cuda error <file>:164 'invalid argument'` and calls `exit()`
   (`custom_collective_common.cuh:24-32`), so the worker dies with no Python traceback. That call exports this GPU's
   own memory; the peer is not involved until it opens the handle (`cudaIpcOpenMemHandle`, `:143`).
3. **Allocator.** With `expandable_segments:True`, PyTorch 2.14.0 backs CUDA-graph private pools with expandable
   (VMM) segments too; only fbcode builds leave private pools out (`c10/cuda/CUDACachingAllocator.cpp:3897-3901` at
   v2.14.0). VMM memory has no legacy CUDA IPC handle, so `cudaIpcGetMemHandle` returns `invalid argument`.
4. **The copy path vLLM already has.** vLLM skips registration and copies graph inputs into its pre-registered
   `cudaMalloc` buffer, but only when the graph pool is the sleep-mode cuMem pool:
   `register = config is None or not config.use_cumem_cudagraph_pool` (`cuda_communicator.py:123-125`, commented
   "cuMem graph buffers cannot be IPC-registered; capture copies them instead"). Nothing reads
   `PYTORCH_CUDA_ALLOC_CONF` for custom all-reduce; `config/vllm.py:1217-1250` checks expandable_segments for KV
   connectors only.

So on a pair with P2P, TP=2 with CUDA graphs, custom all-reduce, and `expandable_segments:True` should fail at
`custom_all_reduce.cuh:164` once the first capture ends, and `expandable_segments:False` should boot with custom
all-reduce. The first capture is the CUDA graph memory profile (`gpu_worker.py:683-692`, logged as `Profiling CUDA
graph memory`), which captures a few graphs inside `graph_capture()` before the KV cache is allocated, so the crash
comes after that log line, before any `Capturing CUDA graphs` progress. A pair whose driver reports no P2P never turns custom all-reduce on; one that reports P2P but cannot map
peer memory would fail in `cudaIpcOpenMemHandle` (`custom_all_reduce.cu:193` at init, or `.cuh:143`), not at `:164`.
The recipe's message quotes no line number.

## Hypotheses

| # | Hypothesis | Confirmed by | Killed by |
|---|---|---|---|
| H1 | The recipe's own `expandable_segments:True` puts graph buffers in VMM memory, which `cudaIpcGetMemHandle` rejects | `default` fails at `custom_all_reduce.cuh:164` and `es-off` serves with `CUSTOM` in its backend list; the probe fails under `True` and passes under `False` | `default` serves with `CUSTOM` |
| H2 | PCIe RTX PRO 6000 pairs have no working P2P path (the recipe's reading) | the P2P gate reads not OK, vLLM's IPC probe fails on `cudaMalloc` memory, or `es-off` fails in `cudaIpcOpenMemHandle` | gate OK, probe passes both ways, `es-off` serves with `CUSTOM` |
| H3 | The custom all-reduce kernel is wrong on sm_120 (vllm#47751's reading) | `es-off` boots with `CUSTOM`, then faults, or scores well under `es-off-car-off` on GSM8K | `es-off` matches `es-off-car-off` on GSM8K and completes every benchmark request |
| H4 | At TP=2 over PCIe, custom all-reduce is no faster than NCCL at decode, so turning it off costs nothing | `es-off` c=1 TPOT within 2% of `es-off-car-off` | `es-off` c=1 TPOT clearly lower |

## Experiment matrix, stage 1

Setup (`setup.sh`, on each box): power limit and PCIe link per GPU; `nvidia-smi topo -p2p w` (`harness up` records
`topo -p2p r` and refuses anything but OK); `torch.cuda.can_device_access_peer` both ways (what vLLM's gate reads);
vLLM's own IPC probe, `gpu_p2p_access_check` (a `cudaMalloc` buffer opened with `cudaIpcOpenMemHandle` in a second
process and written from the peer), both ways; and `vmm_ipc_probe.py`, one `cudaIpcGetMemHandle` on a PyTorch
allocation, under `expandable_segments:True` and `False`, with the segment's own `is_expandable` flag. All of it lands
in `results/dell-deskside-car/jobs/setup.log`. Setup exits, and the box is ended, if the container sets
`PYTORCH_ALLOC_CONF`: torch reads it before `PYTORCH_CUDA_ALLOC_CONF`, so it would hold every variant at one setting.
The image sets neither (its config's `Env`, checked 11-13), and serve_variant.sh exports each variant's env after the
container env, so each server process gets its own value.

Server: the recipe's image, `Qwen/Qwen3-0.6B` at `c1899de2` (1.5 GB, dense, hidden size 1024: a 2 KB bf16 all-reduce
per call at c=1), TP=2, CUDA graphs on, the recipe's container env (:117-119) with only the allocator value moving,
the recipe's flags that do not depend on the model (`--trust-remote-code`, `--gpu-memory-utilization 0.9`,
`--max-num-seqs 512`, `--max-cudagraph-capture-size 512`), and `--max-model-len 8192`. At 3f1bcfd8,
`VLLM_NVFP4_GEMM_BACKEND` no longer exists (`--linear-backend` replaced it; vLLM logs it as an unknown variable) and
`VLLM_WSL2_ENABLE_PIN_MEMORY` is read on WSL only (`platforms/cuda.py:319-322`), so neither changes a run here. Each variant that boots: smoke test, GSM8K 50 (5-shot chat, greedy), then `vllm bench serve`
with random prompts, 1024 tokens in and 512 out, at c=1 (8 prompts) and c=16 (64 prompts).

In run order (H1's two arms first, so a deadline that cuts the run costs `car-off`, which the recipe's own TP=2 runs
already cover):

| Variant | `PYTORCH_CUDA_ALLOC_CONF` | All-reduce flag | Recipe equivalent | Expected if H1 |
|---|---|---|---|---|
| default | `expandable_segments:True` | none | as shipped on a 2-GPU host | BOOT_FAIL, `custom_all_reduce.cuh:164 'invalid argument'` after the first capture |
| es-off | `expandable_segments:False` | none | one env value changed | serves on `['CUSTOM', 'PYNCCL']` |
| es-off-car-off | `expandable_segments:False` | `--disable-custom-all-reduce` | none | serves on `PYNCCL` |
| car-off | `expandable_segments:True` | `--disable-custom-all-reduce` | `NEMOTRON_LIGHTNING_DISABLE_CUSTOM_AR=1` | serves on `PYNCCL` |

From each `server.log`, read the `Using [...] all-reduce backends` line and any `custom_all_reduce.cuh` line. The
crash also shows as the verdict's `fault` (`Cuda error <file>:164 'invalid argument'`) and the `boot_fail` phase line,
which quotes the first log line matching `fail_pattern`. `es-off` against `es-off-car-off` isolates custom all-reduce;
`es-off` against `car-off` is the change the recipe would make.

The small model keeps the all-reduce question apart from Lightning: the repo has no Lightning run at TP=2, so a
Lightning-only boot failure could otherwise hide the answer. Its c=1 gap overstates what Lightning would gain, since a
0.6B model spends a larger share of each decode step in all-reduce.

## Stage 2: the recipe's model (`issues/dell-deskside-car-lightning`)

Run only if stage 1 confirms H1 and kills H2. `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` at `bee75962`
(21.6 GB; hybrid Mamba/MoE, hidden size 2688, 52 layers: a 5.4 KB bf16 all-reduce per call at c=1), TP=2, with the
recipe's container env (:117-119) and serve flags (:127-141). `--served-model-name` is left out, since the harness
names the model by its checkpoint id.

| Variant | `PYTORCH_CUDA_ALLOC_CONF` | All-reduce flag | Expected if H1 |
|---|---|---|---|
| default | `expandable_segments:True` | none | BOOT_FAIL at `custom_all_reduce.cuh:164` |
| car-off | `expandable_segments:True` | `--disable-custom-all-reduce` | serves on `PYNCCL` (the recipe's TP=2 config) |
| es-off | `expandable_segments:False` | none | serves on `CUSTOM`; GSM8K matches car-off; lower c=1 TPOT |

Each arm that boots runs GSM8K 100 and benchmarks at c=1, 16, and 64. Not carried over from the `docker run`:
`--ipc=host`, `--shm-size=8g`, the memlock and stack ulimits, and `--restart`. The TP workers are processes in one
container in both setups, so CUDA IPC between them works the same way.

Out of scope: Super VL, the T4 default model (about 62 GB; the repo's TP=2 run with custom all-reduce off reports
109 output tok/s at c=1 for chat_200_200 in `tests/data/20261019_dbb74c8/sizing.yaml`). Its download and cold boot
(1333 s on the repo's host) do not fit an hour.

## Pod

- 2x RTX PRO 6000 Blackwell, PCIe only. Max-Q first (the T4/T6 card, on Vast), then the Workstation and Server
  Editions on Vast, then both on RunPod secure. Not RTX 5090: its driver reports no P2P, so vLLM would turn custom
  all-reduce off before the bug.
- `require_p2p = true`: `harness up` runs `nvidia-smi topo -p2p r` and refuses a pair that reads anything but OK, and
  the topology gate refuses NV#. Each box's matrix is kept in `results/<issue>/p2p-<pod>.txt`.
- Image pinned exactly as the recipe pins it (digest only).
- Hours: stage 1 about 20-25 min with the image pull (`planned_hours = 0.5`); stage 2 about 35 min
  (`planned_hours = 0.75`).

## Stop rules

- The P2P gate refuses the box: take the next choice. A pair without P2P is the recipe's own case, and the code above
  says vLLM then keeps custom all-reduce off; keep the matrix as the record.
- setup's IPC probe fails on a pair the gate passed: evidence for H2. Run `default` and `es-off` only, then stop.
- `default` serves with `CUSTOM`: H1 is dead on this stack. Finish the matrix (H4 still gets its numbers), skip
  stage 2, and re-plan.
- `es-off` faults or its GSM8K drops: H3. Stop before stage 2.
- Terminate as soon as the matrix is done.

## Fix paths, if H1 holds

- Recipe: with TP>1 and custom all-reduce on, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` (or leave it
  unset), keep `NEMOTRON_LIGHTNING_DISABLE_CUSTOM_AR` as an opt-out, and point the comment and the health and
  diagnose hints at the allocator. Check first that `False` does not cost KV headroom at the 200K context: the recipe
  sets `True` for fragmentation (a soak reported in HyperQwen#163 found no loss at 0.93-0.96 memory utilization on
  that model).
- vLLM: the cuMem branch at `cuda_communicator.py:123-125` already copies graph inputs instead of registering them.
  Taking that branch when expandable_segments is on, or checking each captured input for VMM memory as the SGLang
  fix in `issues/sglang-44291` does, would let the recipe keep `True` with custom all-reduce on. vllm#42609 is the
  tracking issue.

## Files

| File | Purpose |
|---|---|
| `README.md` | This plan, then results |
| `harness.toml` | Stage 1 pod and variants |
| `setup.sh` | P2P and IPC records, the allocator probe, model download |
| `vmm_ipc_probe.py` | One `cudaIpcGetMemHandle` on a PyTorch allocation, per allocator setting |
| `../dell-deskside-car-lightning/` | Stage 2: `harness.toml` and `setup.sh` |

## Results (2026-11-13)

H1 holds and H2 is dead on both boxes tried: the recipe's `expandable_segments:True` alone makes custom all-reduce
abort at `custom_all_reduce.cuh:164` on pairs with working P2P, and `expandable_segments:False` serves with custom
all-reduce on. Stage 2 repeats it on the recipe's own model (`../dell-deskside-car-lightning/README.md`).

Box: Vast pod `sm120-dell-deskside-car-86cf9a`, 2x RTX PRO 6000 Blackwell Workstation Edition (600 W, driver
595.91.07), `topo -m` PHB, PCIe Gen4 x16 at most on this host. Max-Q was the first choice in `harness.toml`; the
harness placed the run on the Workstation Edition, so the recipe's 300 W card is not covered. Image: the recipe's pin,
`vllm/vllm-openai@sha256:cb9bc64f0f038c2c2ebf13b31248fb87aca40c356c3d20f363be087cfcff1fcd`; `env.txt` reports vLLM
0.32.0rc1.dev190+g3f1bcfd82, torch 2.14.0+cu130, CUDA 13.0, FlashInfer 0.7.1.

Peer access and the allocator probe, from `jobs/setup.log` (`topo -p2p w` there and `topo -p2p r` in
`p2p-sm120-dell-deskside-car-86cf9a.txt` read OK both ways):

```
PYTORCH_CUDA_ALLOC_CONF in the image env: None
can_device_access_peer 0->1: True 1->0: True
vllm IPC probe 0->1: True 1->0: True
vmm_ipc_probe expandable_segments:True (segment expandable: True): cudaIpcGetMemHandle failed: CUDART error: invalid argument
vmm_ipc_probe expandable_segments:False (segment expandable: False): cudaIpcGetMemHandle ok
```

Output tok/s from `vllm bench serve` (random 1024 in / 512 out; every request completed in every run), mean TPOT in
ms, the `tp:0` all-reduce backends each `server.log` reports, and rank 0's KV cache at boot:

| Variant | Allocator | Backends | Boot | c=1 | c=16 | TPOT c=1 / 16 | GSM8K 50 | KV cache |
|---|---|---|---|---|---|---|---|---|
| default | `True` | `['CUSTOM', 'PYNCCL']` | BOOT_FAIL, 90 s | - | - | - | - | - |
| es-off | `False` | `['CUSTOM', 'PYNCCL']` | 136 s | 583.8 | 5,169.8 | 1.67 / 2.90 | 23 | 83.85 GiB |
| es-off-car-off | `False` | `['PYNCCL']` | 50 s | 554.3 | 4,973.0 | 1.76 / 3.03 | 24 | 83.67 GiB |
| car-off | `True` | `['PYNCCL']` | 50 s | 553.7 | 4,984.7 | 1.76 / 3.02 | 24 | 83.66 GiB |

- default: both workers print `Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:164 'invalid argument'`
  (`server.log:74`, `:77`) right after the first capture pass (51 PIECEWISE and 2 FULL graphs), before any
  `Graph capturing finished` or `Available KV cache memory` line; the engine then logs `RuntimeError: cancelled` from
  `shm_broadcast.py` and `Engine core initialization failed`. In the arms that boot, that first pass is the CUDA graph
  memory profile: it ends before `Available KV cache memory`. This build logs no `Profiling CUDA graph memory` line,
  and the profile pass draws its own capture bars, so the plan's "before any `Capturing CUDA graphs` progress" was
  wrong in detail; the crash point is the one predicted.
- Custom all-reduce against NCCL (es-off against es-off-car-off, allocator held at `False`): +5.3% at c=1 and +4.0% at
  c=16 output tok/s; mean TPOT at c=1 1.67 against 1.76 ms (-5.1%).
- The recipe's two options (es-off against car-off): +5.4% at c=1, +3.7% at c=16.
- The allocator alone (car-off against es-off-car-off, NCCL in both): -0.1% at c=1, +0.2% at c=16, and KV cache within
  0.01 GiB. The gain is custom all-reduce, and `False` costs no KV cache at boot here.
- The plan expected the 0.6B model to overstate Lightning's c=1 gain. It did not: +5.4% here, +5.5% on Lightning.
- es-off's 136 s boot is its second capture pass (`Graph capturing finished in 59 secs`, against 5 s in the NCCL arms).
  Stage 2's es-off booted faster than its car-off (75 s against 101 s), so one rep does not show a registration cost.
- GSM8K is a correctness check only. Qwen3-0.6B reasons by default; 5, 5, and 3 replies hit the 4,096-token cap, and
  all of them scored wrong. GSM8K keeps 64 requests in flight, so batching, and with it rounding, differ run to run:
  even the two NCCL arms share only 17 of 50 reply texts and disagree on 2 questions (same total). 24 questions are
  wrong in all three arms.

Hypotheses:

- H1 confirmed: default fails at `custom_all_reduce.cuh:164`, es-off serves on `['CUSTOM', 'PYNCCL']`, and the probe
  fails under `True` on a segment that reports itself expandable and passes under `False`.
- H2 killed: P2P gate OK, `can_device_access_peer` True both ways, vLLM's IPC probe True both ways, and es-off serves
  with `CUSTOM`. The failing call is the export at `:164`, not an open at `:143` or `custom_all_reduce.cu:193`.
- H3 killed: es-off served all 72 benchmark requests and GSM8K 50 with no fault, and 23 against 24 is one question.
- H4 killed: es-off's c=1 TPOT is 5.1% under es-off-car-off's, past the 2% bar.

Stage 1 met the condition for stage 2 (H1 confirmed, H2 killed), so stage 2 ran on the next box.

Caveats:

- Cloud boxes with working P2P: 600 W Workstation (here) and Server (stage 2) Editions, native Linux, in a container.
  Dell's towers (Max-Q, 300 W) were not tested, and their P2P status and topology are unknown to us. A pair whose
  driver reports no P2P never turns custom all-reduce on (`custom_all_reduce.py:277-292`), so on such a tower neither
  the crash nor the gain applies.
- One rep per arm. The two NCCL arms, which differ only in the allocator, land within 0.2% of each other; that is the
  only noise reading.
- KV cache at boot only. The fragmentation the recipe sets `True` against would show over a long run, which this kit
  did not do.
