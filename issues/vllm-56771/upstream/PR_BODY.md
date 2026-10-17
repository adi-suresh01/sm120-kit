<!-- markdownlint-disable -->

## Overview

When a safetensors checkpoint is larger than 90% of available RAM, `safetensors_weights_iterator` now evicts each shard from the page cache (`posix_fadvise(DONTNEED)`) once the caller has released its tensors, so a load keeps about one shard cached instead of the whole checkpoint. For DeepSeek-V4.1 this builds on #58857: on main the VL wrapper sorts the full weight stream, which keeps every shard mapped until loading ends, and mapped pages cannot be evicted. Refs #57469, #58625.

## Claims

- Default mmap path only (`--safetensors-load-strategy` unset or `lazy`), Linux only, and only when the checkpoint exceeds 90% of available RAM, the same rule that already turns off auto-prefetch. Checkpoints that fit, and eager, torchao and prefetch loads, behave as before. One info line when it is on.
- A shard is evicted only after the last tensor handed out from it is gone. A caller that holds tensors gets no eviction for those shards.
- DeepSeek-V4.1-Flash at TP4 in a 702 GiB container: page cache stayed at 6.2 to 12.9 GiB (median 9.6) for the first 13 minutes of a 475 GiB load and peaked at the largest shard, 92.1 GiB, instead of growing with the checkpoint. That run also had the same streaming change as #58857 and other fixes applied; details below.

## Validation

**Unit test.** `pytest tests/model_executor/test_weight_utils.py -k "evicts_released_shards or drop_checkpoint_cache"`: 5 passed. The new test writes a 3-shard checkpoint and mocks `os.posix_fadvise` to record which shard it was called on and to assert that every tensor from that shard is already gone.

| Case | Shards evicted |
|---|---|
| caller streams, holding only the current tensor | 1 and 2 (3 is still referenced when the iterator ends) |
| caller holds every tensor, like `sorted()` | none |
| checkpoint fits in RAM | none |

Each case fails when its part of the change is removed (the whole change, the weakref check, the RAM gate). The test is skipped off Linux; I ran it on macOS with `sys.platform` patched to `linux` and a stub `posix_fadvise`, so CI on Linux is its first run as committed. The rest of `test_weight_utils.py` and `model_loader/test_ep_weight_filter.py` pass locally (95 passed, 1 skipped; one test errors at setup because the local run did not load `tests/conftest.py`). `ruff check` and `ruff format` are clean, and mypy reports no new errors in `weight_utils.py`.

**GPU.** 4x RTX PRO 6000 Blackwell Server Edition (sm_120, PCIe), RunPod container with a 702.2 GiB memory limit, `deepseek-ai/DeepSeek-V4.1-Flash` (48 shards, 475.25 GiB), `--tensor-parallel-size 4 --engram-config.cpu_offload true`, DSpark with 5 speculative tokens, CUDA graphs. Both runs on nightly `49cb2bbf` plus #60762 (needed to boot DSv4.1 on sm_120). cgroup memory sampled every 5 s.

| | Before | After |
|---|---|---|
| Extra code | none | the same wrapper change as #58857 (my own overlay of it, without a `clone()`), this PR's eviction, and a `drop_checkpoint_cache` call after the VL wrapper's load (not in this PR) |
| Env | | `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:64`, so torch pins the Engram tables at exact size (the problem #59849 addresses in vLLM) |
| Checkpoint on | network FUSE volume | local overlay disk |
| Load | progress bar finished in 7m35s, then memory climbed about 9 GiB/min with workers in D state | 814 s main weights, 302 s DSpark draft pass, ready in 1339 s |
| Peak container memory | 695.1 GiB, then RunPod stopped the container at 98% of its limit | 290.6 GiB: anon 8.2, shmem 189.2 (pinned Engram tables), other page cache 92.1 |
| Served | never | 10.7K and 106K-token prompts |

The after run is not an isolated test of this PR, since it also changed the wrapper, the pinned allocation size and the disk. What it shows for this change is the cache during the load. The container sat at 204.6 to 211.4 GiB of 702.2 GiB for the whole stream, so the cgroup was never near its limit, yet page cache outside the Engram shmem stayed at 6.2 to 12.9 GiB until the last 25 s. There page cache reached 80.9 and then 92.1 GiB as the two Engram table shards came through. The iterator evicted the first (80.9 to 15.9 GiB by the next sample); the last one is the shard this PR leaves to the kernel, and the post-load `drop_checkpoint_cache` released it (92.1 to 6.2 GiB within 10 s). vLLM logged `Checkpoint size: 475.25 GiB. Available RAM: 499.62 GiB.`, so the 90% gate here would have been on; the code in that run evicted unconditionally.

## Details

**Why.** The default loader reads every shard through an mmap. Once a shard's tensors are on the GPU its pages are clean page cache, and that cache is charged to the container's cgroup. The kernel does not reclaim it until the cgroup reaches its limit or the host runs short, but some runtimes act earlier: RunPod stops a container at 98% of its memory limit, which is how the before run ended. With a checkpoint this size next to large pinned host allocations (189 GiB of Engram tables at TP4 here), the cache alone can carry a container to that line.

**Gate.** It reuses the existing 90%-of-available-RAM rule, so a checkpoint that fits stays cached for the next load, which was the objection to the unconditional release in #59527. Available RAM is read once when the iterator starts, honoring cgroup limits through `_get_available_ram_bytes`.

**Why the last tensor.** Every tensor from a shard keeps the shard mapped, and `DONTNEED` skips mapped pages, so evicting earlier does nothing. Loaders release tensors in order, and only once the next one reaches them, so the iterator keeps a weakref to the last tensor it handed out from each shard and checks them after every yield. If a caller holds tensors, as the DSv4.1 VL wrapper's `sorted()` does on main, those shards are never evicted; that is why DSv4.1 needs #58857 to benefit.

**Not changed here.**

- The last shard is still referenced when the iterator returns, so it stays cached until the kernel needs the memory. For the DSv4.1 VL wrapper a `drop_checkpoint_cache` after its load releases it; that sits on top of #58857's change to the same function, so I left it out of this PR.
- With several TP ranks on one host, a rank that finishes a shard first can evict pages a slower rank has not read yet, and that rank reads them from disk again. This only happens when the checkpoint does not fit. I did not measure load time with and without eviction, so I make no claim about load time.
- `multi_thread_safetensors_weights_iterator`, fastsafetensors, RunAI and instanttensor loading are unchanged. #58625 (opt-in release after loading) is complementary: it frees the cache after the load, this keeps it bounded during the load.
- `_drop_page_cache` returns early off Linux, which also clears the two macOS mypy errors in `drop_checkpoint_cache` that #61836 fixes.
- Not a fix for #57469: that report's remaining failure is pinned host memory (shmem-rss), which is #59849's area.

**AI assistance.** The code, test and this description were drafted with Claude (the commit carries `Assisted-by: Claude`). I reviewed the full diff, and the GPU runs above are mine.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>** (anything written below this line will be removed by GitHub Actions)
