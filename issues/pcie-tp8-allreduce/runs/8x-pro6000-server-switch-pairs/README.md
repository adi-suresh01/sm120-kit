# 2026-11-10, 8x RTX PRO 6000 Server Edition, two GPUs per PCIe switch

`run_sweep.sh --hidden 6144` from commit 09d1e88, run unedited on an owned (not rented) box.

## Box

- 8x NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 615.71.10, 600 W power limit (default), CUDA 13.0.2 (image)
- Dual-socket AMD EPYC 9965 (192 cores each), 2 NUMA nodes
- Two GPUs per PCIe switch, four per socket. `nvidia-smi topo -m`:

```
      GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  NUMA
GPU0   X    PIX   NODE  NODE  SYS   SYS   SYS   SYS   0
GPU1  PIX    X    NODE  NODE  SYS   SYS   SYS   SYS   0
GPU2  NODE  NODE   X    PIX   SYS   SYS   SYS   SYS   0
GPU3  NODE  NODE  PIX    X    SYS   SYS   SYS   SYS   0
GPU4  SYS   SYS   SYS   SYS    X    PIX   NODE  NODE  1
GPU5  SYS   SYS   SYS   SYS   PIX    X    NODE  NODE  1
GPU6  SYS   SYS   SYS   SYS   NODE  NODE   X    PIX   1
GPU7  SYS   SYS   SYS   SYS   NODE  NODE  PIX    X    1
```

## Run

- Image `vllm/vllm-openai:v0.33.0`; torch NCCL 2.30.7, pynccl 2.31.2, FlashInfer 0.7.2.post1
- `--hidden 6144`; NCCL arm at `NCCL_P2P_LEVEL=SYS` (the runner's default level)
- Command: the `docker run ... run_sweep.sh --hidden 6144` line from the issue README, unchanged

## Result

| Step | Exit | Seconds |
|---|---|---|
| probe | 0 | 16 |
| nccl-SYS | 124 | 421 |
| fi-pcie-ipc | 0 | 30 |

`nccl-SYS` finished all 20 pynccl and 20 torch sizes, then hung in `destroy_process_group` until its limit (last size
logged 19:37:16, teardown 19:37:18, SIGTERM 19:43:41), so its rows are complete.

- vLLM left alone picks pynccl. Custom all-reduce is off (more than 2 PCIe GPUs), symmetric memory is off (12.0), and FlashInfer PCIe IPC is available but opt-in.
- FlashInfer chose profile `pcieswitch-pairs` (init 13.9 s), not the `rootcplx-noswitch` of the 2026-11-05 box.
- NCCL connected every pair over P2P (32 channels).
- 0 wrong values: pynccl 94.3B, torch 93.8B, fi_pcie_ipc 40.4B checked.

Microseconds per all-reduce (slowest GPU's median, CUDA graphs), batch = tokens at hidden 6144:

| Size | Batch | NCCL (SYS) | FlashInfer PCIe IPC | FI/NCCL |
|---|---|---|---|---|
| 12 KiB | 1 | 21.0 | 5.8 | 0.28 |
| 24 KiB | 2 | 21.4 | 7.1 | 0.33 |
| 48 KiB | 4 | 22.9 | 12.9 | 0.56 |
| 96 KiB | 8 | 27.6 | 15.5 | 0.56 |
| 192 KiB | 16 | 43.1 | 22.7 | 0.53 |
| 384 KiB | 32 | 78.2 | 39.0 | 0.50 |
| 768 KiB | 64 | 97.0 | 72.6 | 0.75 |
| 1536 KiB | 128 | 174.2 | 144.3 | 0.83 |
| 3072 KiB | 256 | 318.1 | 296.0 | 0.93 |

The full NCCL column (4 KiB to 4 MiB) is in `table.txt`.

## Files

`table.txt`, `rows.jsonl`, `runs.tsv`, the three step logs (`probe.log`, `nccl-SYS.log`, `fi-pcie-ipc.log`),
`<step>.fault.rank<N>.txt`, `nccl-SYS.nccl.host.<pid>.log` and `fi_pcie_ipc_tune.json`, as the runner wrote them,
except for these edits: the machine's hostname is replaced with `host`, and the one GPU UUID (in
`profile_reason`) is replaced with `GPU-<redacted>`.
