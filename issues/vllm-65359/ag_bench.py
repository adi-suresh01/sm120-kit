"""Lei-010's all-gather bench from vllm#65359 (issuecomment-6997879140), kept as written, plus what answers the
question it left open: whether the copy-engine variant, which FlashInfer 0.7.3 admits only on an ordered 4+4 TP8
placement, closes the large-payload gap measured on a host where recursive doubling is the only TP8 variant.

All phases share one workspace, so every rank issues the same collective sequence:
  seed   Lei's loop: NCCL all_gather (list), FlashInfer raw, FlashInfer plus vLLM's post copy, the copy alone; plus
         all_gather_into_tensor, the NCCL call FlashInfer's tuner checks against. FlashInfer runs the policy seed.
  sweep  every launchable config per shape, checked bit for bit against NCCL, then timed with Lei's bench() at a call
         count sized to about 0.5 s per config.
  tune   FlashInfer's own tuner at tune_batches 18/144/288, Lei's row counts (the default 1..512 misses all three).
  tuned  Lei's loop again, on the tuner's winners.
The sweep uses two private 0.7.3 names (_TUNING_SPEC, candidate_configs) so it enumerates exactly what tune() would.
Rank 0 appends one JSON object per row to <out>/<arm>.jsonl and flushes each, so a hang keeps every row before it.
Usage: torchrun --standalone --nproc_per_node=8 ag_bench.py --arm <name> --out <dir> [--phases seed,sweep,tune,tuned]
"""

import argparse
import faulthandler
import json
import os
import time
from importlib.metadata import version

import torch
import torch.distributed as dist

HIDDENS = (256, 5120)
CONCURRENCIES = (1, 8, 16)
K = 6  # dspark k=5 + 1
COLS = 3
TUNE_BATCHES = tuple(c * K * COLS for c in CONCURRENCIES)
MAX_NUMEL = 8 * 1024 * 1024
# A phase that runs longer is hung: every rank dumps its stack and exits, inside setup.sh's outer limit.
PHASE_LIMIT_S = 600


def bench(fn, iters=300, warmup=60):
    """Lei's timing, unchanged: wall time per call on this rank, in us."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def across_ranks(value, op):
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=op)
    return t.item()


def timed(fn):
    """Rank 0's time (what Lei's table prints) and the slowest rank's (what FlashInfer's tuner ranks by)."""
    us = bench(fn)
    return {"rank0": round(us, 2), "slowest": round(across_ranks(us, dist.ReduceOp.MAX), 2)}


def sweep_timed(fn):
    """Ranks one config of the sweep. Small-block configs at 2.9 MB can take tens of ms per call, so the call count
    is sized to about 0.5 s; every rank must issue the same number of collectives, so the count comes from the
    slowest rank's probe."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    per_call_s = across_ranks((time.perf_counter() - t0) / 5, dist.ReduceOp.MAX)
    iters = max(20, min(300, int(0.5 / per_call_s)))
    us = bench(fn, iters=iters, warmup=iters // 5)
    return {"rank0": round(us, 2), "slowest": round(across_ranks(us, dist.ReduceOp.MAX), 2), "iters": iters}


def describe(config):
    return [config.variant.name, config.blocks, config.threads]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--phases", default="seed,sweep,tune,tuned")
    args = parser.parse_args()
    phases = args.phases.split(",")

    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    dev = torch.device("cuda", local)

    stacks = open(os.path.join(args.out, f"{args.arm}.rank{rank}.stacks"), "w")
    faulthandler.enable(file=stacks)
    rows_file = open(os.path.join(args.out, f"{args.arm}.jsonl"), "a") if rank == 0 else None

    def phase(name, limit_s=PHASE_LIMIT_S):
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(limit_s, exit=True, file=stacks)
        if rank == 0:
            print(f"{time.strftime('%H:%M:%S')} phase {name}", flush=True)

    def write(**row):
        if rows_file is not None:
            rows_file.write(json.dumps({"arm": args.arm, **row}) + "\n")
            rows_file.flush()

    import flashinfer.comm as fc
    from flashinfer.comm.pcie_ipc_collectives._ag_rs_tuning import candidate_configs
    from flashinfer.comm.pcie_ipc_collectives.all_gather import _TUNING_SPEC

    # The tuner loads this file when the workspace is built, so a leftover from an earlier attempt would turn the seed
    # phase into a tuned one.
    tune_cache = os.path.join(args.out, f"{args.arm}.tune.json")
    if rank == 0 and os.path.exists(tune_cache):
        os.remove(tune_cache)
    dist.barrier()

    phase("workspace")
    ag_ws = fc.PcieIpcAllGatherWorkspace(
        group=None, max_numel=MAX_NUMEL, dtype=torch.bfloat16, tune_batches=TUNE_BATCHES, tune_cache=tune_cache
    )
    ag_ws.rebind_stream()
    write(
        phase="topology",
        ordered_4plus4=ag_ws.ordered_4plus4,
        reason=ag_ws.ordered_4plus4_reason,
        fingerprint=ag_ws.placement_fingerprint,
        max_blocks=ag_ws.max_blocks,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        nccl_env={k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
        device=torch.cuda.get_device_name(local),
        torch=torch.__version__,
        nccl=".".join(map(str, torch.cuda.nccl.version())),
        flashinfer=version("flashinfer-python"),
    )
    if rank == 0:
        print(f"[topology] ordered_4plus4={ag_ws.ordered_4plus4} reason={ag_ws.ordered_4plus4_reason!r} "
              f"fingerprint={ag_ws.placement_fingerprint}", flush=True)

    def shapes():
        for hidden in HIDDENS:
            for c in CONCURRENCIES:
                rows = c * K * COLS
                inp = torch.randn(rows, hidden, dtype=torch.bfloat16, device=dev)
                yield hidden, c, inp, inp.reshape(-1, inp.shape[-1])

    def reference(flat):
        ref = torch.empty((flat.shape[0] * ws, flat.shape[1]), dtype=flat.dtype, device=dev)
        dist.all_gather_into_tensor(ref, flat)
        return ref

    def matches_everywhere(flat, ref, config=None):
        """Bit-exact on every rank; a config that raises on any rank fails on all of them, as in the tuner."""
        try:
            failed = int(not torch.equal(ag_ws.all_gather(flat, config=config), ref))
        except Exception:  # noqa: BLE001 - one rank raising must not desync the others
            failed = 1
        return across_ranks(failed, dist.ReduceOp.MAX) == 0

    def lei_loop(name):
        phase(name)
        for hidden, c, inp, flat in shapes():
            ref = reference(flat)
            out_nccl = [torch.empty_like(flat) for _ in range(ws)]

            def a():
                dist.all_gather(out_nccl, flat)

            def a_into():
                dist.all_gather_into_tensor(ref, flat)

            def b():
                ag_ws.all_gather(flat)

            def c_fn():
                out_flat = ag_ws.all_gather(flat)
                o = out_flat.reshape((ws,) + inp.size())
                o = o.movedim(0, 1)
                return o.reshape((inp.size()[0] * ws, inp.size()[1]))

            def d():
                o = flat.repeat(ws, 1)
                return o.reshape((ws,) + inp.size()).movedim(0, 1).reshape((inp.size()[0] * ws, inp.size()[1]))

            ok = matches_everywhere(flat, ref)
            times = {"nccl": timed(a), "nccl_into": timed(a_into), "fi": timed(b), "fi_post": timed(c_fn),
                     "copy": timed(d)}
            payload_kb = inp.numel() * 2 / 1024
            config = ag_ws.tuned_launch_config(flat)
            write(phase=name, hidden=hidden, c=c, rows=inp.shape[0], payload_kb=payload_kb, config=describe(config),
                  fi_matches_nccl=ok, us=times)
            if rank == 0:
                r0 = {k: v["rank0"] for k, v in times.items()}
                print(f"hidden={hidden:5d} c={c:2d} rows={inp.shape[0]:4d} payload={payload_kb:7.1f}KB | "
                      f"NCCL {r0['nccl']:7.1f}us  FI-raw {r0['fi']:7.1f}us  FI+post {r0['fi_post']:7.1f}us  "
                      f"copy-only {r0['copy']:7.1f}us | FI/NCCL {r0['fi'] / r0['nccl']:5.2f}x  "
                      f"post-overhead {r0['fi_post'] - r0['fi']:6.1f}us | {describe(config)} match={ok}",
                      flush=True)
            dist.barrier()

    if "seed" in phases:
        lei_loop("seed")

    if "sweep" in phases:
        phase("sweep")
        for hidden, c, inp, flat in shapes():
            ref = reference(flat)
            configs = candidate_configs(
                _TUNING_SPEC,
                world_size=ws,
                shard_numel=flat.numel(),
                element_size=flat.element_size(),
                max_blocks=ag_ws.max_blocks,
                ordered_4plus4=ag_ws.ordered_4plus4,
            )
            for config in configs:
                ok = matches_everywhere(flat, ref, config)
                times = sweep_timed(lambda: ag_ws.all_gather(flat, config=config)) if ok else None
                write(phase="sweep", hidden=hidden, c=c, rows=inp.shape[0], config=describe(config),
                      fi_matches_nccl=ok, us=times)
            if rank == 0:
                print(f"sweep hidden={hidden} rows={inp.shape[0]}: {len(configs)} configs", flush=True)
            dist.barrier()

    if "tune" in phases:
        phase("tune")
        start = time.perf_counter()
        winners = ag_ws.tune(HIDDENS)
        write(phase="tune", seconds=round(time.perf_counter() - start, 1),
              winners={f"{hidden}x{batch}": describe(config) for (hidden, batch), config in winners.items()})

    if "tuned" in phases:
        lei_loop("tuned")

    # Every row is on disk by now. NCCL teardown has hung on this image family before (issues/pcie-tp8-allreduce:
    # destroy_process_group sat until the outer limit), so it gets a short limit instead of the arm's.
    phase("teardown", limit_s=90)
    ag_ws.destroy()
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
