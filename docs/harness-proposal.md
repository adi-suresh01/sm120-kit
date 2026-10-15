# Repro harness proposal (after vllm#49476, sglang#44291 stage 1 and stage 2)

Proposal only. Nothing here is built until Aditya agrees.

## Steps that repeated in all three runs

| Step | vllm#49476 | #44291 stage 1 | #44291 stage 2 | Where it lives today |
|---|---|---|---|---|
| Live stock/price read, approval | yes | yes | yes | plan text |
| Pod create with an sshd entrypoint | yes | yes | yes | plan text (JSON pasted per plan) |
| Container env for SSH sessions (venv image) | - | `/proc/1/environ`, by hand | `export -p` in entrypoint | plan text |
| Kit copy over ssh + tar | yes | yes | yes | plan text |
| Topology gate + `env_report.sh` | yes | yes | yes | `scripts/env_report.sh` |
| Model download to the volume at a pinned revision | yes | yes | yes | per-issue `setup.sh` |
| Toggle upstream patches on the installed package | yes | yes | yes | `scripts/patch_pkg.sh`; stage 2 added a pristine-copy restore |
| One server variant per call: boot, smoke, verdict row | yes | yes | yes | per-issue `run.sh` / `stage2_run.sh` |
| GSM8K + accept length + bench_serving on the live server | - | partial | yes | `stage2_quality.sh` via `POST_SMOKE` |
| Monitor a log for verdict lines | yes | yes | yes | ad hoc (two monitor bugs this time) |
| Pull results, terminate, cost | yes | yes | yes | plan text |

## Proposed pieces

1. `scripts/pod_session.sh`: `ssh`, `copy`, `pull`, `env` subcommands keyed on `POD_IP`/`POD_PORT`, so every pod
   command stops repeating the SSH array, known-hosts file, tar flags (`--no-xattrs` on macOS), and the
   `container_env.sh` source.
2. A checked-in pod spec (`docs/pod-spec.json`) with the sshd + `export -p` entrypoint and the apt list, so plans
   reference it instead of pasting JSON.
3. `scripts/pkg_snapshot.sh save|restore <package>`: the pristine-copy reset from `stage2_run.sh`, generalized,
   used together with `patch_pkg.sh`.
4. `scripts/serve_variant.sh <variants.sh> <name>`: the generic half of `stage2_run.sh` (restore, apply patches,
   wait for idle GPUs, boot with `FAIL_PATTERN`, summary row, process cleanup). Each issue keeps only a small
   variants file mapping names to flags, patches and env.
5. Move `stage2_quality.sh` to `scripts/quality.sh` with the model and protocol as arguments.
6. `scripts/watch_verdicts.sh <log...>`: the Monitor pipeline that worked (`tail -F | grep --line-buffered -oE`),
   so no more `tr` or `cut` in a monitor.

Expected effect: the next issue's local prep shrinks from 7 tasks to writing a variants file and a setup script,
and the pod procedure becomes four commands. Estimated effort: one evening, no GPU.
