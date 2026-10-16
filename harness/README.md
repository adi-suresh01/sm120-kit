# harness

Rents a RunPod pod for one issue, runs server variants on it, pulls the results, and terminates it, inside a GPU
budget Aditya approved for that issue.

## One-time setup

1. RunPod key: see `docs/runpod.md` (a Restricted key in the macOS Keychain, exported as `RUNPOD_API_KEY` by
   `~/.zshrc`). The harness reads it from the environment and never writes or prints it.
2. SSH: the key registered with RunPod must work without a prompt, because ssh runs with `BatchMode=yes`. Either
   load it into ssh-agent (`ssh-add -l` lists it), or set `HARNESS_SSH_KEY` to its private key file; then ssh
   offers only that key (`IdentitiesOnly=yes`), and a key file without a passphrase needs no agent. `up` checks
   this before any money moves. ssh never forwards the agent to the pod.
3. Install from the repo root: `cargo install --path harness --locked`. Run `harness` from anywhere inside the
   repo.
4. Tests need bash 4.4 or newer on `PATH` (`brew install bash` on macOS), python3, curl, and patch.

## An issue folder

```
issues/<repo>-<num>/
  harness.toml        what to rent and what to run (below)
  setup.sh            optional: installs and downloads; runs once per pod, after env_report.sh
  hooks.sh            optional: defines pre_launch and post_stop, each called with the run directory
  patches/<name>.diff patches a variant can apply to an installed Python package
  harness-ledger.json written by the harness: budget, pods, cost, RunPod organization. Commit it.
```

`harness.toml`:

```toml
issue = "vllm-47990"           # must match the folder name
planned_hours = 2.0             # the deadline never exceeds this

[pod]
# pinned by digest
image = "vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
count = 1                       # GPUs on the pod
min_cuda = "13.0"
container_disk_gb = 40
volume_gb = 80                  # /workspace; at least 10; deleted with the pod
require_pcie = false            # default: true when count > 1; refuses anything but a PCIe-only matrix
needs_hf_token = false          # passes the RunPod secret hf_token as HF_TOKEN
apt = ["tmux"]                  # extra packages; openssh-server, curl, procps, and patch are always installed
gpus = [                        # preference order; the first in stock wins, a refused create falls through
  { id = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", cloud = "COMMUNITY" },
  { id = "NVIDIA RTX PRO 6000 Blackwell Server Edition", cloud = "SECURE" },
]

[server]
package = "vllm"                # vllm or sglang: the package served and cleaned up after
model = "nvidia/Qwen3.6-35B-A3B-NVFP4"   # the model name the smoke request sends
launch = ["vllm", "serve", "nvidia/Qwen3.6-35B-A3B-NVFP4"]
base_flags = ["--max-model-len", "262144"]   # every variant gets these; --host and --port are added
port = 8000
fail_pattern = "illegal memory access|cudaErrorIllegalAddress"   # ends a boot early as BOOT_FAIL
boot_timeout_s = 900
patch_strip = 1                 # default patch -p level, relative to the directory holding the package

[variants.baseline-graphs]
flags = ["--moe-backend", "flashinfer_b12x"]

[variants.fixed-graphs-31g]
flags = ["--moe-backend", "flashinfer_b12x", "--gpu-memory-utilization", "0.303"]
patches = [
  "pr60069-b12x-shared-workspace",                 # patches/<name>.diff on server.package
  { name = "fi4827-graph-pinning", package = "flashinfer", strip = 1 },   # any installed Python package
]
env = { MEM_BUDGET_GIB = "31.45" }               # exported for the hooks and the server
smoke_concurrency = 13                           # parallel smoke requests
```

With `package = "sglang"`, a variant can also measure quality and speed on the live server: GSM8K first (its
accept length is the per-request mean), then one `bench_serving` run per `[concurrency, prompts]` pair:

```toml
[variants.a6.quality]
gsm8k = 200
bench = [[1, 8], [16, 64], [64, 256]]
```

Unknown keys are errors, and every named patch must exist. Variants run in file order when `run` names none.
Before every variant the pod restores each package any variant of the issue patches, so patches never leak from
one variant into the next.

## Commands

| Command | Does |
|---|---|
| `harness budget set <issue> <usd> --note <text>` | Records Aditya's approval. Nothing rents without it |
| `harness budget show <issue>` | Spent, reserved, remaining; upgrades estimates to RunPod billing records |
| `harness quote <issue>` | First in-stock choice, $/hr, the deadline `up` would set, worst-case cost; works before `budget set` |
| `harness up <issue> [--no-setup]` | Reserve, create, wait for SSH, topology gate, copy the kit, `env_report.sh`, `setup.sh` |
| `harness exec <issue> <variant>...` | Copies the issue folder again, runs variants on the live pod, pulls each one's results |
| `harness pull <issue>` | Pulls `results/<issue>/` from the pod |
| `harness down <issue>` | Pulls (at most 5 min), terminates, books the cost |
| `harness run <issue> [<variant>...]` | `up`, `exec`, `down`; `down` runs on errors, Ctrl-C, SIGTERM, and SIGHUP too |
| `harness reap` | Settles every ledger against RunPod; every command runs it first |
| `harness ssh <issue> -- <cmd>` | Runs a command on the pod in `/workspace/sm120-kit` with the container env; killed after 15 min |

Exit status is non-zero only for harness errors. A server that crashes is a verdict (`BOOT_FAIL`), not an error.
An error after the pod exists names the pod, its deadline in local time, and `harness down <issue>`.

## What keeps a pod from billing past its budget

1. **Ledger.** `up` writes a reservation (rate x 1.10 x deadline) before it calls create, and refuses a deadline
   under 15 minutes. One live pod per issue. It also refuses to reserve while RunPod lists a pod named for the
   issue that the ledger does not track (another checkout, or a ledger reverted by git), or while the laptop clock
   is more than 2 minutes off RunPod's.
2. **Watchdog on the pod.** The entrypoint runs it as PID 1 on the absolute `HARNESS_DEADLINE_EPOCH`, so a
   container restart cannot extend it. At the deadline it tries `DELETE /v2/pods/$RUNPOD_POD_ID` with the pod's
   own key, then `runpodctl pod remove`, then `runpodctl pod stop`, and keeps retrying; each attempt goes to
   `/workspace/harness-watchdog.log`. `up` ends a pod at once when that log does not show the watchdog armed as
   PID 1 for the ledger's deadline (an image or API that skipped the entrypoint).
3. **Reaper**, first in every command: deletes a pod still listed 2 minutes past its deadline (booking at most
   its reservation), books a pod RunPod no longer lists once a GET agrees, adopts a reservation whose pod
   appeared, releases one whose pod never did (10 minutes), and warns about any `sm120-*` pod no ledger tracks
   without touching it.
4. **Prepaid RunPod credit** is the ceiling under all of these.

A 404 from RunPod never books a pod as gone on its own, since RunPod also answers 404 for a pod the key cannot
see. The ledger records the key's RunPod organization (`GET /v2/organization`) on its first write, and every
command refuses a key of another organization, except `down` and `reap`: they warn and still send the delete,
since a terminate never sent costs more, and the ledger write after it still refuses to book the pod as gone.
That endpoint needs `membership:read` and answers 404 for an account outside any organization; on a 401, 403, or
404 the check is skipped, and nothing is recorded.

## Watching a run

Every event is one JSON line in `results/<issue>/events.jsonl`, mirrored to stdout. Events that belong to no
issue (for example an `sm120-*` pod whose name matches no issue folder) go to `results/harness-events.jsonl`.

```bash
tail -F results/vllm-47990/events.jsonl | grep --line-buffered -oE '"kind":"(phase|verdict|error|warn|terminate)".*'
```

Kinds: `budget, quote, reserve, create, ready, setup, variant_start, phase, verdict, pull, terminate, reap, warn,
error`. A `phase` line carries `variant, phase, detail, t` as `serve_variant.sh` reports each step. A verdict line
carries `variant, verdict (OK|BOOT_FAIL|SMOKE_FAIL|POST_FAIL|TIMEOUT), boot_s, fault, where, gsm8k, accept,
tok_s, result`. An `error` line for a pod that is still up adds `pod, id, deadline, deadline_local, next`.

## On the pod

The kit lands in `/workspace/sm120-kit` (only `harness/pod/` and the issue folder), `HF_HOME=/workspace/hf`, and
results in `/workspace/sm120-kit/results/<issue>/`: `env.txt`, `jobs/<job>.log` (plus `.pid` and `.exit`),
`<variant>/` (server log, responses, quality output, `progress.jsonl`, `verdict.json`), and `verdicts.jsonl`.
After each variant the harness pulls that variant's folder, `jobs/`, and `verdicts.jsonl`.

## Tests

```bash
cd harness && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
shellcheck pod/*.sh tests/pod/*.sh
```

`cargo test` also runs the bash tests in `tests/pod/` (watchdog, package snapshot, `serve_variant.sh` against a
fake server, and stage 2 log replay) and the job start and poll scripts in local bash.
