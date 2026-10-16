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
  hooks.sh            optional: defines pre_launch, post_smoke, and post_stop (below)
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

`avoid_data_centers = ["EUR-IS-2"]` under `[pod]` keeps the issue's pods out of the named RunPod data centers, for
hosts where `setup.sh` cannot finish (see `docs/runpod.md` on host network speed). Ids must have RunPod's shape
(uppercase letters, digits, and dashes, ending in a digit), since they match exactly. Every `gpus` choice must then
be SECURE, since the catalog lists data centers for secure stock only. A choice counts as in stock only where a data
center outside the list has it, and the create names those data centers in `dataCenterIds`. The v2 API schema calls
that list preferred data centers, so `up` also terminates, before setup, a pod that RunPod reports in an avoided
data center. `quote` shows the data centers each choice may use, and warns about an avoided id the catalog lists for
no data center of the first choice's GPU. Empty, the default, sends no list and leaves placement to RunPod.

A variant can also measure quality and speed on the live server: GSM8K on the first `gsm8k` test questions, then one
benchmark run per `[concurrency, prompts]` pair, random prompts of 1024 tokens in and 512 out:

```toml
[variants.a6.quality]
gsm8k = 200
bench = [[1, 8], [16, 64], [64, 256]]
```

- `sglang`: its own `run_eval` (5-shot, 16k token budget) and `bench_serving`. The verdict also carries the
  speculative accept length, the per-request mean over GSM8K.
- `vllm`: `pod/gsm8k_chat.py` and `vllm bench serve`. GSM8K is 5-shot from the first 5 train examples, greedy,
  through `/v1/chat/completions`, scored on the last number in each reply, with every reply kept in
  `<variant>/gsm8k_responses.jsonl`; the data is openai/grade-school-math at a pinned commit, checked against pinned
  sha256s and cached in `/workspace/gsm8k`. `max_tokens` is the smaller of `GSM8K_MAX_TOKENS` (default 4096) and
  `max_model_len - 2048` from `/v1/models`, and under 256 fails the run: vLLM refuses a prompt plus `max_tokens`
  over `max_model_len`, and a model generating garbage runs every reply to the cap. Each reply is written as it
  lands. A request that times out (after 600 s plus 0.25 s per `max_tokens`, or `GSM8K_TIMEOUT_S`) scores as wrong
  and is counted (a model that never stops ends on `max_tokens`, so a timeout means a slow server; read
  `timed_out` next to the score), and a run where every request timed out has no score; one that fails (a 5xx or
  dropped connection after 3 tries, any other error at once) fails the run. The verdict's `gsm8k_counts` carries the
  counts behind the score: `hit_max_tokens` (with the stop on `Question:` a healthy reply ends on its own, so this
  flags garbage even after a right answer), `no_number`, `empty_content` (a reasoning model cut off mid-thought;
  its reasoning is kept in the row but never scored), `timed_out`, `failed`, and `retried`. `gsm8k` is at most
  1319, and the benchmark loads its tokenizer from `server.model`, so that has to be the checkpoint's id or path,
  not a `--served-model-name` alias. `vllm bench serve` has no tokenizer revision flag at v0.25.1 or v0.31.0, so for
  a server pinned with `--revision` the prompts still come from the main revision's tokenizer; both versions then
  trim any prompt that the server's own tokenizer reads as longer than 1024 tokens.

Either way a GSM8K request that failed, or a benchmark that completed fewer requests than it sent, is `POST_FAIL`,
not a low score or a slow run.

Quality and `post_smoke` (below) have to finish inside `serve_variant.sh`'s outer limit of
`server.boot_timeout_s + 3600` seconds, which also covers the boot and the smoke test; a run still going at that
point ends as `TIMEOUT`. A model generating garbage runs every GSM8K reply to `max_tokens` (4096 by default), which
is slow, so keep `gsm8k` around 200 or lower on a variant suspected of it.

`hooks.sh` is sourced before each variant and may define any of:

- `pre_launch <run_dir>`: before the server starts.
- `post_smoke <run_dir> <port> <model>`: on the live server after a passing smoke test, after quality when the
  variant has it, for the issue's own requests (a reporter's exact prompt, long-context prefills). The watchdog
  that covers quality covers it too: the server exiting, a `fail_pattern` line in its log, or 15 quiet minutes stop
  it as `POST_FAIL`, and so does a non-zero exit. It still runs after a failed quality; either failing is
  `POST_FAIL`. A `hook_start` phase line marks where it begins, so a hook can name its own phases freely.
- `post_stop <run_dir>`: after the server is killed, on every way out.

`pre_launch` and `post_smoke` see the variant's `env`, the variant file's variables (`VARIANT`, `PORT`, `MODEL`,
...), and `progress <phase> [detail]`, which adds a phase line to the run's progress; both stop at their first
failing command.

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
2. **Watchdog on the pod.** The entrypoint runs it in the container's main process (on RunPod, the only child of
   docker-init, whose exit ends the container) on the absolute `HARNESS_DEADLINE_EPOCH`, so a container restart
   cannot extend it. At the deadline it tries `DELETE /v2/pods/$RUNPOD_POD_ID` with the pod's own key, then
   `runpodctl remove pod` and `stop pod` (the 1.x that pods ship, and the 2.x spellings), and keeps retrying; each
   attempt goes to `/workspace/harness-watchdog.log`. `up` ends a pod at once when that log does not show the watchdog armed for
   the ledger's deadline in a process that is still the container's main process (an image or API that skipped
   the entrypoint).
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
carries `variant, verdict (OK|BOOT_FAIL|SMOKE_FAIL|POST_FAIL|TIMEOUT), boot_s, fault, where, gsm8k, gsm8k_counts,
accept, tok_s, mem_guard, result`; `mem_guard` is null unless the memory guard (below) fired, and then holds the
reading it fired at. An `error` line for a pod that is still up adds `pod, id, deadline, deadline_local, next`.

When a poll of a running job fails, the harness asks RunPod about the pod. A pod RunPod reports `EXITED`, `ERROR`, or
`TERMINATED`, or no longer lists, ends `exec` (and `up` during setup) at once: its container is gone, and with it
every file on the pod that was not pulled, and the error says to run `harness down <issue>`, since a stopped pod still
bills its disk. A RunPod read that fails only warns; 30 failed polls in a row (about 5 minutes) still end the wait.

## On the pod

The kit lands in `/workspace/sm120-kit` (only `harness/pod/` and the issue folder), `HF_HOME=/workspace/hf`, and
results in `/workspace/sm120-kit/results/<issue>/`: `env.txt`, `jobs/<job>.log` (plus `.pid` and `.exit`),
`<variant>/` (server log, responses, quality output, `progress.jsonl`, `verdict.json`, and the memory guard's
`memory.jsonl`, `mem_guard.log`, and `memory_guard.txt`), and `verdicts.jsonl`.
`exec` clears the laptop's copy of a variant's folder when the variant starts, pulls the folder every 5 minutes while
it runs (2 minute limit, a failure only warns), so a container that dies mid-run leaves its logs up to then on the
laptop, and after it pulls the folder, `jobs/`, and `verdicts.jsonl` over that copy. The last pull also runs when the
wait fails, and deletes nothing.

### Memory guard

RunPod stops a container at about 98% of its memory limit (`docs/runpod.md`) and removes it with every file on it.
From just before the launch until the server is killed, `serve_variant.sh` runs `pod/mem_guard.py`, which reads the
container's cgroup (v2, or v1's names mapped to v2's) and writes `<variant>/memory.jsonl`:

- First a `controls` line: whether `memory.reclaim` and `memory.high` exist and are writable in this container.
- Every `MEM_GUARD_POLL_S` seconds (default 15) `t, current, max` (null for no limit) and the `memory.stat` counters
  `anon, file, active_file, inactive_file, shmem, file_mapped, file_dirty, unevictable` in bytes.
- At `MEM_GUARD_TRIM_FRACTION` of the limit (default 0.80), at most once a minute, a trim: `posix_fadvise(DONTNEED)`
  over the HF hub cache (`HF_HUB_CACHE`, else `$HF_HOME/hub`), then a write of the bytes over that line to
  `memory.reclaim` where the container may write it, since fadvise skips page cache a process has mapped. The line
  carries `trim: {fadvise, reclaim}` (`reclaim` is `ok`, `unavailable`, or an errno name) and `after_trim`, the usage
  after. The guard never writes `memory.high` or `memory.max`.
- At `MEM_GUARD_KILL_FRACTION` (default 0.93, under RunPod's 98%) on a poll after a trim, it fires: a `mem_guard`
  phase line, then `<variant>/memory_guard.txt` (the reading on its first line, all of `memory.stat`, the 40 largest
  processes, each server process's `VmRSS`, `VmLck`, `VmPin`, and `Rss*` lines, and a `py-spy dump --nonblocking` of
  each when `py-spy` is on PATH, all bounded at 60 s), then `pkill -9 -f` of the server's process pattern, and it
  exits. The run then ends the usual way, most likely as `BOOT_FAIL` or `POST_FAIL`, with `mem_guard` in its verdict,
  while the container and its logs still exist.
- With no limit, or one it cannot read, it writes one `warn` line and only logs.

Its own errors go to `<variant>/mem_guard.log` and never fail the run. `serve_variant.sh` stops it on every way out,
and it exits by itself once its parent is gone, so it cannot outlive its variant and kill the next one's server. The
three knobs go in a variant's `env`. `mem_guard.py --drop-cache <path>...` does the fadvise pass once and prints a
summary; `issues/vllm-56771/setup.sh` runs it on `$HF_HOME` after the weights download.

## Tests

```bash
cd harness && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
shellcheck pod/*.sh tests/pod/*.sh
```

`cargo test` also runs the bash tests in `tests/pod/` (watchdog, package snapshot, `serve_variant.sh` against a
fake server with both quality protocols, the memory guard against fake cgroup directories, and stage 2 log replay)
and the job start and poll scripts in local bash.
