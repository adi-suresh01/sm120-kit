# RunPod setup and spend rules

Pods are rented, run, and terminated by `harness` (see `harness/README.md`). This page covers the account setup
it needs and the spend rules it enforces.

## One-time setup (Aditya)

1. Create an account at runpod.io and add prepaid credit ($50 to start; prepaid credit is the spend cap).
2. Settings -> SSH Public Keys: paste `~/.ssh/id_rsa.pub`. Load the key into ssh-agent with `ssh-add`: the
   harness runs ssh in batch mode and cannot answer a passphrase prompt. With several keys in the agent, set
   `HARNESS_SSH_KEY=~/.ssh/id_rsa` in `~/.zshrc` so ssh offers only that one.
3. Settings -> API Keys: create a Restricted key with the management API (`api.runpod.io`) on read/write and
   `api.runpod.ai` on none. Read access covers `GET /v2/organization`, which the harness records in each ledger
   so a key of another account is refused (skipped for an account outside any organization). In a normal
   terminal (not an AI chat, so the key never lands in a transcript), store it in the Keychain; `-w` given
   last makes `security` prompt for the value:
   ```
   security add-generic-password -a "$USER" -s runpod-api-key -w
   ```
4. Add to `~/.zshrc`:
   ```
   export RUNPOD_API_KEY="$(security find-generic-password -a "$USER" -s runpod-api-key -w)"
   ```
5. For issues with gated models: Settings -> Secrets, create `hf_token` holding the Hugging Face token. A
   `harness.toml` with `needs_hf_token = true` passes it to the pod as a secret reference, never as a value.
6. Install and verify in a new shell: `cd ~/Desktop/sm120-kit && cargo install --path harness --locked`, then
   `harness reap` exits 0.
7. Optional, for looking at pods by hand: `brew install runpod/runpodctl/runpodctl`.

## Pod choices

| Use | GPU | $/hr per GPU, community / secure (catalog read 2026-10-14) |
|---|---|---|
| Installs, single-GPU repros | 1x RTX PRO 6000 or 1x RTX 5090 | $1.69 / $2.49 (PRO 6000), $0.69 / $1.19 (5090) |
| TP=2 / EP=2 repros | 2x RTX PRO 6000 | $3.38 / $4.98 |
| DeepSeek-V4 class repros | 4x RTX PRO 6000 | $6.76 / $9.96 |
| TP=8 repros | 8x RTX PRO 6000 | $13.52 / $19.92 |

Community stock is often LOW or empty on hosts with CUDA >= 13.0; secure had stock in 12 data centers.
`harness quote <issue>` reads the live catalog.

Image: the reporter's official image when there is one (e.g. `vllm/vllm-openai:vX`), pinned by digest; otherwise
a PyTorch / CUDA 13 image plus a `setup.sh` that installs the exact vLLM / SGLang version from the issue. The
harness replaces the entrypoint with `harness/pod/entrypoint.sh` (sshd, the container env saved for SSH sessions,
the deadline watchdog), so any Debian-based image with bash works. Weights go to the pod's volume
(`HF_HOME=/workspace/hf`).

## Rules

- Each issue's GPU budget needs Aditya's explicit approval (GPU, count, $/hr, hours, total), recorded with
  `harness budget set <issue> <usd> --note <text>`. The harness sizes every pod's deadline to fit it.
- `harness down` (or `harness run`, which ends with it) terminates the pod when a run ends. The volume goes
  with the pod.
- Never store the API key or HF token in the repo or pass them to a pod as plain values.
- The ledger (`issues/<x>/harness-ledger.json`) records pod id, start, deadline, end, and cost; copy the totals
  into the issue README.
