# RunPod setup and spend rules

## One-time setup (Aditya)

1. Create an account at runpod.io and add prepaid credit ($50 to start; prepaid credit is the spend cap).
2. Settings -> SSH Public Keys: paste `~/.ssh/id_rsa.pub`.
3. Settings -> API Keys: create a read/write key.
4. In a normal terminal (not an AI chat, so the key is never in a transcript):
   ```
   brew install runpod/runpodctl/runpodctl
   runpodctl config --apiKey <key>
   ```
5. Verify: `runpodctl get pod` returns an empty list without errors.

CLI flags change between runpodctl versions. Check `runpodctl create pod --help` before scripting a launch
instead of trusting flags written here from memory.

## Pod choices

| Use | GPU | $/hr per GPU, community / secure (catalog read 2026-10-14) |
|---|---|---|
| Installs, single-GPU repros | 1x RTX PRO 6000 or 1x RTX 5090 | $1.69 / $2.49 (PRO 6000), $0.69 / $1.19 (5090) |
| TP=2 / EP=2 repros | 2x RTX PRO 6000 | $3.38 / $4.98 |
| DeepSeek-V4 class repros | 4x RTX PRO 6000 | $6.76 / $9.96 |
| TP=8 repros | 8x RTX PRO 6000 | $13.52 / $19.92 |

Community stock is often LOW or empty on hosts with CUDA >= 13.0; secure had stock in 12 data centers.
Re-read the catalog before quoting a launch.

Template: a PyTorch / CUDA 13 image, then `pip install` the exact vLLM / SGLang version from the issue. When the
reporter used an official image (e.g. `vllm/vllm-openai:vX`), use that image instead and override the entrypoint
to start sshd; such images do not read `PUBLIC_KEY`. See `issues/vllm-49476/README.md` for the command.
If the image runs Python from a venv set through its ENV (lmsysorg/sglang does), SSH sessions do not inherit
it; export `/proc/1/environ` to a file and source it in every command (see `issues/sglang-44291/README.md`).
Attach a network volume for `HF_HOME` so model weights download once.

## Rules

- Every pod launch needs Aditya's explicit approval: GPU, count, $/hr, hours, total.
- Stop the pod as soon as a run ends; terminate when the issue is done. Volumes cost money too; delete when
  the cluster of issues is finished.
- Never store the API key or HF token in the repo. Export them on the pod for the session only.
- Log pod id, start, stop, and actual cost in the issue README.
