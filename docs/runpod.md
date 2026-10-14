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

| Use | GPU | Approx $/hr (Sep 2026) |
|---|---|---|
| Installs, single-GPU repros | 1x RTX PRO 6000 or 1x RTX 5090 | ~$1.1 / ~$0.7 |
| TP=2 / EP=2 repros | 2x RTX PRO 6000 | ~$2.2 |
| DeepSeek-V4 class repros | 4x RTX PRO 6000 | ~$4.4-9 |
| TP=8 repros | 8x RTX PRO 6000 | ~$9-18 |

Template: a PyTorch / CUDA 13 image, then `pip install` the exact vLLM / SGLang version from the issue.
Attach a network volume for `HF_HOME` so model weights download once.

## Rules

- Every pod launch needs Aditya's explicit approval: GPU, count, $/hr, hours, total.
- Stop the pod as soon as a run ends; terminate when the issue is done. Volumes cost money too; delete when
  the cluster of issues is finished.
- Never store the API key or HF token in the repo. Export them on the pod for the session only.
- Log pod id, start, stop, and actual cost in the issue README.
