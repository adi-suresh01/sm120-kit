# sm120-kit

Reproduction kits for LLM serving bugs on NVIDIA sm_120 GPUs (RTX PRO 6000 Blackwell, RTX 5090) and
PCIe-only multi-GPU servers.

Each folder under `issues/` targets one upstream issue in vLLM, SGLang, or FlashInfer: the reporter's setup,
hypotheses, an experiment matrix, and scripts that run the matrix on a rented GPU. Shared helpers live in
`scripts/`.

## Running a kit on a pod

```
git clone <this repo> && cd sm120-kit
bash scripts/env_report.sh
bash issues/vllm-49476/setup.sh
bash issues/vllm-49476/run.sh <variant>
```

See `docs/runpod.md` for pod setup.
