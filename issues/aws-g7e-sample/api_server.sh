#!/usr/bin/env bash
# One launch line serves every variant, so the vLLM install is picked here: the venv a variant names in VLLM_PYTHON
# (setup.sh's 0.32.0), or else the image's own Python with vLLM 0.15.1, running the module the SageMaker entrypoint
# runs.
set -euo pipefail
exec "${VLLM_PYTHON:-python3}" -m vllm.entrypoints.openai.api_server "$@"
