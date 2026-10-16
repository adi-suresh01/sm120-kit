"""Prints the block-FP8 kernel vLLM's selection picks, under the current env and patches, for kv_a_proj_with_mqa
(576 x 2048) and an aligned layer of the same model (q_proj, 3072 x 2048).

The server logs `Selected <kernel> for Fp8LinearMethod` once per kernel class, not per layer, so this names the
kernel each layer gets. Only meaningful with DeepGEMM disabled: its can_implement needs the model config, which this
probe does not build, so with DeepGEMM enabled the answer differs from the server's.
"""

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key

LAYERS = {"kv_a_proj_with_mqa": (576, 2048), "q_proj": (3072, 2048)}

with set_current_vllm_config(VllmConfig()):
    for name, shape in LAYERS.items():
        try:
            kernel = init_fp8_linear_kernel(
                activation_quant_key=create_fp8_quant_key(static=False, group_shape=GroupShape(1, 128)),
                weight_quant_key=create_fp8_quant_key(static=True, group_shape=GroupShape(128, 128)),
                input_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                weight_shape=shape,
            )
            print(f"{name} {shape}: {type(kernel).__name__}")
        except ValueError as e:
            print(f"{name} {shape}: no kernel: {e}")
