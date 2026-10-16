"""Checks whether a GPU view of pinned host memory follows later host writes on this host (#57224).

Model Runner V2 feeds input ids, positions and block tables through such views; a stale view gives silent garbage
with no error. Writes one JSON line to stdout.
"""

import json

import torch
import vllm  # noqa: F401  (loads the _C ops)

cpu = torch.arange(1, 9, dtype=torch.int32).pin_memory()
view = torch.ops._C.get_cuda_view_from_cpu_tensor(cpu)
torch.cuda.synchronize()
initial = (view + 0).tolist()
cpu.fill_(99)
torch.cuda.synchronize()
after = (view + 0).tolist()
print(json.dumps({"initial": initial, "after_host_write": after, "coherent": after == [99] * 8}))
