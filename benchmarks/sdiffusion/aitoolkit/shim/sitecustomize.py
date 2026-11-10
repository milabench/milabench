"""XPU shim for ai-toolkit: some built-in extensions (omnigen2) query
torch.cuda at import time even when unused. On a torch+xpu build those
calls assert. Provide inert stubs so module imports succeed; actual
SDXL training never touches these paths.
"""

import types

import torch

if not torch.cuda.is_available() and hasattr(torch, "xpu"):

    def _current_device():
        return 0

    def _get_device_properties(device=0):
        p = torch.xpu.get_device_properties(device)
        return types.SimpleNamespace(
            name=p.name,
            major=0,
            minor=0,
            total_memory=getattr(p, "total_memory", 0),
            multi_processor_count=getattr(p, "max_work_group_size", 1),
            warp_size=32,
        )

    torch.cuda.current_device = _current_device
    torch.cuda.get_device_properties = _get_device_properties
