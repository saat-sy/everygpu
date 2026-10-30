"""GPU memory budget for resident weights."""

import torch

from worker.utils import constants
from worker.utils import models as worker_models


def gpu_budget(max_vram_gib: float | None) -> tuple[str, int, int, int]:
    """GPU name, free and total bytes, and the weight budget left after the runtime reserve."""
    if not torch.cuda.is_available():
        raise worker_models.NoGPU("no CUDA GPU visible; the worker needs an NVIDIA GPU")
    free, total = torch.cuda.mem_get_info()
    limit = (
        free if max_vram_gib is None else min(free, int(max_vram_gib * constants.GIB))
    )
    budget = int(limit * constants.WEIGHT_FRACTION)
    return torch.cuda.get_device_name(), free, total, budget
