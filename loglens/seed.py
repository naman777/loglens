"""Global seeding, called at the top of every entry point."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 1337) -> int:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def get_device() -> str:
    """cuda when available, overridable with LOGLENS_DEVICE=cpu (e.g. while the GPU is busy)."""
    dev = os.environ.get("LOGLENS_DEVICE")
    if dev:
        return dev
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
