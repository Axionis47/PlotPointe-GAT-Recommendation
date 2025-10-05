#!/usr/bin/env python3
"""
Seed and determinism utilities (opt-in) for training scripts.
Designed to be safe to import anywhere. Does not modify global state unless called.
"""
from __future__ import annotations
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_determinism(seed: Optional[int] = None) -> None:
    """Enable more deterministic execution (may reduce speed on GPU).
    - Disables cudnn.benchmark and enables cudnn.deterministic
    - Optionally sets seeds across libraries
    - Sets CUBLAS_WORKSPACE_CONFIG for CUDA >= 10.2 as recommended by PyTorch
    """
    if seed is not None:
        set_seeds(seed)
    # cuDNN flags
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    # Some CUDA ops require this env var for determinism
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    # Avoid non-deterministic algorithms where possible
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Older PyTorch versions may not support this; best-effort only
        pass

