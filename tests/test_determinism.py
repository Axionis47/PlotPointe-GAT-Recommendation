#!/usr/bin/env python3
import torch
import numpy as np

from plotpointe.utils.random import enable_determinism, set_seeds


def test_enable_determinism_reproducible():
    enable_determinism(123)
    a = torch.randn(3, 3)
    b = torch.randn(3, 3)

    # Reset and reproduce
    enable_determinism(123)
    a2 = torch.randn(3, 3)
    b2 = torch.randn(3, 3)

    assert torch.allclose(a, a2)
    assert torch.allclose(b, b2)


def test_numpy_python_seed_reproducible():
    set_seeds(999)
    x1 = np.random.randn(5)

    set_seeds(999)
    x2 = np.random.randn(5)

    assert np.allclose(x1, x2)

