import math
import sys
from numbers import Number
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from torch import Tensor


# Either a NumPy array or a JAX array or a PyTorch tensor
ArrayType = TypeVar("ArrayType", "JaxArray", NDArray, "Tensor")


def norm_cdf(x: ArrayType) -> ArrayType:
    """Backend-agnostic cumulative distribution function of N(0, 1)"""
    # Fast path for NumPy arrays
    if isinstance(x, (np.ndarray, Number)):
        return norm.cdf(x)

    if (jax := sys.modules.get("jax")) and isinstance(x, jax.numpy.ndarray):
        return jax.scipy.stats.norm.cdf(x)

    if (torch := sys.modules.get("torch")) and isinstance(x, torch.Tensor):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    raise ValueError(f"Unsupported array type: '{type(x)}'")


def norm_pdf(x: ArrayType) -> ArrayType:
    """Backend-agnostic probability density function of N(0, 1)"""
    # Fast path for NumPy arrays
    if isinstance(x, (np.ndarray, Number)):
        return norm.pdf(x)

    if (jax := sys.modules.get("jax")) and isinstance(x, jax.numpy.ndarray):
        return jax.scipy.stats.norm.pdf(x)

    if (torch := sys.modules.get("torch")) and isinstance(x, torch.Tensor):
        return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

    raise ValueError(f"Unsupported array type: '{type(x)}'")
