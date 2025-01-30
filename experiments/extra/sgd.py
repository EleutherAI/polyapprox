# %%
import sys
import os
from numpy.typing import ArrayLike, NDArray
import numpy as np
from torch import Tensor
from typing import Callable
import torch.nn as nn
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# Now you can import the module
#from polyapprox.ols import ols
#from ..polyapprox.ols import ols
import matplotlib.pyplot as plt
import os
import torch
from torch.distributions import MultivariateNormal
from mnist_2l import Minimal_FFN
from decomp.model import FFNModel


torch.manual_seed(42)
DIR = '/Users/alicerigg/Code/polyapprox/experiments/'
datadict_path = DIR + f'data/new_baseline_uncenteredlong_datadict.pt'
savepaths = {}
use_imports = True
from functional_plotting_utils import *

class GaussianMixture:
    def __init__(
        self,
        means: Tensor,
        covs: Tensor,
        class_probs: Tensor,
        size: int,
        #shape: tuple[int, int, int] = (3, 32, 32),
        shape: tuple[int, int] = (784),
        trf: Callable = lambda x: x,
        eps: float = 1e-6,
        
    ):
        self.class_probs = class_probs
        self.dists = [MultivariateNormal(mean, cov + eps*torch.eye(784)) for mean, cov in zip(means, covs)]
        self.shape = shape
        self.size = size
        self.trf = trf
        self.eps = eps

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of bounds for size {self.size}")

        y = torch.multinomial(self.class_probs, 1).squeeze()
        x = self.dists[y].sample().reshape(self.shape)
        return {
            "pixel_values": self.trf(x),
            "label": y,
        }
    
    def sample(self, num_samples: int) -> list[dict[str, Tensor]]:
        """
        Returns a list of randomly sampled items from the GaussianMixture.

        Args:
            num_samples (int): Number of items to sample.

        Returns:
            torch stack of samples, shape (num_samples, dmodel)
        """
        if isinstance(num_samples, tuple):
            num_samples = num_samples[0]
        if num_samples > self.size:
            raise ValueError(f"Requested {num_samples} samples, but only {self.size} available.")

        sampled_indices = torch.randint(0, self.size, (num_samples,))
        return torch.stack([self[idx]['pixel_values'] for idx in sampled_indices])

    def __len__(self) -> int:
        return self.size

class QuadraticModel(nn.Module):
    def __init__(self):
        super(QuadraticModel, self).__init__()
        self.quadratic_layer = nn.Parameter(torch.randn(10, 784, 784))
        self.linear_layer = nn.Parameter(torch.randn(10, 784))
        self.bias = nn.Parameter(torch.randn(10))

    def from_quad(self, quad):
        self.quadratic_layer = nn.Parameter(quad.unpack_gamma().clone())
        self.linear_layer = nn.Parameter(quad.beta.clone())
        self.bias = nn.Parameter(quad.alpha.clone())
    def forward(self, x):
        # Compute quadratic term
        q = torch.einsum('bi,hij->bhj', x, self.quadratic_layer)
        quad_term = torch.einsum('bj,bhj->bh', x, q)
        #quad_term = torch.einsum('bi,ijh,bj->bh', x, self.quadratic_layer, x)
        # Compute linear term
        #linear_term = torch.einsum('bi,hi->bh',x,self.linear_layer)
        linear_term = x @ self.linear_layer
        # Add quadratic, linear, and bias terms
        output = quad_term + linear_term + self.bias[None,:]
        return output

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear_layer = nn.Parameter(torch.randn(10, 784))
        self.bias = nn.Parameter(torch.randn(10))

    def from_quad(self, quad):
        self.linear_layer = nn.Parameter(quad.beta.clone())
        self.bias = nn.Parameter(quad.alpha.clone())
    def forward(self, x):
        linear_term = x @ self.linear_layer
        # Add linear, and bias terms
        output = linear_term + self.bias[None,:]
        return output