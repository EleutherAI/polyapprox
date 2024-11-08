from torch import nn
from torch import Tensor
import torch
from typing import List, Tuple, Union
from jaxtyping import Float

class Bilinear(nn.Linear):
    """A bilinear layer with optional gate and noise"""
    def __init__(self, d_in: int, d_out: int, bias=False, gate=None) -> None:
        super().__init__(d_in, 2 * d_out, bias=bias)
        self.gate = {"relu":nn.ReLU(), "silu":nn.SiLU(), "gelu":nn.GELU(), None: nn.Identity()}[gate]
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        left, right = super().forward(x).chunk(2, dim=-1)
        return self.gate(left) * right
    
    @property
    def w_l(self):
        return self.weight.chunk(2, dim=0)[0]
    
    @property
    def w_r(self):
        return self.weight.chunk(2, dim=0)[1]


class Linear(nn.Linear):
    """A linear layer with optional gate and noise"""
    def __init__(self, d_in: int, d_out: int, bias=False, gate=None) -> None:
        super().__init__(d_in, d_out, bias=bias)
        self.gate = {"relu": nn.ReLU(), "silu": nn.SiLU(), "gelu": nn.GELU(), None: nn.Identity()}[gate]
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return self.gate(super().forward(x))