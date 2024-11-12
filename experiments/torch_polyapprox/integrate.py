import math
from itertools import combinations, product
import torch
from torch import Tensor
from torch.distributions import Normal
from typing import Callable, Tuple
from scipy.special import owens_t, roots_hermite

def owens_t_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_np = x.cpu().numpy()  # Convert to NumPy
    y_np = y.cpu().numpy()
    result_np = owens_t(x_np, y_np)
    return torch.from_numpy(result_np).to(x.device)  # Convert back to PyTorch with the original device

def roots_hermite_torch(num_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes_np, weights_np = roots_hermite(num_points)
    nodes_torch = torch.from_numpy(nodes_np)  # Convert to Torch tensors
    weights_torch = torch.from_numpy(weights_np)
    return nodes_torch, weights_torch

def bivariate_product_moment(
    h: Tensor,
    k: Tensor,
    rho: Tensor,
    *,
    mean_x: Tensor = 0.0,
    mean_y: Tensor = 0.0,
    std_x: Tensor = 1.0,
    std_y: Tensor = 1.0,
    unconditional: bool = False,
) -> Tensor:
    h = (h - mean_x) / std_x
    k = (k - mean_y) / std_y

    eps = torch.finfo(h.dtype).eps
    rho = torch.clamp(rho, -1.0 + eps, 1.0 - eps)

    # Define constants
    denom = torch.sqrt(1 - rho**2)
    numer = torch.sqrt(h**2 - 2 * rho * h * k + k**2)

    # Z(x): Standard normal PDF
    Z_h = Normal(0, 1).log_prob(h).exp()
    Z_k = Normal(0, 1).log_prob(k).exp()

    # Q(x): Standard normal CDF
    Q_k_given_h = 1 - Normal(rho * h, denom).cdf(k)
    Q_h_given_k = 1 - Normal(rho * k, denom).cdf(h)

    # Compute L(h, k; rho), the probability in the truncated region
    L_hk_rho = bivariate_normal_cdf(-h, -k, rho)

    # Product moment m11 formula
    term1 = rho * L_hk_rho
    term2 = rho * h * Z_h * Q_k_given_h
    term3 = rho * k * Z_k * Q_h_given_k
    term4 = (denom / torch.sqrt(2 * math.pi)) * Normal(0, 1).log_prob(numer / denom).exp()

    # Correct answer if mean_x = mean_y = 0
    m11 = std_x * std_y * (term1 + term2 + term3 + term4)

    # Account for being noncentered
    m10 = (Z_h * Q_k_given_h + rho * Z_k * Q_h_given_k)
    m01 = (rho * Z_h * Q_k_given_h + Z_k * Q_h_given_k)
    m11 += std_x * mean_y * m10 + std_y * mean_x * m01 + mean_x * mean_y * L_hk_rho

    # Divide by the probability that we would end up in the truncated region
    if not unconditional:
        m11 /= L_hk_rho

    return m11


def bivariate_normal_cdf(
    x: Tensor,
    y: Tensor,
    rho: Tensor,
    *,
    mean_x: float = 0.0,
    mean_y: float = 0.0,
    std_x: float = 1.0,
    std_y: float = 1.0,
    tail: bool = False,
) -> Tensor:
    x = (x - mean_x) / std_x
    y = (y - mean_y) / std_y

    if tail:
        x = -x
        y = -y

    eps = torch.finfo(x.dtype).eps
    x = x + torch.where(x < 0, -eps, eps)
    y = y + torch.where(y < 0, -eps, eps)

    rx = (y - rho * x) / (x * torch.sqrt(1 - rho**2))
    ry = (x - rho * y) / (y * torch.sqrt(1 - rho**2))

    mask = (x * y > 0) | ((x * y == 0) & (x + y >= 0))
    beta = torch.where(mask, torch.tensor(0.0), torch.tensor(0.5))

    term1 = 0.5 * (Normal(0, 1).cdf(x) + Normal(0, 1).cdf(y))
    result = term1 - owens_t_torch(x, rx) - owens_t_torch(y, ry) - beta

    backup = 0.25 + torch.asin(rho) / (2 * math.pi)
    fallback = torch.isclose(x, torch.tensor(0.0)) & torch.isclose(y, torch.tensor(0.0))

    return torch.where(fallback, backup, result)


def gauss_hermite(
    f: Callable,
    mu: Tensor = 0.0,
    sigma: Tensor = 1.0,
    num_points: int = 50,
) -> Tensor:
    nodes, weights = roots_hermite_torch(num_points)
    mu = mu.view(-1, 1)
    sigma = sigma.view(-1, 1)

    grid = mu + sigma * torch.sqrt(2) * nodes.view(1, -1)

    return torch.mm(f(grid), weights.reshape(-1, 1)).reshape(-1) / torch.sqrt(torch.tensor(math.pi))


def isserlis(cov: Tensor, indices: list[int]) -> Tensor:
    return sum(
        torch.prod(torch.stack([cov[..., a, b] for a, b in partition]), dim=0)
        for partition in pair_partitions(indices)
    )


def master_theorem(
    mu_x: Tensor,
    cov_x: Tensor,
    mu_y: Tensor,
    var_y: Tensor,
    xcov: Tensor
) -> list[Tensor]:
    *batch_shape, k = mu_x.shape
    *batch_shape2, k2, k3 = cov_x.shape

    assert batch_shape == batch_shape2, "Batch dimensions must match"
    assert k == k2 == k3, "Dimensions of means and covariances must match"
    assert torch.all(var_y > 0.0), "X0 must have positive variance"

    a = xcov / var_y[..., None]
    b = -a * mu_y[..., None] + mu_x

    eps_cov = cov_x - xcov[..., None] * a[..., None, :]

    coefs = []

    for m in range(k + 1):
        coef = torch.zeros(batch_shape)

        for comb in combinations(range(k), m):
            prefix = torch.ones(batch_shape)

            for i in comb:
                prefix = prefix * a[..., i]

            remaining = list(set(range(k)) - set(comb))

            for bitstring in product([0, 1], repeat=len(remaining)):
                num_residuals = sum(bitstring)
                residual_indices = []

                if num_residuals % 2:
                    continue

                term = prefix

                for i, bit in zip(remaining, bitstring):
                    if bit:
                        residual_indices.append(i)
                    else:
                        term = term * b[..., i]

                if residual_indices:
                    iss = isserlis(eps_cov, residual_indices)
                    term = term * iss

                coef = coef + term
        
        coefs.append(coef)

    coefs.reverse()
    return coefs


def pair_partitions(elements: list) -> list:
    """Iterate over all partitions of a list into pairs."""
    if not elements:
        yield []
        return

    pivot = elements[0]
    for i in range(1, len(elements)):
        partner = elements[i]
        remaining = elements[1:i] + elements[i+1:]

        for rest in pair_partitions(remaining):
            yield [(pivot, partner)] + rest