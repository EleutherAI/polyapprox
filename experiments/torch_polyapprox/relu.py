import math
import torch
from torch import Tensor
from torch.distributions import Normal

def relu(x: Tensor) -> Tensor:
    """Rectified Linear Unit (ReLU) activation function"""
    return torch.relu(x)

def relu_prime(x: Tensor) -> Tensor:
    """Derivative of ReLU(x)"""
    return (x > 0).float()

def relu_ev(mu: Tensor, sigma: Tensor) -> Tensor:
    """Expected value of RELU(x) under N(mu, sigma)"""
    normal_dist = Normal(mu, sigma)
    return mu * normal_dist.cdf(mu) + sigma * normal_dist.log_prob(mu).exp()

def relu_prime_ev(mu: Tensor, sigma: Tensor) -> Tensor:
    """Expected value of RELU'(x) under N(mu, sigma)
    Got an error here at Normal(mu, sigma) # before .cdf. Error in mu, sigma
    """
    print(type(mu), type(sigma))
    print(mu.shape)
    print(sigma.shape)
    return Normal(mu, sigma).cdf(mu)

def relu_poly_ev(n: int, mu: Tensor, sigma: Tensor) -> Tensor:
    """
    Compute E[x^n * ReLU(x)] analytically where x ~ N(mu, sigma^2)

    Parameters:
    n     : int, the exponent n in x^n * ReLU(x)
    mu    : Tensor, the mean(s) of the normal distribution(s)
    sigma : Tensor, the standard deviation(s) of the normal distribution(s)

    Returns:
    result : Tensor, the computed expected value(s)
    """
    mu = mu.clone()
    sigma = sigma.clone()

    loc = -mu / sigma
    expected_value = torch.zeros_like(mu)

    # Precompute standard normal PDF and CDF at loc
    phi_loc = Normal(0, 1).log_prob(loc).exp()  # PDF of standard normal at loc
    Phi_loc = Normal(0, 1).cdf(loc)  # CDF of standard normal at loc

    # Compute M_0 and M_1
    M = [Phi_loc, -phi_loc]

    # Compute higher-order M_k recursively
    for k in range(2, n + 2):
        M.append(-loc ** (k - 1) * phi_loc + (k - 1) * M[k - 2])

    # Sum over k from 0 to n+1
    for k in range(n + 2):
        binom_coeff = math.comb(n + 1, k)
        mu_power = mu ** (n + 1 - k)
        sigma_power = sigma ** k

        # We need to compute an "upper" integral from loc to infinity, but all the
        # formulas are for lower integrals from -infinity to loc. We compute the
        # full integral and then we can get the upper by subtracting the lower.
        if k == 0:
            full = 1
        elif k % 2:
            full = 0
        else:
            full = math.factorial(k - 1)

        expected_value += binom_coeff * mu_power * sigma_power * (full - M[k])

    return expected_value