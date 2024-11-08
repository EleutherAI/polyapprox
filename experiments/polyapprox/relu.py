import math

from numpy.typing import ArrayLike, NDArray
from scipy.special import factorial2
from scipy.stats import norm
import numpy as np

def relu(x):
    """Rectified Linear Unit (ReLU) activation function"""
    return np.maximum(0, x)


def relu_prime(x):
    """Derivative of ReLU(x)"""
    return np.where(x > 0, 1.0, 0.0)


def relu_ev(mu, sigma):
    """Expected value of RELU(x) under N(mu, sigma)"""
    return mu * norm.cdf(mu / sigma) + sigma * norm.pdf(mu / sigma)


def relu_prime_ev(mu, sigma):
    """Expected value of RELU'(x) under N(mu, sigma)"""
    return norm.cdf(mu / sigma)


def relu_poly_ev(n: int, mu: ArrayLike, sigma: ArrayLike) -> NDArray:
    """
    Compute E[x^n * ReLU(x)] analytically where x ~ N(mu, sigma^2)

    Parameters:
    n     : int, the exponent n in x^n * ReLU(x)
    mu    : ArrayLike, the mean(s) of the normal distribution(s)
    sigma : ArrayLike, the standard deviation(s) of the normal distribution(s)

    Returns:
    result : NDArray, the computed expected value(s)
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    loc = -mu / sigma
    expected_value = np.zeros_like(mu)

    # Precompute standard normal PDF and CDF at loc
    phi_loc = norm.pdf(loc)  # PDF of standard normal at loc
    Phi_loc = norm.cdf(loc)  # CDF of standard normal at loc

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
            full = factorial2(k - 1)

        expected_value += binom_coeff * mu_power * sigma_power * (full - M[k])

    return expected_value