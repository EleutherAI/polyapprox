import math

import array_api_compat
from scipy.special import factorial2

from .backends import ArrayType, norm_cdf, norm_pdf


def relu_ev(mu: ArrayType, sigma: ArrayType) -> ArrayType:
    """Expected value of RELU(x) under N(mu, sigma)"""
    return mu * norm_cdf(mu / sigma) + sigma * norm_pdf(mu / sigma)


def relu_prime_ev(mu: ArrayType, sigma: ArrayType) -> ArrayType:
    """Expected value of RELU'(x) under N(mu, sigma)"""
    return norm_cdf(mu / sigma)


def relu_poly_ev(n: int, mu: ArrayType, sigma: ArrayType) -> ArrayType:
    """
    Compute E[x^n * ReLU(x)] analytically where x ~ N(mu, sigma^2)

    Parameters:
    n     : int, the exponent n in x^n * ReLU(x)
    mu    : ArrayLike, the mean(s) of the normal distribution(s)
    sigma : ArrayLike, the standard deviation(s) of the normal distribution(s)

    Returns:
    result : NDArray, the computed expected value(s)
    """
    xp = array_api_compat.array_namespace(mu, sigma)

    loc = -mu / sigma
    expected_value = xp.zeros_like(mu)

    # Precompute standard normal PDF and CDF at loc
    phi_loc = norm_pdf(loc)  # PDF of standard normal at loc
    Phi_loc = norm_cdf(loc)  # CDF of standard normal at loc

    # Compute M_0 and M_1
    M = [Phi_loc, -phi_loc]

    # Compute higher-order M_k recursively
    for k in range(2, n + 2):
        M.append(-(loc ** (k - 1)) * phi_loc + (k - 1) * M[k - 2])

    # Sum over k from 0 to n+1
    for k in range(n + 2):
        binom_coeff = math.comb(n + 1, k)
        mu_power = mu ** (n + 1 - k)
        sigma_power = sigma**k

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
