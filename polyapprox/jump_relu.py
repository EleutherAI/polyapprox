import math

import array_api_compat
from scipy.special import factorial2

from .backends import ArrayType, norm_cdf, norm_pdf

# All of these functions assume theta = 1.0, which loses no generality because
# we can always rescale our weights and biases to get an equivalent network.

def jump_relu_ev(mu: ArrayType, sigma: ArrayType) -> ArrayType:
    """Expected value of JumpReLU(x) under N(mu, sigma)"""
    return mu * (1-norm_cdf((1-mu) / sigma)) + sigma * norm_pdf((1-mu) / sigma)

# By default, we include the Dirac delta term in the expected value of the derivative
# since relu_prime is equal to theta(x-1) + dirac_delta(x-1), but since we're also
# doing some estimates of it approximately, we need the example without the Dirac delta
# because the Monte Carlo sampler will not sample exactly x=1.
def jump_relu_prime_ev(mu: ArrayType, sigma: ArrayType, include_dirac_delta_term: bool = True) -> ArrayType:
    """Expected value of JumpReLU'(x) under N(mu, sigma)"""
    normal_term = norm_cdf((mu-1) / sigma)

    if include_dirac_delta_term:
        return normal_term + norm_pdf((mu-1) / sigma) # Include the Dirac delta term
    else:
        return normal_term


def jump_relu_poly_ev(n: int, mu: ArrayType, sigma: ArrayType) -> ArrayType:
    """
    Compute E[x^n * JumpReLU(x)] analytically where x ~ N(mu, sigma^2)

    Parameters:
    n     : int, the exponent n in x^n * ReLU(x)
    mu    : ArrayLike, the mean(s) of the normal distribution(s)
    sigma : ArrayLike, the standard deviation(s) of the normal distribution(s)

    Returns:
    result : NDArray, the computed expected value(s)
    """
    xp = array_api_compat.array_namespace(mu, sigma)

    loc = (1-mu) / sigma # Single change from relu
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
