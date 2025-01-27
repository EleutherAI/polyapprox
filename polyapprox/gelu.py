import math

import array_api_compat
from scipy.special import gamma
from scipy.stats import nct

from .backends import ArrayType, norm_cdf, norm_pdf


def gelu(x: ArrayType) -> ArrayType:
    """Gaussian Error Linear Unit (GELU) activation function"""
    return x * norm_cdf(x)


def gelu_prime(x: ArrayType) -> ArrayType:
    """Derivative of GELU(x)"""
    return norm_cdf(x) + x * norm_pdf(x)


def gelu_ev(mu: ArrayType, sigma: ArrayType) -> ArrayType:
    """Expected value of GELU(x) under N(mu, sigma)"""
    denom = (1 + sigma**2) ** 0.5
    return mu * norm_cdf(mu / denom) + (sigma**2 / denom) * norm_pdf(mu / denom)


def gelu_prime_ev(mu: ArrayType, sigma: ArrayType) -> ArrayType:
    """Expected value of GELU'(x) under N(mu, sigma)"""
    denom = (1 + sigma**2) ** 0.5

    inner = mu / denom - mu * sigma**2 / denom**3
    return norm_cdf(mu / denom) + norm_pdf(mu / denom) * inner


def gelu_poly_ev(n: int, mu: ArrayType, sigma: ArrayType) -> ArrayType:
    """Compute E[x^n * GELU(x)] analytically where x ~ N(mu, sigma^2)"""
    xp = array_api_compat.array_namespace(mu, sigma)
    ev = xp.zeros_like(mu)

    for k in range(n + 2):
        coef = math.comb(n + 1, k) * mu ** (n + 1 - k) * sigma**k
        ev += coef * integral_noncentral_t_whole_line(k, sigma, mu)

    return ev


def integral_noncentral_t(n: int, a, b):
    """
    Compute the integral of x^n * G'(x) * G(ax + b) dx from 0 to infinity
    using the noncentral t-distribution formula.

    Parameters:
    - n: int, the power of x in the integral
    - a: float, coefficient of x inside the CDF G(ax + b)
    - b: float, shift in the argument of the CDF G(ax + b)

    Returns:
    - result: float, integral value using the noncentral t-distribution formula
    """
    f = n + 1  # degrees of freedom
    delta = -b  # noncentrality parameter
    term1 = gamma((n + 1) / 2) * 2 ** ((n - 1) / 2) / math.sqrt(2 * math.pi)
    prob_term = nct.cdf(
        a * math.sqrt(n + 1), f, delta
    )  # Pr{T_f < a sqrt(n+1) | delta=-b}
    return term1 * prob_term


def integral_noncentral_t_whole_line(n, a, b):
    """
    Compute the integral of x^n * G'(x) * G(ax + b) dx from -infinity to infinity.
    """
    # First part: Integral from 0 to infinity (given by the nct formula)
    positive_part = integral_noncentral_t(n, a, b)
    negative_part = integral_noncentral_t(n, -a, b)

    # Second part: Integral from -infinity to 0
    # Equivalent to the integral of (-x)^n * G'(-x) * G(-ax + b) dx from 0 to infinity
    # Even n: (-x)^n = x^n, so 2nd part is the same form with a replaced by -a
    # Odd n: (-x)^n = -x^n, so 2nd part is negative of original with a replaced by -a
    if n % 2:
        negative_part = -negative_part

    # Total integral over the whole real line
    return positive_part + negative_part
