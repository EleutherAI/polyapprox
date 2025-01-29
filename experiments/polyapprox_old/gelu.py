import math

from numpy.typing import ArrayLike, NDArray
from scipy.special import gamma
from scipy.stats import nct, norm
import numpy as np


def gelu(x):
    """Gaussian Error Linear Unit (GELU) activation function"""
    return x * norm.cdf(x)


def gelu_prime(x):
    """Derivative of GELU(x)"""
    return norm.cdf(x) + x * norm.pdf(x)


def gelu_dbl_prime(x):
    """Second derivative of GELU(x)"""
    return (2 - x ** 2) * norm.pdf(x)


def gelu_ev(mu, sigma):
    """Expected value of GELU(x) under N(mu, sigma)"""
    denom = np.sqrt(1 + sigma ** 2)
    return mu * norm.cdf(mu / denom) + (sigma ** 2 / denom) * norm.pdf(mu / denom)


def gelu_prime_ev(mu, sigma):
    """Expected value of GELU'(x) under N(mu, sigma)"""
    denom = np.sqrt(1 + sigma ** 2)

    inner = mu / denom - mu * sigma ** 2 / denom ** 3
    return norm.cdf(mu / denom) + norm.pdf(mu / denom) * inner


def gelu_dbl_prime_ev(mu, sigma):
    """Expected value of GELU''(x) under N(mu, sigma)"""
    denom = np.sqrt(1 + sigma ** 2)

    inner1 = 1 / denom - 3 * mu ** 2 / denom ** 3
    inner2 = -3 * mu / denom ** 3 + 9 * mu ** 3 / denom ** 5
    return norm.cdf(mu / denom) + norm.pdf(mu / denom) * inner1 + norm.pdf(mu / denom) * inner2


def gelu_poly_ev(n: int, mu: ArrayLike, sigma: ArrayLike) -> NDArray:
    """Compute E[x^n * GELU(x)] analytically where x ~ N(mu, sigma^2)"""
    ev = np.zeros_like(mu)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    for k in range(n + 2):
        coef = math.comb(n + 1, k) * mu ** (n + 1 - k) * sigma ** k
        ev += coef * integral_noncentral_t_whole_line(k, sigma, mu)
    
    return ev


def integral_noncentral_t(n, a, b):
    """
    Compute the integral of x^n * G'(x) * G(ax + b) dx from 0 to infinity
    using the noncentral t-distribution formula.
    
    Parameters:
    - n: int, the power of x in the integral
    - a: float, coefficient of x inside the CDF G(ax + b)
    - b: float, shift in the argument of the CDF G(ax + b)
    
    Returns:
    - result: float, the computed integral value using the noncentral t-distribution formula
    """
    f = n + 1  # degrees of freedom
    delta = -b  # noncentrality parameter
    term1 = gamma((n + 1) / 2) * 2 ** ((n - 1) / 2) / np.sqrt(2 * np.pi)
    prob_term = nct.cdf(a * np.sqrt(n + 1), f, delta)  # Pr{T_f < a sqrt(n+1) | delta=-b}
    return term1 * prob_term


def integral_noncentral_t_whole_line(n, a, b):
    """
    Compute the integral of x^n * G'(x) * G(ax + b) dx from -infinity to infinity.
    """
    # First part: Integral from 0 to infinity (already given by the noncentral t-distribution formula)
    positive_part = integral_noncentral_t(n, a, b)
    negative_part = integral_noncentral_t(n, -a, b)

    # Second part: Integral from -infinity to 0
    # This is equivalent to the integral of (-x)^n * G'(-x) * G(-ax + b) dx from 0 to infinity
    # For even n, (-x)^n = x^n, so the second part is the same form with `a` replaced by `-a`
    # For odd n, (-x)^n = -x^n, so the second part is the negative of the original with `a` replaced by `-a`
    if n % 2:
        negative_part = -negative_part
    
    # Total integral over the whole real line
    return positive_part + negative_part