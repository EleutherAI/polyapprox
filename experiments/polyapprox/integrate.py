from itertools import combinations, product
from typing import Callable

from numpy.typing import ArrayLike, NDArray
from scipy.special import owens_t, roots_hermite
from scipy.stats import norm
import numpy as np


def bivariate_product_moment(
    h,
    k,
    rho,
    *,
    mean_x: ArrayLike = 0.0,
    mean_y: ArrayLike = 0.0,
    std_x: ArrayLike = 1.0,
    std_y: ArrayLike = 1.0,
    unconditional = False,
):
    h = np.asarray((h - mean_x) / std_x)
    k = np.asarray((k - mean_y) / std_y)

    mean_x = np.asarray(mean_x)
    mean_y = np.asarray(mean_y)
    std_x = np.asarray(std_x)
    std_y = np.asarray(std_y)

    eps = np.finfo(float).eps
    rho = np.clip(rho, -1.0 + eps, 1.0 - eps)

    # Define constants
    denom = np.sqrt(1 - rho**2)
    numer = np.sqrt(h**2 - 2 * rho * h * k + k**2)

    # Z(x): Standard normal PDF
    Z_h = norm.pdf(h)
    Z_k = norm.pdf(k)

    # Q(x): Standard normal CDF
    Q_k_given_h = 1 - norm.cdf((k - rho * h) / denom)
    Q_h_given_k = 1 - norm.cdf((h - rho * k) / denom)
    
    # Compute L(h, k; rho), the probability in the truncated region
    L_hk_rho = bivariate_normal_cdf(-h, -k, rho)
    
    # Product moment m11 formula
    term1 = rho * L_hk_rho
    term2 = rho * h * Z_h * Q_k_given_h
    term3 = rho * k * Z_k * Q_h_given_k
    term4 = (denom / np.sqrt(2 * np.pi)) * norm.pdf(numer / denom)
    
    # Correct answer if mean_x = mean_y = 0
    m11 = std_x * std_y * (term1 + term2 + term3 + term4)

    # Account for being noncentered
    # E[(s_x z_x + m_x) (s_y z_y + m_y)] =
    # s_x s_y E[z_x * z_y] + m_x s_y E[z_y] + m_y s_x E[z_x] + m_x m_y
    # Compute E[z_x] and E[z_y] using the truncated first moments
    m10 = (Z_h * Q_k_given_h + rho * Z_k * Q_h_given_k)
    m01 = (rho * Z_h * Q_k_given_h + Z_k * Q_h_given_k)
    m11 += std_x * mean_y * m10 + std_y * mean_x * m01 + mean_x * mean_y * L_hk_rho

    # Divide by the probability that we would end up in the truncated region
    if not unconditional:
        m11 /= L_hk_rho

    return m11


def bivariate_normal_cdf(
    x,
    y,
    rho,
    *,
    mean_x = 0.0,
    mean_y=0.0,
    std_x=1.0,
    std_y=1.0,
    tail: bool = False,
):
    """
    Computes the bivariate normal cumulative distribution function.
    """
    # Normalize x and y
    x = np.asarray((x - mean_x) / std_x)
    y = np.asarray((y - mean_y) / std_y)

    # Compute the tail probability if asked
    if tail:
        x = -x
        y = -y

    # Nudge x and y away from zero to avoid division by zero
    eps = np.finfo(x.dtype).tiny
    x = x + np.where(x < 0, -eps, eps)
    y = y + np.where(y < 0, -eps, eps)

    rx = (y - rho * x) / (x * np.sqrt(1 - rho**2))
    ry = (x - rho * y) / (y * np.sqrt(1 - rho**2))

    # Subtract 1/2 when x and y have the opposite sign
    mask = (x * y > 0) | ((x * y == 0) & (x + y >= 0))
    beta = np.where(mask, 0.0, 0.5)

    # Calculate the value of the bivariate CDF using the provided formula
    term1 = 0.5 * (norm.cdf(x) + norm.cdf(y))
    result = term1 - owens_t(x, rx) - owens_t(y, ry) - beta

    # Numerically stable fallback for when x and y are close to zero
    backup = 0.25 + np.arcsin(rho) / (2 * np.pi)
    fallback = np.isclose(x, 0.0) & np.isclose(y, 0.0)

    return np.where(fallback, backup, result)


def gauss_hermite(
    f: Callable,
    mu: ArrayLike = 0.0,
    sigma: ArrayLike = 1.0,
    num_points: int = 50,
) -> NDArray:
    """
    Compute E[f(x)] where x ~ N(mu, sigma^2) using Gauss-Hermite quadrature.

    Parameters:
    - mu: array-like, means
    - sigma: array-like, standard deviations
    - num_points: int, number of quadrature points

    Returns:
    - expectations: array-like, E[f(x)] for each (mu, sigma)
    """
    # Obtain Gauss-Hermite nodes and weights
    nodes, weights = roots_hermite(num_points)  # Nodes: z_i, Weights: w_i

    # Reshape for broadcasting
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)

    # See example in https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
    grid = mu[:, None] + sigma[:, None] * np.sqrt(2) * nodes

    # Compute the weighted sum
    return np.dot(f(grid), weights) / np.sqrt(np.pi)


def isserlis(cov: np.ndarray, indices: list[int]):
    """Compute `E[prod_{i=1}^n X_i]` for jointly Gaussian X_i with covariance `cov`.

    This is an implementation of Isserlis' theorem, also known as Wick's formula. It is
    super-exponential in the number of indices, so it is only practical for small `n`.
    
    Args:
        cov: Covariance matrix or batch of covariance matrices of shape (..., n, n).
        indices: List of indices 0 < i < n for which to compute the expectation.
    """
    return sum(
        np.prod([cov[..., a, b] for a, b in partition], axis=0)
        for partition in pair_partitions(indices)
    )


def master_theorem(
    mu_x: np.ndarray,
    cov_x: np.ndarray,
    mu_y: np.ndarray,
    var_y: np.ndarray,
    xcov: np.ndarray
):
    """Reduce the multivariate integral E[g(y) * x1 * x2 ...] to k univariate integrals."""
    *batch_shape, k = mu_x.shape
    *batch_shape2, k2, k3 = cov_x.shape

    assert batch_shape == batch_shape2, "Batch dimensions must match"
    assert k == k2 == k3, "Dimension of means and covariances must match"

    # TODO: Make this work for constant X0 by choosing a "pivot" variable
    # from among the X_i with the largest variance, then computing all the
    # conditional expectations with respect to that variable.
    assert np.all(var_y > 0.0), "X0 must have positive variance"

    # Coefficients and intercepts for each conditional expectation
    a = xcov / var_y[..., None]
    b = -a * mu_y[..., None] + mu_x

    # Covariance matrix of the residuals
    eps_cov = cov_x - xcov[..., None] * a[..., None, :]

    # Polynomial coefficients get appended here
    coefs = []

    # Iterate over polynomial terms
    for m in range(k + 1):
        # Running sum of terms in the coefficient
        coef = np.zeros(batch_shape)

        # Enumerate every combination of m unique a terms
        for comb in combinations(range(k), m):
            prefix = np.ones(batch_shape)

            # Multiply together the a terms
            for i in comb:
                prefix = prefix * a[..., i]

            # Get a list of indices that are left over. The correctness of this
            # depends on combinations returning the indices in sorted order.
            remaining = list(range(k))
            for idx in reversed(comb):
                del remaining[idx]

            # Iterate over all bitstrings of length k - m, and each bit in
            # the bitstring tells us whether to pick a `b` factor
            # or a residual factor.
            for bitstring in product([0, 1], repeat=len(remaining)):
                num_residuals = sum(bitstring)
                residual_indices = []

                # Skip terms with an odd number of residuals
                if num_residuals % 2:
                    continue

                # Running product of factors
                term = prefix

                # Multiply together the b factors
                for i, bit in zip(remaining, bitstring):
                    if bit:
                        residual_indices.append(i)
                    else:
                        term = term * b[..., i]
                
                # Apply Isserlis' theorem to the residual factors
                if residual_indices:
                    iss = isserlis(eps_cov, residual_indices)
                    term = term * iss

                # Add the term to the coefficient
                coef = coef + term
        
        coefs.append(coef)

    # Make descending order
    coefs.reverse()
    return coefs


def pair_partitions(elements: list):
    """Iterate over all partitions of a list into pairs."""
    # The empty set can be "partitioned" into the empty partition
    if not elements:
        yield []
        return

    pivot = elements[0]
    for i in range(1, len(elements)):
        partner = elements[i]
        remaining = elements[1:i] + elements[i+1:]

        for rest in pair_partitions(remaining):
            yield [(pivot, partner)] + rest
