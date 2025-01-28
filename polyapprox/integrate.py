import math
from itertools import combinations, product
from typing import Callable

import array_api_compat
import numpy as np
from scipy.special import roots_hermite

from .backends import ArrayType


def gauss_hermite(
    f: Callable,
    mu: ArrayType,
    sigma: ArrayType,
    num_points: int = 50,
) -> ArrayType:
    """
    Compute E[f(x)] where x ~ N(mu, sigma^2) using Gauss-Hermite quadrature.

    Parameters:
    - mu: array-like, means
    - sigma: array-like, standard deviations
    - num_points: int, number of quadrature points

    Returns:
    - expectations: array-like, E[f(x)] for each (mu, sigma)
    """
    xp = array_api_compat.array_namespace(mu, sigma)

    # Obtain Gauss-Hermite nodes and weights
    nodes, weights = map(xp.asarray, roots_hermite(num_points))

    # See example in https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
    grid = mu[..., None] + sigma[..., None] * math.sqrt(2) * nodes

    # Compute the weighted sum
    prods = xp.einsum("...i,...i->...", f(grid), weights)
    return prods / math.sqrt(math.pi)


def isserlis(cov: ArrayType, indices: list[int]) -> ArrayType:
    """Compute `E[prod_{i=1}^n X_i]` for jointly Gaussian X_i with covariance `cov`.

    This is an implementation of Isserlis' theorem, also known as Wick's formula. It is
    super-exponential in the number of indices, so it is only practical for small `n`.

    Args:
        cov: Covariance matrix or batch of covariance matrices of shape (..., n, n).
        indices: List of indices 0 < i < n for which to compute the expectation.
    """
    xp = array_api_compat.array_namespace(cov)
    res = xp.zeros(cov.shape[:-2])

    for partition in pair_partitions(indices):
        res += xp.prod(xp.stack([cov[..., a, b] for a, b in partition]), axis=0)

    return res


def master_theorem(
    mu_x: ArrayType,
    var_x: ArrayType,
    mu_y: ArrayType,
    cov_y: ArrayType,
    xcov: ArrayType,
) -> list[ArrayType]:
    """Reduce multivariate integral E[g(x) * y1 * y2 ...] to k univariate integrals."""
    xp = array_api_compat.array_namespace(mu_y, cov_y, mu_x, var_x, xcov)

    *batch_shape, k = mu_y.shape
    *batch_shape2, k2, k3 = cov_y.shape

    assert batch_shape == batch_shape2, "Batch dimensions must match"
    assert k == k2 == k3, "Dimension of means and covariances must match"

    # TODO: Make this work for constant X by choosing a "pivot" variable
    # from among the Y_i with the largest variance, then computing all the
    # conditional expectations with respect to that variable.
    assert xp.all(var_x > 0.0), "X must have positive variance"

    # Coefficients and intercepts for each conditional expectation
    a = xcov / var_x[..., None]
    b = -a * mu_x[..., None] + mu_y

    # Covariance matrix of the residuals
    eps_cov = cov_y - xcov[..., None] * a[..., None, :]

    # Polynomial coefficients get appended here
    coefs = []

    # Iterate over polynomial terms
    for m in range(k + 1):
        # Running sum of terms in the coefficient
        coef = xp.zeros(batch_shape)

        # Enumerate every combination of m unique a terms
        for comb in combinations(range(k), m):
            prefix = xp.ones(batch_shape)

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
        remaining = elements[1:i] + elements[i + 1 :]

        for rest in pair_partitions(remaining):
            yield [(pivot, partner)] + rest


def quadratic_feature_mean_cov(
    mu: ArrayType, Sigma: ArrayType
) -> tuple[ArrayType, ArrayType]:
    """
    Compute the mean and covariance of psi(X) = [X, phi(X)] in a fully vectorized way.
    X ~ N(mu, Sigma) in R^d, and phi(X) is all pairwise products X_i X_j (i <= j).

    Returns:
      mean_psi (shape: d + d(d+1)//2)
      cov_psi  (shape: (d + d(d+1)//2, d + d(d+1)//2))
    """
    xp = array_api_compat.array_namespace(mu, Sigma)
    d = mu.shape[0]

    # Indices for upper-triangular pairs (i, j) with i <= j
    I_triu, J_triu = map(xp.asarray, np.triu_indices(d))
    n_features = len(I_triu)  # = d(d+1)//2

    # ----------------------------------------------------------------
    # 1) Mean of psi(X) = [ E[X],  E[phi(X)] ]
    #    - E[X] = mu
    #    - E[X_i X_j] = Sigma[i,j] + mu[i]*mu[j]
    #      can be vectorized using the upper-tri part of (Sigma + mu mu^T).
    # ----------------------------------------------------------------

    # Vectorized approach to E[phi(X)]:
    # We'll take the upper-triangular part of (Sigma + mu mu^T).
    M = Sigma + xp.outer(mu, mu)  # (d, d)
    mean_phi = M[I_triu, J_triu]  # (n_features,)

    mean_psi = xp.concat([mu, mean_phi])  # (d + n_features,)

    # ----------------------------------------------------------------
    # 2) Covariance blocks
    #    Cov[psi(X)] = [[ Cov(X,X), Cov(X, phi(X))    ],
    #                   [ Cov(phi(X), X), Cov(phi(X), phi(X)) ]]
    # ----------------------------------------------------------------

    # A) Cov(X, X) = Sigma
    cov_xx = Sigma  # (d, d)

    # B) Cov(phi(X), phi(X)) => can use the fully vectorized approach
    #    from the previous discussion.  We'll call it cov_phi here.
    SIR = Sigma[I_triu[:, None], I_triu[None, :]]  # shape (n_features, n_features)
    SJS = Sigma[J_triu[:, None], J_triu[None, :]]
    SIS = Sigma[I_triu[:, None], J_triu[None, :]]
    SJR = Sigma[J_triu[:, None], I_triu[None, :]]

    Mii = mu[I_triu]  # shape (n_features,)
    Mjj = mu[J_triu]  # shape (n_features,)

    cov_phi = (
        SIR * SJS
        + SIS * SJR
        + xp.outer(Mii, Mii) * SJS
        + xp.outer(Mii, Mjj) * SJR
        + xp.outer(Mjj, Mii) * SIS
        + xp.outer(Mjj, Mjj) * SIR
    )  # shape (n_features, n_features)

    # C) Cov(X, phi(X)) => B in block matrix
    #    Cov(X_i, X_j X_k) = mu_j * Sigma[i,k] + mu_k * Sigma[i,j].
    #
    # Vectorized build: for each pair (j,k),
    # we want column idx = mu_j * Sigma[:,k] + mu_k * Sigma[:,j].
    # I_triu[idx] = j, J_triu[idx] = k
    # so B[:, idx] = mu[j] * Sigma[:, k] + mu[k] * Sigma[:, j].

    # sigma_jk = Sigma[:, J_triu] => shape (d, n_features)
    # sigma_kj = Sigma[:, I_triu] => shape (d, n_features)
    # multiply columns by mu[j], mu[k]
    B = (
        mu[I_triu][None, :] * Sigma[:, J_triu] + mu[J_triu][None, :] * Sigma[:, I_triu]
    )  # shape (d, n_features)

    # Now we assemble everything into a (d + n_features)-dim block matrix
    cov_psi = xp.zeros((d + n_features, d + n_features))

    # top-left block: Cov(X, X)
    cov_psi[0:d, 0:d] = cov_xx
    # top-right block: Cov(X, phi(X)) = B
    cov_psi[0:d, d:] = B
    # bottom-left block: Cov(phi(X), X) = B^T
    cov_psi[d:, 0:d] = B.T
    # bottom-right block: Cov(phi(X), phi(X))
    cov_psi[d:, d:] = cov_phi

    return mean_psi, cov_psi
