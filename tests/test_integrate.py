import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from scipy.integrate import dblquad, quad
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm

from polyapprox.extra import sigmoid, swish
from polyapprox.gelu import gelu
from polyapprox.integrate import (
    bivariate_normal_cdf,
    gauss_hermite,
    isserlis,
    master_theorem,
)


@given(st.floats(-0.999, 0.999), st.floats(-1e6, 1e6), st.floats(-1e6, 1e6))
def test_bivariate_normal_cdf(rho, x, y):
    ours = bivariate_normal_cdf(x, y, rho)
    scipy = mvn.cdf([x, y], cov=[[1, rho], [rho, 1]])

    # Sometimes SciPy returns NaNs for large values while we don't
    # Don't penalize us for being more numerically stable
    if not np.isnan(scipy):
        np.testing.assert_allclose(ours, scipy, atol=1e-12)


@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_gauss_hermite(mu, sigma):
    funcs = [gelu, sigmoid, swish, np.tanh]

    for f in funcs:
        numerical, err = quad(
            lambda x: f(x) * norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.inf
        )

        analytical = gauss_hermite(f, np.array(mu), np.array(sigma), num_points=5000)
        assert abs(numerical - analytical) < err + np.finfo(float).eps

        # Test PyTorch backend
        mu_torch = torch.tensor(mu, dtype=torch.double)
        sigma_torch = torch.tensor(sigma, dtype=torch.double)
        analytical_torch = gauss_hermite(f, mu_torch, sigma_torch, num_points=5000)
        assert abs(numerical - analytical_torch) < err + torch.finfo(torch.double).eps


def test_isserlis():
    B = np.random.randn(2, 2) / np.sqrt(2)
    K = B @ B.T

    analytical = isserlis(K, [0] * 4 + [1] * 6)
    numerical, error = dblquad(
        lambda y, x: x**4 * y**6 * mvn.pdf([x, y], cov=K),
        -np.inf,
        np.inf,
        -np.inf,
        np.inf,
    )
    np.testing.assert_allclose(numerical, analytical, atol=error)


def test_master_theorem():
    B = np.random.randn(2, 2) / np.sqrt(2)
    K = B @ B.T

    analytical = master_theorem(K, [0] * 4 + [1] * 6)
    numerical, error = dblquad(
        lambda y, x: x**4 * y**6 * mvn.pdf([x, y], cov=K),
        -np.inf,
        np.inf,
        -np.inf,
        np.inf,
    )
    np.testing.assert_allclose(numerical, analytical, atol=error)
