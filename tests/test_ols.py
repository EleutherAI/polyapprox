import numpy as np
import pytest
import statsmodels.api as sm
from scipy.stats import multivariate_normal as mvn
from statsmodels.regression.linear_model import OLS

from polyapprox.gelu import gelu
from polyapprox.ols import ols


def relu(x):
    return np.maximum(0, x)


def test_ols_relu():
    d = 10
    nil = np.zeros(d)

    W1 = np.random.randn(d, d)
    W2 = np.random.randn(d, d)

    # When x ~ N(0, 1) and there are no biases, the coefficients take the intuitive
    # form: 0.5 * W2 @ W1
    lin_res = ols(W1, nil, W2, nil, act="relu", order="linear")
    quad_res = ols(W1, nil, W2, nil, act="relu", order="quadratic")
    np.testing.assert_allclose(lin_res.beta.T, 0.5 * W2 @ W1)
    np.testing.assert_allclose(quad_res.beta.T, 0.5 * W2 @ W1)

    # Monte Carlo check that the FVU is below 1
    n = 10_000
    x = np.random.randn(n, d)
    y = relu(x @ W1.T) @ W2.T

    # Quadratic FVU should be lower than linear FVU, which should be lower than 1
    lin_fvu = np.square(y - lin_res(x)).sum() / np.square(y).sum()
    quad_fvu = np.square(y - quad_res(x)).sum() / np.square(y).sum()
    assert quad_fvu < lin_fvu < 1

    # Check the trivial case where the activation is the identity
    lin_res = ols(W1, nil, W2, nil, act="identity", order="linear")
    quad_res = ols(W1, nil, W2, nil, act="identity", order="quadratic")
    np.testing.assert_allclose(lin_res.beta.T, W2 @ W1)
    np.testing.assert_allclose(quad_res.beta.T, W2 @ W1)

    assert lin_res.gamma is None and quad_res.gamma is not None
    np.testing.assert_allclose(lin_res.alpha, 0)
    np.testing.assert_allclose(quad_res.alpha, 0)
    np.testing.assert_allclose(quad_res.gamma, 0)


@pytest.mark.parametrize("act", ["gelu", "relu"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_ols_monte_carlo(act: str, k: int):
    # Determinism
    np.random.seed(0)

    # Choose activation function
    act_fn = gelu if act == "gelu" else relu

    # Implement the MLP
    def mlp(x, W1, b1, W2, b2):
        return act_fn(x @ W1.T + b1) @ W2.T + b2

    d_in = 10
    d_inner = 1_000
    d_out = 1

    # Construct a random Gaussian mixture with k components
    # random psd matrix d_in x d_in
    A = np.random.randn(k, d_in, d_in) / np.sqrt(d_in)
    cov_x = A @ A.mT
    mu_x = np.random.randn(k, d_in)

    W1 = np.random.randn(d_inner, d_in) / np.sqrt(d_in)
    W2 = np.random.randn(d_out, d_inner) / np.sqrt(d_inner)
    b1 = np.random.randn(d_inner) / np.sqrt(d_in)
    b2 = np.random.randn(d_out)

    # Compute analytic coefficients
    analytic = ols(W1, b1, W2, b2, act=act, cov=cov_x.squeeze(), mean=mu_x.squeeze())

    # Generate Monte Carlo data
    x = np.concatenate(
        [mvn.rvs(mean=mu_x[i], cov=cov_x[i], size=10_000) for i in range(k)]
    )
    y = mlp(x, W1, b1, W2, b2)

    # Use statsmodels to approximate the coefficients
    X = sm.add_constant(x)
    empirical = OLS(y.squeeze(), X).fit()

    # Check that analytic coefficients are within the confidence interval
    lo, hi = empirical.conf_int(0.01).T
    analytic_beta = analytic.beta.squeeze()

    assert lo[0] < analytic.alpha < hi[0]
    assert np.all((lo[1:] < analytic_beta) & (analytic_beta < hi[1:]))
