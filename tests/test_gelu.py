from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.stats import norm

from polyapprox.gelu import gelu, gelu_ev, gelu_poly_ev, gelu_prime, gelu_prime_ev

jax.config.update("jax_enable_x64", True)


def jax_test(f, target, *args, atol=1e-12):
    res = f(*map(jnp.array, args))
    np.testing.assert_allclose(res, target, atol=atol)


def torch_test(f, target, *args, atol=1e-12):
    res = f(*map(lambda x: torch.tensor(x, dtype=torch.double), args))
    np.testing.assert_allclose(res, target, atol=atol)


@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_gelu_evs(mu, sigma):
    analytic_ev = gelu_ev(np.array(mu), np.array(sigma))
    numerical_ev, err = quad(
        lambda x: gelu(x) * norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.inf
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err)
    torch_test(gelu_ev, numerical_ev, mu, sigma, atol=err)


@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_gelu_prime_evs(mu, sigma):
    analytic_ev = gelu_prime_ev(np.array(mu), np.array(sigma))
    numerical_ev, err = quad(
        lambda x: gelu_prime(x) * norm.pdf(x, loc=mu, scale=sigma),
        -np.inf,
        np.inf,
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err)
    torch_test(gelu_prime_ev, numerical_ev, mu, sigma, atol=err)


@given(st.integers(0, 5), st.floats(-4, 4), st.floats(0.1, 10))
def test_gelu_poly_evs(n, mu, sigma):
    analytic_ev = gelu_poly_ev(n, np.array(mu), np.array(sigma))
    numerical_ev, err = quad(
        lambda x: x**n * gelu(x) * norm.pdf(x, loc=mu, scale=sigma),
        -np.inf,
        np.inf,
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err)
    torch_test(partial(gelu_poly_ev, n), numerical_ev, mu, sigma, atol=err)
