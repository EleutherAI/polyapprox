from dataclasses import dataclass
from functools import partial
from typing import Literal

from numpy.typing import ArrayLike, NDArray
import numpy as np

from .extra import sigmoid, sigmoid_prime, swish, swish_prime
from .gelu import gelu_ev, gelu_prime_ev, gelu_poly_ev
from .integrate import bivariate_product_moment, gauss_hermite, master_theorem
from .relu import relu_ev, relu_prime_ev, relu_poly_ev


@dataclass(frozen=True)
class OlsResult:
    alpha: NDArray
    """Intercept of the linear model."""

    beta: NDArray
    """Coefficients of the linear model."""

    gamma: NDArray | None = None
    """Coefficients for second-order interactions, if available."""

    fvu: float | None = None
    """Fraction of variance unexplained, if available.
    
    Currently only implemented for ReLU activations.
    """

    def __call__(self, x: NDArray) -> NDArray:
        """Evaluate the linear model at the given inputs."""
        y = x @ self.beta + self.alpha

        if self.gamma is not None:
            outer = np.einsum('ij,ik->ijk', x, x)

            rows, cols = np.tril_indices(x.shape[1])
            y += outer[:, rows, cols] @ self.gamma

        return y


# Mapping from activation functions to EVs
ACT_TO_EVS = {
    'gelu': gelu_ev,
    'relu': relu_ev,
    'sigmoid': partial(gauss_hermite, sigmoid),
    'swish': partial(gauss_hermite, swish),
    'tanh': partial(gauss_hermite, np.tanh),
}
# Mapping from activation functions to EVs of their derivatives
ACT_TO_PRIME_EVS = {
    'gelu': gelu_prime_ev,
    'relu': relu_prime_ev,
    'sigmoid': partial(gauss_hermite, sigmoid_prime),
    'swish': partial(gauss_hermite, swish_prime),
    'tanh': partial(gauss_hermite, lambda x: 1 - np.tanh(x)**2),
}
ACT_TO_POLY_EVS = {
    'gelu': gelu_poly_ev,
    'relu': relu_poly_ev,
}


def ols(
    W1: NDArray, 
    b1: NDArray,
    W2: NDArray,
    b2: NDArray,
    *,
    act: str = 'gelu',
    mean: NDArray | None = None,
    cov: NDArray | None = None,
    order: Literal['linear', 'quadratic'] = 'linear',
    return_fvu: bool = False,
) -> OlsResult:
    """Ordinary least squares approximation of a single hidden layer MLP.

    Args:
        W1: Weight matrix of the first layer.
        b1: Bias vector of the first layer.
        W2: Weight matrix of the second layer.
        b2: Bias vector of the second layer.
        mean: Mean of the input distribution. If None, the mean is zero.
        cov: Covariance of the input distribution. If None, the covariance is the
            identity matrix.
        return_fvu: Whether to compute the fraction of variance unexplained.
            This is only available for ReLU activations, and can be computationally
            expensive for large networks.
    """
    d_input = W1.shape[1]

    # Preactivations are Gaussian; compute their mean and standard deviation
    if cov is not None:
        print(W1.shape, cov.shape, W1.T.shape)
        preact_cov = W1 @ cov @ W1.T
        cross_cov = cov @ W1.T
    else:
        preact_cov = W1 @ W1.T
        cross_cov = W1.T

    preact_mean = b1.copy()
    preact_var = np.diag(preact_cov)
    preact_std = np.sqrt(preact_var)
    if mean is not None:
        preact_mean += W1 @ mean

    try:
        act_ev = ACT_TO_EVS[act]
        act_prime_ev = ACT_TO_PRIME_EVS[act]
    except KeyError:
        raise ValueError(f"Unknown activation function: {act}")

    # Apply Stein's lemma to compute cross-covariance of the input
    # with the activations. We need the expected derivative of the
    # activation function with respect to the preactivation.
    act_prime_mean = act_prime_ev(preact_mean, preact_std)
    output_cross_cov = (cross_cov * act_prime_mean) @ W2.T

    # Compute expectation of act_fn(x) for each preactivation
    act_mean = act_ev(preact_mean, preact_std)
    output_mean = W2 @ act_mean + b2

    # beta = Cov(x)^-1 Cov(x, f(x))
    if cov is not None:
        beta = np.linalg.solve(cov, output_cross_cov)
    else:
        beta = output_cross_cov

    alpha = output_mean
    if mean is not None:
        alpha -= beta.T @ mean

    if order == 'quadratic':
        cov = cov_x = cov if cov is not None else np.eye(d_input)
        mu = mean if mean is not None else np.zeros(d_input)
        rows, cols = np.tril_indices(d_input)

        cov_y = W1 @ W1.T
        xcov = cov_x @ W1.T

        cov_x = cov
        mean_y = mu @ W1.T + b1
        var_x = np.diag(cov_x)

        Cov_x = np.array([
            [var_x[rows], cov_x[rows, cols]],
            [cov_x[cols, rows], var_x[cols]],
        ]).T

        Xcov = np.stack([
            xcov[rows],
            xcov[cols],
        ]).T

        Mean_x = np.array([mu[rows], mu[cols]]).T
        Var_y = np.diag(cov_y)

        coefs = master_theorem(
            Mean_x, Cov_x, mean_y[..., None], Var_y[..., None], Xcov
        )

        # Compute univariate integrals
        try:
            poly_ev = ACT_TO_POLY_EVS[act]
        except KeyError:
            raise ValueError(f"Quadratic not implemented for activation: {act}")
        
        quad = poly_ev(2, preact_mean, preact_std)
        lin = poly_ev(1, preact_mean, preact_std)
        const = poly_ev(0, preact_mean, preact_std)
        E_gy_x1x2 = (
            coefs[0] * quad[:, None] +
            coefs[1] * lin[:, None] +
            coefs[2] * const[:, None]
        )
        quad_xcov = W2 @ (E_gy_x1x2 - np.outer(const, (rows == cols)))
        gamma = quad_xcov / (1 + (rows == cols))

        # adjust constant term
        alpha -= (rows == cols) @ gamma.T
    else:
        gamma = None

    # For ReLU, we can compute the covariance matrix of the activations, which is
    # useful for computing the fraction of variance unexplained in closed form.
    if act == 'relu' and return_fvu:
        # TODO: Figure out what is wrong with our implementation for non-zero means
        assert mean is None, "FVU computation is not implemented for non-zero means"
        rhos = preact_cov / np.outer(preact_std, preact_std)

        # Compute the raw second moment matrix of the activations
        act_m2 = bivariate_product_moment(
            0.0, 0.0, rhos,
            mean_x=preact_mean[:, None],
            mean_y=preact_mean[None],
            std_x=preact_std[:, None],
            std_y=preact_std[None],
            unconditional=True,
        )

        # E[MLP(x)^T MLP(x)]
        mlp_scale = np.trace(W2 @ act_m2 @ W2.T) + 2 * act_mean.T @ W2.T @ b2 + b2 @ b2

        # E[g(x)^T MLP(x)] where g(x) is the linear predictor
        x_moment = cross_cov + (np.outer(mean, output_mean) if mean is not None else 0)
        inner_prod = np.trace(beta.T @ x_moment) + alpha.T @ output_mean

        # E[g(x)^T g(x)] where g(x) is the linear predictor
        inner = 2 * mean.T @ beta @ alpha if mean is not None else 0
        lin_scale = np.trace(beta.T @ cov @ beta) + inner + alpha.T @ alpha

        # Fraction of variance unexplained
        denom = mlp_scale - output_mean @ output_mean
        fvu = (mlp_scale - 2 * inner_prod + lin_scale) / denom
    else:
        fvu = None

    return OlsResult(alpha, beta, fvu=fvu, gamma=gamma)


def glu_mean(
    W: NDArray,
    V: NDArray,
    b1: ArrayLike = 0.0,
    b2: ArrayLike = 0.0,
    *,
    act: str = 'sigmoid',
    mean: NDArray | None = None,
    cov: NDArray | None = None,
):
    """Analytically compute the mean output of a gated linear unit (GLU).

    See "GLU Variants Improve Transformer" <https://arxiv.org/abs/2002.05202>
    by Shazeer (2020) for more details.
    """
    # The network takes the form σ(W @ x + b1) * (V @ x + b2)
    # Let y = W @ x + b1 and z = V @ x + b2
    if cov is not None:
        # Cross-covariance matrix of y and z
        cross_cov = W @ cov @ V.T

        y_std = np.diag(W @ cov @ W.T) ** 0.5
        # z_std = np.diag(V @ cov @ V.T) ** 0.5
    else:
        cross_cov = W @ V.T
        y_std = np.linalg.norm(W, axis=1)
        # z_std = np.linalg.norm(V, axis=1)

    y_mean = np.array(b1)
    if mean is not None:
        y_mean += W @ mean

    z_mean = np.array(b2)
    if mean is not None:
        z_mean += V @ mean
    
    try:
        act_ev = ACT_TO_EVS[act]
        act_prime_ev = ACT_TO_PRIME_EVS[act]
    except KeyError:
        raise ValueError(f"Unknown activation function: {act}")
    
    # Apply Stein's lemma to compute
    # E[GLU(x)]_i = E[σ(y_i) * z_i] = Cov(σ(y_i), z_i) + E[σ(y_i)] * E[z_i]
    # The lemma says that Cov(σ(y_i), z_i) = Cov(y_i, z_i) * E[σ'(y_i)]
    # so we need to compute E[σ'(y_i)] for each i
    act_mean = act_ev(y_mean, y_std)
    output_mean = np.diag(cross_cov) * act_prime_ev(y_mean, y_std) + act_mean * z_mean

    return output_mean
