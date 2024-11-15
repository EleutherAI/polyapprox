from dataclasses import dataclass
from functools import partial
from typing import Literal
from math import floor
from numpy.typing import ArrayLike, NDArray
#import numpy as np
import torch
from torch import Tensor

from .extra import sigmoid, sigmoid_prime, swish, swish_prime
from .gelu import gelu_ev, gelu_prime_ev, gelu_poly_ev
from .integrate import bivariate_product_moment, gauss_hermite, master_theorem
from .relu import relu_ev, relu_prime_ev, relu_poly_ev


@dataclass(frozen=True)
class OlsResult:
    alpha: Tensor
    """Intercept of the linear model."""

    beta: Tensor
    """Coefficients of the linear model."""

    gamma: Tensor | None = None
    """Coefficients for second-order interactions, if available."""

    fvu: float | None = None
    """Fraction of variance unexplained, if available.
    
    Currently only implemented for ReLU activations.
    """
    @property
    def device(self) -> torch.device:
        return self.alpha.device
    
    def cpu(self) -> 'OlsResult':
        """Create a copy of this OlsResult with tensors moved to the CPU."""
        return OlsResult(
            alpha=self.alpha.cpu(),
            beta=self.beta.cpu(),
            gamma=self.gamma.cpu() if self.gamma is not None else None,
            fvu=self.fvu  # fvu is a float, no need to move to CPU
        )
        
    def get_gamma_tensor(self):
        nrows = floor((2*self.gamma.shape[-1]) ** 0.5)
        gamma_tensor = torch.zeros((self.gamma.shape[0], nrows, nrows))
        rows, cols = torch.tril_indices(nrows, nrows)
        gamma_tensor[:, rows, cols] = self.gamma
        return 0.5 * (gamma_tensor + gamma_tensor.mT)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the linear model at the given inputs."""
        y = x @ self.beta + self.alpha
        if self.gamma is not None:
            a = torch.einsum('bi,hij->bhj', x, self.get_gamma_tensor())
            y += torch.einsum('bj,bhj->bh', x, a)

        return y    


# Mapping from activation functions to EVs
ACT_TO_EVS = {
    'gelu': gelu_ev,
    'relu': relu_ev,
    'sigmoid': partial(gauss_hermite, sigmoid),
    'swish': partial(gauss_hermite, swish),
    'tanh': partial(gauss_hermite, torch.tanh),
}
# Mapping from activation functions to EVs of their derivatives
ACT_TO_PRIME_EVS = {
    'gelu': gelu_prime_ev,
    'relu': relu_prime_ev,
    'sigmoid': partial(gauss_hermite, sigmoid_prime),
    'swish': partial(gauss_hermite, swish_prime),
    'tanh': partial(gauss_hermite, lambda x: 1 - torch.tanh(x)**2),
}
ACT_TO_POLY_EVS = {
    'gelu': gelu_poly_ev,
    'relu': relu_poly_ev,
}


def ols(
    W1: torch.Tensor, 
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    *,
    act: str = 'gelu',
    mean: torch.Tensor | None = None,
    cov: torch.Tensor | None = None,
    order: Literal['linear', 'quadratic'] = 'linear',
    return_fvu: bool = False,
    debug_mode: bool = False
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
    
    def debug_print(*args):
        if debug_mode:
            print(*args)
    
    d_input = W1.shape[1]
    if mean is not None:
        mean = mean.squeeze()
    
    cov = torch.eye(d_input) if cov is None else cov
    mean = torch.zeros(d_input) if mean is None else mean
    device = W1.device
    # Preactivations are Gaussian; compute their mean and standard deviation
    # For MNIST, add actual instance values in comments
    debug_print(f'0. First handle linear case.')
    debug_print(f'According to Nora, linear and quad should be split into separate cases, so only trust up to 4.')
    if cov is not None:
        debug_print(f'1. Cov provided, relevant shapes: W1 {W1.shape}, cov {cov.shape}, W1^T {W1.T.shape}')
        #MNIST: W1 [256,  784], cov [ 784,  784], W1.T [ 784, 256]
        #CIFAR: W1 [256, 3072], cov [3072, 3072], W1.T [3072, 256]
        preact_cov = W1 @ cov @ W1.T
        cross_cov = cov @ W1.T
        debug_print(f'Computing preact_cov = W1 @ cov @ W1.T, shape f{preact_cov.shape}')
        debug_print(f'Computing cross_cov = cov @ W1.T, shape f{cross_cov.shape}')
        #MNIST: preact_cov [ 256, 256], cross_cov [ 784,  256]
        #CIFAR: preact_cov [ 256, 256], cross_cov [3072,  256]
    else:
        debug_print(f'1. No cov provided, assuming identity. relevant shapes: W1 {W1.shape}, W1^T {W1.T.shape}')
        preact_cov = W1 @ W1.T
        cross_cov = W1.T
        debug_print(f'Computing preact_cov = W1 @ Id @ W1.T, shape f{preact_cov.shape}')
        debug_print(f'Computing cross_cov = Id @ W1.T, shape f{cross_cov.shape}')
        #MNIST: preact_cov [ 256, 256], cross_cov [ 784,  256]
        #CIFAR: preact_cov [ 256, 256], cross_cov [3072,  256]

    preact_mean = b1.clone().to(device)
    preact_var = torch.diag(preact_cov).to(device)
    preact_std = torch.sqrt(preact_var).to(device)
    debug_print(f'2. Preactivation mean (from b1): {preact_mean.shape}, variance: {preact_var.shape}, std: {preact_std.shape}')
    #MNIST: preact_mean [256], preact_var [256], preact_std [256]
    #CIFAR: preact_mean [256], preact_var [256], preact_std [256]
    
    if mean is not None:
        
        #preact_mean += W1 @ mean # OG. Apparently migrating to torch means need mean.T?
        debug_print(f'2. Mean not none, updated preact_mean += W1 @ mean')
        debug_print(f'W1 {W1.shape}, mean {mean.shape}, preact_mean {preact_mean.shape}')
        preact_mean += W1 @ mean.squeeze()
        debug_print(f'2. Mean not none, updated preact_mean += W1 @ mean')
        #MNIST: W1 [256,  784], mean [ 784], preact_mean [256]
        #CIFAR: W1 [256, 3072], mean [3072], preact_mean [256]

    try:
        act_ev = ACT_TO_EVS[act]
        act_prime_ev = ACT_TO_PRIME_EVS[act]
    except KeyError:
        raise ValueError(f"Unknown activation function: {act}")

    # Apply Stein's lemma to compute cross-covariance of the input
    # with the activations. We need the expected derivative of the
    # activation function with respect to the preactivation.
    debug_print('3. Applying Stein\'s lemma to compute the cross-covariance of the input. Uses preact mean & std. Stores in output_cross_cov')
    debug_print('Stein\'s lemma says that E[g(X)X^n] can be computed as a linear combination of E[g^(k)(X)] terms, kth derivatives')
    debug_print('Important to remember!! n is rarely larger than 2, IMO Nora overly generalized this. We will nonetheless roll with it.')
    # error rn, preact_std is numpy instead of torch. Check this.
    debug_print(f'preact_std {preact_std.shape}')
    # MNIST & CIFAR: preact_std [256]
    act_prime_mean = act_prime_ev(preact_mean, preact_std) # ideally same device.
    output_cross_cov = (cross_cov * act_prime_mean) @ W2.T
    debug_print(f'out_cross_cov = (cross_cov * act_prime_mean) @ W2.T')
    debug_print(f'cross_cov {cross_cov.shape},  {act_prime_mean.shape},  W2.T {W2.T.shape}')
    debug_print(f'Output cross covariance shape before bias: {output_cross_cov.shape}')
    # Compute expectation of act_fn(x) for each preactivation
    act_mean = act_ev(preact_mean, preact_std)
    output_mean = W2 @ act_mean + b2

    # beta = Cov(x)^-1 Cov(x, f(x))
    debug_print(f'4b. Compute linear term beta, using cov of the input distribution vs output distribution. Input cov be invertible!!')
    if cov is not None:
        debug_print(f'Cov provided. Computes beta = np.linalg.solve(cov, output_cross_cov) ~ linear algebra')
        debug_print(f'Equivalent to computing beta = Cov(x)^-1 Cov(x, f(x)). Only possible if Cov(x) invertible')
        debug_print(f'cov: {cov.shape}, output_cross_cov: {output_cross_cov.shape}')
        # need to fix: Cov is currently singular. cov is [784, 784]. Right. Need to take a tril.
        beta = torch.linalg.solve(cov, output_cross_cov) # prev np
        debug_print(f'beta: {beta.shape}')
    else:
        debug_print(f'No cov provided, identity assumed. Then beta = output_cross_cov')
        beta = output_cross_cov
        debug_print(f'beta: {beta.shape}')

    debug_print('4a. Compute intercept term alpha. Subtract off input_mean if provided')
    alpha = output_mean
    debug_print(f'alpha = output_mean. Shape = {alpha.shape}')
    if mean is not None:
        debug_print(f'input mean provided, adjust accordingly: alpha -= beta.T @ mean. Double check:\nbeta.T: {beta.T.shape}, mean: {mean.shape}')
        alpha -= beta.T @ mean

    debug_print(f'End of linear work. Next handles quadratic, which currently only works for N(0,1)')
    if order == 'quadratic':
        #assert cov == None and mean == None, ValueError('Only N(0,1) known to work!!')
        debug_print(f'5.1 check')
        # Need to validate, update code s.t. all tensors on same device. All torch tensor inits default to 'cpu'. 
        
        # sets cov_x to cov, if cov exists
        # sets cov and cov_x to identity, if cov is not provided. So that cov can be acted on as if it's identity.
        # Could have just done that in the beginning and omitted some if statements...
        #cov = cov_x = cov if cov is not None else torch.eye(d_input).to(device) # prev np
        #mu = mean if mean is not None else torch.zeros(d_input).to(device) # prev np
        rows, cols = torch.tril_indices(d_input, d_input).to(device)

        cov_y = W1 @ cov @ W1.T
        xcov = cov @ W1.T

        #cov_x = cov
        mean_y = mean @ W1.T + b1
        var_x = torch.diag(cov).to(device) # prev np
        debug_print(f'5.2 check')

        #Cov_x = torch.array([ # prev np. Need torch.Tensor rather than np.array. To fix.
        #    [var_x[rows], cov_x[rows, cols]],
        #    [cov_x[cols, rows], var_x[cols]],
        #]).T
        
        debug_print(f'Attempting to produce matrix Cov_x: \n',
                    f'[var_x[rows] {var_x[rows].shape}, cov_x[rows, cols] {cov[rows, cols].shape}],\n',
                    f'[cov_x[cols, rows] {cov[cols, rows].shape}, var_x[cols] {var_x[cols].shape}],].T')
        Cov_x = torch.Tensor([ # prev np. Need torch.Tensor rather than np.array. To fix.
            [var_x[rows].cpu().numpy(), cov[rows, cols].cpu().numpy()],
            [cov[cols, rows].cpu().numpy(), var_x[cols].cpu().numpy()],
        ]).to(device).T
        debug_print(f'Cov_x {Cov_x.shape}')
        # iiinteresting. swapping to .numpy() works, obviously this is not desirable as we want things
        # completely on-device, which wont work if using device='cuda:0'.

        debug_print(f'Creating cross covariance matrix. xcov {xcov.shape}, xcov[rows] {xcov[rows].shape}, xcov[cols] {xcov[cols].shape}')
        Xcov = torch.stack([ # prev np
            xcov[rows],
            xcov[cols],
        ]).T

        #Mean_x = np.array([mu[rows], mu[cols]]).T # prev np
        #Var_y = np.diag(cov_y) # prev np
        debug_print(f'Attempting to produce matrix Mean_x: \n',
                    f'Mean_x = [mu[rows] {mean[rows].shape}, mu[cols] {mean[cols].shape}].T')
        #Mean_x = torch.Tensor([mu[rows], mu[cols]]).T # prev np
        Mean_x = torch.Tensor([mean[rows].cpu().numpy(), mean[cols].cpu().numpy()]).to(device).T # prev np
        Var_y = torch.diag(cov_y).to(device) # prev np

        
        debug_print(f'5.3 check')
        # on GPU, goes OOM
        debug_print(f'Attempting master_theorem. On device {device}. Takes args:\n',
        f'Mean_x {Mean_x.shape}, Cov_x {Cov_x.shape}, mean_y[..., None] {mean_y[..., None].shape}, XCov {Xcov.shape}')
        
        #MNIST: [307720, 2],  [307720, 2, 2],  [256, 1],  [256, 307720, 2]
        coefs = master_theorem(
            Mean_x, Cov_x, mean_y[..., None], Var_y[..., None], Xcov
        )
        
        debug_print(f'5.4 check on master thm coefficients, quad: {coefs[0].shape}, lin: {coefs[1].shape}, const: {coefs[2].shape}')
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
        
        debug_print(f'5.5 check')
        #quad_xcov = W2 @ (E_gy_x1x2 - np.outer(const, (rows == cols))) # prev np
        quad_xcov = W2 @ (E_gy_x1x2 - torch.outer(const, (rows == cols))) # prev np
        
        # 
        gamma = quad_xcov / (1 + (rows == cols)) #wtf?
        
        debug_print(f'5.6 check, rows {type(rows)}')
        # adjust constant term
        alpha -= (rows == cols).float() @ gamma.T
    else:
        gamma = None

    # For ReLU, we can compute the covariance matrix of the activations, which is
    # useful for computing the fraction of variance unexplained in closed form.
    if act == 'relu' and return_fvu:
        # TODO: Figure out what is wrong with our implementation for non-zero means
        assert mean is None, "FVU computation is not implemented for non-zero means"
        #rhos = preact_cov / np.outer(preact_std, preact_std) # prev np
        rhos = preact_cov / torch.outer(preact_std, preact_std) # prev np

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
        #mlp_scale = np.trace(W2 @ act_m2 @ W2.T) + 2 * act_mean.T @ W2.T @ b2 + b2 @ b2
        mlp_scale = torch.trace(W2 @ act_m2 @ W2.T) + 2 * act_mean.T @ W2.T @ b2 + b2 @ b2 # prev np

        # E[g(x)^T MLP(x)] where g(x) is the linear predictor
        x_moment = cross_cov + (torch.outer(mean, output_mean) if mean is not None else 0) # prev np
        #inner_prod = np.trace(beta.T @ x_moment) + alpha.T @ output_mean # prev np
        inner_prod = torch.trace(beta.T @ x_moment) + alpha.T @ output_mean # prev np

        # E[g(x)^T g(x)] where g(x) is the linear predictor
        inner = 2 * mean.T @ beta @ alpha if mean is not None else 0
        #lin_scale = np.trace(beta.T @ cov @ beta) + inner + alpha.T @ alpha # prev np
        lin_scale = torch.trace(beta.T @ cov @ beta) + inner + alpha.T @ alpha # prev np

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

        #y_std = np.diag(W @ cov @ W.T) ** 0.5 # prev np
        y_std = torch.diag(W @ cov @ W.T) ** 0.5 # prev np
        # z_std = np.diag(V @ cov @ V.T) ** 0.5
    else:
        cross_cov = W @ V.T
        #y_std = np.linalg.norm(W, axis=1) # prev np
        y_std = torch.linalg.norm(W, axis=1) # prev np
        # z_std = np.linalg.norm(V, axis=1)

    #y_mean = np.array(b1) # prev np
    z_mean = torch.Tensor(b2) # prev np.array
    if mean is not None:
        y_mean += W @ mean

    #z_mean = np.array(b2) # prev np
    z_mean = torch.Tensor(b2) # prev np
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
    #output_mean = np.diag(cross_cov) * act_prime_ev(y_mean, y_std) + act_mean * z_mean # prev np
    output_mean = torch.diag(cross_cov) * act_prime_ev(y_mean, y_std) + act_mean * z_mean # prev np

    return output_mean
