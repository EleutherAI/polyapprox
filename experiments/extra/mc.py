import numpy as np
from scipy.stats import norm
from tqdm import tqdm

def compute_E_xf(W1, W2, b1):
    """
    Computes the analytical expectation E[x f(x)^T] for a single hidden layer ReLU network.
    Parameters:
    - W1: numpy.ndarray, shape (k, n)
        Weight matrix of the first layer (W^{(1)}).
    - W2: numpy.ndarray, shape (m, k)
        Weight matrix of the second layer (W^{(2)}).
    - b1: numpy.ndarray, shape (k,)
        Bias vector of the first layer (b^{(1)}).
    Returns:
    - E_xf: numpy.ndarray, shape (n, m)
        The computed expectation E[x f(x)^T].
    """
    # Compute sigma_y: standard deviations of y_j
    sigma_y = np.linalg.norm(W1, axis=1)  # Shape: (k,)

    # Avoid division by zero in case sigma_y has zeros
    sigma_y_safe = np.where(sigma_y == 0, 1e-8, sigma_y)

    # Compute alpha: standardized biases
    alpha = b1 / sigma_y_safe  # Shape: (k,)

    # Compute Phi(alpha)
    Phi_alpha = norm.cdf(alpha)  # CDF of standard normal at alpha_j

    # Construct diagonal matrix D
    D = np.diag(Phi_alpha)  # Shape: (k, k)

    # Compute E[x f(x)^T] analytically
    E_xf = W1.T @ D @ W2.T  # Shape: (n, m)

    return E_xf

def monte_carlo_E_xf(W1, W2, b1, b2=None, N_samples=100_000, hide_progress=True):
    """
    Estimates E[x f(x)^T] using Monte Carlo simulation.
    """
    n = W1.shape[1]  # Input dimension
    m = W2.shape[0]  # Output dimension

    # Initialize accumulator for E[x f(x)^T]
    E_xf_mc = np.zeros((n, m))

    # Generate N_samples of x ~ N(0, I)
    x_samples = np.random.randn(N_samples, n)  # Shape: (N_samples, n)

    # Compute f(x) for each sample
    # First compute y = W1 x + b1
    y_samples = x_samples @ W1.T + b1  # Shape: (N_samples, k)

    # Apply ReLU activation
    relu_y = np.maximum(0, y_samples)  # Shape: (N_samples, k)
    # Compute f(x) = W2 ReLU(y) + b2 (we can ignore b2 since E[x] = 0)
    f_samples = relu_y @ W2.T  # Shape: (N_samples, m)

    if b2 is not None:
        print('Warning: Code samples assuming E[x]=0. May be off distribution')
        print('Proceed with caution (ignore later)')
        relu_y += b2
    # Compute x f(x)^T for each sample and accumulate
    for i in tqdm(range(N_samples), disable=hide_progress):
        E_xf_mc += np.outer(x_samples[i], f_samples[i])

    # Divide by number of samples to get the expectation
    E_xf_mc /= N_samples

    return E_xf_mc