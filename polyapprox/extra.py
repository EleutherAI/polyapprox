import array_api_compat
from scipy.special import expit as sigmoid

from .backends import ArrayType
from .integrate import gauss_hermite


def id_poly_ev(n: int, mu: ArrayType, sigma: ArrayType) -> ArrayType:
    """
    Compute E[x^n * x] analytically where x ~ N(mu, sigma^2)

    Parameters:
    n     : int, the exponent n in x^n * x
    mu    : ArrayType, the mean(s) of the normal distribution(s)
    sigma : ArrayType, the standard deviation(s) of the normal distribution(s)
    """
    return normal_moment(n + 1, mu, sigma)


def normal_moment(n: int, mu: ArrayType, sigma: ArrayType) -> ArrayType:
    """Compute E[x^n] analytically where x ~ N(mu, sigma^2)."""
    xp = array_api_compat.array_namespace(mu, sigma)

    E0 = xp.ones_like(mu)  # E[x^0] = 1
    if n == 0:
        return E0

    E1 = mu  # E[x^1] = mu
    if n == 1:
        return E1

    for k in range(2, n + 1):
        E2 = mu * E1 + (k - 1) * sigma**2 * E0
        E0, E1 = E1, E2

    return E1


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def swish(x):
    return x * sigmoid(x)


def swish_prime(x):
    return sigmoid(x) + swish(x) - sigmoid(x) * swish(x)


def sigmoid_poly_ev(n: int, mu: ArrayType, sigma: ArrayType) -> ArrayType:
    """
    Compute E[x^n * ReLU(x)] analytically where x ~ N(mu, sigma^2)

    Parameters:
    n     : int, the exponent n in x^n * ReLU(x)
    mu    : ArrayType, the mean(s) of the normal distribution(s)
    sigma : ArrayType, the standard deviation(s) of the normal distribution(s)
    """
    return gauss_hermite(lambda x: x**n * sigmoid(x), mu, sigma)


def swish_poly_ev(n: int, mu: ArrayType, sigma: ArrayType) -> ArrayType:
    """
    Compute E[x^n * ReLU(x)] analytically where x ~ N(mu, sigma^2)

    Parameters:
    n     : int, the exponent n in x^n * ReLU(x)
    mu    : ArrayType, the mean(s) of the normal distribution(s)
    sigma : ArrayType, the standard deviation(s) of the normal distribution(s)
    """
    return gauss_hermite(lambda x: x**n * swish(x), mu, sigma)
