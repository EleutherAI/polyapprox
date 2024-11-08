from scipy.special import expit as sigmoid


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_dbl_prime(x):
    return sigmoid(x) * (1 - sigmoid(x)) * (1 - 2 * sigmoid(x))

def swish(x):
    return x * sigmoid(x)

def swish_prime(x):
    return sigmoid(x) + swish(x) - sigmoid(x) * swish(x)