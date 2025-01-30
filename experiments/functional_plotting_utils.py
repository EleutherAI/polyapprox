import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.distributions import MultivariateNormal
from mnist_2l import Minimal_FFN
from decomp.model import FFNModel

torch.manual_seed(42)

def ckpt_to_model(ckpt, configs):
    model = FFNModel.from_config(**configs)
    model.load_state_dict(state_dict=ckpt)
    model = Minimal_FFN(model.get_layer_data())
    return model
# Helper functions
def kl_divergence(px, qx):
    '''
    Expects px, qx to be shape [batch_dim, 10], to be the final logits. 

    To compute KL divergence between two models, must compose with evaluate, and return logits

    use `kl_div_between_models` instead, if expecting to provide d_input space objects
    '''
    assert px.shape == qx.shape, ValueError(f'Shapes must match! {px.shape} != {qx.shape}')
    # Expecting -1 dim to be class logits in probability form. Softmax just in case
    p_x = torch.nn.functional.log_softmax(px, dim=-1)
    q_x = torch.nn.functional.log_softmax(qx, dim=-1)
    kl_divergence = torch.sum(torch.exp(p_x) * (p_x - q_x), dim=-1)
    return kl_divergence

# ========== Dataset-Related Utility Functions ========== #
def compute_dataset_statistics(dataset: torch.Tensor, clamp_min=1e-6):
    '''
    Returns mean [784], cov [784, 784], cholesky_factor [784, 784]

    Cholesky factor can be used to create a stable MultivariateNormal(mu,cov),
    without worrying about numerical precision errors

    '''
    #print(x.shape)
    #x = dataset.float().view(-1,784)
    x = dataset.view(-1,784)
    print(x.shape)
    mean = torch.mean(x, dim=0, keepdim=True)
    centered_data = x - mean
    cov = torch.matmul(centered_data.T, centered_data) / (x.size(0) - 1)

    eigvals, eigvecs = torch.linalg.eigh(cov)
    clamped_eigvals = torch.clamp(eigvals, min=clamp_min)
    sqrt_eigvals = torch.sqrt(clamped_eigvals)
    # Handle potential numerical instability for covariance matrix
    cholesky_factor = (eigvecs * sqrt_eigvals) @ eigvecs.T

    return mean, cov, cholesky_factor

def compute_dataset_statistics_old(dataset, clamp_min=1e-6):
    '''
    Returns mean [784], cov [784, 784], cholesky_factor [784, 784]

    Cholesky factor can be used to create a stable MultivariateNormal(mu,cov),
    without worrying about numerical precision errors

    '''
    #print(x.shape)
    x = dataset.x.flatten(start_dim=1)
    print(x.shape)
    mean = torch.mean(x, dim=0)
    centered_data = x - mean
    cov = torch.matmul(centered_data.T, centered_data) / (x.size(0) - 1)

    eigvals, eigvecs = torch.linalg.eigh(cov)
    clamped_eigvals = torch.clamp(eigvals, min=clamp_min)
    sqrt_eigvals = torch.sqrt(clamped_eigvals)
    # Handle potential numerical instability for covariance matrix
    cholesky_factor = (eigvecs * sqrt_eigvals) @ eigvecs.T

    return mean, cov, cholesky_factor

def sample_dataset(dataset, idx=None):
    if idx is None:
        idx = torch.randint(0, len(dataset.x), (1,)).item()
    return dataset.x[idx].reshape(1, -1)

def sample_gaussian_n(mean=None, cholesky_factor=None, num_samples=1):
    '''
    Must provide mean, to determine size.

    If not provided, defaults to hardcoded torch.zeros(784)
    '''
    d = mean.size(0) if mean is not None else 784

    if mean is None:
        mean = torch.zeros(d)
    #d = mean.size(0)

    if cholesky_factor is None:
        cholesky_factor = torch.eye(d)
    gaussian = MultivariateNormal(torch.zeros(d), torch.eye(d))
    z = gaussian.sample((num_samples,))
    x = mean + torch.mm(cholesky_factor, z.T).T
    return x


def compute_mean_cov_classmeans_classcovs(dataset, clamp_min=1e-6, cholesky=False):
    '''
    Takes [B,*] data, reshapes to [B,784]

    returns mean, cov, Bmean, Bcov (gaussian, gaussian mixture)
    '''
    class_means = []
    class_covariances = []

    # Assuming dataset.y contains class labels and dataset.x contains the data
    unique_classes = torch.unique(dataset.y)

    for cls in unique_classes:
        class_data = dataset.x[dataset.y == cls].view(-1,784)
        mean = torch.mean(class_data, dim=0)
        centered_data = class_data - mean
        cov = torch.matmul(centered_data.T, centered_data) / (class_data.size(0) - 1)
        
        class_means.append(mean)
        class_covariances.append(cov)

    class_means = torch.stack(class_means)
    class_covariances = torch.stack(class_covariances)
    total_mean = torch.mean(dataset.x.view(-1,784), dim=0)
    tcentered_data = dataset.x.view(-1, 784) - total_mean
    total_cov = torch.matmul(tcentered_data.T, tcentered_data) / (tcentered_data.size(0) - 1)

    if cholesky:
        eigvals, eigvecs = torch.linalg.eigh(cov)
        class_eigvals, class_eigvecs = torch.linalg.eigh(class_covariances)
        clamped_eigvals = torch.clamp(eigvals, min=clamp_min)
        clamped_class_eigvals = torch.clamp(class_eigvals, min=clamp_min)
        sqrt_eigvals = torch.sqrt(clamped_eigvals)
        sqrt_class_eigvals = torch.sqrt(clamped_class_eigvals)
        # Handle potential numerical instability for covariance matrix
        cholesky_factor = (eigvecs * sqrt_eigvals) @ eigvecs.T
        class_left = torch.einsum('cij,ci->cij',class_eigvecs, sqrt_class_eigvals)
        print(f'class_left {class_left.shape}, class_eigvecs {class_eigvecs.shape}')
        print(f'vals {sqrt_class_eigvals.shape}')
        class_cholesky_factors = torch.einsum('cij,ckj->cik', class_left, class_eigvecs)
        return total_mean, total_cov, class_means, \
            class_covariances, cholesky_factor, class_cholesky_factors
    
    return total_mean, total_cov, class_means, class_covariances
# ========== Model Evaluation and Transformation ========== #

def evaluate_model(model, dataset, loss_fn=None, inputs=None, proj=None, add=None, add_noise_std=None,
                   num_examples=None, transform=None, return_logits=True):
    if loss_fn is None:
        acc_loss = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
        loss_fn = acc_loss
    
    if hasattr(dataset, 'y') and hasattr(dataset, 'x'):
        labels = dataset.y[:num_examples] if inputs is None else None
        inputs = dataset.x[:num_examples].flatten(start_dim=1) if inputs is None else inputs.flatten(start_dim=1)
    else:
        labels = None
        inputs = dataset[:num_examples].flatten(start_dim=1)
    
    inputs = inputs.clone()
    inputs = transform(inputs) if transform is not None else inputs
    if proj is not None:
        inputs = torch.einsum('ij,bj->bi', proj, inputs)
    if add is not None:
        inputs += add
    if add_noise_std is not None:
        inputs += torch.randn_like(inputs) * add_noise_std

    fwd = model(inputs)
    if labels is None:
        loss = -1.0 # indicating invalid
    else:
        loss = loss_fn(fwd, labels).item()
    return (fwd, loss) if return_logits else loss

def kl_div_between_models(pmodel, qmodel, dataset, **kwargs):
    p_logits, _ = evaluate_model(pmodel, dataset, **kwargs)
    q_logits, _ = evaluate_model(qmodel, dataset, **kwargs)
    kl_div = kl_divergence(p_logits, q_logits)
    return kl_div.mean().item()

def fvu_between_models(pmodel, qmodel, dataset, epsilon=1e-8, **kwargs):
    p_logits, _ = evaluate_model(pmodel, dataset, **kwargs)
    q_logits, _ = evaluate_model(qmodel, dataset, **kwargs)
    residual_variance = torch.var(p_logits - q_logits, unbiased=False)
    total_variance = torch.var(p_logits, unbiased=False)
    return residual_variance / (total_variance+epsilon)

def compute_top_eigenvectors(gamma_matrix, topk=3, orientation=None):
    eigvals, eigvecs = torch.linalg.eigh(gamma_matrix)
    eigvecs = orient_eigenvectors(eigvecs, orientation) if orientation is not None else eigvecs
    top_eigvecs = eigvecs[:, -topk:]
    top_eigvals = eigvals[-topk:]
    return top_eigvals, top_eigvecs

# Helper to orient eigenvectors consistently
def orient_eigenvectors(vectors, orientation):
    pos_mass_norm = torch.clamp(vectors, min=0).norm(dim=0)
    neg_mass_norm = torch.clamp(vectors, max=0).norm(dim=0)
    mask = (pos_mass_norm >= neg_mass_norm)  # True for positiive dominant components
    if orientation == -1:  # Make all negative dominant
        vectors[:, mask] *= -1
    elif orientation == 1:  # Make all positive dominant
        vectors[:, ~mask] *= -1
    return vectors

# ========== SVD and Eigenvalue Attacks ========== #

def svd_attack_projection(beta, topk=1):
    u_mat, _, _ = torch.linalg.svd(beta, full_matrices=False)
    proj_matrix = torch.eye(u_mat.size(0))
    for i in range(topk):
        proj_matrix -= torch.outer(u_mat[:, i], u_mat[:, i])
    return proj_matrix

# ========== Plotting Utilities ========== #

def plot_eigen_spectrum(eigvals, description=''):
    plt.figure(figsize=(8, 6))
    plt.plot(eigvals, marker='o')
    plt.title(f'Eigenvalues for {description}')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.show()


def plot_top_eigenvectors(eigvecs, eigvals, shape, topk=3, cmap='RdBu', vmax=0.2):
    vmin = -vmax
    plt.figure(figsize=(12, 6))
    for i in range(topk):
        plt.subplot(2, topk, i + 1)
        plt.imshow(eigvecs[:, -(i + 1)].reshape(shape), cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(f'Top Positive {i + 1}: {eigvals[-(i + 1)]:.2f}')
        plt.axis('off')
        plt.subplot(2, topk, i + topk + 1)
        plt.imshow(eigvecs[:, i].reshape(shape), cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(f'Top Negative {i + 1}: {eigvals[i]:.2f}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def startup(filepath, idx=-1):
    '''
    Returns:

    datadict - all available  data
    
    ckpt[idx] - statedict to be used with Minimal_FFN. Pass ckpt and config to `ckpt_to_model`.

    config - config file used to train the model. Redundant from datadict.

    dataset - dataset from config file. Redundant from datadict. NOTE possible leakage: it is possible dataset
    initialization from saving 'config' does not preserve transforms used on the model. Thus, it can only be
    trusted for the uncentered case, currently.

    model - `ckpt_to_model(ckpt, config)`. Redundant from datadict.

    ols - `model.approx_fit('quadratic')`. 
    '''
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The specified file path does not exist: {filepath}")
    
    print('filepath: ', filepath)
    datadict = torch.load(filepath)
    ckpt = datadict['ckpts'][idx]
    config = datadict['config']
    dataset = config['test']
    #mu, scale = dataset.recenter()
    model = ckpt_to_model(ckpt, config)
    ols = model.approx_fit('quadratic')
    return datadict, ckpt, config, dataset, model, ols

def print_dict_recursively(dictionary, indent=0):
    for key, val in dictionary.items():
        print('-' * indent + f">Key: {key}")
        if isinstance(val, tuple):
            print('-' * (indent + 2) + f">Tuple with {len(val)} elements")
        elif isinstance(val, torch.Tensor):
            print('-' * (indent + 2) + f">Tensor with shape {val.shape}")
        elif isinstance(val, dict):
            print('-' * (indent + 2) + ">Dictionary:")
            print_dict_recursively(val, indent + 4)

        elif isinstance(val, list):
            print('-' * (indent + 2) + f">List with length {len(val)}")
            if len(val) > 0:
                print('-' * (indent + 4) + f">Type of first element: {type(val[0])}")
        else:
            print('-' * (indent + 2) + f">Value: {val}")

def get_eig_attack_pinv(gamma_tensor, topk=1, return_vecs=False, orientation=None):
    '''
    Expects gamma_tensor inputted. You can get that from ols by using:

    ols.get_gamma_tensor(). Should be shape [10, 784, 784] for MNIST
    '''

    # will be [10,784] and [10,784,784]
    top_eigvals, top_eigvecs = compute_top_eigenvectors(gamma_tensor, topk=topk, orientation=orientation)

    #print('inside get_eig_attack_pinv, eigvecs shape:', top_eigvecs.shape)
    top_eigvecs = top_eigvecs.permute(0, 2, 1)
    top_eigvecs = top_eigvecs.reshape(10 * topk, -1) # this could be messing with it

    pseudoinverse = torch.linalg.pinv(top_eigvecs)

    if return_vecs:
        return top_eigvecs, pseudoinverse
    else:
        return pseudoinverse
def eig_attack_add(pseudoinverse: torch.Tensor, digit: int):
    '''
    Expects pseudoinverse matrix as input, and the digit you want to steer towards.
    Compose with `get_eig_attack_pinv` if you want to instead provide a gamma tensor and choose the value of topk.
    Returns an unnormalized pseudoinverse vector in the input space.
    
    If the raw vector of this is passed to the gamma tensor, expected behavior is it sets the top eig activation to approx 1.
    '''
    topk = pseudoinverse.size(-1) // 10
    return pseudoinverse[:,topk * digit]