# %%
import matplotlib.pyplot as plt
import os
import torch
from torch.distributions import MultivariateNormal
from mnist_2l import Minimal_FFN
from decomp.model import FFNModel


torch.manual_seed(42)
DIR = '/Users/alicerigg/Code/polyapprox/experiments/figs/'
savepaths = {}
use_imports = True
if use_imports:
    from functional_plotting_utils import (
        ckpt_to_model, kl_divergence, compute_dataset_statistics,
        sample_dataset, sample_gaussian_n, evaluate_model,
        kl_div_between_models, fvu_between_models, orient_eigenvectors,
        svd_attack_projection, plot_eigen_spectrum, plot_top_eigenvectors,
        startup, print_dict_recursively, compute_top_eigenvectors,
        get_eig_attack_pinv, eig_attack_add,
        )
else:
    def ckpt_to_model(ckpt, configs):
        model = FFNModel.from_config(**configs)
        model.load_state_dict(state_dict=ckpt)
        model = Minimal_FFN(model.get_layer_data())
        return model
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
    def compute_dataset_statistics(dataset, clamp_min=1e-6):
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
    def sample_gaussian_n(mean, cholesky_factor, num_samples=1):
        d = mean.size(0)
        gaussian = MultivariateNormal(torch.zeros(d), torch.eye(d))
        z = gaussian.sample((num_samples,))
        x = mean + torch.mm(cholesky_factor, z.T).T
        return x
    def evaluate_model(model, dataset, loss_fn=None, inputs=None, proj=None, add=None, add_noise_std=None,
                    num_examples=None, transform=None, return_logits=True):
        if loss_fn is None:
            acc_loss = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
            loss_fn = acc_loss
        
        labels = dataset.y[:num_examples] if inputs is None else None
        inputs = dataset.x[:num_examples].flatten(start_dim=1) if inputs is None else inputs.flatten(start_dim=1)
        inputs = inputs.clone()
        inputs = transform(inputs) if transform is not None else inputs
        if proj is not None:
            inputs = torch.einsum('ij,bj->bi', proj, inputs)
        if add is not None:
            #print(inputs.shape, add.shape)
            inputs += add
        if add_noise_std is not None:
            inputs += torch.randn_like(inputs) * add_noise_std

        fwd = model(inputs)
        return (fwd, loss_fn(fwd, labels).item()) if return_logits else loss_fn(fwd, labels).item()
    def kl_div_between_models(pmodel, qmodel, dataset, **kwargs):
        p_logits, _ = evaluate_model(pmodel, dataset, **kwargs)
        q_logits, _ = evaluate_model(qmodel, dataset, **kwargs)
        kl_div = kl_divergence(p_logits, q_logits)
        return kl_div.mean().item()
    def fvu_between_models(pmodel, qmodel, dataset, **kwargs):
        p_logits, _ = evaluate_model(pmodel, dataset, **kwargs)
        q_logits, _ = evaluate_model(qmodel, dataset, **kwargs)
        residual_variance = torch.var(p_logits - q_logits, unbiased=False)
        total_variance = torch.var(p_logits, unbiased=False)
        return residual_variance / total_variance
    def orient_eigenvectors(vectors, orientation):
        
        pos_mass_norm = torch.clamp(vectors, min=0).norm(dim=-2)
        neg_mass_norm = torch.clamp(vectors, max=0).norm(dim=-2)
        mask = (pos_mass_norm >= neg_mass_norm)  # True for positiive dominant components
        mask = mask.unsqueeze(-2)
        if orientation == -1:  # Make all negative dominant
            #vectors[..., mask] *= -1
            vectors = torch.where(mask, -vectors, vectors)
        elif orientation == 1:  # Make all positive dominant
            vectors = torch.where(~mask, -vectors, vectors)
            #vectors[..., ~mask] *= -1
        return vectors
    def svd_attack_projection(beta, topk=1):
        u_mat, _, _ = torch.linalg.svd(beta, full_matrices=False)
        proj_matrix = torch.eye(u_mat.size(0))
        for i in range(topk):
            proj_matrix -= torch.outer(u_mat[:, i], u_mat[:, i])
        return proj_matrix
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
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The specified file path does not exist: {filepath}")
        
        datadict = torch.load(filepath)
        ckpt = datadict['ckpts'][idx]
        config = datadict['config']
        dataset = config['test']
        mu, scale = dataset.recenter()
        model = ckpt_to_model(ckpt, config)
        ols = model.approx_fit('quadratic')
        return datadict, ckpt, config, dataset, mu, scale, model, ols
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
    def compute_top_eigenvectors(gamma_matrix, digit=None, topk=3, withneg=False, orientation=None):
        '''
        Computes top eigenvectors and values. If digit not provided, provides for all digits.

        if withneg enabled, returns shape [2,topk] eigvals and [2,topk,784] eigvecs.
        # [0] for positive (increasing order), [1] for negative (increasing order)

        '''
        eigvals, eigvecs = torch.linalg.eigh(gamma_matrix)
        eigvecs = orient_eigenvectors(eigvecs, orientation) if orientation is not None else eigvecs
        # eigvecs[10,dmodel,idx]
        top_eigvecs = eigvecs[:, :, -topk:].flip(dims=[-1])
        # to get the top one, index at [-1].
        top_eigvals = eigvals[:, -topk:].flip(dims=[-1])

        if digit is not None:
            return top_eigvals[digit], top_eigvecs[digit]
        return top_eigvals, top_eigvecs
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
datadict_path = f'/Users/alicerigg/Code/polyapprox/experiments/data/new_baseline_uncenteredlong_datadict.pt'
#datadict_path = f'/Users/alicerigg/Code/polyapprox/experiments/datadicts/baseline_centeredlong_datadict.pt'
datadict, ckpt, config, dataset, model, quad = startup(datadict_path, idx=-1)

if True:
    print_dict_recursively(datadict)

if False:
    ckpts = datadict['ckpts']
    config = datadict['config']
    dataset = config['test']
    mu, scale = dataset.recenter()
    models = [ckpt_to_model(ckpt, config) for ckpt in ckpts]

    fvus = []
    kls = []

    with torch.no_grad():
        for model in models:
            quad = model.approx_fit('quadratic')
            kls.append(kl_div_between_models(model, quad, dataset))
            fvus.append(fvu_between_models(model, quad, dataset))

    print(kls,fvus)
# %%
# need to pass mean, cov data. 
linear = model.approx_fit('linear', 'master')
# %%
print_dict_recursively(datadict['train_measures'])
#datadict['train_measures'].keys()
# %%
metrics = datadict['metrics']
metrics[0].keys()
# %%
topk = 10
digit = 3
mag = torch.linspace(1e-3,1e2,16)
quad = model.approx_fit('quadratic', 'master')
gamma = quad.unpack_gamma()
pseudoinverse = get_eig_attack_pinv(gamma, topk=topk, orientation=1)

add = eig_attack_add(pseudoinverse, digit)

attack_vecs = add[None,:] * mag[:,None]

relu_acc = torch.zeros_like(mag)
quad_acc = torch.zeros_like(mag)
#print(relu_acc)
for i in range(13):
    relu_acc[i] = torch.tensor(evaluate_model(model, dataset, add=attack_vecs[i], return_logits=False))
    quad_acc[i] = torch.tensor(evaluate_model(quad, dataset, add=attack_vecs[i], return_logits=False))

print(relu_acc, quad_acc)
print(attack_vecs.norm(), gamma.norm(), add.norm(), digit, mag.norm())

plt.figure(figsize=(10, 6))
plt.plot(mag, relu_acc, label='ReLU Model Accuracy', marker='o')
plt.plot(mag, quad_acc, label='Quadratic Model Accuracy', marker='x')
plt.xlabel('Magnitude')
plt.ylabel('Accuracy')
plt.title('Comparison of ReLU and Quadratic Model Accuracies')
plt.legend()
plt.grid(True)

# %% # DONE: create eigenvector attacks for every digit for topk between 1 and 10.
gamma = quad.unpack_gamma()
vecs, pinv = get_eig_attack_pinv(gamma, 1, True, 1)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(vecs[i, :].reshape(28, 28), cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax.set_title(f'pinv[:,{2 * i}]')
    ax.axis('off')
plt.tight_layout()
plt.show()
for k in range(1,11):
    #print(f'Top-{k} Adversarial digit construction')
    #print('-'*30)
    vecs, pinv = get_eig_attack_pinv(gamma, topk=k, return_vecs=True, orientation=1)

    #  pseudo-inverse adversasrial view
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(pinv[:, i*k].reshape(28, 28), cmap='RdBu', vmin=-0.3, vmax=0.3)
        ax.set_title(f'pinv[:,{2 * i}]')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
# %%
ckpts = datadict['ckpts']
config = datadict['config']
dataset = config['test']
#mu, scale = dataset.recenter()
models = [ckpt_to_model(ckpt, config) for ckpt in ckpts]

fvus = []
kls = []

with torch.no_grad():
    for model in models:
        quad = model.approx_fit('quadratic')
        kls.append(kl_div_between_models(model, quad, dataset))
        fvus.append(fvu_between_models(model, quad, dataset))

print(kls,fvus)
# %%
plt.figure(figsize=(10, 5))
plt.plot(kls, label='KL Divergence', marker='o')
plt.plot(fvus, label='FVU', marker='x')
plt.xlabel('Model Index')
plt.ylabel('Value')
plt.yscale('log')
plt.title('KL Divergence and FVU for Models')
plt.legend()
plt.grid(True)
plt.show()

# %%
print(dataset.x.norm())
# %%
metrics = dataset['metrics']

metrics[-1]
# %%

with torch.no_grad():
    for model in models:
        linear = model.approx_fit('linear')
        kls.append(kl_div_between_models(model, quad, dataset))
        fvus.append(fvu_between_models(model, quad, dataset))
