# %%
import sys
import os
from numpy.typing import ArrayLike, NDArray
import numpy as np
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import the module
from polyapprox.ols import ols
#from ..polyapprox.ols import ols
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
        get_eig_attack_pinv, eig_attack_add, compute_means_covs
        )
# need sample_gaussian_n, compute_dataset_statistics
datadict_path = f'/Users/alicerigg/Code/polyapprox/experiments/datadicts/baseline_centeredlong_datadict.pt'
datadict, ckpt, config, dataset, mu, scale, model, quad = startup(datadict_path, idx=-1)

if False:
    print_dict_recursively(datadict)

# %% Dataset statistics, creating samplers
_, _, class_means, class_covariances = compute_means_covs(dataset)
total_mean, total_cov, cholesky_factor = compute_dataset_statistics(dataset)

n_samples = 10000
x01 = sample_gaussian_n(num_samples = n_samples)
xmusig = sample_gaussian_n(total_mean, cholesky_factor, num_samples = n_samples)

xmixture = None # TODO: implement mixture sampling. For now, try above
#total_cov = torch.mean(dataset.x.view(-1,784), dim=0)
# %% Skippable. checks equivalence
linear = model.approx_fit('linear', 'master', class_means, class_covariances, True)
#linear_default = model.approx_fit('linear', 'stable', total_mean, total_cov)
linear_default_new = model.approx_fit('linear', 'master', total_mean, total_cov)
linear_default_old = model.approx_fit('linear', 'old', total_mean, total_cov)
# new version produces slightly different mean and var


#linear_default01 = model.approx_fit('linear', 'stable')
linear_default01_new = model.approx_fit('linear', 'master')
linear_default01_old = model.approx_fit('linear', 'old')

# checks equivalence

# List of linear variants
linear_variants = [
    linear, # 0.2 similarity with others conditioned on data statistics. 0.01 with 01. Cool.
    #linear_default, # my torch version. Skull emoji, mine is off.
    # OK, nora's is correct
    linear_default_new, # matches v
    linear_default_old, # matches ^
    #linear_default01, # of course mine is wrong too
    linear_default01_new,
    linear_default01_old
]

# Initialize a matrix to store the L2 norm differences

num_variants = len(linear_variants)
# Initialize a matrix to store the cosine similarities
beta_sim_matrix = torch.zeros((num_variants, num_variants))
alpha_sim_matrix = torch.zeros((num_variants, num_variants))
# Compute the cosine similarities between all pairs of linear variants
for i in range(num_variants):
    for j in range(num_variants):
        #if i != j:
        # Compute the cosine similarity between the weights of the two models
        weight_i = linear_variants[i].beta.flatten()
        weight_j = linear_variants[j].beta.flatten()
        ai = linear_variants[i].alpha.flatten()
        aj = linear_variants[j].alpha.flatten()
        #print(type(weight_i))
        if isinstance(weight_i, np.ndarray):
            weight_i = torch.tensor(weight_i).clone()
            weight_j = torch.tensor(weight_j).clone()
            ai = torch.tensor(ai).clone()
            aj = torch.tensor(aj).clone()
        weight_i = weight_i if isinstance(weight_i, torch.Tensor) else torch.tensor(weight_i)
        weight_j = weight_j if isinstance(weight_j, torch.Tensor) else torch.tensor(weight_j)
        ai = ai if isinstance(ai, torch.Tensor) else torch.tensor(ai)
        aj = aj if isinstance(aj, torch.Tensor) else torch.tensor(aj)
        beta_cosine_similarity = torch.nn.functional.cosine_similarity(weight_i, weight_j, dim=0)
        alpha_cosine_similarity = torch.nn.functional.cosine_similarity(ai, aj, dim=0)
        beta_sim_matrix[i, j] = beta_cosine_similarity
        alpha_sim_matrix[i, j] = alpha_cosine_similarity

print("Cosine Similarity Matrix (Weights):")
print(beta_sim_matrix)
print(alpha_sim_matrix)

# Skippable. use master whenever. Master is consistent throughout.
# adversarial quadratic
digit = 3
mag = torch.linspace(1e-3,1e2,16)
gamma = quad.get_gamma_tensor()
pseudoinverse = get_eig_attack_pinv(gamma, topk=3, orientation=1)

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
# %% 
total_mean.std(), total_mean.min(), total_mean.max()
# %%

plt.figure(figsize=(6, 6))
plt.imshow(total_mean.view(28,28), cmap='RdBu', interpolation='nearest')
plt.colorbar(label='Value')
plt.title('Total Mean Visualization')
plt.axis('off')  # Hide axis labels
plt.show()

# %%
models = [ckpt_to_model(ckpt, config) for ckpt in datadict['ckpts']]
def compute_all(models=models, data=dataset, statistics=None):
    #models = [ckpt_to_model(ckpt, config) for ckpt in ckpts]
    accs, kls, fvus = {'ols_mixture': [], 'ols_musig': [], 'ols_01': []}, \
        {'ols_mixture': [], 'ols_musig': [], 'ols_01': []}, {'ols_mixture': [], 'ols_musig': [], 'ols_01': []}
        
    if statistics is None:
        total_mean, total_cov, class_means, class_covariances = compute_means_covs(dataset)
    else:
        total_mean, total_cov, class_means, class_covariances = statistics
    with torch.no_grad():
        for i, model in enumerate(models):
            #print(i)
            ols_fits = {}
            ols_fits['ols_mixture'] = model.approx_fit('linear', 'master', class_means, class_covariances)
            ols_fits['ols_musig'] = model.approx_fit('linear', 'master', total_mean, total_cov)
            ols_fits['ols_01'] = model.approx_fit('linear', 'master')
            for key, ols_fit in ols_fits.items():
                accs[key].append(evaluate_model(ols_fit, data, return_logits=False))
                kls[key].append(kl_div_between_models(model, ols_fit, data))
                fvus[key].append(fvu_between_models(model, ols_fit, data))
    return accs, kls, fvus

eval_datasets = {'eval_test': dataset,
                 'eval_*test': config['test'],
                 'eval_*train': config['train'],
                 'eval_01': x01,
                 'eval_musig': xmusig,
                 'eval_mixture': xmixture,}
all_accs, all_kls, all_fvus = {}, {}, {}

#kls, fvus = {'mixture': [], 'musig': [], '01': []}, {'mixture': [], 'musig': [], '01': []}
for key, eval_dataset in eval_datasets.items():
    if eval_dataset is not None:
        print(key)
        accs, kls, fvus = compute_all(models, eval_dataset)
        all_kls[key] = kls
        all_fvus[key] = fvus
        if eval_dataset == dataset:
            all_accs[key] = accs
    else:
        continue # not supported

# Need to produce sample of n01 data and mixture data.

#test_kls, test_fvus = compute_all()
#n01_kls, n01_fvus = compute_all()
# %%
for eval_key in eval_datasets.keys():
    #print(eval_key)
    if eval_key == 'eval_mixture':
        continue
    plt.figure(figsize=(10, 5))
    for key in all_kls[eval_key].keys():
        if key == 'eval_mixture':
            continue
        #print('---',key)
        if eval_key == 'eval_test':
            plt.plot(all_accs[eval_key][key], label=f'Accuracy {key}')
        #plt.plot(all_kls[eval_key][key], label=f'KL fit {key}', marker='o')
        #plt.plot(all_fvus[eval_key][key], label=f'FVU fit {key}', marker='x')
    plt.xlabel('Model Index')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.title(f'KL Divergence and FVU for Models on {eval_key}')
    plt.legend()
    plt.grid(True)
    plt.show()
# %%
all_accs['eval_test']
# %%
model.approx_fit('linear', 'master', class_means, class_covariances)
# %%
import matplotlib.cm as cm
subtitles = {'eval_test': 'Datadict Test set',
             'eval_*test': 'Config Test set',
             'eval_*train': 'config train set',
             'eval_01':r'$\mathcal{N}(0,1)$', 
             'eval_musig': r'$\mathcal{N}(\mu,\Sigma)$'}
legends = {'ols_mixture': 'Mixture',
           'ols_01': r'$\mathcal{N}(0,1)$',
           'ols_musig': r'$\mathcal{N}(\mu,\Sigma)$'}
#kl_colors = {'light_blue': '#ADD8E6', 'medium_blue': '#0000CD', 'dark_blue': '#00008B'}
kl_cmap = cm.get_cmap('Blues')
fvu_cmap = cm.get_cmap('Reds')
kl_colors = {'ols_01': kl_cmap(0.4), 'ols_musig': kl_cmap(0.6), 'ols_mixture': kl_cmap(0.8)}
fvu_colors = {'ols_01': fvu_cmap(0.4), 'ols_musig': fvu_cmap(0.6), 'ols_mixture': fvu_cmap(0.8)}


fig, axes = plt.subplots(nrows=1, ncols=len(subtitles), figsize=(15, 5))
fig.suptitle('KL Divergence and FVU between Model and OLS logits')
for idx, eval_key in enumerate(eval_datasets.keys()):
    if eval_key == 'eval_mixture':
        continue
    ax = axes[idx]
    ax.set_title(f'Dataset: {subtitles[eval_key]}')
    for ols_key in all_kls[eval_key].keys():
        ax.plot(all_kls[eval_key][ols_key], label=f'{legends[ols_key]} KL', marker='o', color=fvu_colors[ols_key])
    for ols_key in all_fvus[eval_key].keys():
        ax.plot(all_fvus[eval_key][ols_key], label=f'{legends[ols_key]} FVU', marker='x', color=kl_colors[ols_key])
    ax.set_xlabel('Model Index')
    ax.set_ylabel('Value')
    ax.set_yscale('log')
    if eval_key == 'eval_musig':
        ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.show()
# %%
