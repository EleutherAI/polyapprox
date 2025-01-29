# %% first cell
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
        get_eig_attack_pinv, eig_attack_add,
        )
# need sample_gaussian_n, compute_dataset_statistics
datadict_path = f'/Users/alicerigg/Code/polyapprox/experiments/datadicts/baseline_centeredlong_datadict.pt'
datadict, ckpt, config, dataset, mu, scale, model, quad = startup(datadict_path, idx=-1)
# dataset original here, derived from config['test_set']
if False:
    print_dict_recursively(datadict)

# second cell: does use dataset
# Compute mean and covariance for each class separately
def compute_means_covs(dataset):
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
    return total_mean, total_cov, class_means, class_covariances

total_mean, total_cov, class_means, class_covariances = compute_means_covs(dataset)
#total_cov = torch.mean(dataset.x.view(-1,784), dim=0)
# 
linear = model.approx_fit('linear', 'master', class_means, class_covariances, True)
linear_default = model.approx_fit('linear', 'stable', total_mean, total_cov)
linear_default_new = model.approx_fit('linear', 'master', total_mean, total_cov)
linear_default_old = model.approx_fit('linear', 'old', total_mean, total_cov)
# new version produces slightly different mean and var


linear_default01 = model.approx_fit('linear', 'stable')
linear_default01_new = model.approx_fit('linear', 'master')
linear_default01_old = model.approx_fit('linear', 'old')

# third cell: does not use dataset
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

# fourth cell, uses but does not change dataset
def compute_all(ckpts=datadict['ckpts'], data=dataset):
    models = [ckpt_to_model(ckpt, config) for ckpt in ckpts]
    kls, fvus = {'mixture': [], 'musig': [], '01': []}, {'mixture': [], 'musig': [], '01': []}
    total_mean, total_cov, class_means, class_covariances = compute_means_covs(dataset)
    with torch.no_grad():
        for i, model in enumerate(models):
            #print(i)

            mixture = model.approx_fit('linear', 'master', class_means, class_covariances)
            musig = model.approx_fit('linear', 'master', total_mean, total_cov)
            zero1 = model.approx_fit('linear', 'master')
            kls['mixture'].append(kl_div_between_models(model, mixture, data))
            kls['musig'].append(kl_div_between_models(model, musig, data))
            kls['01'].append(kl_div_between_models(model, zero1, data))
            fvus['mixture'].append(fvu_between_models(model, mixture, data))
            fvus['musig'].append(fvu_between_models(model, musig, data))
            fvus['01'].append(fvu_between_models(model, zero1, data))
    return kls, fvus

# Need to produce sample of n01 data and mixture data.
test_kls, test_fvus = compute_all()
n01_kls, n01_fvus = compute_all()

# fifth cell
plt.figure(figsize=(10, 5))
for key in test_kls.keys():
    plt.plot(test_kls[key], label='KL Divergence', marker='o')
    plt.plot(test_fvus[key], label='FVU', marker='x')
    plt.xlabel('Model Index')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.title(f'KL Divergence and FVU for Models {key}')
    plt.legend()
    plt.grid(True)
    plt.show()
# %%
