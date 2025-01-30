# %% Init
import sys
from tqdm import tqdm
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
from decomp.datasets import MNIST
from extra.sgd import GaussianMixture
from functional_plotting_utils import (
    startup, compute_mean_cov_classmeans_classcovs, ckpt_to_model, evaluate_model,
    sample_gaussian_n, fvu_between_models, kl_div_between_models,
    
)

torch.manual_seed(42)
DIR = '/Users/alicerigg/Code/polyapprox/experiments/figs/'
savepaths = {}


# need sample_gaussian_n, compute_dataset_statistics
datadict_path = f'/Users/alicerigg/Code/polyapprox/experiments/datadicts/baseline_centeredlong_datadict.pt'
datadict, _, config, dataset, model, quad = startup(datadict_path, idx=-1)

print(config.keys())
# %% figure 2 data. COMMENT: does not work if mean is on train???
train = MNIST(train=True)
test_set = config['test']
mean, cov, class_means, class_covs = compute_mean_cov_classmeans_classcovs(test_set)
all_models = {}
baseline_models = [ckpt_to_model(ckpt, config) for ckpt in datadict['ckpts']]
linear_01 = [model.approx_fit('linear') for model in baseline_models]
linear_ms = [model.approx_fit('linear', 'master', mean, cov) for model in baseline_models]
linear_mixture = [model.approx_fit('linear', 'master', class_means, class_covs) for model in baseline_models]
quadratic_01 = [model.approx_fit('quadratic') for model in baseline_models]

all_models['baseline'] = baseline_models
all_models['linear_01'] = linear_01
all_models['linear_ms'] = linear_ms
all_models['linear_mixture'] = linear_mixture
all_models['quadratic_01'] = quadratic_01

fig2_test_accuracies = {}
for model_type, family in all_models.items():
    fig2_test_accuracies[model_type] = []
    for ckpt in family:
        ckpt_acc = evaluate_model(ckpt, test_set, return_logits=False)
        fig2_test_accuracies[model_type].append(ckpt_acc)
#  FIGURE 2
plt.figure(figsize=(10, 6))
for model_type, accuracies in fig2_test_accuracies.items():
    plt.plot(accuracies, marker='o', label=model_type)

plt.title('Test Set Accuracies for Different Model Types')
plt.xlabel('Checkpoint Index')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('main_paper_figs/fig2.pdf')
plt.show()
# %% Figure 3 data
test_set = MNIST(train=False)
train = MNIST(train=True)
class_probs = torch.bincount(train.y) /len(train.y)
mean, cov, class_means, class_covs, chol, chols = compute_mean_cov_classmeans_classcovs(train, cholesky=True)
mixture = GaussianMixture(class_means, class_covs, class_probs, 10000)
gaussian_01 = sample_gaussian_n(num_samples=10000)
gaussian_ms = sample_gaussian_n(mean, chol, 10000)
gaussian_mixture = mixture.sample(10000)


datasets = {'test_set': test_set,
            'train_set': train,
            'gaussian_01': gaussian_01,
            'gaussian_ms': gaussian_ms,
            'gaussian_mixture': gaussian_mixture}

# models = all_models
#measures = {'KL Divergence': {}, 'FVU': {}}
fig3_fvu = {}
fig4_kl = {}
with torch.no_grad():
    for _, (datatype, dataset) in tqdm(enumerate(datasets.items())):
        fig3_fvu[datatype] = {}
        fig4_kl[datatype] = {}
        for idx, (model_type, family) in enumerate(all_models.items()):
            if model_type != 'baseline':
                pmodel = all_models['baseline'][idx]
                fig3_fvu[datatype][model_type] = []
                fig4_kl[datatype][model_type] = []
                for ckpt in family:
                    ckpt_fvu = fvu_between_models(pmodel, ckpt, dataset)
                    ckpt_kl = kl_div_between_models(pmodel, ckpt, dataset)
                    fig3_fvu[datatype][model_type].append(ckpt_fvu)
                    fig4_kl[datatype][model_type].append(ckpt_kl)

# %%
for datatype in datasets.keys():
    plt.figure(figsize=(10, 6))
    for model_type, fvus in fig3_fvu[datatype].items():
        plt.plot(fvus, marker='o', label=model_type)

    plt.title(f'FVU for Different Models on {datatype}')
    plt.xlabel('Checkpoint Index')
    plt.ylabel('FVU')
    plt.yscale('log')
    plt.ylim(1e-3,1e3)
    plt.axhline(y=1, color='red', linestyle='--', linewidth=4)
    plt.legend()
    plt.grid(True)
    #plt.savefig('main_paper_figs/fig3.pdf')
    plt.show()
# 
for datatype in datasets.keys():
    plt.figure(figsize=(10, 6))
    for model_type, kls in fig4_kl[datatype].items():
        plt.plot(kls, marker='o', label=model_type)

    plt.title(f'KL Divergence for Different Models on {datatype}')
    plt.xlabel('Checkpoint Index')
    plt.ylabel('KL Divergence')
    plt.yscale('log')
    plt.ylim(1e-3,1e3)
    plt.axhline(y=1, color='red', linestyle='--', linewidth=4)
    plt.legend()
    plt.grid(True)
    #plt.savefig('main_paper_figs/fig3.pdf')
    plt.show()
# %% fig 7 data: Visualize beta.

