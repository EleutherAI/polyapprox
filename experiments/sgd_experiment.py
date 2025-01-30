# %%
import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

current_dir = os.getcwd()
print(current_dir)
target_folder = os.path.abspath(os.path.join(current_dir, "..", "schedule_free"))
#print(two_up)
#target_folder = os.path.join(two_up, "schedule_free")
# Navigate 1 folder down (replace 'target_folder' with the actual folder name)

sys.path.append(target_folder)

from schedulefree.wrap_schedulefree import ScheduleFreeWrapper

from extra.sgd import QuadraticModel, LinearModel, GaussianMixture
from functional_plotting_utils import startup, compute_mean_cov_classmeans_classcovs, ckpt_to_model

import matplotlib.pyplot as plt

from torch.distributions import MultivariateNormal
#from mnist_2l import Minimal_FFN
#from decomp.model import FFNModel

def train_model(model, target_model, sampler, optimizer, fixed_dataset=None, num_epochs=100, bsz=1000, num_samples=10000,
                print_log_scale=False):
    model.train()
    target_model.eval()  # Ensure target model is in evaluation mode
    criterion = nn.MSELoss()
    #pbar = tqdm(total=num_epochs, desc="training Progress", unit="epoch")
    for epoch in range(num_epochs):
    #with tqdm(range(num_epochs), desc='Training Progress', unit='epoch') as pbar:
        #for epoch in pbar:
        running_loss = 0.0
        for _ in range(num_samples):
            # Sample inputs from the distribution
            if fixed_dataset is not None:
                inputs = fixed_dataset.clone()
            else:
                inputs = sampler.sample((bsz,))

            # Get target outputs from the target model
            with torch.no_grad():
                target_outputs = target_model(inputs)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, target_outputs)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Print statistics
        loss_print = running_loss / num_samples
        if print_log_scale:
            loss_print = torch.log(torch.tensor(loss_print)).item()
        #pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
        #pbar.set_postfix({'loss: ': f'{loss_print:.4f}'})
        #pbar.update(1)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_print:.4f}")
    #pbar.close()
            #print(f'Epoch [{epoch + 1}/{num_epochs}], logLoss: {loss_print:.4f}')

# need sample_gaussian_n, compute_dataset_statistics
path1 = f'datadicts/baseline_centeredlong_datadict.pt'
path2 = f'data/new_baseline_uncenteredlong_datadict.pt'
idx = -1
datadict, model, config, dataset, target_model, quad = startup(path2, idx=idx)

mean, cov, means, covs = compute_mean_cov_classmeans_classcovs(dataset)
class_probs = torch.bincount(dataset.y) /len(dataset.y)
#print("Class counts:", class_counts)

cov = cov + 1e-6 * torch.eye(784)
covs = covs + 1e-6 * torch.eye(784)
if True:
    models = [ckpt_to_model(model, config) for model in datadict['ckpts']]
    linears01 = [model.approx_fit() for model in models]
    linearsms = [model.approx_fit('linear','master',mean,cov) for model in models]
    linears_mixture = [model.approx_fit('linear','master',means,covs) for model in models]
    linears_mixture_sgd = [LinearModel() for linear in linears01]
    quads01 = [model.approx_fit('quadratic') for model in models]
    quads_mixture_sgd = [QuadraticModel() for quad in quads01]
    for i, (lin_sgd, quad_sgd) in enumerate(zip(linears_mixture_sgd, quads_mixture_sgd)):
        lin_sgd.from_quad(linears01[i])
        quad_sgd.from_quad(quads01[i])

    gaussian = MultivariateNormal(torch.zeros(784), torch.eye(784))
    gausian_ms = MultivariateNormal(mean, cov)
    mixture_sampler = GaussianMixture(means, covs, class_probs, 100000)

# %% train all fits.
#gaussian = MultivariateNormal(torch.zeros(784), torch.eye(784))
#gausian_ms = MultivariateNormal(mean, cov)
#mixture_sampler = GaussianMixture(means, class_covs, class_probs, 100000)
# mixture sampling gets to 0.42 after 50 epochs, slows down.
for i, (lin_sgd, quad_sgd) in tqdm(enumerate(zip(linears_mixture_sgd, quads_mixture_sgd))):
    #quad_sgd.from_quad(quads01[i])
    #quad_sgd.train()
    optim_lin = torch.optim.SGD(lin_sgd.parameters(), lr=1.8e-2)
    optim_quad = torch.optim.SGD(quad_sgd.parameters(), lr=1.8e-2)
    optim_lin = ScheduleFreeWrapper(optim_lin, momentum=0.9)
    optim_quad = ScheduleFreeWrapper(optim_quad, momentum=0.9)
    optim_lin.train()
    optim_quad.train()
    #quad_sgd.train()
    #optim_quad.train()
    train_model(lin_sgd, models[i], mixture_sampler, optim_lin,
                num_epochs=20, num_samples=10, bsz=2**10, print_log_scale=True)
    train_model(quad_sgd, models[i], mixture_sampler, optim_quad,
                num_epochs=20, num_samples=10, bsz=2**10, print_log_scale=True)
# %% figure 2
from functional_plotting_utils import evaluate_model
from decomp.datasets import MNIST

train = MNIST(train=True)
test_set = config['test']
#mean, cov, class_means, class_covs = compute_mean_cov_classmeans_classcovs(test_set)
all_models = {}
#baseline_models = [ckpt_to_model(ckpt, config) for ckpt in datadict['ckpts']]
#linear_01 = [model.approx_fit('linear') for model in baseline_models]
#linear_ms = [model.approx_fit('linear', 'master', mean, cov) for model in baseline_models]
#linear_mixture = [model.approx_fit('linear', 'master', means, class_covs) for model in baseline_models]
#quadratic_01 = [model.approx_fit('quadratic') for model in baseline_models]

all_models['baseline'] = models
all_models['linear_01'] = linears01
all_models['linear_ms'] = linearsms
all_models['linear_mixture'] = linears_mixture
all_models['linear_mixture_sgd'] = linears_mixture
all_models['quadratic_01'] = quads01
all_models['quadratic_mixture_sgd'] = quads_mixture_sgd

with torch.no_grad():
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
from functional_plotting_utils import sample_gaussian_n, fvu_between_models, kl_div_between_models
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
        for model_type, family in all_models.items():
            if model_type != 'baseline':
                fig3_fvu[datatype][model_type] = []
                fig4_kl[datatype][model_type] = []
                for idx, ckpt in enumerate(family):
                    pmodel = all_models['baseline'][idx]
                    
                    #for ckpt in family:
                    ckpt_fvu = fvu_between_models(pmodel, ckpt, dataset)
                    ckpt_kl = kl_div_between_models(pmodel, ckpt, dataset)
                    fig3_fvu[datatype][model_type].append(ckpt_fvu)
                    fig4_kl[datatype][model_type].append(ckpt_kl)

# %% Figures 3,4
for datatype in datasets.keys():
    plt.figure(figsize=(10, 6))
    for model_type, fvus in fig3_fvu[datatype].items():
        plt.plot(fvus, marker='o', label=model_type)

    plt.title(f'FVU for Different Models on {datatype}')
    plt.xlabel('Checkpoint Index')
    plt.ylabel('FVU')
    plt.yscale('log')
    plt.ylim(1e-4,1e1)
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
    plt.ylim(1e-4,1e1)
    plt.axhline(y=1, color='red', linestyle='--', linewidth=4)
    plt.legend()
    plt.grid(True)
    #plt.savefig('main_paper_figs/fig3.pdf')
    plt.show()
# %%
def _fvu_between_models(pmodel, qmodel, dataset, epsilon=1e-8, **kwargs):
    p_logits, _ = evaluate_model(pmodel, dataset, **kwargs)
    q_logits, _ = evaluate_model(qmodel, dataset, **kwargs)
    residual_variance = torch.var(p_logits - q_logits, unbiased=False)
    total_variance = torch.var(p_logits, unbiased=False)
    return residual_variance / (total_variance+epsilon), p_logits, q_logits

last_model = all_models['baseline'][-1]
last_quad = quads_mixture[-1]

fvu, p, q = _fvu_between_models(last_model, last_quad, dataset)

print(fvu)
# %%
# mixture sampling gets to 0.42 after 50 epochs, slows down.
for i, quad_init in tqdm(enumerate(quads_mixture)):
    quad_init.from_quad(quads01[i])
    optimizer = torch.optim.SGD(quad_init.parameters(), lr=0.001)
    train_model(quad_init, models[i], mixture_sampler, optimizer, num_epochs=1, num_samples=10, bsz=2**10)

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
