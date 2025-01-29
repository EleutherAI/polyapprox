
# %% init, show first 10 images of dataset (for mnist + variants)
import torch
from tqdm import tqdm
#from mnist_2l import Minimal_FFN
from decomp.model import FFNModel
from decomp.datasets import MNIST, CIFAR10
#from mean_cov import MeanCovarianceCalculator
#from extra.ipynb_utils import init_configs, test_model_acc, D_INPUTS
from extra.plotting import Plotter, Utilizer, Evaluator, MultiPlotter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Literal, Union
#from torch.distributions import MultivariateNormal
from kornia.augmentation import RandomGaussianNoise
#from functools import partial
from copy import deepcopy

torch.manual_seed(42)

DIR = '/Users/alicerigg/Code/polyapprox/experiments/results/descent/'

'''
Ablations defaults
- Training noise 0.01
- 0.0 wd

Ablations Full list:
- [ ] No training noise
- [ ] No weight decay
- [ ] No LR schedule
- [ ] SGD
- [x] All at once [AA]
- [ ] defaults (baseline)
- [ ] Preprocess MNIST -> N(0,1) with Transform
'''

dataset = 'mnist' # works with 'fmnist', 'emnist'

default = {
    'lr': 1e-3, # default: 1e-3
    'wd': 0.0,
    'epochs': 2**4,
    'd_input': 784,
    'bias': True,
    'batch_size': 2**11,
    'device': 'cpu',
    'train': MNIST(train=True, device='cpu'),
    'test': MNIST(train=False, device='cpu'),
    'noise': None,
    #'noise': None,
    'n_layer': 1,
    'optim': 'AdamW',
    'scheduler': 'cosine',
}

baseline = {
    'lr': 1e-5, # default: 1e-3
    'wd': 0.1,
    'epochs': 2**11,
    'd_input': 784,
    'bias': True,
    'batch_size': 2**5,
    'device': 'cpu',
    'train': MNIST(train=True, device='cpu'),
    'test': MNIST(train=False, device='cpu'),
    'noise': RandomGaussianNoise(std=0.05),
    #'noise': None,
    'n_layer': 1,
    'optim': 'AdamW',
    'scheduler': 'cosine',
}

ablate_lr_scheduler = deepcopy(baseline)
ablate_noise = deepcopy(baseline)
ablate_weight_decay = deepcopy(baseline)
sgd_optimizer = deepcopy(baseline)
all_ablations = deepcopy(baseline)

ablate_lr_scheduler['scheduler'] = None
ablate_noise['noise'] = None
ablate_weight_decay['wd'] = 0.0
sgd_optimizer['optim'] = 'SGD'

all_ablations['scheduler'] = None
all_ablations['noise'] = None
all_ablations['wd'] = 0.0
all_ablations['optim'] = 'SGD'

full_configs_list = {
    'Baseline': baseline,
    'All Ablations': all_ablations,
    'No LR Scheduler': ablate_lr_scheduler,
    'No Training Noise': ablate_noise,
    'No Weight Decay': ablate_weight_decay,
    'SGD Optimizer': sgd_optimizer,
}
savepaths = {
    'Baseline': 'baseline',
    'All Ablations': 'all_ablations',
    'No LR Scheduler': 'no_lr_scheduler',
    'No Training Noise': 'no_noise',
    'No Weight Decay': 'no_weight_decay',
    'SGD Optimizer': 'sgd_optimizer',
}

def change_all_configs(config_list, key, new_val):
    new_cfg_list = {}
    for _key, _val in config_list.items():
        new_config = deepcopy(_val)
        new_config[key] = new_val
        new_cfg_list[_key] = new_config
    return new_cfg_list
def ckpt_to_model(ckpt, configs):
    model = FFNModel.from_config(**configs)
    model.load_state_dict(state_dict=ckpt)
    return model
def process_checkpoint(config):
    relu_1l = FFNModel.from_config(**config)

    ckpts, metrics = relu_1l.fit(
        config['train'],
        config['test'],
        checkpoint_steps=[2**i for i in range(20)],
        return_ckpts=True
    )
    processed_ckpts = [Plotter(ckpt_to_model(ckpt, config), config['test']) for ckpt in ckpts]
    #eval_ckpts = [Plotter(ckpt_to_model(ckpt, config), config['test']) for ckpt in ckpts]
    processed_ckpts[-1].eval.init_generator(use_psd_sqrt=True)

    datadict = {'config': config, 'ckpts': ckpts,
                'test_set': config['test'], 'metrics': metrics}
    #datadict['metrics'] = metrics

    return datadict, processed_ckpts, metrics
def plot_checkpoint(multiplotter: MultiPlotter, measures=None, digit=3, show_beta=False,
                    description=None, separate_axis=False,
                    eig_figsize=(15,6), col_start=0, col_end=None):
    if measures is None:
        measures = multiplotter.evaluate_quantitative_measures()
    multiplotter.plot_accuracies(savepath=description)
    multiplotter.plot_fvu(savepath=description)
    multiplotter.plot_kl_divergence(savepath=description)
    #plot_fvu_kl_divergence(measures, description, separate_axis=separate_axis)
    show_beta = digit if show_beta else -1
    multiplotter.plot_eigvecs_over_list(show_beta=show_beta, savepath=description, out_dir=digit,
                           figsize=eig_figsize, col_start=col_start, col_end=col_end)
def full_sweep(config_list, separate_axis=False):
    for key, config in tqdm(config_list.items()):
        ckpts = process_checkpoint(config)
        multiplotter = MultiPlotter(ckpts)
        measures = multiplotter.evaluate_quantitative_measures()
        plot_checkpoint(ckpts, measures, description=key, separate_axis=separate_axis)
def plot_max_svds(full_data, key: Literal['fvu', 'kl'] = 'fvu', ymin=1e-3, ymax=1):
    assert key in ['fvu', 'kl'], ValueError('Invalid Key! Expecting fvu or kl')
    steps = [2**i for i in range(multiplotter.n)]
    linestyles = {'fvu': '--', 'kl': ':'}
    #linestyle = ['-', ':', '--']
    #colors = {'acc': cm.viridis}
    runs = ['relu', 'linear', 'quadratic']
    scale = np.linspace(0.9, 0.4, 3)

    max_data = {}
    baseline_data = {}
    plt.figure(figsize=(10,6))
    max_data[key] = full_data[key][:,:,-1] # shape (n,3)
    baseline_data[key] = full_data[key][:,:,0]
    for i, run in enumerate(runs):
        plt.plot(steps, max_data[key], color=cm.viridis(scale[i]),
                    linestyle=linestyles[key], label=f'{run} - Max {key.upper()}')
        plt.plot(steps, baseline_data[key], color=cm.viridis(scale[i]),
                    linestyle=linestyles[key], label=f'{run} - Base {key.upper()}')
            #plt.ylabel(key)
    
    plt.xscale('log')
    plt.ylim(ymin=ymin,ymax=ymax)
    plt.yscale('log')
    plt.xlabel('Training Steps')
    plt.ylabel(f'{key.upper()}')
    plt.legend()
    #plt.plot(steps, max_data['acc'][:,i])
def plot_eig_attack_data(data, metric='acc', std=torch.linspace(0, 1, 20),
                         ylogscale=False, xlogscale=False, ymin=None, ymax=None,
                         description='', savepath=None, format='pdf'):
    """
    Plot the results of the eig_attack_data against standard deviation values.
    
    Parameters:
        data (dict): Dictionary containing the results with keys 'acc', 'kl', 'fvu'.
        metric (str): Metric to plot ('acc', 'kl', or 'fvu').
        std (Tensor): The standard deviation values used for the x-axis.
    """
    # Ensure the provided metric is valid
    dir = 'eig_attack'
    assert metric in data, f"Metric '{metric}' is not in data. Available metrics: {list(data.keys())}"
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define styles
    colors = ['r', 'g', 'b']  # Colors for relu, linear, quadratic
    linestyles = ['-', '--', ':']  # Line styles for attack, noise, and baseline
    
    # Legend labels
    model_names = ['ReLU', 'Linear', 'Quadratic']
    intervention_names = ['Attack', 'Noise', 'Baseline']
    
    # Generate plots
    for i, color in enumerate(colors):  # Iterate over models
        for j, linestyle in enumerate(linestyles):  # Iterate over interventions
            ax.plot(
                std,  # x-axis: standard deviation
                data[metric][i, j, :],  # y-axis: values for the given metric
                label=f"{model_names[i]} - {intervention_names[j]}",  # Legend label
                color=color,
                linestyle=linestyle
            )
    
    # Add labels, legend, and title
    ax.set_xlabel("Magnitude", fontsize=14)
    ax.set_ylabel(metric.upper(), fontsize=14)
    if xlogscale:
        ax.set_xscale('log')  # Set x-axis to log scale
    if ylogscale:
        ax.set_yscale('log')  # Set y-axis to log scale
    if ymin is not None:
        ax.set_ylim(ymin=ymin)
    if ymax is not None:
        ax.set_ylim(ymax=ymax)
    ax.set_title(f"{metric.upper()} vs Magnitude", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    # Show plot
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(f'{DIR}{savepath}/{description}_{metric}.{format}', format=format)
    plt.show()


# Full MNIST sweep
_dir = 'new_baseline_uncentered'
config = deepcopy(baseline)
config['noise'] = None
#config['epochs'] = 2 ** 7
#config['test'].recenter()
#config['train'].recenter()
config['wd'] = 0.1
# 
#test = config['test']
#test.compute_mean(), test.compute_std()

# 
#config['lr'] = 1e-5
datadict, processed_ckpts, metrics = process_checkpoint(config)
print(datadict['metrics'])
models = [ckpt_to_model(ckpt, config) for ckpt in datadict['ckpts']]

config = datadict['config']
#test_plotter_list = [Plotter(ckpt)]
multiplotter = MultiPlotter(processed_ckpts)
measures = multiplotter.evaluate_quantitative_measures()

savepath=None
#_dir = 'new_baseline_uncentered'
long = 'long'
if True:
    ckpt_save_path = f'/Users/alicerigg/Code/polyapprox/experiments/data/{_dir}{long}_datadict.pt'
    torch.save(datadict, ckpt_save_path)
    print(f'checkpoints saved to {ckpt_save_path}')
#multiplotter.plot_accuracies(savepath=savepath)
# 
#multiplotter.plot_fvu(savepath=savepath,ylogscale=True)#, ymin=1e-3, ymax=1)
#multiplotter.plot_kl_divergence(savepath=savepath,ylogscale=True)
# %%
config = deepcopy(baseline)
relu_1l = FFNModel.from_config(**config)

ckpts, metrics = relu_1l.fit(
    config['train'],
    config['test'],
    checkpoint_steps=[2**i for i in range(20)],
    return_ckpts=True
)

# %%
import matplotlib.pyplot as plt

# Extracting data from metrics
train_loss = [m['train/loss'] for m in metrics]
train_acc = [m['train/acc'] for m in metrics]
val_loss = [m['val/loss'] for m in metrics]
val_acc = [m['val/acc'] for m in metrics]

# Plotting
plt.figure(figsize=(6, 6))

# Plot training and validation loss
#plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
# %%
# Plot training and validation accuracy
plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# %%
_dir = 'baseline_centered'
datadict['test_measures'] = measures
if True:

    ckpt_save_path = f'/Users/alicerigg/Code/polyapprox/experiments/data/{_dir}_datadict.pt'
    torch.save(datadict, ckpt_save_path)
    print(f'checkpoints saved to {ckpt_save_path}')
# %%
# %%
#savepath= dir + 'default'
savepath=None
multiplotter.plot_accuracies(savepath=savepath)
# 
multiplotter.plot_fvu(savepath=savepath,ylogscale=True)#, ymin=1e-3, ymax=1)
multiplotter.plot_kl_divergence(savepath=savepath,ylogscale=True)

# %% Free space. Compute logit contributions
n = multiplotter.n
quads = [multiplotter.plotters[i].util.ols for i in range(n)]
#linears = [multiplotter.plotters[i].util.linear for i in range(n)]

all_gammas = torch.zeros(n,10,784,784)
all_beta = torch.zeros(n,784,10)
all_alpha = torch.zeros(n,10)
for i in range(n):
    all_gammas[i] = quads[i].get_gamma_tensor()
    all_beta[i] = quads[i].beta
    all_alpha[i] = quads[i].alpha

eigvals = torch.linalg.eigvalsh(all_gammas)
print(eigvals.shape)
# %%
print(torch.norm(all_beta[:,:,3], dim=(1)))
print(eigvals[:,3,-1])

# n: checkpoints
# 3: ell, B, alpha
# 10: digits. By default just select 3rd one here.
ell_B_alpha = torch.zeros(n, 3, 10)
ell_B_alpha[:,0] = eigvals[:,:,-1]
ell_B_alpha[:,1] = torch.norm(all_beta, dim=1)
ell_B_alpha[:,2] = all_alpha

torch_save_path = f'/Users/alicerigg/Code/polyapprox/experiments/data/ell_B_alpha.pt'
torch.save(ell_B_alpha, torch_save_path)
print(f'ell_B_alpha saved to {torch_save_path}')

#print(torch.norm(all_alpha, dim=1))
# %%
# Load the ell_B_alpha tensor from the saved file
ell_B_alpha_load_path = f'/Users/alicerigg/Code/polyapprox/experiments/data/ell_B_alpha.pt'
ell_B_alpha_loaded = torch.load(ell_B_alpha_load_path)
print(f'ell_B_alpha loaded from {ell_B_alpha_load_path}')

# %% Pickle
import pickle

#run_data = {'measures': measures, 'ckpts': ckpts}
# Save measures to a pickle file
measures_save_path = f'{DIR}run_data.pkl'
with open(measures_save_path, 'wb') as f:
    pickle.dump(measures, f)
print(f'Measures saved to {measures_save_path}')
# %%
import pickle
# Load measures from the pickle file
with open(measures_save_path, 'rb') as f:
    loaded_measures = pickle.load(f)
print('Measures loaded from pickle file')
multiplotter = MultiPlotter(plotter_list=[], measures=loaded_measures)
# Verify that the loaded measures are the same as the original
#assert measures == loaded_measures, "Loaded measures do not match the original"
#print('Loaded measures match the original measures')

# %%
# "Main plots". No need for MNIST, fast to run
savepath= dir + 'default'

multiplotter.plot_accuracies(savepath=savepath)
# 
multiplotter.plot_fvu(savepath=savepath,ylogscale=True)#, ymin=1e-3, ymax=1)
multiplotter.plot_kl_divergence(savepath=savepath,ylogscale=True)#, ymin=1e-3, ymax=1)
#multiplotter.plot_eigvecs_over_list(show_beta=3, savepath=savepath, orientation=1)
# %%
custom_params = {
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'axes.titlepad': 20,
            'axes.labelpad': 10,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'grid.alpha': 0.7,
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
        }
# 
multiplotter.plot_accuracies(savepath=savepath, custom_params=custom_params)
# %%
subplot_dict = {'left': 0.025, 'right': 0.99, 'top': 0.82, 'bottom': 0.03,
                'wspace': 0.1, 'hspace': 0.03}
text_pos = {'text_x': 0.017, 'text_y1': 0.635, 'text_y2': 0.22}
multiplotter.plot_eigvecs_over_list(subplots_adjust=subplot_dict, show_beta=3, figsize=(21,3),
                                    savepath=savepath, text_pos=text_pos, orientation=1)
# %% 
idx = multiplotter.n - 2
print(f'Step 2^{idx}')
std = torch.linspace(0,1,5)
svd_attack_data = multiplotter.plotters[idx].get_svd_attack_data()
eig_attack_data = multiplotter.plotters[idx].get_eig_attack_data(topk=5, std=std)

# 
eigpath=dir + f'eig_attack'
description=f'ckpt_{idx}'
plot_eig_attack_data(eig_attack_data, 'acc', std=std, savepath=eigpath, description=description)
plot_eig_attack_data(eig_attack_data, 'fvu', std=std, ylogscale=True, ymin=1e-2, ymax=1e1, savepath=eigpath, description=description)
plot_eig_attack_data(eig_attack_data, 'kl', std=std, ylogscale=True, ymin=1e-2, ymax=1e1, savepath=eigpath, description=description)

# %%
plot._plot_svd_attack(svd_attack_data, mode=['all'], ylogscale=True, ymax=2)
# %% MVP for SVD attack. TODO: pick 1 SVD checkpoint (find 1 good idx) to dump NOW.
#idx = -1
for plot in multiplotter.plotters:
    plot = multiplotter.plotters[-1]
    data = plot.get_svd_attack_data()
    plot._plot_svd_attack(data, mode=['all'], ylogscale=True, ymax=2)
# %% SVD pass
keys = ['acc', 'fvu', 'kl']
all_data = {key: torch.zeros(multiplotter.n,3,12) for key in keys}
for i, plotter in tqdm(enumerate(multiplotter.plotters)):
    data = plotter.get_svd_attack_data()
    for key in keys:
        all_data[key][i][:,:-1] = data[key]
        all_data[key][i][:,-1] = data[key].max(dim=-1)[0]

    plotter.plot_svd_attack(data, mode=['all'], ylogscale=True,
                            xlogscale=False, ymin=1e-3, ymax=1, multiplot=True)

#%%
plot_max_svds(all_data, ymin=1e-3, ymax=None, key='kl')

# %%
