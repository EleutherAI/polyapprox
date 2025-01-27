import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import itertools
import einops
from collections import defaultdict
import copy
from einops import einsum
from kornia.augmentation import RandomGaussianNoise

from decomp.model import Model, FFNModel, _Config
from decomp.datasets import MNIST, FMNIST, CIFAR10 #, EMNIST, SVHN, MNIST1D
from decomp.plotting import plot_explanation, plot_eigenspectrum
#from mnist_2l import Minimal_FFN
from tqdm import tqdm
from torch_polyapprox.ols import ols

device = 'cpu'

D_INPUTS = {
    'mnist': 784,
    'fmnist': 784,
    'emnist': 784,
    'mnist1d': 40,
    'cifar': 3072,
    'svhn': 3072,
}
def init_datasets(device='cpu'):
    datasets = {
        'mnist': (MNIST(train=True, device=device), MNIST(train=False, device=device)),
        'fmnist': (FMNIST(train=True, device=device), FMNIST(train=False, device=device)),
        #'emnist': (EMNIST(train=True, device=device), EMNIST(train=False, device=device)),
        #'mnist1d': (MNIST1D(train=True, device=device), MNIST1D(train=False, device=device)),
        'cifar': (CIFAR10(train=True, device=device), CIFAR10(train=False, device=device)),
        #'svhn': (SVHN(train=True, device=device), SVHN(train=False, device=device))
    }
    return datasets
#if __name__ == "__main__":

datasets = init_datasets()
dataset = 'mnist'

def init_configs(dataset=None, device='cpu'):
    DEFAULT_CONFIG = {
        'lr': 1e-3, # default: 1e-3
        'wd': 0.2,
        'epochs': 128,
        'd_input': D_INPUTS[dataset],
        'bias': True,
        'bsz': 2**11,
        'device': device,
        'train': datasets[dataset][0],
        'test': datasets[dataset][1],
        'noise': RandomGaussianNoise(std=0.1),
        #'noise': None,
        'n_layer': 6
    }
    return DEFAULT_CONFIG

CPU_DATASETS = {
    'mnist': (MNIST(train=True, device='cpu'), MNIST(train=False, device='cpu')),
    'cifar': (CIFAR10(train=True, device='cpu'), CIFAR10(train=False, device='cpu'))}
DEFAULT_CONFIG = {
    'lr': 1e-3, # default: 1e-3
    'wd': 0.2,
    'epochs': 128,
    'd_input': D_INPUTS[dataset],
    'bias': True,
    'bsz': 2**11,
    'device': device,
    'train': datasets[dataset][0],
    'test': datasets[dataset][1],
    'noise': RandomGaussianNoise(std=0.1),
    #'noise': None,
    'n_layer': 6
}

def approx_fit(model, order='linear'):
    W1 = model.w_e.detach()
    W2 = model.w_u.detach()
    b1 = model.embed.bias.detach()#.cpu().data.numpy()
    b2 = model.head.bias.detach()
    return ols(W1, b1, W2, b2, act='relu', order=order)

def lin_fit_list(list_of_models):
    lin_fits = []
    for model in tqdm(list_of_models):
        lin_fits.append(approx_fit(model))
    return lin_fits

def quad_fit_list(list_of_models):
    quad_fits = []
    for model in tqdm(list_of_models):
        quad_fits.append(approx_fit(model, order='quadratic'))
    return quad_fits

def test_model_acc(model,
                   test_set,
                   num_samples=10000,
                   transform = None,
                   proj = None,
                   loss = None):
    assert 0 <= num_samples <= test_set.y.size(0), 'num_samples too large!!'
    
    #print(test.x.shape)
    x = test_set.x
    inp = test_set.x.flatten(start_dim=1)
    #print(inp, inp.shape)
    #assert inp.device == model.device
    if transform is not None:
        #print(inp.shape)
        inp = transform(test_set.x).flatten(start_dim=1)
    if proj is not None:
        inp = torch.einsum('ij,bj->bi', proj, inp)
    else:
        pass
        #inp = test.x.flatten(start_dim=1)
    fwd = model(inp[:num_samples])
    if loss is None: # if not provided, use accuracy
        loss = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
    return loss(fwd, test_set.y[:num_samples]).item()
    #else: # assume kl, i know, horrible practice
    #    return loss(fwd)


def np_gamma_to_B(gamma_mat):
    gamma_entries = gamma_mat.shape[-1]
    #print(gamma_entries)
    row_dim = int(np.floor(np.sqrt(2*gamma_entries)))
    full_mat = np.zeros((row_dim, row_dim, gamma_mat.shape[0]))
    tril_indices = np.tril_indices(row_dim)
    
    full_mat[tril_indices] = gamma_mat.T
    full_mat = 0.5 * (full_mat + full_mat.transpose(1, 0, 2))
    return full_mat.permute(2, 0, 1)

def gamma_to_B(gamma_mat):
    gamma_entries = gamma_mat.shape[-1]
    print(gamma_entries)
    row_dim = math.floor((2*gamma_entries)**0.5)
    full_mat = torch.zeros((gamma_mat.shape[0], row_dim, row_dim))
    tril_indices = torch.tril_indices(row_dim, row_dim)
    
    full_mat[:, tril_indices[0], tril_indices[1]] = gamma_mat
    full_mat = 0.5 * (full_mat + full_mat.mT)
    return full_mat

def test_inner(x, quad):
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    
    full_mat = gamma_to_B(quad.gamma)
    print(full_mat.shape, x.shape)
    prod = torch.einsum('hij,bi,bj->bh', full_mat, x, x)
    return prod

def test_outer(x, quad):
    outer = torch.einsum('ij,ik->ijk', x, x)
    rows, cols = torch.tril_indices(x.shape[1], x.shape[1])
    print(outer[:, rows, cols].shape, quad.gamma.shape)
    return outer[:, rows, cols] @ quad.gamma.T

def fit_approxs(model, debug_mode=False):
    W1 = model.w_e.detach()
    W2 = model.w_u.detach()
    b1 = model.embed.bias.detach()#.cpu().data.numpy()
    b2 = model.head.bias.detach()

    #print(type(W1), type(W2), type(b1), type(b2))
    lin_01 = ols(W1, b1, W2, b2, act='relu')
    quad01 = ols(W1, b1, W2, b2, act='relu', order='quadratic', debug_mode=debug_mode)
    return lin_01, quad01

import pandas as pd

def adv_svd(quad_approx, u_mat = None, topk=1):
    if u_mat is None: # uses inputted umat by default. Otherwise, computes it for given quad.
        u_mat, _, _ = torch.svd(quad_approx.beta)
    P = torch.eye(u_mat.shape[0])
    for i in range(topk):
        P -= torch.outer(u_mat[:,i], u_mat[:,i])
    return P

def test_svd_perf(base_model, approxs=None, topk=10, transform=None, loss=None, n=10000):
    if approxs is None:
        lin_fit, quad_fit = fit_approxs(base_model)
    else:
        lin_fit, quad_fit = approxs[0], approxs[1]
    _u, _, _ = torch.svd(quad_fit.beta)
    base_accs, line_accs, quad_accs = [], [], []
    for i in range(topk+1):
        base_accs.append(test_model_acc(base_model, transform=transform, proj=adv_svd(_u,topk=i), loss=loss, num_samples=n))
        line_accs.append(test_model_acc(lin_fit, transform=transform, proj=adv_svd(_u,topk=i), loss=loss, num_samples=n))
        quad_accs.append(test_model_acc(quad_fit, transform=transform, proj=adv_svd(_u,topk=i), loss=loss, num_samples=n))
    return base_accs, line_accs, quad_accs

def show_results(model_pair, topk=10, transform_pair=(None,None), loss=None): # add reg_model, unreg_model as args with defaults.
    rows = []
    reg_base_accs, reg_line_accs, reg_quad_accs = test_svd_perf(model_pair[0], transform=transform_pair[0], loss=loss)
    #print(reg_base_accs)
    unr_base_accs, unr_line_accs, unr_quad_accs = test_svd_perf(model_pair[1], transform=transform_pair[1], loss=loss)
    #print(unr_base_accs)
    for i in range(topk+1):
        rows.append((i, reg_base_accs[i], reg_line_accs[i], reg_quad_accs[i],
                        unr_base_accs[i], unr_line_accs[i], unr_quad_accs[i]))

    df = pd.DataFrame(rows, columns=['TopK', 'regRelu', 'regLinear', 'regQuadratic',
                                       'unregRelu', 'unregLinear', 'unregQuadratic'])
    
    return df

def plot_svd_results(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['TopK'], df['regRelu'], color='b', label='Reg ReLU', linestyle='-')
    plt.plot(df['TopK'], df['regLinear'], color='b', label='Reg Linear', linestyle=':')
    plt.plot(df['TopK'], df['regQuadratic'], color='b', label='Reg Quadratic', linestyle='--')
    plt.plot(df['TopK'], df['unregRelu'], color='r', label='Unreg ReLU', linestyle='-')
    plt.plot(df['TopK'], df['unregLinear'], color='r', label='Unreg Linear', linestyle=':')
    plt.plot(df['TopK'], df['unregQuadratic'], color='r', label='Unreg Quadratic', linestyle='--')

    # Adding a title and labels
    plt.title('SVD Performance Across Different Models and TopK')
    plt.xlabel('TopK')
    plt.ylabel('Accuracy')
    
    # Add grid and legend, then display it.
    plt.grid()
    plt.legend()
    plt.show()
