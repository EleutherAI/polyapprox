import torch.nn as nn
from decomp.model import FFNModel
from decomp.datasets import MNIST, CIFAR10
from kornia.augmentation import RandomGaussianNoise
from extra.ipynb_utils import test_model_acc
from torch_polyapprox.ols import ols
import torch
from torch import Tensor
from schedulefree import ScheduleFreeWrapper
from tqdm import tqdm
from typing import List
from itertools import combinations, chain
import matplotlib.pyplot as plt
import numpy as np

def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
def compose_n_modules(Rs: List):
    if not Rs:
        return lambda x: x
    else:
        first, *rest = Rs
        return lambda x: first(compose_n_modules(rest)(x))
def nk_weave(comb, List1, List2):
    n = len(List1)
    assert len(List1) == len(List2)
    new_list = []
    for i in range(n):
        if i in comb:
            new_list.append(List1[i])
        else:
            new_list.append(List2[i])
    return new_list
def gamma_decompose(out_vec, gamma_mat):
    if isinstance(out_vec, Tensor):
        q = torch.einsum(gamma_mat, out_vec, 'h,hij->ij')
    elif isinstance(out_vec, int):
        q = gamma_mat[out_vec]
    q = 0.5 * (q + q.mT)
    print(q.shape)
    eigvals, eigvecs = torch.linalg.eigh(q)
    return eigvals, eigvecs
class Minimal_FFN(nn.Module):
    def __init__(self, data, device='cpu'):
        super().__init__()
        self.W1 = nn.Parameter(data['W1']).to(device)
        self.W2 = nn.Parameter(data['W2']).to(device)
        self.activation = nn.ReLU()
        self.b1 = nn.Parameter(data['b1']).to(device)
        self.b2 = nn.Parameter(data['b2']).to(device)

    def enc(self, x: Tensor):
        x = x.flatten(start_dim=1)
        x = self.activation(x @ self.W1.T + self.b1)
        return x
    def dec(self, x: Tensor):
        return x @ self.W2.T + self.b2
    def forward(self, x: Tensor):
        return self.dec(self.enc(x))

    def approx_fit(self, order='linear'):
        W1 = self.W1.detach()
        W2 = self.W2.detach()
        b1 = self.b1.detach()
        b2 = self.b2.detach()
        return ols(W1,b1,W2,b2,order=order,act='relu')

if __name__ == "__main__":
    device = 'cuda:4'
    dataset = 'mnist'
    fitting_all = True
    fitting_quadratic = True
    d_inputs = {
        'mnist': 784,
        'fmnist': 784,
        'cifar': 3072
    }
    datasets = {
        'mnist': (MNIST(train=True, device=device), MNIST(train=False, device=device)),
        'cifar': (CIFAR10(train=True, device=device), CIFAR10(train=False, device=device))}
    cpu_datasets = {
        'mnist': (MNIST(train=True, device='cpu'), MNIST(train=False, device='cpu')),
        'cifar': (CIFAR10(train=True, device='cpu'), CIFAR10(train=False, device='cpu'))}
    cfg = {
        'lr': 1e-3, # default: 1e-3
        'wd': 0.0,
        'epochs': 128,
        'd_input': d_inputs[dataset],
        'bias': True,
        'bsz': 2**11,
        'device': device,
        'train': datasets[dataset][0],
        'test': datasets[dataset][1],
        'noise': RandomGaussianNoise(std=0.1),
        #'noise': None,
        'n_layer': 1
    }
    relu_model = FFNModel.from_config(
            lr=cfg['lr'],
            wd=cfg['wd'],
            epochs=cfg['epochs'],
            batch_size=cfg['bsz'],
            d_input=cfg['d_input'],
            bias=cfg['bias'],
            n_layer=cfg['n_layer']
    ).to(device)
    train, test = datasets[dataset][0], datasets[dataset][1]
    metrics = relu_model.fit(train, test, cfg['noise'], checkpoint_epochs=[1,2,4,8,16,32,64,128])
    cpu_test = cpu_datasets[dataset][1]
    if fitting_all:
        data = [relu_model.get_layer_data(i) for i in range(relu_model.n_layer)][::-1]
        assert torch.allclose(data[-1]['W1'], relu_model.blocks[0].cpu().weight)
        Relus = [Minimal_FFN(datum) for datum in data]
        linear_fits = [Relu.approx_fit() for Relu in tqdm(Relus)]
        quad_fits = [Relu.approx_fit('quadratic') for Relu in tqdm(Relus)]
        all_subsets = powerset(range(cfg['n_layer']))
        print('-'*20, 'Testing linear fits', '-'*20)
        for comb in all_subsets:
            franken = compose_n_modules(nk_weave(comb, linear_fits, Relus))
            print(f'{comb}: {test_model_acc(franken, test_set=cpu_test)}')
        print('-'*20, 'Testing quadratic fits', '-'*17)
        for comb in all_subsets:
            franken = compose_n_modules(nk_weave(comb, quad_fits, Relus))
            print(f'{comb}: {test_model_acc(franken, test_set=cpu_test)}')
    
    # take quad_fit_1. Compute SVD
    g, b, a = quad_fits[0].get_gamma_tensor(), quad_fits[0].beta, quad_fits[0].alpha

    u, s, v = torch.svd(b) # print s
    out_logit = 7
    #print(f'g {g.shape}')
    #eigvals, eigvecs = gamma_decompose(out_logit, g)
    #eigvals, eigvecs = torch.linalg.eig(g[out_logit])
    eigvals = torch.linalg.eigvals(g[out_logit])
    print(eigvals.shape)
    print(f'singular values: {s}')
    #print(f'eigvals shape {eigvals.shape}\n eigvals\n{eigvals}')
    # now check output logit top eigs.
    #x = torch.arange(0, 784).numpy()
    #y = eigvals.numpy()
    #z = s.numpy()
    print(f'sample norm: {torch.norm(test.x[0])}')
    plotting=False
    if plotting:
        plt.figure(figsize=(10, 5))  # Optional, adjust the figure size as desired
        plt.plot(x, y, label='Trace 1', color='blue')  # First trace
        plt.plot(x, z, label='Trace 2', color='orange')  # Second trace

        # Adding labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Line Graph of Two Traces')
        plt.legend()

        # Display the plot
        plt.show()
    else:
        #pass
        print(f'L2: singular {torch.norm(s)}, eig7 {torch.norm(eigvals)}')
        print(f'L1: singular {torch.norm(s,p=1)}, eig7 {torch.norm(eigvals,p=1)}')
        
    print("Training complete.")
    
# Next up: 