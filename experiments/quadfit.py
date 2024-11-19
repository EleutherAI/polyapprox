import torch
import torch.nn as nn
import torch.optim as optim
from decomp.model import FFNModel
from decomp.datasets import MNIST, FMNIST, CIFAR10
from kornia.augmentation import RandomGaussianNoise
from tqdm import tqdm
from pandas import DataFrame
from schedulefree import ScheduleFreeWrapper
from transformers import PretrainedConfig, PreTrainedModel
#from torch.nn import Pretr
# Define the quadratic model training

class BilinearModel(nn.Module):
    def __init__(self, d_input=784, d_hidden=64, n_layer=1, device='cpu'):
        super().__init__()
        self.W = nn.Parameter(torch.randn((10, d_input)))
        self.V = nn.Parameter(torch.randn((10, d_input)))
        self.B = nn.Parameter(torch.randn((10, d_input))).to(device)
        self.C = nn.Parameter(torch.randn(10)).to(device)

    def forward(self, x):
        _x = x.flatten(start_dim=1)
        a1 = torch.einsum('bi,hi,hj->bhj', _x, self.W, self.V)
        a2 = torch.einsum('bj,bhj->bh', _x, a1)
        b = x @ self.B.T
        c = self.C
        return a2+b+c   
class QuadraticModel(nn.Module):
    def __init__(self, d_input=784, d_hidden=64, n_layer=1, device='cpu'):
        super().__init__()
        if n_layer == 1:
            self.A = nn.Parameter(torch.randn((10, d_input, d_input))).to(device)
            self.B = nn.Parameter(torch.randn((10, d_input))).to(device)
            self.C = nn.Parameter(torch.randn(10)).to(device)
            #self.A = nn.Parameter(torch.randn((10, d_input, d_input)),device=device)#.to(device)
            #self.B = nn.Parameter(torch.randn((10, d_input)),device=device)#.to(device)
            #self.C = nn.Parameter(torch.randn(10),device=device)#.to(device)
        elif n_layer == 2:
            self.A = nn.ParameterList([
                torch.randn((d_hidden, d_input, d_input),device=device),
                torch.randn((10, d_hidden, d_hidden),device=device)
                ])
            self.B = nn.ParameterList([
                torch.randn((d_hidden, d_input),device=device),
                torch.randn((10, d_hidden),device=device)
                ])
            self.C = nn.ParameterList([
                torch.randn((d_hidden),device=device),
                torch.randn((10),device=device)
                ])
        else:
            pass
        #self.A = nn.ParameterList([
        #    torch.randn((d_hidden, d_input, d_input),device=device) for _ in range(n_layer)])
        #self.B = nn.ParameterList([torch.randn((10, d_input),device=device) for _ in range(n_layer)])
        #self.C = nn.Parameter(torch.randn(10)).to(device)

    def forward(self, x):
        _x = x.flatten(start_dim=1)
        #a = torch.einsum('bi,bj,hij->bh', _x, _x, self.A) A: 10,3072,3072 ~ 30M
        a1 = torch.einsum('bi,hij->bhj', _x, self.A) # [1,3072], [10,3072,3072] -> [1,10,3072]
        a2 = torch.einsum('bj,bhj->bh', _x, a1)
        b = x @ self.B.T
        c = self.C
        return a2+b+c


def train_quadratic_model(model_to_approx, cfg, modelClass=QuadraticModel, num_epochs=500, learning_rate=3e-2):
    device = model_to_approx.device
    bsz = cfg['bsz']
    model = modelClass(d_input=cfg['d_input']).to(device)
    #base_optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0025)
    
    if modelClass == QuadraticModel:
        optimizer = optim.Adam([model.A, model.B, model.C], lr=learning_rate, betas=(0.9,0.999))
    else:
        optimizer = optim.Adam([model.W, model.V, model.B, model.C], lr=learning_rate, betas=(0.9,0.999))
    #optimizer = ScheduleFreeWrapper(
    #    base_optimizer, momentum=0.9, weight_decay_at_y=0.1)
    #optimizer = optim.Adam([model.A, model.B, model.C], lr=learning_rate)
    criterion = nn.MSELoss()
    history = []
    model.train()
    for epoch in tqdm(range(num_epochs), disable=True):
        x = torch.randn(bsz, cfg['d_input'], device=device, requires_grad=True)#.to(device)
        with torch.no_grad():
            #x = torch.randn(bsz, cfg['d_input']).to(device)
            pass
            #y = model_to_approx(x.detach())
        y = model_to_approx(x)
        #print(x.grad_fn)
        #print(y.grad_fn)
        optimizer.zero_grad()
        output = model.forward(x)
        #print(output.grad_fn)
        loss = criterion(output, y)
        #print(loss.grad_fn)
        loss = loss / float(bsz)
        #print(loss)
        #print(loss.grad_fn)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            metrics = {
                'Epoch': epoch+1,
                'MSE': loss.item(),
                'Val/acc': evaluate_approx(model, cfg).item(),
            }
            #e, l, a = epoch, loss.item(), evaluate_approx(model, cfg)
            print(f'Epoch {epoch+1}, Loss: {metrics['MSE']}, Val/acc: {metrics['Val/acc']}')
            history.append(metrics)
            #print(f'Epoch {epoch}, Loss: {loss.item()}, Val/acc: {evaluate_approx(model, cfg)}')

    return model, history

def init_model(config):
    model = FFNModel.from_config(
            wd=config['wd'],
            epochs=config['epochs'],
            d_input=config['d_input'],
            bias=config['bias']).to(config['device'])
    return model
    
def load_checkpoint(epoch, config, model_label='1l_mnist', mode='with_noise'):
    print(f'Model label: {model_label}')
    path = f'/mnt/ssd-1/mechinterp/alice/polyapprox/polyapprox/experiments/ckpts/{mode}/{model_label}/'
    model_name = f'relu_model_epoch{str(epoch).zfill(4)}.pth'
    print(f'Full path:\n{path}{model_name}')
    model = init_model(config)
    checkpoint = torch.load(path + model_name)
    
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def evaluate_approx(model, cfg):
    test_x = cfg['test'].x.flatten(start_dim=1)
    test_y = cfg['test'].y
    accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
    y_hat = model(test_x)
    return accuracy(y_hat, test_y)

# Main execution
if __name__ == "__main__":
    _device = 'cuda:0'
    dataset = 'mnist'
    d_inputs = {
        'mnist': 784,
        'fmnist': 784,
        'cifar': 3072
    }
    datasets = {
        'mnist': (MNIST(train=True, device=_device), MNIST(train=False, device=_device)),
        'fmnist': (FMNIST(train=True, device=_device), FMNIST(train=False, device=_device)),
        'cifar': (CIFAR10(train=True, device=_device), CIFAR10(train=False, device=_device))
    }
    cfg = {
        'wd': 0.2,
        'epochs': 1,
        'd_input': d_inputs[dataset],
        'bias': True,
        'bsz': 2**14,
        'device': _device,
        'train': datasets[dataset][0],
        'test': datasets[dataset][1],
        'noise': RandomGaussianNoise(std=0.4)
    }
    #relu_model = load_checkpoint(128, cfg, f'1l_{dataset}', 'without_noise')
    relu_model = init_model(cfg)
    relu_model.fit(cfg['train'], cfg['test'], transform=cfg['noise'])
    relu_model.eval()
    #quadratic_model, qmetrics = train_quadratic_model(relu_model, cfg=cfg)
    bilinear_model, bmetrics = train_quadratic_model(relu_model, cfg)
    accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
    test_x = cfg['test'].x.flatten(start_dim=1)
    test_y = cfg['test'].y
    #y = quadratic_model(test_x)
    y = bilinear_model
    acc = accuracy(y, test_y)
    print(f'MNIST test set accuracy: {acc}')
    print("Training complete.")
    print(DataFrame.from_records(bmetrics, columns=['Epoch', 'MSE', 'Val/acc']))