import torch.nn as nn
from decomp.model import FFNModel
from decomp.datasets import MNIST
from kornia.augmentation import RandomGaussianNoise
from extra.ipynb_utils import test_model_acc
from torch_polyapprox.ols import ols
import torch
from schedulefree import ScheduleFreeWrapper

def compose_modules(R2, R1):
    return lambda x: R2(R1(x))
class Minimal_FFN(nn.Module):
    def __init__(self, data, device='cpu'):
        super().__init__()
        self.w_e = nn.Parameter(data['We']).to(device)
        self.w_u = nn.Parameter(data['Wu']).to(device)
        self.activation = nn.ReLU()
        self.b_e = nn.Parameter(data['be']).to(device)
        self.b_u = nn.Parameter(data['bu']).to(device)

    def enc(self, x):
        x = x.flatten(start_dim=1)
        x = self.activation(x @ self.w_e.T + self.b_e)
        return x
    def dec(self, x):
        return x @ self.w_u.T + self.b_u
    def forward(self, x):
        return self.dec(self.enc(x))

    def approx_fit(self, order='linear', mode='all'):
        W1 = self.w_e.detach()
        W2 = self.w_u.detach()
        b1 = self.b_e.detach()#.cpu().data.numpy()
        b2 = self.b_u.detach()
        print(W1.shape, W2.shape, b1.shape, b2.shape)
        if mode == 'enc':
            W2 = torch.eye(W2.shape[0])
            b2 = torch.zeros_like(b2)
        #elif mode == 'dec':
        #    W1 = torch.zeros_like(W1)
        return ols(W1,b1,W2,b2,order=order,act='relu')

if __name__ == "__main__":
    device = 'cpu'
    dataset = 'mnist'
    d_inputs = {
        'mnist': 784,
        'fmnist': 784,
        'cifar': 3072
    }
    datasets = {'mnist': (MNIST(train=True, device=device), MNIST(train=False, device=device))}
    cpu_datasets = {'mnist': (MNIST(train=True, device='cpu'), MNIST(train=False, device='cpu'))}
    cfg = {
        'wd': 0.2,
        'epochs': 4,
        'd_input': d_inputs[dataset],
        'bias': True,
        'bsz': 2**11,
        'device': device,
        'train': datasets[dataset][0],
        'test': datasets[dataset][1],
        'noise': RandomGaussianNoise(std=0.4)
    }
    relu_model = FFNModel.from_config(
            wd=cfg['wd'],
            epochs=cfg['epochs'],
            batch_size=cfg['bsz'],
            d_input=cfg['d_input'],
            bias=cfg['bias'],
            n_layer=2
    ).to(device)
    train, test = datasets[dataset][0], datasets[dataset][1]
    metrics = relu_model.fit(train, test, cfg['noise'])
    
    # convert relu_model to data_dict object
    data1, data2 = {}, {}
    data1['We'] = relu_model.blocks[0].weight.data.cpu().detach()
    data1['be'] = relu_model.blocks[0].bias.data.cpu().detach()
    data1['Wu'] = relu_model.blocks[1].weight.data.cpu().detach()
    data1['bu'] = relu_model.blocks[1].bias.data.cpu().detach()
    
    data2['We'] = data1['Wu']
    data2['be'] = data1['bu']
    data2['Wu'] = relu_model.head.weight.data.cpu().detach()
    data2['bu'] = relu_model.head.bias.data.cpu().detach()
    
    Relu1 = Minimal_FFN(data1)
    Relu2 = Minimal_FFN(data2)
    lin1 = Relu1.approx_fit(mode='enc').cpu()
    Lin1 = Relu1.approx_fit().cpu()
    lin2 = Relu2.approx_fit().cpu()
    quad1 = Relu1.approx_fit(order='quadratic',mode='enc').cpu()
    Quad1 = Relu1.approx_fit(order='quadratic').cpu()
    quad2 = Relu2.approx_fit(order='quadratic').cpu()
    cpu_test = cpu_datasets[dataset][1]
    r1 = lambda x: Relu1.enc(x)
    r2 = lambda x: Relu2.dec(Relu2.activation(x))
    r2r1 = lambda x: Relu2.dec(Relu2.activation(Relu1(x)))
    
    l2l1 = lambda x: lin2(lin1(x)) # 84% with enc mode
    l2L1 = lambda x: lin2(lin1(x)) # 84% with enc mode
    q2q1 = lambda x: quad2(quad1(x)) # 91.5% with quad mode
    l2q1 = lambda x: lin2(quad1(x)) # bad fit might be q1 incorrectly fit, factoring output. should be just enc.
    q2l1 = lambda x: quad2(lin1(x))
    
    q2r1 = lambda x: quad2(Relu1.enc(x)) # 95.9% without enc mode, 92% with
    l2r1 = lambda x: lin2(Relu1.enc(x)) # 95.5% without enc mode, 92% with
    r2q1 = lambda x: Relu2.dec(Relu2.activation(quad1(x))) # 95% without enc mode, 13.7% with
    r2l1 = lambda x: Relu2.dec(Relu2.activation(lin1(x))) # 78% without enc mode,  12.3% with
    #Composed_quad = 
    #print('-'*10, 'Relu relu baseline', '-'*10)
    print(f'r2r1: {test_model_acc(r2r1, test_set=cpu_test)}')
    #print(f'relu_model: {test_model_acc(relu_model, test_set=cpu_test)}')
    print(f'Lin + quad:')
    print(f'l2l1: {test_model_acc(l2l1, test_set=cpu_test)}')
    print(f'l2l1: {test_model_acc(l2l1, test_set=cpu_test)}')
    print(f'q2q1: {test_model_acc(q2q1, test_set=cpu_test)}')
    print(f'l2q1: {test_model_acc(l2q1, test_set=cpu_test)}')
    print(f'q2l1: {test_model_acc(q2l1, test_set=cpu_test)}')
    print('-'*10, 'Relu+quad/lin', '-'*10)
    print(f'q2r1: {test_model_acc(q2r1, test_set=cpu_test)}')
    print(f'l2r1: {test_model_acc(l2r1, test_set=cpu_test)}')
    print(f'r2q1: {test_model_acc(r2q1, test_set=cpu_test)}')
    print(f'r2l1: {test_model_acc(r2l1, test_set=cpu_test)}')
    #print(f'Accuracies: Quad2(Quad1) {test_model_acc(Composed_quad, test_set=cpu_test)}')
    #print(f'Accuracies: Lin1 {test_model_acc(lin1, test_set=cpu_test)}, Lin2 {test_model_acc(lin2, test_set=cpu_test)}')
    print("Training complete.")