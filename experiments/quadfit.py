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

class FFNModel(PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        torch.manual_seed(config.seed)

        d_input, d_hidden, d_output = config.d_input, config.d_hidden, config.d_output
        bias = config.bias
        in_bias = config.in_bias
        out_bias = config.out_bias
        n_layer = config.n_layer
        if bias:
            in_bias = True
            out_bias = True

        self.activation = nn.ReLU()
        self.blocks = nn.ModuleList(
            [Linear(d_input, d_hidden, bias=in_bias)] + [
             Linear(d_hidden, d_hidden, bias=in_bias) for _ in range(n_layer-1)])
        self.head = Linear(d_hidden, d_output, bias=out_bias)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Float[Tensor, "... inputs"]) -> Float[Tensor, "... outputs"]:
        x = x.flatten(start_dim=1)
        for layer in self.blocks:
            x = self.activation(layer(x))
        return self.head(x)

    def accuracy(self, y_hat, y):
        return (y_hat.argmax(dim=-1) == y).float().mean()
    @property
    def w_e(self):
        return self.embed.weight.data

    @property
    def w_block(self):
        return torch.stack([self.blocks[i].weight.data for i in range(self.n_layer-1)], dim=0)
    @property
    def w_u(self):
        return self.head.weight.data

    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(_Config(*args, **kwargs))

    @classmethod
    def from_config_obj(cls, config: _Config):
        #return cls(
        #    lr=config.lr,
        #    wd=config.wd,
        #    epochs=config.epochs,
        #)
        kwargs = {attr: getattr(config, attr) for attr in vars(config) if hasattr(config, attr)}
        return cls(**kwargs)
    
    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        new = cls(_Config(*args, **kwargs))
        new.load_state_dict(torch.load(path))
        return new

    def step(self, x, y):
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)

        return loss, accuracy

    def save_checkpoint(model, metrics, epoch, filename='relu_model'):
        folder = '/mnt/ssd-1/mechinterp/alice/polyapprox/polyapprox/experiments/ckpts/'
        #filepath=os.path.join(base_filepath, f'{filename}_epoch{str(epoch).zfill(4)}.pth')
        filepath = folder + f'{filename}_epoch{str(epoch).zfill(4)}.pth'
        #directory = os.path.dirname(base_filepath)
        current_file_path = os.path.abspath(__file__)
        #print(current_file_path)
        # Check if the directory exists, and if not, create it
        #if not os.path.exists(directory):
        #    os.makedirs(directory)
        #filepath = f'{base_filepath}_epoch{str(epoch).zfill(4)}'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, filepath)

    def fit(self, train, test, transform=None, disable=False, test_transform=False, checkpoint_epochs=None):
        torch.manual_seed(self.config.seed)
        torch.set_grad_enabled(True)

        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)

        loader = DataLoader(train, batch_size=self.config.batch_size, shuffle=True, drop_last=True, collate_fn=_collator(transform))
        test_transf = transform if test_transform else None
        test_loader = DataLoader(test, batch_size=test.y.size(0), shuffle=False, drop_last=True, collate_fn=_collator(test_transf))
        #test_x, test_y = test.x, test.y

        pbar = tqdm(range(self.config.epochs), disable=disable)
        history = []
        
        #checkpoints = {}
        
        checkpoint_epochs = checkpoint_epochs or []

        for epoch_num in pbar:
            epoch = []
            for x, y in loader:
                loss, acc = self.train().step(x, y)
                epoch += [(loss.item(), acc.item())]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            test_x, test_y = next(iter(test_loader))
            val_loss, val_acc = self.eval().step(test_x, test_y)

            metrics = {
                "train/loss": sum(loss for loss, _ in epoch) / len(epoch),
                "train/acc": sum(acc for _, acc in epoch) / len(epoch),
                "val/loss": val_loss.item(),
                "val/acc": val_acc.item()
            }

            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3f}" for k, v in metrics.items()))
            
            # Checkpoint saving
            if epoch_num+1 in checkpoint_epochs:
                self.save_checkpoint(self, epoch_num+1)

        torch.set_grad_enabled(False)
        return DataFrame.from_records(history, columns=['train/loss', 'train/acc', 'val/loss', 'val/acc'])
class QuadraticModel(nn.Module):
    def __init__(self, d_input=784, device='cpu'):
        super().__init__()
        self.A = nn.Parameter(torch.randn(10, d_input, d_input)).to(device)
        self.B = nn.Parameter(torch.randn(10, d_input)).to(device)
        self.C = nn.Parameter(torch.randn(10)).to(device)

    def forward(self, x):
        _x = x.flatten(start_dim=1)
        #a = torch.einsum('bi,bj,hij->bh', _x, _x, self.A) A: 10,3072,3072 ~ 30M
        a1 = torch.einsum('bi,hij->bhj', _x, self.A) # [1,3072], [10,3072,3072] -> [1,10,3072]
        a2 = torch.einsum('bj,bhj->bh', _x, a1)
        b = x @ self.B.T
        c = self.C
        return a2+b+c


def train_quadratic_model(model_to_approx, cfg, num_epochs=500, learning_rate=3e-2):
    device = model_to_approx.device
    bsz = cfg['bsz']
    model = QuadraticModel(d_input=cfg['d_input']).to(device)
    #base_optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0025)
    
    base_optimizer = optim.Adam([model.A, model.B, model.C], lr=learning_rate, betas=(0.9,0.999))
    optimizer = ScheduleFreeWrapper(
        base_optimizer, momentum=0.9, weight_decay_at_y=0.1)
    #optimizer = optim.SGD([model.A, model.B, model.C], lr=learning_rate)
    criterion = nn.MSELoss()
    history = []
    for epoch in tqdm(range(num_epochs), disable=True):
        with torch.no_grad():
            x = torch.randn(bsz, cfg['d_input']).to(device)
            y = model_to_approx(x)
            
        optimizer.zero_grad()
        output = model.forward(x)
        loss = criterion(output, y)
        loss = loss / float(bsz)
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

def load_checkpoint(epoch, config, model_label='1l_mnist', mode='with_noise'):
    print(f'Model label: {model_label}')
    path = f'/mnt/ssd-1/mechinterp/alice/polyapprox/polyapprox/experiments/ckpts/{mode}/{model_label}/'
    model_name = f'relu_model_epoch{str(epoch).zfill(4)}.pth'
    print(f'Full path:\n{path}{model_name}')
    model = FFNModel.from_config(
            wd=config['wd'],
            epochs=config['epochs'],
            d_input=config['d_input'],
            bias=config['bias']).to(config['device'])
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
    _device = 'cuda:3'
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
        'epochs': 128,
        'd_input': d_inputs[dataset],
        'bias': True,
        'bsz': 2**14,
        'device': _device,
        'train': datasets[dataset][0],
        'test': datasets[dataset][1],
        'noise': RandomGaussianNoise(std=0.4)
    }
    relu_model = load_checkpoint(128, cfg, f'1l_{dataset}', 'without_noise')
    quadratic_model, metrics = train_quadratic_model(relu_model, cfg=cfg)
    accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
    test_x = cfg['test'].x.flatten(start_dim=1)
    test_y = cfg['test'].y
    y = quadratic_model(test_x)
    acc = accuracy(y, test_y)
    print(f'MNIST test set accuracy: {acc}')
    print("Training complete.")
    print(DataFrame.from_records(metrics, columns=['Epoch', 'MSE', 'Val/acc']))