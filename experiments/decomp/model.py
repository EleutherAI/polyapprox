import os
import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from transformers import PretrainedConfig, PreTrainedModel
from jaxtyping import Float
from tqdm import tqdm
from pandas import DataFrame
from einops import *
from .components import Linear, Bilinear

def _collator(transform=None):
    def inner(batch):
        x = torch.stack([item[0] for item in batch]).float()
        y = torch.stack([item[1] for item in batch])
        return (x, y) if transform is None else (transform(x), y)
    return inner

class Config(PretrainedConfig):
    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.5,
        epochs: int = 100,
        batch_size: int = 2048,
        d_hidden: int = 256,
        n_layer: int = 1,
        d_input: int = 784,
        d_output: int = 10,
        bias: bool = False,
        residual: bool = False,
        seed: int = 42,
        **kwargs
    ):
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
    
        self.d_hidden = d_hidden
        self.n_layer = n_layer
        self.d_input = d_input
        self.d_output = d_output
        self.bias = bias
        self.residual = residual
        
        
        
        super().__init__(**kwargs)

class _Config(PretrainedConfig):
    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.5,
        epochs: int = 100,
        batch_size: int = 2048,
        d_hidden: int = 256,
        n_layer: int = 1,
        d_input: int = 784,
        d_output: int = 10,
        bias: bool = False,
        in_bias: bool = False,
        out_bias: bool = False,
        residual: bool = False,
        seed: int = 42,
        **kwargs
    ):
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        self.d_hidden = d_hidden
        self.n_layer = n_layer
        self.d_input = d_input
        self.d_output = d_output
        self.bias = bias
        self.in_bias = in_bias
        self.out_bias = out_bias
        self.residual = residual



        super().__init__(**kwargs)

class QuadraticModel(PreTrainedModel):
    pass

class FFNModel(PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        torch.manual_seed(config.seed)

        d_input, d_hidden, d_output = config.d_input, config.d_hidden, config.d_output
        bias = config.bias
        in_bias = config.in_bias
        out_bias = config.out_bias
        n_layer = config.n_layer
        self.n_layer = n_layer
        if bias:
            in_bias = True
            out_bias = True

        self.activation = nn.ReLU()
        #self.embed = Linear(d_input, d_hidden, bias=in_bias)
        self.blocks = nn.ModuleList([Linear(d_input, d_hidden, bias=in_bias)] +
             [Linear(d_hidden, d_hidden, bias=in_bias) for _ in range(2*n_layer-2)] +
             [Linear(d_hidden, d_output, bias=out_bias)])
        #self.head = Linear(d_hidden, d_output, bias=out_bias)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Float[Tensor, "... inputs"]) -> Float[Tensor, "... outputs"]:
        x = x.flatten(start_dim=1)
        #x = self.activation(self.embed(x))
        for idx in range(len(self.blocks)):
            x = self.blocks[idx](x)
            if idx % 2 == 0: #preact. 1 ~> postact
                x = self.activation(x)
        return x

    def get_layer_data(self, layer=0):
        data = {}
        data['W1'] = self.blocks[2*layer].weight.data.detach().cpu()
        data['b1'] = self.blocks[2*layer].bias.data.detach().cpu()
        data['W2'] = self.blocks[2*layer+1].weight.data.detach().cpu()
        data['b2'] = self.blocks[2*layer+1].bias.data.detach().cpu()
        return data

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

class Model(PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        torch.manual_seed(config.seed)
        
        d_input, d_hidden, d_output = config.d_input, config.d_hidden, config.d_output
        bias, n_layer = config.bias, config.n_layer
        
        self.embed = Linear(d_input, d_hidden, bias=False)
        self.blocks = nn.ModuleList([Bilinear(d_hidden, d_hidden, bias=bias) for _ in range(n_layer)])
        self.head = Linear(d_hidden, d_output, bias=False)
        
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
    
    def forward(self, x: Float[Tensor, "... inputs"]) -> Float[Tensor, "... outputs"]:
        x = self.embed(x.flatten(start_dim=1))
        
        for layer in self.blocks:
            x = x + layer(x) if self.config.residual else layer(x)
        
        return self.head(x)
    
    @property
    def w_e(self):
        return self.embed.weight.data
    
    @property
    def w_u(self):
        return self.head.weight.data
    
    @property
    def w_lr(self):
        return torch.stack([rearrange(layer.weight.data, "(s o) h -> s o h", s=2) for layer in self.blocks])
    
    @property
    def w_l(self):
        return self.w_lr.unbind(1)[0]
    
    @property
    def w_r(self):
        return self.w_lr.unbind(1)[1]
    
    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(Config(*args, **kwargs))

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        new = cls(Config(*args, **kwargs))
        new.load_state_dict(torch.load(path))
        return new
    
    def step(self, x, y):
        y_hat = self(x)
        
        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        
        return loss, accuracy
    
    def fit(self, train, test, transform=None):
        torch.manual_seed(self.config.seed)
        torch.set_grad_enabled(True)
        
        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        
        loader = DataLoader(train, batch_size=self.config.batch_size, shuffle=True, drop_last=True, collate_fn=_collator(transform))
        test_x, test_y = test.x, test.y
        
        pbar = tqdm(range(self.config.epochs))
        history = []
        
        for _ in pbar:
            epoch = []
            for x, y in loader:
                loss, acc = self.train().step(x, y)
                epoch += [(loss.item(), acc.item())]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            val_loss, val_acc = self.eval().step(test_x, test_y)

            metrics = {
                "train/loss": sum(loss for loss, _ in epoch) / len(epoch),
                "train/acc": sum(acc for _, acc in epoch) / len(epoch),
                "val/loss": val_loss.item(),
                "val/acc": val_acc.item()
            }
            
            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3f}" for k, v in metrics.items()))
        
        torch.set_grad_enabled(False)
        return DataFrame.from_records(history, columns=['train/loss', 'train/acc', 'val/loss', 'val/acc'])

    def decompose(self):
        """The function to decompose a single-layer model into eigenvalues and eigenvectors."""
        
        # Split the bilinear layer into the left and right components
        l, r = self.w_lr[0].unbind()
        
        # Compute the third-order (bilinear) tensor
        b = einsum(self.w_u, l, r, "cls out, out in1, out in2 -> cls in1 in2")
        
        # Symmetrize the tensor
        b = 0.5 * (b + b.mT)

        # Perform the eigendecomposition
        vals, vecs = torch.linalg.eigh(b)
        
        # Project the eigenvectors back to the input space
        vecs = einsum(vecs, self.w_e, "cls emb comp, emb inp -> cls comp inp")
        
        # Return the eigenvalues and eigenvectors
        return vals, vecs