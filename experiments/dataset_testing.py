# %%
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNIST(Dataset):
    """Wrapper class that loads MNIST onto the GPU for speed reasons."""
    def __init__(self, train=True, download=True, device="cpu"):
        dataset = datasets.MNIST(root='./data', train=train, download=download)
        self.x = dataset.data.float().to(device).unsqueeze(1) / 255.0
        self.y = dataset.targets.to(device)
        self.mean = None
        self.var = None
        self.shape = self.x[0].shape
        self.d_input = 784
        self.name = 'mnist'
        # ok I'm being dumb. Need to test this tomorrow with a fresh mind.
    
    def normalize_dataset(self):
        """Normalize x. End result should be that each x[i] has approximately norm 1.0"""
        self.mean = self.x.mean()
        self.var = self.x.var() * torch.tensor(self.d_input)
        # Right now this produces norm sqrt(d_input)
        # batch-wide normalization does not actually have super good
        # per image normalization. Also: check test mean/var vs train. Should use train norm for test.
        self.x = (self.x - self.mean) / torch.sqrt(self.var + 1e-6)
        #self.x = self.x / torch.sqrt(torch.tensor(self.d_input))
    
    def unnormalize_dataset(self):
        if self.mean is not None and self.var is not None:
            self.x = self.x * torch.sqrt(self.var + 1e-6) + self.mean
        else:
            raise ValueError("Mean and variance have not been computed. Please call normalize() first.")
        
    def normalize(self, x):
        original_shape = x.shape
        x = x.view(-1, *self.shape)  # Reshape to self.shape, preserving batch dimension
        if self.mean is not None and self.var is not None:
            x = (x - self.mean) / torch.sqrt(self.var + 1e-6)  # Perform normalization
        else:
            print("Mean and variance have not been computed. Please call normalize() first.")
        x = x.view(original_shape)  # Return to original shape
        return x

        
    def unnormalize(self, x):
        """Transform normalized input x back to its original scale using stored mean and variance."""
        original_shape = x.shape
        x = x.view(-1, *self.shape)  # Reshape to self.shape, preserving batch dimension
        if self.mean is not None and self.var is not None:
            #x = (x - self.mean) / torch.sqrt(self.var + 1e-6)  # Perform normalization
            x = x * torch.pow(self.var, 0.5) + self.mean
        else:
            print("Mean and variance have not been computed. Please call normalize() first.")
        x = x.view(original_shape)  # Return to original shape
        return x

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)
    
# %%
# Initialize the MNIST dataset
mnist_dataset = MNIST(train=True, download=True, device="cpu")
x0 = mnist_dataset.x[0].clone()
mnist_dataset.normalize_dataset()
x1 = mnist_dataset.x[0].clone()
x2 = mnist_dataset.unnormalize(x1).clone()
print(f'Original norm: {torch.norm(x0)}') # 28
print(f'Normalized: {torch.norm(x1)}') # 28
print(f'Unnormalized: {torch.norm(x2)}') # 28
# %%
mnist_dataset.unnormalize(mnist_dataset.x[0]).shape
# %%
# Plot the first element using matplotlib
import matplotlib.pyplot as plt

first_image, first_label = mnist_dataset[0]

plt.imshow(first_image.squeeze(), cmap='gray')
plt.title(f"Label: {first_label.item()}")
plt.axis('off')
plt.show()

# %%
mnist_dataset.x.shape
# %%
mean, var = mnist_dataset.x.mean(dim=(1,2,3)), mnist_dataset.x.var(dim=(1,2,3))
# %%
mean.mean(), mean.var()
# %%
var.mean(), var.var()
# %%
Mean, Var = mnist_dataset.x.mean(), mnist_dataset.x.var()
Mean, Var
# %%
