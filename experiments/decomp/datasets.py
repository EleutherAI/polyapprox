import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class BaseDataset(Dataset):
    """Base dataset class with shared utilities."""
    def __init__(self):
        super().__init__()
        self.mean = None  # Holds the cached mean
        self.std = None   # Holds the cached standard deviation

    def compute_mean(self):
        """
        Compute the mean of the dataset across all examples and store it.
        This assumes that `self.x` exists and is a tensor.
        """
        if not hasattr(self, "x"):
            raise AttributeError("Subclasses of BaseDataset must define 'self.x'.")
        
        # Compute and cache the mean only if it's not already computed

        return self.x.mean().item()

    def compute_std(self):
        """
        Compute the standard deviation of the dataset across all examples and store it.
        This assumes that `self.x` exists and is a tensor.
        """
        if not hasattr(self, "x"):
            raise AttributeError("Subclasses of BaseDataset must define 'self.x'.")
        
        # Compute and cache the std only if it's not already computed
        #if self.std is None:
        self.std = self.x.std().item()
        return self.std

    def recenter(self, mean=None, scale=None):
        """
        Recenter the dataset by subtracting the mean and scaling by the standard deviation.
        
        Args:
            mean (float, optional): The mean to subtract from the dataset.
                                    If None, it will use the stored or computed mean.
            scale (float, optional): The scale (e.g., std) to divide by for normalization.
                                     If None, it will use the stored or computed standard deviation.
        """
        if not hasattr(self, "x"):
            raise AttributeError("Subclasses of BaseDataset must define 'self.x'.")
        
        # Use the provided mean/scale, or default to the computed values
        mean = mean if mean is not None else self.compute_mean()
        scale = scale if scale is not None else self.compute_std()
        #print(mean, scale, 'inside recenter method of datasets.py')
        # Subtract the mean and normalize by the scale
        #self.x = (self.x - mean) / scale
        return mean, scale

class MNIST(BaseDataset):
    """Wrapper class that loads MNIST onto the GPU for speed reasons."""
    def __init__(self, train=True, download=True, device="cpu"):
        dataset = datasets.MNIST(root='./data', train=train, download=download)
        self.x = dataset.data.float().to(device).unsqueeze(1) / 255.0

        self.recenter()
        self.y = dataset.targets.to(device)
        self.name = 'mnist'
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)

class FMNIST(BaseDataset):
    """Wrapper class that loads F-MNIST onto the GPU for speed reasons."""
    def __init__(self, train=True, download=True, device="cpu"):
        dataset = datasets.FashionMNIST(root='./data', train=train, download=download)
        self.x = dataset.data.float().to(device).unsqueeze(1) / 255.0
        self.y = dataset.targets.to(device)
        self.name = 'fmnist'
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)
    
class CIFAR10(BaseDataset):
    """Wrapper class that loads CIFAR-10 onto the GPU for speed reasons."""
    def __init__(self, train=True, download=True, device="cpu"):
        # Define a transformation to normalize the images
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the images to PyTorch tensors
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize with mean and std
        ])
        
        dataset = datasets.CIFAR10(root='./data', train=train, download=download, transform=self.transform)
        self.x = dataset.data  # This is still in NumPy format
        self.y = torch.tensor(dataset.targets).to(device) # this is just a list
        
        self.x = torch.tensor(self.x).float().to(device) / 255.0  # Convert to float tensor and scale
        self.x = self.x.permute(0, 3, 1, 2)  # Reorder dimensions from HWC to CHW for PyTorch
        #self.y = torch.tensor(self.y).to(device)  # Convert targets to tensor on the specified device
        self.name = 'cifar'
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)

    def remove_label(self, label):
        """Remove all examples with the specified label."""
        mask = self.y != label
        self.x = self.x[mask]
        self.y = self.y[mask]