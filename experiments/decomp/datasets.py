import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNIST(Dataset):
    """Wrapper class that loads MNIST onto the GPU for speed reasons."""
    def __init__(self, train=True, download=True, device="cuda"):
        dataset = datasets.MNIST(root='./data', train=train, download=download)
        self.x = dataset.data.float().to(device).unsqueeze(1) / 255.0
        self.y = dataset.targets.to(device)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)

class FMNIST(Dataset):
    """Wrapper class that loads F-MNIST onto the GPU for speed reasons."""
    def __init__(self, train=True, download=True, device="cuda"):
        dataset = datasets.FashionMNIST(root='./data', train=train, download=download)
        self.x = dataset.data.float().to(device).unsqueeze(1) / 255.0
        self.y = dataset.targets.to(device)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)

class CIFAR10(Dataset):
    """Wrapper class that loads CIFAR-10 onto the GPU for speed reasons."""
    def __init__(self, train=True, download=True, device="cuda"):
        # Define a transformation to normalize the images
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the images to PyTorch tensors
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize with mean and std
        ])
        
        dataset = datasets.CIFAR10(root='./data', train=train, download=download, transform=self.transform)
        self.x = dataset.data.float().to(device).unsqueeze(1) / 255.0  # This is still in NumPy format
        self.y = dataset.targets.to(device)
        
        #self.x = torch.tensor(self.x).float().to(device) / 255.0  # Convert to float tensor and scale
        #self.x = self.x.permute(0, 3, 1, 2)  # Reorder dimensions from HWC to CHW for PyTorch
        #self.y = torch.tensor(self.y).to(device)  # Convert targets to tensor on the specified device
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)
    
class _CIFAR10(Dataset):
    """Wrapper class that loads CIFAR-10 onto the GPU for speed reasons."""
    def __init__(self, train=True, download=True, device="cuda"):
        # Define a transformation to normalize the images
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the images to PyTorch tensors
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize with mean and std
        ])
        
        dataset = datasets.CIFAR10(root='./data', train=train, download=download, transform=self.transform)
        self.x = dataset.data  # This is still in NumPy format
        self.y = dataset.targets
        
        self.x = torch.tensor(self.x).float().to(device) / 255.0  # Convert to float tensor and scale
        self.x = self.x.permute(0, 3, 1, 2)  # Reorder dimensions from HWC to CHW for PyTorch
        self.y = torch.tensor(self.y).to(device)  # Convert targets to tensor on the specified device
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)