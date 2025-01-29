# %%
import torch
from torchvision import datasets, transforms
from decomp.datasets import MNIST as oldMNIST
from decomp.model import FFNModel, _Config
from functional_plotting_utils import compute_dataset_statistics
def load_mnist(train=True, download=True, data_dir='./data'):
    """Load the MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_dataset = datasets.MNIST(root=data_dir, train=train, download=download, transform=transform)
    return mnist_dataset

def whiten_dataset(dataset, mean, whitening_matrix):
    """
    Apply whitening to the dataset using the precomputed mean and whitening matrix.
    Args:
        dataset (torch.utils.data.Dataset): Dataset to whiten.
        mean (torch.Tensor): Mean vector (784,) to subtract.
        whitening_matrix (torch.Tensor): Whitening matrix (784, 784).
    Returns:
        torch.Tensor: Whitened dataset, shape (N, 784).
    """
    original_shape = dataset.shape
    # Load the dataset as a single tensor
    #all_data = torch.stack([transforms.ToTensor()(image).view(-1) for image, _ in dataset])  # Shape: (N, 784)
    all_data = dataset.clone().view(-1, 784)
    # Subtract mean and apply whitening transformation
    centered = all_data - mean
    whitened = centered @ whitening_matrix.T  # Apply whitening

    return whitened.reshape(original_shape)
# Example usage:
#train = load_mnist(train=True)
#test = load_mnist(train=False)
train = oldMNIST(train=True)
test = oldMNIST(train=False)
mean, cov, cholesky = compute_dataset_statistics(train.x)
train_data = whiten_dataset(train.x, mean, cholesky)
test_data = whiten_dataset(test.x, mean, cholesky)
train.x = train_data
test.x = test_data
print(mean.norm())
# 
config = _Config(out_bias=True, epochs=16)
#config['epochs'] = 16
_model = FFNModel(config)
#train = train.test_data.reshape(-1,784) / 255.0
#test = test.test_data.reshape(-1,784) / 255.0
#train = train.transform(train.numpy())
_model.fit(train, test)
# %%
from mnist_2l import Minimal_FFN
from functional_plotting_utils import *
model = Minimal_FFN(_model.get_layer_data())

linear_01 = model.approx_fit()
linear_ms = model.approx_fit(mean=mean, cov=cov)

scores = []
# NOTE: Default `evaluate` on dataset type objects fetches directly from .x
# while i did set it above to be whitened, it could be that it is not whitened.
# supports passing whitened data directly
scores.append(evaluate_model(model, train, return_logits=False))
scores.append(evaluate_model(linear_01, train, return_logits=False))
scores.append(evaluate_model(linear_ms, train, return_logits=False))
scores.append(evaluate_model(model, test, return_logits=False))
scores.append(evaluate_model(linear_01, test, return_logits=False))
scores.append(evaluate_model(linear_ms, test, return_logits=False))
# 
new_train = oldMNIST(train=True)
new_test = oldMNIST(train=False)

scores.append(evaluate_model(model, new_train, return_logits=False))
scores.append(evaluate_model(linear_01, new_train, return_logits=False))
scores.append(evaluate_model(linear_ms, new_train, return_logits=False))
scores.append(evaluate_model(model, new_test, return_logits=False))
scores.append(evaluate_model(linear_01, new_test, return_logits=False))
scores.append(evaluate_model(linear_ms, new_test, return_logits=False))
scores
# %%
test_data.
# %%
_mean, _cov, _cholesky = compute_dataset_statistics(train_data, clamp_min=1e-5)
_mean.norm(), _cov.norm()
# %%
mean.norm()
# %%
zero_count = torch.sum(_cov != 0).item()
# %%
zero_count
# %%
nonzero_indices = torch.nonzero(_cov, as_tuple=True)
first_nonzero_index = nonzero_indices[0][0].item() if nonzero_indices[0].numel() > 0 else None
first_nonzero_index

# %%
zero_rows = torch.sum(_cov == 0, dim=0)
zero_all = torch.sum(zero_rows == 16)
zero_all
# %%
eigvals, eigvecs = torch.linalg.eigh(_cov)

plt.figure(figsize=(10, 6))
plt.plot(eigvals, marker='o', linestyle='-')
plt.title('Eigenvalues of the Covariance Matrix')
plt.xlabel('Index')
plt.yscale('log')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.show()

# %%
(_cholesky - torch.diag(torch.diag(_cholesky))).norm()
# %%
