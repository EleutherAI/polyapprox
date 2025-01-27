# %%
import numpy as np
import matplotlib.pyplot as plt
import torch

ell_B_alpha_load_path = f'/Users/alicerigg/Code/polyapprox/experiments/data/ell_B_alpha.pt'
LBA = torch.load(ell_B_alpha_load_path)
print(f'ell_B_alpha loaded from {ell_B_alpha_load_path}')

LBA.shape

# %%
# Example data: 3x20
x = np.arange(20)  # X-axis
y1 = np.random.rand(20) * 10
y2 = np.random.rand(20) * 10
y3 = np.random.rand(20) * 10

# Combine all 3 into an array
y = np.vstack([y1, y2, y3])

# Stackplot to visualize the decomposition
plt.figure(figsize=(10, 6))
plt.stackplot(x, y1, y2, y3, labels=["Component 1", "Component 2", "Component 3"], alpha=0.8)

plt.title("Stacked Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Values")
plt.legend(loc="upper left")
plt.show()
# %%
# Combine all 3 into an array
# Convert numpy arrays to torch tensors
y1_tensor = torch.tensor(y1)
y2_tensor = torch.tensor(y2)
y3_tensor = torch.tensor(y3)

# Stack the tensors vertically
y_tensor = torch.vstack([y1_tensor, y2_tensor, y3_tensor])

# Convert tensors back to numpy for plotting
y1_np, y2_np, y3_np = y_tensor.numpy()

# Stackplot to visualize the decomposition
plt.figure(figsize=(10, 6))
plt.stackplot(x, y1_np, y2_np, y3_np, labels=["Component 1", "Component 2", "Component 3"], alpha=0.8)

plt.title("Stacked Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Values")
plt.legend(loc="upper left")
plt.show()
# %%
# Extract data from LBA for plotting
ell_values = LBA[:, 0, 3].numpy()  # Eigenvalues for the 3rd digit
beta_norms = LBA[:, 1, 3].numpy()  # Norms of beta for the 3rd digit
alpha_values = LBA[:, 2, 3].numpy()  # Alpha values for the 3rd digit
# %%
d_ratio = LBA[:, 1] / LBA[:, 0]

print(d_ratio[:,:5])

# %%
# Create a heatmap of d_ratio
def heatmap(data, title='',
            xlabel='', ylabel='checkpoints (ex)',
            colorbarlabel='function', vmin=None, vmax=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(data, aspect='auto', cmap='viridis',
               interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(label=colorbarlabel)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    #plt.xticks(ticks=data.size(1))
    plt.show()

# %% 

# X-axis for plotting
x = np.arange(len(ell_values))

# Stackplot to visualize the decomposition
plt.figure(figsize=(10, 6))
plt.stackplot(x, ell_values, beta_norms, alpha_values, labels=["Eigenvalues", "Beta Norms", "Alpha Values"], alpha=0.8)

plt.title("Stacked Plot of LBA Components for Digit 3")
plt.xlabel("Checkpoints")
plt.ylabel("Values")
plt.legend(loc="upper left")
plt.show()

# %%
num_steps = 10000
x = torch.linspace(0,10,num_steps)[:,None,None,None] # shape [20]

# shape (50,19,3,10)
g = LBA[None, :, 0]
b = LBA[None, :, 1]
a = LBA[None, :, 2]

#print(g.shape, b.shape, a.shape)
y = torch.zeros(num_steps, 19, 1, 10)

numerator = b*x#  [1,19,1,10]

x, y, z = torch.abs(a), torch.abs(b*x), torch.abs(g*(x**2))
denominator = x+y+z

r0, r1, r2 = x / denominator, y / denominator, z / denominator
r0, r1, r2 = r0.squeeze(), r1.squeeze(), r2.squeeze()

print(r0.shape, r1.shape, r2.shape)

# %% no_alpha

denominator_a = y+z

ra1 = y / denominator_a
ra2 = z / denominator_a

ra1, ra2 = ra1.squeeze(), ra2.squeeze()
#ra1 = 1-ra1
#ra2 = 1-ra2
heatmap(ra1[:,:,3], xlabel='checkpoints', ylabel='preactivation magnitude')
heatmap(ra2[:,:,3], xlabel='checkpoints', ylabel='preactivation magnitude', vmin=0)
#print(r0+r1+r2)

# %%
ra1 + ra2
heatmap(ra1[:,:,3]+ra2[:,:,3], xlabel='checkpoints', ylabel='preactivation magnitude', vmin=0, vmax=1)
# %%
negative_count_r0 = torch.sum(r0 < 0).item()
negative_count_r1 = torch.sum(r1 < 0).item()
negative_count_r2 = torch.sum(r2 < 0).item()

mass = 10000 * 19 * 10
print("Number of instances where r0 is negative:", negative_count_r0/mass)
print("Number of instances where r1 is negative:", negative_count_r1/mass)
print("Number of instances where r2 is negative:", negative_count_r2/mass)

# %%



# %%
# Find indices where both r1 and r2 are negative simultaneously
negative_simultaneous_count = torch.sum((r1 < 0) & (r2 < 0)).item()

print("Number of instances where r1 and r2 are negative simultaneously:", negative_simultaneous_count)
# %%
r0.shape
# %%
heatmap(r0[:,:,3], xlabel='checkpoints', ylabel='preactivation magnitude')
heatmap(r1[:,:,3], xlabel='checkpoints', ylabel='preactivation magnitude')
heatmap(r2[:,:,3], xlabel='checkpoints', ylabel='preactivation magnitude')
# %%
heatmap(y[:,:,3], xlabel='checkpoints', ylabel='x value', vmin=0, vmax=2)
# %%
# Example usage with broadcasting
c = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(0)  # Shape (1, 3)
z = torch.tensor([0.5, 1.0]).unsqueeze(1)       # Shape (2, 1)
B = 1.0  # Scalar
ell = 0.5  # Scalar
alpha = 0.1  # Scalar

result = f(c, z, B, ell, alpha)
print(result)