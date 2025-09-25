#%%
from tensor.layers import TensorConvolutionTrainLayer, TensorNetworkLayer
from tensor.network import SumOfNetworks
from tensor.bregman import XEAutogradBregman
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from tensor.utils import visualize_tensornetwork
import numpy as np
import pandas as pd
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
torch.set_default_dtype(torch.float64)

# Transformations for the CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_path = "/work3/aveno/CIFAR10/data"

# Load the training dataset
train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Concatenate all training samples and labels
train_samples = []
train_labels = []
for images, labels in train_loader:
    train_samples.append(images)
    train_labels.append(labels)
xinp_train = torch.cat(train_samples, dim=0).cuda()
y_train = torch.cat(train_labels, dim=0).cuda()

KERNEL_SIZE = 4
STRIDE = 4
num_swipes =5
N = 2
r = [1, 3, 3, 3, 3]
CB = [1,3, 3, 3, 3]
eps = 5.0
bsize = 2**9
print("batch:", bsize, "rankR:", r, "rankC:", CB)
epsdecay = 1e-1
method = 'ridge_cholesky'
lr = 1

# Preprocess the training data
xinp_train = F.unfold(xinp_train, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), stride=(STRIDE,STRIDE), padding=0).transpose(-2, -1)
xinp_train = torch.cat((xinp_train, torch.zeros((xinp_train.shape[0], 1, xinp_train.shape[2]), device=xinp_train.device)), dim=-2)
xinp_train = torch.cat((xinp_train, torch.zeros((xinp_train.shape[0], xinp_train.shape[1], 1), device=xinp_train.device)), dim=-1)
xinp_train[..., -1, -1] = 1.0
y_train = F.one_hot(y_train, num_classes=10).to(dtype=torch.float64)

# Do a random split of train into train and val
seed = 42
validation_split = 0.1
n = xinp_train.shape[0]
idx = np.arange(n)
rng = np.random.RandomState(seed)
rng.shuffle(idx)
split = int(n * (1 - validation_split))
train_idx, val_idx = idx[:split], idx[split:]
xinp_train, xinp_val = xinp_train[train_idx], xinp_train[val_idx]
y_train, y_val = y_train[train_idx], y_train[val_idx]


# Load the test dataset
test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                    download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Concatenate all test samples and labels
test_samples = []
test_labels = []
for images, labels in test_loader:
    test_samples.append(images)
    test_labels.append(labels)
xinp_test = torch.cat(test_samples, dim=0).cuda()
y_test = torch.cat(test_labels, dim=0).cuda()

# Preprocess the test data
xinp_test = F.unfold(xinp_test, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), stride=(STRIDE,STRIDE), padding=0).transpose(-2, -1)
xinp_test = torch.cat((xinp_test, torch.zeros((xinp_test.shape[0], 1, xinp_test.shape[2]), device=xinp_test.device)), dim=-2)
xinp_test = torch.cat((xinp_test, torch.zeros((xinp_test.shape[0], xinp_test.shape[1], 1), device=xinp_test.device)), dim=-1)
xinp_test[..., -1, -1] = 1.0
y_test = F.one_hot(y_test, num_classes=10).to(dtype=torch.float64)

trajectory = []
epoch = 0
def convergence_criterion():
    global epoch
    epoch += 1
    y_pred_val = layer(xinp_val)
    y_pred_val = torch.cat((y_pred_val, torch.zeros_like(y_pred_val[:, :1])), dim=1)
    accuracy_test = accuracy_score(y_val.argmax(dim=-1).cpu().numpy(), y_pred_val.argmax(dim=-1).cpu().numpy())
    trajectory.append({
        'epoch': epoch,
        'val_accuracy': accuracy_test
    })
    print(f"Epoch {epoch}, val accuracy: {accuracy_test}")
    return False

# Define Bregman function
nets = []
for i in range(1, N+1):
    if i == 1:
        num_patches = xinp_train.shape[1]
        patch_pixels = xinp_train.shape[2]
    else:
        num_patches = xinp_train.shape[1] - 1
        patch_pixels = xinp_train.shape[2] - 1
    net = TensorConvolutionTrainLayer(num_carriages=i, bond_dim=r[i-1], num_patches=num_patches, patch_pixels=patch_pixels, output_shape=y_train.shape[1]-1, convolution_bond=CB[i-1]).tensor_network
    nets.append(net)
layer = TensorNetworkLayer(
    SumOfNetworks(
        nets,
        train_operators=True
    )
).cuda(-1)
print('Num params:', layer.num_parameters())
#%%
visualize_tensornetwork(layer.tensor_network)
#%%
with torch.inference_mode():
    y_pred = layer.forward(xinp_train[:64], to_tensor=True)
    w = 1/y_pred.std().item()
    #del y_pred
bf = XEAutogradBregman(w=w)
# layer.tensor_network.forward = torch.compile(layer.tensor_network.forward)
#%%
layer.tensor_network.accumulating_swipe(xinp_train, y_train, bf, eps=eps, eps_decay=epsdecay, batch_size=bsize, lr=lr, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=num_swipes)
#%%

data = []
for traj in trajectory:
      data.append((traj['epoch'], traj['val_accuracy']))
val_acc = [i[1] for i in data]
print('==========================================================================')
print('==========================================================================')
print("Best:", max(val_acc))
print("batch:", bsize, "rankR:", r, "rankC:", CB)
print('==========================================================================')
print('==========================================================================')
# df = pd.DataFrame(data, columns=['Epoch', 'Val Accuracy'])
# df.to_csv(f'tt_convolution_N{N}_r{r}_cb{CB}_swipes{num_swipes}_P{layer.num_parameters()}_fit_mnist.csv', index=False)
# %%
