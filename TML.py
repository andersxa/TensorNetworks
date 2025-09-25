#%%
import torch
from tensor.layers import TensorTrainLayer
torch.set_default_dtype(torch.float64)
#%%
N = 3
r = 4
NUM_SWIPES = 4
method = 'ridge_cholesky'
# epss = np.geomspace(0.07542717629430484, 0.00000000000722857583, 2*NUM_SWIPES).tolist()
# Define Bregman function
eps = [1,1,1,1,1,1,1]
S, F = 1000, 15
X = torch.randn(S, F)
def fbasis(X):
    Input = []
    for i in range(X.shape[-1]):
        T =torch.stack([torch.cos(X[:, i]), torch.sin(X[:,i])], dim=-1)
        Input.append(T)
        print(T.shape)
    return Input
x_train = fbasis(X)
# print(x_train)
layer = TensorTrainLayer(N, r, x_train[0].shape[1], output_shape=1, constrict_bond=True, perturb=True, seed=42)
# layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=512, lr=1.0, eps=epss, eps_r=0.5, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=NUM_SWIPES, skip_second=False, direction='l2r', disable_tqdm=True)
# convergence_criterion()
# print("Train:",(y_train.real - layer(x_train).real).square().mean().sqrt())
# print("Val:",(y_val.real - layer(x_val).real).square().mean().sqrt())
#%%
