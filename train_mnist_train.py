import os
import pandas as pd
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score

# Assuming these are custom local modules
from tensor.layers import TensorConvolutionTrainLayer, TensorNetworkLayer
from tensor.network import SumOfNetworks
from tensor.bregman import XEAutogradBregman
from models.tensor_train import EarlyStopping

# This should be set before any torch imports that might initialize CUDA
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_default_dtype(torch.float64)


def error_rate_torch(y_true, y_pred):
    """Calculates the error rate (1 - accuracy) from torch tensors."""
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true_labels = y_true.cpu().numpy()
    if y_true_labels.ndim > 1 and y_true_labels.shape[1] > 1:
        y_true_labels = np.argmax(y_true_labels, axis=1)
    return 1.0 - accuracy_score(y_true_labels, y_pred_labels)

def train_model(args):
    """
    Trains and evaluates the Tensor Network model based on provided arguments.

    Args:
        args: An argparse.Namespace object containing all hyperparameters.

    Returns:
        A dictionary containing the results of the run.
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # --- 1. Data Loading and Preprocessing ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if args.dataset == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root=args.data_path, train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root=args.data_path, train=False, transform=transform, download=True)
    elif args.dataset == 'FashionMNIST':
        train_dataset = torchvision.datasets.FashionMNIST(root=args.data_path, train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root=args.data_path, train=False, transform=transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Load all data into memory
    xinp_train = torch.cat([images for images, _ in train_loader], dim=0)
    y_train = torch.cat([labels for _, labels in train_loader], dim=0)
    xinp_test = torch.cat([images for images, _ in test_loader], dim=0)
    y_test = torch.cat([labels for _, labels in test_loader], dim=0)

    # Preprocess and move to GPU
    def preprocess_data(x, y):
        x = F.unfold(x, kernel_size=(args.kernel_size, args.kernel_size), stride=(args.kernel_stride, args.kernel_stride), padding=0).transpose(-2, -1)
        x = torch.cat((x, torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device)), dim=-2)
        x = torch.cat((x, torch.zeros((x.shape[0], x.shape[1], 1), device=x.device)), dim=-1)
        x[..., -1, -1] = 1.0
        y = F.one_hot(y, num_classes=10).to(dtype=torch.float64)
        return x.cuda(), y.cuda()

    xinp_train, y_train = preprocess_data(xinp_train, y_train)
    xinp_test, y_test = preprocess_data(xinp_test, y_test)

    # Create validation split
    n = xinp_train.shape[0]
    idx = np.arange(n)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(idx)
    split = int(n * (1 - args.validation_split))
    train_idx, val_idx = idx[:split], idx[split:]
    xinp_train, xinp_val = xinp_train[train_idx], xinp_train[val_idx]
    y_train, y_val = y_train[train_idx], y_train[val_idx]

    # --- 2. Model Definition ---
    if 'type1' in args.model_type:
        nets = []
        for i in range(1, args.N + 1):
            if i == 1:
                num_patches = xinp_train.shape[1]
                patch_pixels = xinp_train.shape[2]
            else:
                num_patches = xinp_train.shape[1] - 1
                patch_pixels = xinp_train.shape[2] - 1
            net = TensorConvolutionTrainLayer(num_carriages=i, bond_dim=args.r, num_patches=num_patches, patch_pixels=patch_pixels, output_shape=y_train.shape[1] - 1, convolution_bond=args.CB).tensor_network
            nets.append(net)
        layer = TensorNetworkLayer(SumOfNetworks(nets, train_operators=True)).cuda()
    else:
        layer = TensorConvolutionTrainLayer(num_carriages=args.N, bond_dim=args.r, num_patches=xinp_train.shape[1], patch_pixels=xinp_train.shape[2], output_shape=y_train.shape[1] - 1, convolution_bond=args.CB).cuda()
    
    if args.verbose > 0:
        print('Num params:', layer.num_parameters())

    # --- 3. Training Setup ---
    def model_predict(x):
        y_pred = layer(x)
        y_pred = torch.cat((y_pred, torch.zeros_like(y_pred[:, :1])), dim=1)
        return y_pred

    early_stopper = EarlyStopping(
        xinp_val, y_val,
        model_predict=model_predict,
        get_model_weights=layer.node_states,
        loss_fn=error_rate_torch,
        abs_err=args.abs_err,
        rel_err=args.rel_err,
        early_stopping=args.early_stopping,
        verbose=args.verbose
    )

    with torch.inference_mode():
        y_pred_sample = layer.forward(xinp_train[:64], to_tensor=True)
        w = 1 / y_pred_sample.std().item()
    
    bf = XEAutogradBregman(w=w)
    
    # --- 4. Training ---
    layer.tensor_network.accumulating_swipe(
        xinp_train, y_train, bf, batch_size=2048, lr=1.0, 
        convergence_criterion=early_stopper.convergence_criterion, 
        orthonormalize=False, method='ridge_cholesky', 
        eps=args.eps, eps_decay=args.eps_decay, 
        verbose=args.verbose, num_swipes=args.num_swipes, 
        disable_tqdm=args.verbose < 3
    )

    # --- 5. Evaluation ---
    layer.load_node_states(early_stopper.best_state_dict, set_value=True)

    with torch.inference_mode():
        y_pred_test = model_predict(xinp_test)
        accuracy_test = accuracy_score(y_test.argmax(dim=-1).cpu().numpy(), y_pred_test.argmax(dim=-1).cpu().numpy())
        if args.verbose > 0:
            print(f"Test Accuracy: {accuracy_test * 100:.2f}%")

    # --- 6. Return Results ---
    return {
        'dataset': args.dataset,
        'model_type': args.model_type,
        'N': args.N,
        'r': args.r,
        'CB': args.CB,
        'test_accuracy': accuracy_test * 100,
        'num_parameters': layer.num_parameters()
        'num_swipes': args.num_swipes,
        'eps': args.eps,
        'eps_decay': args.eps_decay,
        'kernel_size': args.kernel_size,
        'kernel_stride': args.kernel_stride,
        'early_stopping': args.early_stopping,
        'abs_err': args.abs_err,
        'rel_err': args.rel_err,
        'seed': args.seed,
        'validation_split': args.validation_split
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Tensor Network model on image classification.")
    
    # Model and Data Hyperparameters
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'FashionMNIST'], help='Dataset to use.')
    parser.add_argument('--data_path', type=str, default='/work3/aveno/MNIST/data')
    parser.add_argument('--model_type', type=str, default='tt_type1', help='Type of tensor network model.')
    parser.add_argument('--N', type=int, default=4, help='Number of carriages or sum of networks.')
    parser.add_argument('--r', type=int, default=10, help='Bond dimension (rank) of the tensor train.')
    parser.add_argument('--CB', type=int, default=4, help='Convolution bond dimension.')
    parser.add_argument('--kernel_size', type=int, default=4, help='Size of the unfolding kernel.')
    parser.add_argument('--kernel_stride', type=int, default=4, help='Stride of the unfolding kernel.')

    # Training Hyperparameters
    parser.add_argument('--num_swipes', type=int, default=5, help='Maximum number of training swipes.')
    parser.add_argument('--eps', type=float, default=5.0, help='Initial regularization parameter for ridge regression.')
    parser.add_argument('--eps_decay', type=float, default=0.25, help='Decay factor for the regularization parameter.')
    
    # Early Stopping Hyperparameters
    parser.add_argument('--early_stopping', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--abs_err', type=float, default=1e-4, help='Absolute error tolerance for early stopping.')
    parser.add_argument('--rel_err', type=float, default=1e-3, help='Relative error tolerance for early stopping.')

    # General Configuration
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Fraction of training data to use for validation.')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level (0=silent, 1=progress, 2=detailed, 3=debug).')
    
    args = parser.parse_args()

    if args.verbose > 0:
        print("Starting training with the following configuration:")
        print(args)

    results_dict = train_model(args)
    
    # Format the accuracy to two decimal places for the final print
    results_dict['test_accuracy'] = f"{results_dict['test_accuracy']:.2f}"
    
    # Print the final results string as requested
    output_string = ",".join(map(str, results_dict.values()))
    print(output_string)
