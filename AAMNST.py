import os
import pandas as pd
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from datetime import datetime

# Assuming these are custom local modules
from tensor.layers import TensorConvolutionTrainLayer, TensorNetworkLayer
from tensor.network import SumOfNetworks
from tensor.bregman import XEAutogradBregman
from models.tensor_train import EarlyStopping

# This should be set before any torch imports that might initialize CUDA
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_default_dtype(torch.float64)

Batches = 32
def error_rate_torch(y_true, y_pred):
    """Calculates the error rate (1 - accuracy) from torch tensors."""
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true_labels = y_true.cpu().numpy()
    if y_true_labels.ndim > 1 and y_true_labels.shape[1] > 1:
        y_true_labels = np.argmax(y_true_labels, axis=1)
    return 1.0 - accuracy_score(y_true_labels, y_pred_labels)


def get_config_from_env():
    """Get configuration from environment variables with fallback defaults."""
    config = {}
    
    # Model and Data Hyperparameters
    config['dataset'] = os.environ.get('DATASET', 'MNIST') # MNIST of FashionMNIST
    config['data_path'] = '/work3/aveno/'+config['dataset']+'/data'
    config['model_type'] = os.environ.get('MODEL_TYPE', 'tt_type1')
    config['N'] = int(os.environ.get('N', '4'))
    config['r'] = int(os.environ.get('R', '10'))
    config['CB'] = int(os.environ.get('CB', '4'))
    config['seed'] = int(os.environ.get('SEED', '42'))
    config['kernel_size'] = int(os.environ.get('KERNEL_SIZE', '4'))
    config['kernel_stride'] = int(os.environ.get('KERNEL_STRIDE', '4'))
    
    # Training Hyperparameters
    config['num_swipes'] = int(os.environ.get('NUM_SWIPES', '5'))
    config['eps'] = float(os.environ.get('EPS', '5.0'))
    config['eps_decay'] = float(os.environ.get('EPS_DECAY', '0.25'))
    
    # Early Stopping Hyperparameters
    config['early_stopping'] = int(os.environ.get('EARLY_STOPPING', '10'))
    config['abs_err'] = float(os.environ.get('ABS_ERR', '1e-4'))
    config['rel_err'] = float(os.environ.get('REL_ERR', '1e-3'))
    
    # General Configuration
    config['validation_split'] = float(os.environ.get('VALIDATION_SPLIT', '0.1'))
    config['verbose'] = int(os.environ.get('VERBOSE', '1'))
    
    # Results file path
    config['results_path'] = os.environ.get('RESULTS_PATH', '/zhome/6b/e/212868/Desktop/code/TensorNetworksFork')
    config['results_filename'] = os.environ.get('RESULTS_FILENAME', 'nicco_results.csv')
    config['results_file'] = config['results_path']+'/'+config['results_filename']
    
    return config


def save_results_to_dataset(results_dict, results_file):
    """Save results to a CSV dataset file, appending to existing data."""
    # Add timestamp
    results_dict['timestamp'] = datetime.now().isoformat()
    
    # Convert to DataFrame
    df_new = pd.DataFrame([results_dict])
    
    # Check if file exists and append/create accordingly
    if os.path.exists(results_file):
        try:
            # Read existing data and append
            df_existing = pd.read_csv(results_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(results_file, index=False)
            print(f"Results appended to existing file: {results_file}")
        except Exception as e:
            print(f"Error reading existing file, creating new one: {e}")
            df_new.to_csv(results_file, index=False)
            print(f"Results saved to new file: {results_file}")
    else:
        # Create new file
        df_new.to_csv(results_file, index=False)
        print(f"Results saved to new file: {results_file}")


def train_model(config):
    """
    Trains and evaluates the Tensor Network model based on provided configuration.

    Args:
        config: A dictionary containing all hyperparameters from environment variables.

    Returns:
        A dictionary containing the results of the run.
    """
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # --- 1. Data Loading and Preprocessing ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if config['dataset'] == 'MNIST':
        print(config['data_path'])
        train_dataset = torchvision.datasets.MNIST(root=config['data_path'], train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root=config['data_path'], train=False, transform=transform, download=True)
    elif config['dataset'] == 'FashionMNIST':
        train_dataset = torchvision.datasets.FashionMNIST(root=config['data_path'], train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root=config['data_path'], train=False, transform=transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batches, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Batches, shuffle=False)

    # Load all data into memory
    xinp_train = torch.cat([images for images, _ in train_loader], dim=0)
    y_train = torch.cat([labels for _, labels in train_loader], dim=0)
    xinp_test = torch.cat([images for images, _ in test_loader], dim=0)
    y_test = torch.cat([labels for _, labels in test_loader], dim=0)

    # Preprocess and move to GPU
    def preprocess_data(x, y):
        x = F.unfold(x, kernel_size=(config['kernel_size'], config['kernel_size']), stride=(config['kernel_stride'], config['kernel_stride']), padding=0).transpose(-2, -1)
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
    rng = np.random.RandomState(config['seed'])
    rng.shuffle(idx)
    split = int(n * (1 - config['validation_split']))
    train_idx, val_idx = idx[:split], idx[split:]
    xinp_train, xinp_val = xinp_train[train_idx], xinp_train[val_idx]
    y_train, y_val = y_train[train_idx], y_train[val_idx]

    # --- 2. Model Definition ---
    if 'type1' in config['model_type']:
        nets = []
        for i in range(1, config['N'] + 1):
            if i == 1:
                num_patches = xinp_train.shape[1]
                patch_pixels = xinp_train.shape[2]
            else:
                num_patches = xinp_train.shape[1] - 1
                patch_pixels = xinp_train.shape[2] - 1
            net = TensorConvolutionTrainLayer(num_carriages=i, bond_dim=config['r'], num_patches=num_patches, patch_pixels=patch_pixels, output_shape=y_train.shape[1] - 1, convolution_bond=config['CB']).tensor_network
            nets.append(net)
        layer = TensorNetworkLayer(SumOfNetworks(nets, train_operators=True)).cuda()
    else:
        layer = TensorConvolutionTrainLayer(num_carriages=config['N'], bond_dim=config['r'], num_patches=xinp_train.shape[1], patch_pixels=xinp_train.shape[2], output_shape=y_train.shape[1] - 1, convolution_bond=config['CB']).cuda()
    
    if config['verbose'] > 0:
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
        abs_err=config['abs_err'],
        rel_err=config['rel_err'],
        early_stopping=config['early_stopping'],
        verbose=config['verbose']
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
        eps=config['eps'], eps_decay=config['eps_decay'], 
        verbose=config['verbose'], num_swipes=config['num_swipes'], 
        disable_tqdm=config['verbose'] < 3
    )

    # --- 5. Evaluation ---
    layer.load_node_states(early_stopper.best_state_dict, set_value=True)

    with torch.inference_mode():
        y_pred_test = model_predict(xinp_test)
        accuracy_test = accuracy_score(y_test.argmax(dim=-1).cpu().numpy(), y_pred_test.argmax(dim=-1).cpu().numpy())
        if config['verbose'] > 0:
            print(f"Test Accuracy: {accuracy_test * 100:.2f}%")

    # --- 6. Return Results ---
    return {
        'dataset': config['dataset'],
        'model_type': config['model_type'],
        'N': config['N'],
        'r': config['r'],
        'CB': config['CB'],
        'test_accuracy': round(accuracy_test * 100, 2),
        'num_parameters': layer.num_parameters(),
        'num_swipes': config['num_swipes'],
        'eps': config['eps'],
        'eps_decay': config['eps_decay'],
        'kernel_size': config['kernel_size'],
        'kernel_stride': config['kernel_stride'],
        'early_stopping': config['early_stopping'],
        'abs_err': config['abs_err'],
        'rel_err': config['rel_err'],
        'seed': config['seed'],
        'validation_split': config['validation_split']
    }


if __name__ == "__main__":
    # Get configuration from environment variables
    config = get_config_from_env()
    
    if config['verbose'] > 0:
        print("Starting training with the following configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")

    # Train the model
    results_dict = train_model(config)
    
    # Save results to dataset file
    save_results_to_dataset(results_dict, config['results_file'])
    
    if config['verbose'] > 0:
        print("Training completed successfully!")
