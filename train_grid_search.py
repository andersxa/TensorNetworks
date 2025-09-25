import os
import argparse
import torch
import numpy as np
torch.set_default_dtype(torch.float64)
from load_ucirepo import get_ucidata
from models.tensor_train import TensorTrainRegressor
from tensor.bregman import AutogradLoss, XEAutogradBregman
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score

seeds = [2582264110, 1011906602, 1729841207, 3653141251, 826749499, 1399498030, 358916519, 328531638, 787755760, 4192420569]

test_seeds = [836578142, 895435625, 2631647123, 2487125586, 3323088614, 3313309148, 3300558450, 4053540165, 2318036890, 4234260150]

def evaluate_model(model, X, y_true, metric='accuracy'):
    y_pred = model.predict(X)
    if metric == 'accuracy':
        if y_true.ndim == 2:
            y_true = y_true.argmax(-1)
        y_true = y_true.cpu().numpy()
        acc = accuracy_score(y_true, y_pred.argmax(-1))
        return acc
    elif metric == 'rmse':
        if y_true.ndim == 2:
            y_true = y_true.squeeze(-1)
        y_true = y_true.cpu().numpy()
        rmse = root_mean_squared_error(y_true, y_pred)
        return rmse
    elif metric == 'r2':
        if y_true.ndim == 2:
            y_true = y_true.squeeze(-1)
        y_true = y_true.cpu().numpy()
        r2 = r2_score(y_true, y_pred)
        return r2
    else:
        raise ValueError(f"Unknown metric: {metric}")

def train_model(args, data=None, test=False):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    if data is None:
        data = get_ucidata(args.dataset_id, args.task, args.data_device)
    X_train, y_train, X_val, y_val, X_test, y_test = data

    # For each y, if it is not 2D, add a dimension
    if y_train.ndim == 1 and args.task == 'regression':
        y_train = y_train.unsqueeze(-1)
        y_val = y_val.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)
    elif (y_train.ndim == 1 or y_train.shape[1] == 1) and args.task == 'classification':
        num_classes = len(torch.unique(y_train.to(dtype=torch.long)))
        y_train = torch.nn.functional.one_hot(y_train.to(dtype=torch.long), num_classes=num_classes).squeeze(1)
        y_val = torch.nn.functional.one_hot(y_val.to(dtype=torch.long), num_classes=num_classes).squeeze(1)
        y_test = torch.nn.functional.one_hot(y_test.to(dtype=torch.long), num_classes=num_classes).squeeze(1)

    # Model setup and training (unified for all models)
    output_dim = y_train.shape[1] if args.task == 'regression' else y_train.shape[1]-1
        
    X_train = X_train.to(torch.float64)
    y_train = y_train.to(torch.float64)
    X_val = X_val.to(torch.float64)
    y_val = y_val.to(torch.float64)
    X_test = X_test.to(torch.float64)
    y_test = y_test.to(torch.float64)
    
    if args.task == 'regression':
        bf = AutogradLoss(torch.nn.MSELoss(reduction='none')) 
    else:
        bf = XEAutogradBregman(w=1)

    # Use torch tensors for tensor train
    model = TensorTrainRegressor(
        N=args.N,
        r=args.r,
        output_dim=output_dim,
        linear_dim=args.lin_dim,
        bf=bf,
        constrict_bond=False,
        perturb=False,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        eps_start=args.eps_start,
        eps_decay=args.eps_decay,
        batch_size=args.batch_size,
        method=args.method,
        num_swipes=args.num_swipes,
        model_type=args.model_type,
        cum_sum=args.cum_sum,
        task=args.task,
        verbose=args.verbose,
        early_stopping=args.early_stopping if args.early_stopping > 0 else None,
    )
    # Add num parameters to config
    model.fit(X_train, y_train, X_val, y_val)
    # Unified evaluation
    metric = 'accuracy' if args.task == 'classification' else 'rmse'
    val_score = evaluate_model(model, X_val, y_val, metric)

    num_params = model._model.num_parameters()
    converged_epoch = model._early_stopper.epoch

    report_dict = {}
    if args.task == 'classification':
        report_dict['val_rmse'] = np.nan
        report_dict['val_r2'] = np.nan
        report_dict['val_accuracy'] = val_score
        report_dict['num_params'] = num_params
        report_dict['converged_epoch'] = converged_epoch
    else:
        # Calculate R2 score as well
        r2_val = evaluate_model(model, X_val, y_val, metric='r2')
        report_dict['val_rmse'] = val_score
        report_dict['val_r2'] = r2_val
        report_dict['val_accuracy'] = np.nan
        report_dict['num_params'] = num_params
        report_dict['converged_epoch'] = converged_epoch

    if test:
        test_score = evaluate_model(model, X_test, y_test, metric)
        if args.task == 'classification':
            report_dict['test_rmse'] = np.nan
            report_dict['test_r2'] = np.nan
            report_dict['test_accuracy'] = test_score
        else:
            r2_test = evaluate_model(model, X_test, y_test, metric='r2')
            report_dict['test_rmse'] = test_score
            report_dict['test_r2'] = r2_test
            report_dict['test_accuracy'] = np.nan
    return report_dict
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensor Network Training for Tabular Data')
    parser.add_argument('--dataset_id', type=int, required=True, help='UCI dataset ID to load')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (for saving results)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to store the dataset (cpu or cuda)')
    parser.add_argument('--model_type', type=str, default='tt', required=True, help='Type of model to train: tt, cpd, _type1, etc.')
    parser.add_argument('--task', type=str, default='regression', choices=['regression', 'classification'], help='Task type: regression or classification')

    # Tensor Train hyperparameters
    parser.add_argument('--N', type=int, default=3, help='Number of carriages for tensor train')
    parser.add_argument('--r', type=int, default=3, help='Bond dimension for tensor train')
    parser.add_argument('--num_swipes', type=int, default=30, help='Number of swipes for tensor train')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate for tensor train')
    parser.add_argument('--method', type=str, default='ridge_exact', help='Method for tensor train')
    parser.add_argument('--eps_start', type=float, default=1.0, help='Initial Epsilon for tensor train')
    parser.add_argument('--eps_decay', type=float, default=0.75, help='Epsilon decay factor for tensor train')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for tensor train')
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity level for tensor train')
    parser.add_argument('--lin_dim', type=int, default=None, help='Linear dimension for tensor train (if any)')
    parser.add_argument('--cum_sum', action='store_true', help='Use cumulative sum layer instead of tensor train')
    parser.add_argument('--early_stopping', type=int, default=5, help='Early stopping patience for tensor train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')


    args = parser.parse_args()
    
    result = train_model(args, test=True)  # loads data inside main by default
    print(result)
