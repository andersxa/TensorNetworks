import sys
import os
import torch
import numpy as np
torch.set_default_dtype(torch.float64)
from datetime import datetime
from dotdict import DotDict
import pandas as pd
from train_tnml_MNIST import get_image_data
from models.tensor_train import TensorTrainRegressor
from tensor.bregman import AutogradLoss, XEAutogradBregman
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score

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
        data = get_image_data(dataset_name=args.dataset_name, data_path=args.data_path, device=args.data_device)
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

    # Use torch tensors for tensor train with CPD model
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
        cum_sum=getattr(args, 'cum_sum', False),
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
    args = DotDict()
    args.device = 'cuda'
    args.data_device = 'cuda'
    args.task = 'classification'
    args.model_type = 'cpd'

    # Choose dataset: 'mnist' or 'fashionmnist'
    args.dataset_name = 'mnist'  # Change to 'fashionmnist' to use FashionMNIST
    args.data_path = "/work3/aveno/MNIST/data"  # Specify path to data

    # CPD hyperparameters
    Ns = [2] #[2, 3, 4, 5, 6]
    rs = [4] #[4, 8, 12, 16, 64, 144, 256]
    args.num_swipes = 100
    args.lr = 1.0
    args.eps_start = 5.0
    args.eps_decay = 0.25
    args.batch_size = 128
    args.verbose = 1
    args.method = 'ridge_cholesky'
    args.lin_dim = None
    args.cum_sum = False

    # Load data once
    data = get_image_data(dataset_name=args.dataset_name, data_path=args.data_path, device=args.data_device)
    num_features = data[0].shape[1]
    args.early_stopping = max(10, num_features+1)

    print(f"Dataset: {args.dataset_name}", file=sys.stdout, flush=True)
    print(f"Number of features: {num_features}", file=sys.stdout, flush=True)
    print(f"Training samples: {data[0].shape[0]}", file=sys.stdout, flush=True)
    print(f"Validation samples: {data[2].shape[0]}", file=sys.stdout, flush=True)
    print(f"Test samples: {data[4].shape[0]}", file=sys.stdout, flush=True)

    seeds = list(range(42, 42+5))

    # Perform grid search
    results = []
    for N in Ns:
        for r in rs:
            args.N = N
            args.r = r
            for seed in seeds:
                try:
                    args.seed = seed
                    print(f"Training {args.dataset_name} with N={N}, r={r}, seed={seed}", file=sys.stdout, flush=True)
                    result = train_model(args, data=data, test=False)
                    results.append((args.dataset_name, N, r, np.nan, result['val_rmse'], result['val_r2'], result['val_accuracy'], result['num_params'], result['converged_epoch'], seed))
                    print(f"Result: {result}", file=sys.stdout, flush=True)
                except KeyboardInterrupt:
                    print("Interrupted by user, exiting...", file=sys.stdout, flush=True)
                    exit(0)
                except Exception as e:
                    print(f"Failed with error: {e}", file=sys.stdout, flush=True)
                    torch.cuda.empty_cache()
                    continue

    df = pd.DataFrame(results, columns=['dataset', 'N', 'r', 'lin_dim', 'val_rmse', 'val_r2', 'val_accuracy', 'num_params', 'converged_epoch', 'seed'])
    df['num_swipes'] = args.num_swipes
    df['eps_start'] = args.eps_start
    df['eps_decay'] = args.eps_decay
    df['early_stopping'] = args.early_stopping
    df['model_type'] = args.model_type

    if len(df) == 0:
        exit(0)

    os.makedirs('./image_results', exist_ok=True)
    df.to_csv(f'./image_results/{args.dataset_name}_ablation_results_{args.model_type}.csv', index=False)

    # Take the best one and run it on the test set
    # First we aggregate over seeds to find the best (N, r) pair
    df_agg = df.groupby(['N', 'r']).agg({'val_rmse': 'mean', 'val_accuracy': 'mean'}).reset_index()

    if args.task == 'regression':
        best_row = df_agg.loc[df_agg['val_rmse'].idxmin()]
    else:
        best_row = df_agg.loc[df_agg['val_accuracy'].idxmax()]

    args.N = int(best_row['N'])
    args.r = int(best_row['r'])
    print(f"Best hyperparameters: N={args.N}, r={args.r}", file=sys.stdout, flush=True)
    print(f"Best validation accuracy: {best_row['val_accuracy']:.4f}", file=sys.stdout, flush=True)

    # Run 5 test runs with different seeds
    test_results = []
    test_seeds = [1337, 2024, 3141, 4242, 5555]
    for test_seed in test_seeds:
        args.seed = test_seed
        print(f"Final evaluation on test set for {args.dataset_name} with N={args.N}, r={args.r}, seed {test_seed}", file=sys.stdout, flush=True)
        result = train_model(args, data=data, test=True)
        test_results.append(result['test_accuracy'])
        print(f"Final Result: {result}", file=sys.stdout, flush=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('./image_results', exist_ok=True)
        with open(f'./image_results/test_results_{args.model_type}.csv', 'a+') as f:
            f.write(f"{timestamp},{args.model_type},{args.dataset_name},{args.N},{args.r},{np.nan},{result['test_rmse']},{result['test_r2']},{result['test_accuracy']},{result['num_params']},{result['converged_epoch']}\n")

    print(f"Final test accuracy: {np.mean(test_results):.4f} ± {np.std(test_results):.4f}", file=sys.stdout, flush=True)
