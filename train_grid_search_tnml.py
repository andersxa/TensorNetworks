import sys
import torch
import numpy as np
torch.set_default_dtype(torch.float64)
from load_ucirepo import get_ucidata
from tensor.bregman import AutogradLoss, XEAutogradBregman
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score
from train_grid_search import seeds, test_seeds
from datetime import datetime
from dotdict import DotDict
torch.set_default_dtype(torch.float64)
import pandas as pd

from models.tnml import TNMLRegressor

datasets = [
    ('adult', 2, 'classification'),
    ('iris', 53, 'classification'),
    ('hearth', 45, 'classification'),
    ('winequalityc', 186, 'classification'),
    ('breast', 17, 'classification'),
    ('bank', 222, 'classification'),
    ('wine', 109, 'classification'),
    ('car_evaluation', 19, 'classification'),
    ('student_dropout', 697, 'classification'),
    ('mushrooms', 73, 'classification'),
    ('student_perf', 320, 'regression'),
    ('abalone', 1, 'regression'),
    ('obesity', 544, 'regression'),
    ('bike', 275, 'regression'),
    ('realstate', 477, 'regression'),
    ('energy_efficiency', 242, 'regression'),
    ('concrete', 165, 'regression'),
    ('ai4i', 601, 'regression'),
    # ('appliances', 374, 'regression'),
    ('popularity', 332, 'regression'),
    ('seoulBike', 560, 'regression'),
]


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
    if data is None:
        data = get_ucidata(args.dataset_id, args.task, device=args.data_device)
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
    model = TNMLRegressor(
        r=args.r,
        output_dim=output_dim,
        bf=bf,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        eps_start=args.eps_start,
        eps_decay=args.eps_decay,
        batch_size=args.batch_size,
        method=args.method,
        num_swipes=args.num_swipes,
        model_type=args.model_type,
        task=args.task,
        verbose=args.verbose,
        early_stopping=args.early_stopping if args.early_stopping > 0 else None,
        basis='polynomial' if 'poly' in args.model_type else 'sin-cos',
        degree=args.degree
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
    skip_grid_search = False# Set to True to skip grid search and load from CSV

    args = DotDict()
    args.device = 'cuda'
    args.data_device = 'cuda'

    rs = [4, 8, 12, 16]
    args.num_swipes = 100
    args.lr = 1.0
    args.eps_start = 5.0
    args.eps_decay = 0.25

    args.batch_size = 16384

    args.verbose = 1
    args.method = 'ridge_cholesky'
    args.lin_dim = None

    for basis_func in ['polynomial', 'sin-cos']: #'sin-cos', 
        is_poly = basis_func == 'polynomial'
        degrees = [1,2,3,4,5,6] if is_poly else [np.nan]
        args.model_type = f'tnml_{basis_func}'
        for dataset, dataset_id, task in datasets:
            data = get_ucidata(dataset_id, task, args.data_device)
            num_features = data[0].shape[1]
            args.early_stopping = max(10, num_features+1)
            args.task = task

            rerun = True
            if skip_grid_search:
                # Load existing results from CSV
                try:
                    df = pd.read_csv(f'./results/{dataset}_ablation_results_{args.model_type}.csv')
                    print(f"Loaded existing results for {dataset}", file=sys.stdout, flush=True)
                    rerun = False
                except FileNotFoundError:
                    print(f"No existing results found for {dataset}, running grid search.", file=sys.stdout, flush=True)
                    rerun = True
            if rerun:
                # Perform grid search
                results = []
                for degree in degrees:
                    for r in rs:
                        for seed in seeds:
                            try:
                                args.N = degree if is_poly else np.nan
                                args.degree = degree
                                args.r = r
                                args.seed = seed
                                print(f"Training {dataset} with r={r}, basis={basis_func}, degree={degree} and early_stopping={args.early_stopping}", file=sys.stdout, flush=True)
                                result = train_model(args, data=data, test=False)
                                results.append((dataset, args.degree, args.r, np.nan, result['val_rmse'], result['val_r2'], result['val_accuracy'], result['num_params'], result['converged_epoch'], seed))
                                print(f"Result: {result}", file=sys.stdout, flush=True)
                            except KeyboardInterrupt:
                                print("Interrupted by user, exiting...", file=sys.stdout, flush=True)
                                exit(0)
                            except:
                                print("Failed, skipping...", file=sys.stdout, flush=True)
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

                df.to_csv(f'/zhome/6b/e/212868/Desktop/code/TensorNetworksFork/results/{dataset}_ablation_results_{args.model_type}.csv', index=False)

            # Take the best one and run it on the test set
            # First we aggregate over seeds to find the best (N, r) pair
            group_by_cols = ['N', 'r'] if is_poly else ['r']
            df_agg = df.groupby(group_by_cols).agg({'val_rmse': 'mean', 'val_accuracy': 'mean'}).reset_index()

            if task == 'regression':
                best_row = df_agg.loc[df_agg['val_rmse'].idxmin()]
            else:
                best_row = df_agg.loc[df_agg['val_accuracy'].idxmax()]
            
            if is_poly:
                args.N = int(best_row['N']) if not np.isnan(best_row['N']) else np.nan
                args.degree = int(best_row['N']) if not np.isnan(best_row['N']) else np.nan
            args.r = int(best_row['r'])

            # Run 5 test runs with different seeds
            for test_seed in test_seeds:
                args.seed = test_seed
                print(f"Final evaluation on test set for {dataset} with seed {test_seed}", file=sys.stdout, flush=True)
                result = train_model(args, data=data, test=True)
                print(f"Final Result: {result}", file=sys.stdout, flush=True)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                with open(f'/zhome/6b/e/212868/Desktop/code/TensorNetworksFork/results/test_results_{args.model_type}.csv', 'a+') as f:
                    f.write(f"{timestamp},{args.model_type},{dataset},{args.N},{args.r},{np.nan},{result['test_rmse']},{result['test_r2']},{result['test_accuracy']},{result['num_params']},{result['converged_epoch']}\n")
