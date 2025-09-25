import sys
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from load_ucirepo import get_ucidata
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, WhiteKernel, ConstantKernel as C
from datetime import datetime
from dotdict import DotDict
import pandas as pd

def root_mean_squared_error_torch(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return root_mean_squared_error(y_true, y_pred)

def error_rate_torch(y_true, y_pred):
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true_labels = y_true.cpu().numpy()
    if y_true_labels.ndim > 1 and y_true_labels.shape[1] > 1:
        y_true_labels = np.argmax(y_true_labels, axis=1)
    return 1.0 - accuracy_score(y_true_labels, y_pred_labels)

datasets = [
    ('iris', 53, 'classification'),            # 150
    ('wine', 109, 'classification'),           # 178
    ('hearth', 45, 'classification'),          # 303
    ('realstate', 477, 'regression'),          # 414
    ('breast', 17, 'classification'),          # 569
    ('student_perf', 320, 'regression'),       # 649
    ('energy_efficiency', 242, 'regression'),  # 768
    ('concrete', 165, 'regression'),           # 1030
    ('car_evaluation', 19, 'classification'),  # 1728
    ('obesity', 544, 'regression'),            # 2111
    ('abalone', 1, 'regression'),              # 4177
    ('student_dropout', 697, 'classification'),# 4424
    ('winequalityc', 186, 'classification'),   # 6497
    ('mushrooms', 73, 'classification'),       # 8124
    ('ai4i', 601, 'regression'),               # 10000
    ('bike', 275, 'regression'),               # 17379
    ('appliances', 374, 'regression'),         # 19735
    ('popularity', 332, 'regression'),         # 39644
    ('bank', 222, 'classification'),           # 45211
    ('adult', 2, 'classification'),            # 48842
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

class GaussianProcess:
    """
    A unified wrapper for Gaussian Processes that handles both regression and classification tasks.
    """
    def __init__(self, task='regression', kernel=None, alpha=1e-10, seed=42):
        self.task = task
        self.kernel = kernel
        self.alpha = alpha

        if self.task == 'regression':
            self.model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, random_state=seed)
        elif self.task == 'classification':
            self.model = GaussianProcessClassifier(kernel=kernel, random_state=seed)
        else:
            raise ValueError(f"Unknown task: {task}")

    def fit(self, X, y):
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()

        if self.task == 'classification' and y_np.ndim == 2:
            y_np = y_np.argmax(axis=-1)

        self.model.fit(X_np, y_np)

    def predict(self, X):
        X_np = X.cpu().numpy()
        if self.task == 'classification':
            return self.model.predict_proba(X_np)
        else:
            return self.model.predict(X_np)

def train_model(args, data=None, test=False):
    if data is None:
        data = get_ucidata(args.dataset_id, args.task, device=args.data_device)
    X_train, y_train, X_val, y_val, X_test, y_test = data

    # Model setup
    model = GaussianProcess(task=args.task, kernel=args.kernel, alpha=args.alpha)
    model.fit(X_train, y_train)

    # Unified evaluation
    metric = 'accuracy' if args.task == 'classification' else 'rmse'
    val_score = evaluate_model(model, X_val, y_val, metric)

    num_params = np.nan
    converged_epoch = np.nan

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
    skip_grid_search = False  # Set to True to skip grid search and load from CSV

    args = DotDict()
    args.device = 'cuda'
    args.data_device = 'cuda'

    args.model_type = 'gp'

    # Hyperparameter grids - GP specific
    alphas = [None]

    def create_kernels(n_features):
        kernels = []
        kernel_names = []

        # DotProduct + WhiteKernel
        kernels.append(DotProduct() + WhiteKernel())
        kernel_names.append('DotProduct_plus_WhiteKernel')
        # RBF + WhiteKernel
        kernels.append(C(1.0) * RBF(length_scale=[1.0] * n_features) + WhiteKernel())
        kernel_names.append('RBF_ARD_plus_WhiteKernel')
        # DotProduct + RBF + WhiteKernel
        kernels.append(C(1.0) * RBF(length_scale=[1.0] * n_features) + DotProduct() + WhiteKernel())
        kernel_names.append('RBF_ARD_plus_DotProduct_plus_WhiteKernel')

        # # Base kernels with scalar length scales
        # base_kernels_scalar = [
        #     (C(1.0) * RBF(), 'RBF'),
        #     (C(1.0) * Matern(nu=2.5), 'Matern25'),
        #     (DotProduct(), 'DotProduct'),
        #     (C(1.0) * RBF() + DotProduct(), 'RBF_plus_DotProduct')
        # ]

        # # Base kernels with array length scales (ARD)
        # base_kernels_ard = [
        #     (C(1.0) * RBF(length_scale=[1.0] * n_features), 'RBF_ARD'),
        #     (C(1.0) * Matern(nu=2.5, length_scale=[1.0] * n_features), 'Matern25_ARD'),
        #     (C(1.0) * RBF(length_scale=[1.0] * n_features) + DotProduct(), 'RBF_ARD_plus_DotProduct')
        # ]

        # # 1. Base kernels with scalar hyperparameters
        # for kernel, name in base_kernels_scalar:
        #     kernels.append(kernel)
        #     kernel_names.append(name)

        # # 2. Base kernels with array hyperparameters (ARD)
        # for kernel, name in base_kernels_ard:
        #     kernels.append(kernel)
        #     kernel_names.append(name)

        # # 3. Base kernels with scalar hyperparameters + WhiteKernel
        # for kernel, name in base_kernels_scalar:
        #     kernels.append(kernel + WhiteKernel())
        #     kernel_names.append(f'{name}_WhiteKernel')

        # # 4. Base kernels with array hyperparameters + WhiteKernel
        # for kernel, name in base_kernels_ard:
        #     kernels.append(kernel + WhiteKernel())
        #     kernel_names.append(f'{name}_WhiteKernel')

        return kernels, kernel_names

    seeds = list(range(42, 42 + 5))

    for dataset, dataset_id, task in datasets:
        data = get_ucidata(dataset_id, task, args.data_device)
        X_train, y_train, X_val, y_val, X_test, y_test = data
        n_features = X_train.shape[1]

        args.task = task
        args.dataset_id = dataset_id

        # Create kernels based on the number of features
        # If less than 4000 samples then create all kernels:
        if X_train.shape[0] < 4000:
            kernels, kernel_names = create_kernels(n_features)
            print(f"Using full kernel set for dataset with {X_train.shape[0]} samples.", file=sys.stdout, flush=True)
        else:
            # Use only RBF with ARD + DotProduct + WhiteKernel
            kernels = [C(1.0) * RBF(length_scale=[1.0] * n_features) + DotProduct() + WhiteKernel()]
            kernel_names = ['RBF_ARD_plus_DotProduct_WhiteKernel']
            print("Using reduced kernel set due to large dataset size.", file=sys.stdout, flush=True)
        

        rerun = True
        if skip_grid_search:
            try:
                # Load existing results from CSV
                df = pd.read_csv(f'./results/{dataset}_ablation_results_{args.model_type}.csv')
                print(f"Loaded existing results for {dataset}", file=sys.stdout, flush=True)
                rerun = False
            except FileNotFoundError:
                rerun = True
                
        if rerun:
            # Perform grid search
            results = []
            for kernel, kernel_name in zip(kernels, kernel_names):
                for alpha in alphas:
                    try:
                        args.kernel = kernel
                        args.kernel_name = kernel_name
                        args.alpha = alpha

                        print(f"Training {dataset} with kernel {kernel_name}, alpha {alpha}", file=sys.stdout, flush=True)
                        result = train_model(args, data=data, test=False)

                        results.append((
                            dataset,
                            kernel_name,
                            alpha,
                            result['val_rmse'],
                            result['val_r2'],
                            result['val_accuracy'],
                            result['num_params']
                        ))

                        print(f"Result: {result}", file=sys.stdout, flush=True)
                    except KeyboardInterrupt:
                        print("Interrupted by user, exiting...", file=sys.stdout, flush=True)
                        exit(0)
                    except Exception as e:
                        print(f"Failed with error: {e}, skipping...", file=sys.stdout, flush=True)
                        continue

            # Build per-dataset results frame
            df = pd.DataFrame(
                results,
                columns=['dataset', 'kernel_name', 'alpha', 'val_rmse', 'val_r2', 'val_accuracy', 'num_params']
            )
            df['model_type'] = args.model_type

            if len(df) == 0:
                exit(0)

            df.to_csv(f'./results/{dataset}_ablation_results_{args.model_type}.csv', index=False)

        # Aggregate across seeds
        group_by_cols = ['kernel_name']
        df_agg = df.groupby(group_by_cols).agg({'val_rmse': 'mean', 'val_accuracy': 'mean'}).reset_index()

        if task == 'regression':
            best_row = df_agg.loc[df_agg['val_rmse'].idxmin()]
        else:
            best_row = df_agg.loc[df_agg['val_accuracy'].idxmax()]

        # Reconstruct best setting and evaluate on test set multiple times
        best_kernel_name = best_row['kernel_name']
        best_alpha = np.nan

        # Find the corresponding kernel object
        best_kernel_idx = kernel_names.index(best_kernel_name)
        args.kernel = kernels[best_kernel_idx]
        args.kernel_name = best_kernel_name
        args.alpha = best_alpha

        # Run 5 test runs with different seeds
        test_seeds = [1337, 2024, 3141, 4242, 5555]
        for test_seed in test_seeds:
            print(f"Final evaluation on test set for {dataset} with seed {test_seed}", file=sys.stdout, flush=True)
            result = train_model(args, data=data, test=True)
            print(f"Final Result: {result}", file=sys.stdout, flush=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(f'./results/test_results_{args.model_type}.csv', 'a+') as f:
                f.write(
                    f"{timestamp},{args.model_type},{dataset},{best_kernel_name},"
                    f"{best_alpha},{result['test_rmse']},{result['test_r2']},"
                    f"{result['test_accuracy']},{result['num_params']},{result['converged_epoch']}\n"
                )