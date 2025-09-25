from datetime import datetime
from dotdict import DotDict
import torch
torch.set_default_dtype(torch.float64)
from train_grid_search import train_model, seeds, test_seeds
from load_ucirepo import get_ucidata
import pandas as pd
import numpy as np

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

if __name__ == '__main__':
    skip_grid_search = False# Set to True to skip grid search and load from CSV

    args = DotDict()
    args.device = 'cuda'
    args.data_device = 'cuda'
    args.model_type = 'cpd_type1'

    Ns = [2, 3, 4, 5, 6]
    rs = [4, 8, 12, 16, 64, 144, 256]
    args.num_swipes = 100
    args.lr = 1.0
    args.eps_start = 5.0
    args.eps_decay = 0.25

    args.batch_size = 1024
    args.early_stopping = 10

    args.verbose = 1
    args.method = 'ridge_cholesky'
    args.lin_dim = None

    for dataset, dataset_id, task in datasets:
        print(f"Starting dataset {dataset} with id {dataset_id} for task {task}")
        data = get_ucidata(dataset_id, task, args.data_device)
        args.task = task

        if skip_grid_search:
            # Load existing results from CSV
            df = pd.read_csv(f'./results/{dataset}_ablation_results_{args.model_type}.csv')
            print(f"Loaded existing results for {dataset}")
        else:
            # Perform grid search
            results = []
            for N in Ns:
                for r in rs:
                    args.N = N
                    args.r = r
                    for seed in seeds:
                        try:
                            args.seed = seed
                            print(f"Training {dataset} with N={N}, r={r}")
                            result = train_model(args, data=data, test=False)
                            results.append((dataset, N, r, np.nan, result['val_rmse'], result['val_r2'], result['val_accuracy'], result['num_params'], result['converged_epoch'], seed))
                            print(f"Result: {result}")
                        except:
                            print("Failed, skipping...")
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
        df_agg = df.groupby(['N', 'r']).agg({'val_rmse': 'mean', 'val_accuracy': 'mean'}).reset_index()
        if task == 'regression':
            best_row = df_agg.loc[df_agg['val_rmse'].idxmin()]
        else:
            best_row = df_agg.loc[df_agg['val_accuracy'].idxmax()]
        
        try:
            args.N = int(best_row['N'])
            args.r = int(best_row['r'])
        except:
            args.N = 3
            args.r = 12
            print("============================================")
            print("============================================")
            print("WARNING!: Failed to select best hyperparameters, using default N=3, r=12")
            print(f"WARNING!: Failed for {dataset}, task={task}")
            print("============================================")
            print("============================================")
        
        # Run 5 test runs with different seeds
        for test_seed in test_seeds:
            args.seed = test_seed
            print(f"Final evaluation on test set for {dataset} with N={args.N}, r={args.r}, seed {test_seed}")
            result = train_model(args, data=data, test=True)
            print(f"Final Result: {result}")

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(f'/zhome/6b/e/212868/Desktop/code/TensorNetworksFork/results/test_results_{args.model_type}.csv', 'a+') as f:
                f.write(f"{timestamp},{args.model_type},{dataset},{args.N},{args.r},{np.nan},{result['test_rmse']},{result['test_r2']},{result['test_accuracy']},{result['num_params']},{result['converged_epoch']}\n")
