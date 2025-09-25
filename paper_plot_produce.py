#%%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error

def root_mean_squared_error_torch(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return root_mean_squared_error(y_true, y_pred)

def unexplained_variance(y_true, y_pred):
    y_mean = torch.mean(y_true, dim=0, keepdim=True)
    ss_tot = torch.sum((y_true - y_mean) ** 2, dim=1, keepdim=True)
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=1, keepdim=True)
    return (ss_res / ss_tot).mean().item()

from tensor.bregman import SquareBregFunction, AutogradLoss
from tensor.module import TensorTrainRegressorEarlyStopping, EarlyStopping
from tensor.layers import CPDLayer, TensorTrainLayer
from scipy import special
from sklearn.preprocessing import PolynomialFeatures
import math
import time

import pandas as pd
from tqdm import tqdm

from data import RandomPolynomial, RandomPolynomialRange, RandomIndependentPolynomial

def get_data(d, degree, num_train_points, num_val_points, num_test_points, random_state=42):
    rng = np.random.default_rng(random_state)
    X_train = rng.uniform(-1, 1, size=(num_train_points, d)) #rng.normal(1, 0.1, size=(num_train_points, d))
    X_val = rng.uniform(-1, 1, size=(num_val_points, d)) #rng.normal(1, 0.1, size=(num_val_points, d))
    X_test = rng.uniform(-1, 1, size=(num_test_points, d)) #rng.normal(1, 0.3, size=(num_test_points, d))

    # poly = RandomPolynomial(
    #     d=d,
    #     degree=degree,
    #     sigma0=0.02,
    #     r=1,
    #     include_bias=True,
    #     interaction_only=False,
    #     random_state=random_state
    # )
    poly = RandomPolynomialRange(
        d=d,
        degree=degree,
        input_range=(-1, 1),
        random_state=random_state
    )
    # poly = RandomIndependentPolynomial(
    #     d=d,
    #     degree=degree,
    #     coeff_sigma=5,
    #     r=1,
    #     include_bias=True,
    #     interaction_only=False,
    #     random_state=random_state
    # )
    y_train = poly.evaluate(X_train)
    y_val = poly.evaluate(X_val)
    y_test = poly.evaluate(X_test)

    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_val.ndim == 1:
        y_val = y_val.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_tensor_train(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    max_degree,      # formerly "carriages"
    rank,
    eps_start=1e-12,
    eps_end=1e-12,
    early_stopping=5,
    split_train=False,
    constrain_bond=False,
    method='ridge_cholesky',
    model_type='tt',
    random_state=42,
    verbose=0
):
    if max_degree > 2:
        tt = TensorTrainRegressorEarlyStopping(
            eps_start=eps_start,
            eps_end=eps_end,
            early_stopping=early_stopping,
            rel_err=1e-4,
            abs_err=1e-6,
            split_train=split_train,
            N=max_degree,
            r=rank,
            output_dim=1,
            batch_size=-1,
            constrict_bond=constrain_bond,
            seed=random_state,
            device='cuda',
            bf=SquareBregFunction(),
            lr=1.0,
            method=method,
            model_type=model_type,
            verbose=verbose
        )
        #try:
        # pass validation data into fit()
        tt.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        singular = tt._singular
        # except Exception as e:
        #     singular = True
        #     print(f"Error during fitting: {e}")

        # evaluate on the test set
        y_pred_test = tt.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = root_mean_squared_error(y_test, y_pred_test)
        best_degree = tt._best_degree
        num_params = sum(p.tensor.numel() for i, p in enumerate(tt._model.tensor_network.train_nodes) if i < tt._best_degree)
        val_history = tt._early_stopping.val_history
        time_history = tt._early_stopping.time_history
    else:
        tt = None
        singular = True
        best_degree = np.nan
        r2_test = np.nan
        rmse_test = np.nan
        num_params = None
        val_history = None

    return r2_test, rmse_test, singular, tt, best_degree, num_params, val_history, time_history

def fit_poly_mononomial(X, y, degree, include_bias=True):
    """
    X_train: Numpy array of shape (B, d)
    y: Numpy array of shape (B,1)
    degree: total degree (int)
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    Phi = poly.fit_transform(X)          # (B, n_terms)
    y = y.reshape(-1, 1)

    Phi = torch.tensor(Phi, dtype=torch.float64, device='cuda')
    y = torch.tensor(y, dtype=torch.float64, device='cuda')
    lstsq = torch.linalg.lstsq(Phi, y)
    coeffs = lstsq.solution  # (n_terms, 1)
    rank = lstsq.rank
    
    coeffs = coeffs.cpu().numpy()  # convert to numpy array

    return poly, coeffs, rank

def evaluate_poly(X, coeffs, poly):
    """
    Evaluate the polynomial with coefficients coeffs at X.

    X: Numpy array of shape (B, d)
    coeffs: Numpy array of shape (n_terms, 1)
    poly: PolynomialFeatures instance used to generate coeffs
    """
    Phi = poly.transform(X)          # (B, n_terms)
    y_pred = Phi.dot(coeffs)         # (B, 1)
    return y_pred

def evaluate_polynomial_regression(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    max_degree,
    d,
    abs_err=0.0,
    rel_err=0.0,
    early_stopping=5,
    verbose=0
):
    current = {'poly': None, 'coeffs': None, 'rank': None}

    def model_predict(X):
        return evaluate_poly(X, current['coeffs'], current['poly'])

    def get_weights():
        return (current['poly'], current['coeffs'], current['rank'])

    loss_fn = root_mean_squared_error

    es = EarlyStopping(
        X_train, y_train, X_val, y_val,
        model_predict=model_predict,
        get_model_weights=get_weights,
        loss_fn=loss_fn,
        abs_err=abs_err,
        rel_err=rel_err,
        early_stopping=early_stopping,
        verbose=verbose,
        start_degree=1
    )

    for deg in range(1, max_degree + 1):
        if special.comb(d + deg, d) >= 12_000:
            break

        poly, coeffs, rank = fit_poly_mononomial(
            X_train, y_train, degree=deg, include_bias=True
        )

        current['poly'], current['coeffs'], current['rank'] = poly, coeffs, rank

        if es.convergence_criterion():
            break
    
    best_summary = es.best_summary()
    best_poly, best_coeffs, best_rank = best_summary['best_state_dict']
    best_degree = best_summary['best_degree']
    val_history = es.val_history
    time_history = es.time_history
    if best_poly is None:
        return np.nan, np.nan, None, None, np.nan, None, None, None

    y_test_pred = evaluate_poly(X_test, best_coeffs, best_poly)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    num_params = np.prod(best_coeffs.shape)

    return test_r2, test_rmse, best_poly, best_coeffs, best_degree, num_params, best_rank, val_history, time_history

def prepare_torch_data(
    X, y,
    X_val=None, y_val=None,
    validation_split=0.2,
    split_train=False,
    seed=42,
    device='cuda'
):
    # Convert to torch tensors if needed
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float64, device=device)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float64, device=device)
    if y.ndim == 1:
        y = y.unsqueeze(1)

    # Validation split if not provided
    if X_val is None or y_val is None:
        if split_train:
            n = X.shape[0]
            idx = np.arange(n)
            rng = np.random.RandomState(seed)
            rng.shuffle(idx)
            split = int(n * (1 - validation_split))
            train_idx, val_idx = idx[:split], idx[split:]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = X, y
    else:
        if isinstance(X_val, np.ndarray):
            X_val = torch.tensor(X_val, dtype=torch.float64, device=device)
        if isinstance(y_val, np.ndarray):
            y_val = torch.tensor(y_val, dtype=torch.float64, device=device)
        if y_val.ndim == 1:
            y_val = y_val.unsqueeze(1)
        X_train, y_train = X, y

    # Append 1 to X for bias term
    X_train = torch.cat((X_train, torch.ones((X_train.shape[0], 1), dtype=torch.float64, device=device)), dim=1)
    X_val = torch.cat((X_val, torch.ones((X_val.shape[0], 1), dtype=torch.float64, device=device)), dim=1)

    return X_train, y_train, X_val, y_val


def cpd_predict(cpd, X, device='cuda'):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float64, device=device)
    # Append 1 to X for bias term
    X = torch.cat((X, torch.ones(X.shape[0], 1, dtype=torch.float64, device=device)), dim=1)
    y_pred = cpd(X)
    return y_pred.detach().cpu().numpy()

def evaluate_cpd(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    max_degree,
    rank,
    num_swipes=5,
    eps_start=1e-12,
    eps_end=1e-10,
    early_stopping=5,
    split_train=False,
    method='ridge_cholesky',
    timeout=120,
    random_state=42,
    verbose=0
):
    # Set torch seed
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    X_train, y_train, X_val, y_val = prepare_torch_data(
        X_train, y_train,
        X_val, y_val,
        split_train=split_train,
        seed=random_state,
        device='cuda'
    )

    early_stopping = EarlyStopping(
        X_train, y_train, X_val, y_val,
        model_predict=lambda X: None, #cpd,
        get_model_weights=lambda: None, #lambda: cpd.node_states(),
        loss_fn=unexplained_variance,
        abs_err=1e-6,
        rel_err=1e-4,
        early_stopping=early_stopping,
        verbose=verbose,
        start_degree=2
    )

    loss_fn = AutogradLoss(torch.nn.MSELoss(reduction='none'))

    time_start = time.time()

    for deg in range(2, max_degree + 1):
        cpd = CPDLayer(
            num_factors=deg,
            rank=rank,
            input_features=X_train.shape[1],
            output_shape=1,
        ).cuda()
        early_stopping.model_predict = cpd
        early_stopping.get_model_weights = lambda: cpd.node_states()
        converged = cpd.tensor_network.accumulating_swipe(
            X_train, y_train, loss_fn,
            batch_size=-1,
            convergence_criterion=None,
            eps=np.geomspace(eps_start, eps_end, num=num_swipes*2).tolist(),
            method=method,
            skip_second=False,
            lr=1.0,
            orthonormalize=False,
            verbose=verbose == 1,
            num_swipes=num_swipes,
            direction='l2r',
            disable_tqdm=True
        )

        if early_stopping.convergence_criterion() or (time.time() - time_start) > timeout:
            break
    
    best_summary = early_stopping.best_summary()
    best_state_dict = best_summary['best_state_dict']
    best_degree = best_summary['best_degree']

    singular = not converged

    # Restore best state if available
    cpd = CPDLayer(
        num_factors=best_degree,
        rank=rank,
        input_features=X_train.shape[1],
        output_shape=1,
    ).cuda()
    if best_state_dict is not None:
        cpd.load_node_states(best_state_dict, set_value=True)
    y_pred_test = cpd_predict(cpd, X_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    num_params = sum(p.tensor.numel() for i, p in enumerate(cpd.tensor_network.train_nodes))
    val_history = early_stopping.val_history
    time_history = early_stopping.time_history
    return r2_test, rmse_test, singular, cpd, best_degree, num_params, val_history, time_history

def tt_full_predict(tt_model, X, device='cuda'):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float64, device=device)
    # Append 1 to X for bias term
    X = torch.cat((X, torch.ones(X.shape[0], 1, dtype=torch.float64, device=device)), dim=1)
    y_pred = tt_model(X)
    return y_pred.detach().cpu().numpy()

def evaluate_tt_full(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    max_degree,
    rank,
    num_swipes=5,
    eps_start=1e-12,
    eps_end=1e-10,
    early_stopping=5,
    split_train=False,
    method='ridge_cholesky',
    timeout=120,
    random_state=42,
    verbose=0
):
    # Set torch seed
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    X_train, y_train, X_val, y_val = prepare_torch_data(
        X_train, y_train,
        X_val, y_val,
        split_train=split_train,
        seed=random_state,
        device='cuda'
    )

    early_stopping = EarlyStopping(
        X_train, y_train, X_val, y_val,
        model_predict=lambda X: None, #cpd,
        get_model_weights=lambda: None, #lambda: cpd.node_states(),
        loss_fn=unexplained_variance,
        abs_err=1e-6,
        rel_err=1e-4,
        early_stopping=early_stopping,
        verbose=verbose,
        start_degree=2
    )

    loss_fn = SquareBregFunction()

    time_start = time.time()

    for deg in range(2, max_degree + 1):
        tt_model = TensorTrainLayer(deg, rank, X_train.shape[1],
                               output_shape=1,
                               constrict_bond=False,
                               perturb=False,
                               seed=random_state).cuda()
        early_stopping.model_predict = tt_model
        early_stopping.get_model_weights = lambda: tt_model.node_states()
        converged = tt_model.tensor_network.accumulating_swipe(
            X_train, y_train, loss_fn,
            batch_size=-1,
            convergence_criterion=None,
            eps=np.geomspace(eps_start, eps_end, num=num_swipes*2).tolist(),
            method=method,
            skip_second=False,
            lr=1.0,
            orthonormalize=False,
            verbose=verbose == 1,
            num_swipes=num_swipes,
            direction='l2r',
            disable_tqdm=True
        )

        if early_stopping.convergence_criterion() or (time.time() - time_start) > timeout:
            break
    
    best_summary = early_stopping.best_summary()
    best_state_dict = best_summary['best_state_dict']
    best_degree = best_summary['best_degree']

    singular = not converged

    # Restore best state if available
    tt_model = TensorTrainLayer(best_degree, rank, X_train.shape[1],
                                output_shape=1,
                                constrict_bond=False,
                                perturb=False,
                                seed=random_state).cuda()
    if best_state_dict is not None:
        tt_model.load_node_states(best_state_dict, set_value=True)
    y_pred_test = tt_full_predict(tt_model, X_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    num_params = sum(p.tensor.numel() for i, p in enumerate(tt_model.tensor_network.train_nodes))
    val_history = early_stopping.val_history
    time_history = early_stopping.time_history
    return r2_test, rmse_test, singular, tt_model, best_degree, num_params, val_history, time_history


if __name__ == "__main__":
    eps = 3e-13
    def collect_results(seeds=range(42, 42+20), degree=3, d=1, rank=24, cpd_rank=100, max_degree=10, eps=3e-13, one_value_only=False):
        rows = []
        n_values = [math.comb(i+d, d) for i in range(degree,degree+5)]
        if one_value_only:
            n_values = [n_values[0]]  # Only use the first value
        with tqdm(total=len(seeds) * len(n_values), desc="Collecting results") as pbar:
            for seed in seeds:
                for n in n_values:
                    # Generate data once per seed and N
                    X_train, y_train, X_val, y_val, X_test, y_test = get_data(
                        d=d,
                        degree=degree,
                        num_train_points=n,
                        num_val_points=n,
                        num_test_points=2, #test set is unused
                        random_state=seed
                    )

                    # Tensor Train model
                    tt_r2, tt_rmse, tt_singular, tt_model, tt_degree, tt_params, tt_history, tt_time_history = evaluate_tensor_train(
                        X_train, y_train,
                        X_val, y_val,
                        X_test, y_test,
                        early_stopping=max_degree,
                        max_degree=max_degree,
                        eps_start=eps,
                        eps_end=eps,
                        rank=rank, #change eps to 5e-12 to support higher rank
                        constrain_bond=False,
                        method='ridge_cholesky',
                        split_train=False,
                        random_state=seed,
                        verbose=0
                    )
                    for deg in tt_history.keys():
                        tt_loss = tt_history[deg]
                        tt_time = tt_time_history[deg]
                        rows.append((seed, "Growing TT", n, deg, float(tt_loss), tt_time, tt_r2, tt_rmse, tt_degree, tt_params))

                    # # Polynomial baseline
                    poly_r2, poly_rmse, poly_model, poly_coeffs, poly_degree, poly_params, poly_rank, poly_history, poly_time_history = evaluate_polynomial_regression(
                        X_train, y_train,
                        X_val, y_val,
                        X_test, y_test,
                        max_degree=max_degree,
                        d=d,
                        abs_err=1e-5,
                        rel_err=1e-4,
                        early_stopping=max_degree,
                        verbose=0
                    )
                    for deg in poly_history.keys():
                        poly_loss = poly_history[deg]
                        poly_time = poly_time_history[deg]
                        rows.append((seed, "Poly.Reg.", n, deg, float(poly_loss), poly_time, poly_r2, poly_rmse, poly_degree, poly_params))

                    # # CPD baseline
                    cpd_r2, cpd_rmse, cpd_singular, cpd_model, cpd_degree, cpd_params, cpd_history, cpd_time_history = evaluate_cpd(
                        X_train, y_train,
                        X_val, y_val,
                        X_test, y_test,
                        max_degree=max_degree,
                        rank=cpd_rank,
                        eps_start=eps,
                        eps_end=eps,
                        early_stopping=max_degree,
                        split_train=False,
                        method='ridge_cholesky',
                        random_state=seed,
                        verbose=0
                    )
                    for deg in cpd_history.keys():
                        cpd_loss = cpd_history[deg]
                        cpd_time = cpd_time_history[deg]
                        rows.append((seed, "CPD", n, deg, float(cpd_loss), cpd_time, cpd_r2, cpd_rmse, cpd_degree, cpd_params))

                    # Tensor Train without Growing
                    tt_full_r2, tt_full_rmse, tt_full_singular, tt_full_model, tt_full_degree, tt_full_params, tt_full_history, tt_full_time_history = evaluate_tt_full(
                        X_train, y_train,
                        X_val, y_val,
                        X_test, y_test,
                        max_degree=max_degree,
                        rank=rank,
                        eps_start=eps,
                        eps_end=eps,
                        early_stopping=max_degree,
                        split_train=False,
                        method='ridge_cholesky',
                        random_state=seed,
                        verbose=0
                    )
                    for deg in tt_full_history.keys():
                        tt_full_loss = tt_full_history[deg]
                        tt_full_time = tt_full_time_history[deg]
                        rows.append((seed, "TT", n, deg, float(tt_full_loss), tt_full_time, tt_full_r2, tt_full_rmse, tt_full_degree, tt_full_params))

                    # CPD with growing (using evaluate_tensor_train with model_type='cpd')
                    cpd_grow_r2, cpd_grow_rmse, cpd_grow_singular, cpd_grow_model, cpd_grow_degree, cpd_grow_params, cpd_grow_history, cpd_grow_time_history = evaluate_tensor_train(
                        X_train, y_train,
                        X_val, y_val,
                        X_test, y_test,
                        early_stopping=max_degree,
                        max_degree=max_degree,
                        rank=cpd_rank,
                        eps_start=1e-12,
                        eps_end=1e-12,
                        split_train=False,
                        method='ridge_cholesky',
                        model_type='cpd',
                        random_state=seed,
                        verbose=0
                    )
                    for deg in cpd_grow_history.keys():
                        cpd_grow_loss = cpd_grow_history[deg]
                        cpd_grow_time = cpd_grow_time_history[deg]
                        rows.append((seed, "Growing CPD", n, deg, float(cpd_grow_loss), cpd_grow_time, cpd_grow_r2, cpd_grow_rmse, cpd_grow_degree, cpd_grow_params))

                    pbar.update(1)

        df = pd.DataFrame(rows, columns=["seed", "name", "N", "degree", "loss", "time", "r2", "rmse", "best_degree", "num_params"])
        return df

    parameters = [
        # {'degree': 3, 'max_degree': 8, 'd': 1, 'rank': 6, 'cpd_rank': 100},
        # {'degree': 3, 'max_degree': 8, 'd': 3, 'rank': 12, 'cpd_rank': 100},
        {'degree': 3, 'max_degree': 8, 'd': 7, 'rank': 24, 'cpd_rank': 100},
        # {'degree': 5, 'max_degree': 10, 'd': 1, 'rank': 12, 'cpd_rank': 100},
        {'degree': 5, 'max_degree': 10, 'd': 3, 'rank': 24, 'cpd_rank': 100},
        # {'degree': 5, 'max_degree': 10, 'd': 7, 'rank': 38, 'cpd_rank': 100},
    ]
    all_dfs = []
    for params in parameters:
        print(f"Collecting for params: {params}")
        df = collect_results(
            degree=params['degree'],
            d=params['d'],
            rank=params['rank'],
            cpd_rank=params['cpd_rank'],
            max_degree=params['max_degree'],
            eps=eps
        )
        df['max_degree'] = params['max_degree']
        all_dfs.append(df)
        # Save to csv
        df.to_csv(f"results_d{params['d']}_deg{params['degree']}_rank{params['rank']}_cpdrank{params['cpd_rank']}_r2.csv", index=False)
