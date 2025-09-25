import numpy as np
from time import time
from functools import partial
import torch
import math
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, root_mean_squared_error, accuracy_score
from tensor.layers import TensorTrainLayer
from tensor.bregman import SquareBregFunction

def fbasis(X):
    Input = []
    for i in range(X.shape[-1]):
        T = torch.stack([torch.cos((0.5*math.pi)*X[:, i]), torch.sin((0.5*math.pi)*X[:,i])], dim=-1)
        Input.append(T)
    return Input

def polynomial_basis(X, degree=3):
    Input = []
    for i in range(X.shape[-1]):
        T = torch.stack([X[:, i]**d for d in range(degree+1)], dim=-1)
        Input.append(T)
    return Input

def root_mean_squared_error_torch(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return root_mean_squared_error(y_true, y_pred)

def unexplained_variance(y_true, y_pred):
    y_mean = torch.mean(y_true, dim=0, keepdim=True)
    ss_tot = torch.sum((y_true - y_mean) ** 2, dim=1, keepdim=True)
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=1, keepdim=True)
    return (ss_res / ss_tot).mean().item()

def error_rate_torch(y_true, y_pred):
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true_labels = y_true.cpu().numpy()
    if y_true_labels.ndim > 1 and y_true_labels.shape[1] > 1:
        y_true_labels = np.argmax(y_true_labels, axis=1)
    return 1.0 - accuracy_score(y_true_labels, y_pred_labels)

class EarlyStopping:
    def __init__(self, X_val, y_val, model_predict, get_model_weights=None, loss_fn=None, abs_err=0.0, rel_err=0.0, early_stopping=5, verbose=0):
        self.X_val = X_val
        self.y_val = y_val
        self.model_predict = model_predict
        self.get_model_weights = get_model_weights
        self.loss_fn = loss_fn
        self.abs_err = abs_err
        self.rel_err = rel_err
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.early_stop_count = 0
        self.best_val_loss = np.inf
        self.val_history = {}
        weights = self.get_model_weights()
        self.best_state_dict = weights if weights is not None else None
        self.start_time = time()
        self.time_history = {}
        self.epoch = 0

    def convergence_criterion(self):
        elapsed_time = time() - self.start_time
        self.epoch += 1
        # Compute losses
        y_pred_val = self.model_predict(self.X_val)
        val_loss = self.loss_fn(self.y_val, y_pred_val)

        self.val_history[self.epoch] = val_loss
        self.time_history[self.epoch] = elapsed_time

        # Measure improvement relative to previous best
        prev_best = self.best_val_loss
        improvement = prev_best - val_loss
        meets_abs = improvement >= self.abs_err
        meets_rel = improvement >= self.rel_err * abs(prev_best)
        meets_threshold = meets_abs or meets_rel

        if improvement > 0:
            # Always save the new best
            self.best_val_loss = val_loss
            if self.get_model_weights is not None:
                self.best_state_dict = self.get_model_weights()

            # Only reset patience if the improvement meets thresholds
            if meets_threshold:
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1
        else:
            # No improvement at all
            self.early_stop_count += 1

        # Early stopping check
        if self.early_stop_count >= self.early_stopping:
            if self.verbose > 0:
                print(f"Converged with best loss: {self.best_val_loss:.4f}")
            return True

        return False

class TNMLRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, r=8, output_dim=1, seed=42,
                 device='cuda', bf=None,
                 lr=1.0, eps_start=1.0, eps_decay=0.5,
                 abs_err=1e-6, rel_err=1e-4,
                 batch_size=512, method='ridge_cholesky',
                 num_swipes=30,
                 model_type='tt',
                 task='regression',
                 train_operator=False,
                 early_stopping=0,
                 basis='sin-cos', # 'sin-cos' or 'polynomial
                 degree=3, # for polynomial basis
                 verbose=0,
                 constrict_bond= True):
        self.r = r
        self.input_dim = degree+1 if basis == 'polynomial' else 2
        self.output_dim = output_dim
        self.constrict_bond = constrict_bond 
        self.perturb = False
        self.seed = seed
        self.device = device
        self.bf = bf if bf is not None else SquareBregFunction()
        self.lr = lr
        self.eps = eps_start
        self.eps_decay = eps_decay
        self.abs_err = abs_err
        self.rel_err = rel_err
        self.batch_size = batch_size
        self.method = method
        self.num_swipes = num_swipes
        self.model_type = model_type
        self.task = task
        self.train_operator = train_operator
        self.early_stopping = early_stopping
        self.basis = basis
        self.degree = degree
        self.verbose = verbose
        print(self.eps)
        self._model = None
        if self.perturb and self.output_dim > 1:
            raise ValueError("perturb not supported for output dim > 1")

    def _initialize_model(self):
        if self.input_dim is None:
            raise ValueError("input_dim must be set")
        self._model = TensorTrainLayer(self.N, self.r, self.input_dim,
                                    output_shape=self.output_dim,
                                    constrict_bond=self.constrict_bond,
                                    perturb=self.perturb,
                                    seed=self.seed).to(self.device)
        if self.verbose > 2:
            print("Number of parameters:", self._model.num_parameters())

    def fit(self, X, y, X_val=None, y_val=None, validation_split=0.1, split_train=True):
        # X, y: numpy arrays or torch tensors
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float64, device=self.device)

        if self._model is None:
            self.N = X.shape[1]
            self._initialize_model()
        
        if self.verbose > 0:
            print("Number of parameters:", self._model.num_parameters())

        # Validation split if not provided
        if X_val is None or y_val is None:
            if split_train:
                n = X.shape[0]
                idx = np.arange(n)
                rng = np.random.RandomState(self.seed)
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
                X_val = torch.tensor(X_val, dtype=torch.float64, device=self.device)
            if isinstance(y_val, np.ndarray):
                y_val = torch.tensor(y_val, dtype=torch.float64, device=self.device)
            # if y_val.ndim == 1:
            #     y_val = y_val.unsqueeze(1)
            X_train, y_train = X, y

        if self.basis == 'sin-cos':
            X_train = fbasis(X_train)
            X_val = fbasis(X_val)
        elif self.basis == 'polynomial':
            X_train = polynomial_basis(X_train, degree=self.degree)
            X_val = polynomial_basis(X_val, degree=self.degree)
        
        def model_predict(X_batch):
            y_pred = self._model.tensor_network.forward_batch(X_batch, self.batch_size)
            if self.task == 'classification':
                y_pred = torch.cat([y_pred, torch.zeros_like(y_pred[..., :1])], dim=-1)
            return y_pred
        
        self._early_stopper = EarlyStopping(
            X_val, y_val,
            model_predict=model_predict,
            get_model_weights=self._model.node_states,
            loss_fn=root_mean_squared_error_torch if self.task == 'regression' else error_rate_torch,
            abs_err=self.abs_err,
            rel_err=self.rel_err,
            early_stopping=self.early_stopping,
            verbose=self.verbose
        )

        self._model.tensor_network.orthonormalize_left()
        # Call accumulating_swipe
        self._model.tensor_network.accumulating_swipe(
            X_train, y_train, self.bf,
            batch_size=self.batch_size,
            lr=self.lr,
            eps=self.eps,
            eps_decay=self.eps_decay,
            convergence_criterion=self._early_stopper.convergence_criterion,
            orthonormalize=True,
            method=self.method,
            verbose=self.verbose,
            num_swipes=self.num_swipes,
            skip_second=False,
            direction='l2r',
            disable_tqdm=self.verbose < 3,
        )

        # Load best weights
        if self._early_stopper.best_state_dict is not None:
            self._model.load_node_states(self._early_stopper.best_state_dict, set_value=True)

        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        if self.basis == 'sin-cos':
            X = fbasis(X)
        elif self.basis == 'polynomial':
            X = polynomial_basis(X, degree=self.degree)
        y_pred = self._model.tensor_network.forward_batch(X, self.batch_size)
        if self.task == 'classification':
            y_pred = torch.cat([y_pred, torch.zeros_like(y_pred[..., :1])], dim=-1)
        return y_pred.detach().cpu().numpy()

    def score(self, X, y_true):
        # Return R2 score on X, y_true
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        if not isinstance(y_true, np.ndarray):
            y_true = y_true.cpu().numpy()
        y_pred = self.predict(X)
        return r2_score(y_true, y_pred) if self.task == 'regression' else accuracy_score(y_true, np.argmax(y_pred, axis=1))
