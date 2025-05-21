import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # 设置为你想用的核心数


import sys
import argparse
import time

import torch
import numpy as np
import math
from sklearn.model_selection import KFold

from utils1 import jumpgp_ld_wrapper
# from VI_utils_gpu_acc_UZ_qumodified_cor import *
from check_VI_utils_gpu_acc_UZ_qumodified_cor import *
from JumpGP_test import *

def parse_args():
    parser = argparse.ArgumentParser(description='Test DeepJumpGP model with CV for Q and m2')
    parser.add_argument('--folder_name', type=str, required=True, 
                        help='Folder name containing dataset.pkl')
    parser.add_argument('--Q', type=int, default=2,
                        help='Latent dimensionality (overridden if using CV)')
    parser.add_argument('--m1', type=int, default=5,
                        help='Number of inducing points per region')
    parser.add_argument('--m2', type=int, default=20,
                        help='Number of global inducing points (overridden if using CV)')
    parser.add_argument('--n', type=int, default=100,
                        help='Number of neighbors per region')
    parser.add_argument('--num_steps', type=int, default=500,
                        help='Number of VI training steps')
    parser.add_argument('--MC_num', type=int, default=5,
                        help='Number of Monte Carlo samples for prediction')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    # CV-specific
    parser.add_argument('--cv_splits', type=int, default=3,
                        help='Number of folds for cross-validation')
    parser.add_argument('--Q_list', type=int, nargs='+', default=[3, 5, 7],
                        help='List of Q candidates for CV')
    parser.add_argument('--m2_list', type=int, nargs='+', default=[20],
                        help='List of m2 candidates for CV')
    parser.add_argument('--better_init', action='store_true',
                        help='Whether to use better initialization from jumpgp_ld_wrapper')
    parser.add_argument('--use_cv', dest='use_cv', action='store_true', help='Use CV to select Q and m2')
    parser.add_argument('--no_use_cv', dest='use_cv', action='store_false', help='Do not use CV')
    parser.set_defaults(use_cv=False)
    return parser.parse_args()

# def initialize_model(X_train, Y_train, X_test, Y_test, args, device):
#     """
#     Initialize all model parameters for DJGP.
    
#     Args:
#         X_train: Training input features
#         Y_train: Training targets
#         X_test: Test input features
#         Y_test: Test targets
#         args: Arguments containing model hyperparameters
#         device: Device to place tensors on
        
#     Returns:
#         regions: List of region dictionaries
#         V_params: Dictionary of V parameters
#         u_params: List of u parameters
#         hyperparams: Dictionary of hyperparameters
#     """
#     T, D = X_test.shape
#     Q  = args.Q
#     m1 = args.m1
#     m2 = args.m2
#     n  = args.n

#     # Build neighborhoods
#     neighborhoods = find_neighborhoods(
#         X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=n
#     )
#     regions = []
#     for i in range(T):
#         X_nb = neighborhoods[i]['X_neighbors'].to(device)   # (n, D)
#         y_nb = neighborhoods[i]['y_neighbors'].to(device)   # (n,)
#         regions.append({
#             'X': X_nb,
#             'y': y_nb,
#             'C': torch.randn(m1, Q, device=device)          # random init
#         })

#     # Initialize V_params
#     V_params = {
#         'mu_V':    torch.randn(m2, Q, D, device=device, requires_grad=True),
#         'sigma_V': torch.rand( m2, Q, D, device=device, requires_grad=True),
#     }

#     # Initialize u_params
#     u_params = []
#     for _ in range(T):
#         u_params.append({
#             'U_logit':     torch.zeros(1, device=device, requires_grad=True),
#             'mu_u':        torch.randn(m1, device=device, requires_grad=True),
#             'Sigma_u':     torch.eye(m1, device=device, requires_grad=True),
#             'sigma_noise': torch.tensor(0.5, device=device, requires_grad=True),
#             'sigma_k':torch.tensor(0.5, device=device, requires_grad=True),
#             'omega':       torch.randn(Q+1, device=device, requires_grad=True),
#         })

#     # Initialize hyperparams
#     X_train_mean = X_train.mean(dim=0)
#     X_train_std  = X_train.std(dim=0)
#     Z = X_train_mean + torch.randn(m2, D, device=device) * X_train_std

#     hyperparams = {
#         'Z':             Z,                      # (m2, D)
#         'X_test':        X_test,                 # (T, D)
#         'lengthscales': torch.rand(Q, device=device, requires_grad=True),
#         'var_w':         torch.tensor(1.0, device=device, requires_grad=True),
#     }
    
#     return regions, V_params, u_params, hyperparams

def initialize_model(X_train, Y_train, X_test, Y_test, args, device):
    """
    Initialize all model parameters for DJGP using SIR for dimension reduction.
    
    Args:
        X_train: Training input features
        Y_train: Training targets
        X_test: Test input features
        Y_test: Test targets
        args: Arguments containing model hyperparameters
        device: Device to place tensors on
        
    Returns:
        regions: List of region dictionaries
        V_params: Dictionary of V parameters
        u_params: List of u parameters
        hyperparams: Dictionary of hyperparameters
    """
    T, D = X_test.shape
    Q  = args.Q
    m1 = args.m1
    m2 = args.m2
    n  = args.n

    # 1. Apply SIR dimension reduction
    from sklearn.cluster import KMeans
    from JumpGP_test_CV import SIR
    
    # Initialize and fit SIR
    sir = SIR(H=D, K=Q)  # Using default H=20 as in JumpGP_test_CV
    X_train_np = X_train.cpu().numpy()
    Y_train_np = Y_train.cpu().numpy()
    sir.fit(X_train_np, Y_train_np)
    transform_matrix = torch.from_numpy(sir.transform()).to(device).to(X_train.dtype)  # (D, Q)
    
    # Transform training data
    X_train_reduced = torch.matmul(X_train, transform_matrix)  # (N, Q)
    X_test_reduced = torch.matmul(X_test, transform_matrix)  # (T, Q)
    
    # 2. Initialize Z using K-means on origining data
    kmeans = KMeans(n_clusters=m2, random_state=42)
    Z_centers = kmeans.fit(X_train.cpu().numpy()).cluster_centers_
    Z = torch.from_numpy(Z_centers).to(device).to(X_train.dtype).requires_grad_(True)  # (m2, Q)
    
    # 3. Build neighborhoods and initialize regions
    neighborhoods = find_neighborhoods(
        X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=n
    )
    regions = []
    for i in range(T):
        X_nb = neighborhoods[i]['X_neighbors'].to(device).to(X_train.dtype)   # (n, D)
        y_nb = neighborhoods[i]['y_neighbors'].to(device).to(X_train.dtype)   # (n,)
        
        # Transform neighbors to reduced space
        X_nb_reduced = torch.matmul(X_nb, transform_matrix)  # (n, Q)
        
        # Get K-means centers for this neighborhood
        kmeans = KMeans(n_clusters=m1, random_state=42)
        C_centers = kmeans.fit(X_nb_reduced.cpu().numpy()).cluster_centers_
        C = torch.from_numpy(C_centers).to(device).to(X_train.dtype)  # (m1, Q)
        
        regions.append({
            'X': X_nb,
            'y': y_nb,
            'C': C
        })

    # print("(n,D),(n,1),(m1,Q)",X_nb.shape,y_nb.shape,C.shape)

    # 4. Initialize V_params using SIR transformation matrix
    V_params = {
        'mu_V':    transform_matrix.t().unsqueeze(0).repeat(m2, 1, 1).requires_grad_(True),  # (m2, Q, D)
        'sigma_V': torch.full((m2, Q, D), 2.0, device=device, dtype=X_train.dtype, requires_grad=True),
    }
    # print("(m2,Q,D)",V_params['mu_V'].shape, V_params['sigma_V'].shape)

    # 5. Initialize u_params using jumpgp_ld_wrapper
    u_params = []
    for t in tqdm(range(T)):
        if args.better_init:
            neigh = neighborhoods[t]
            X_neigh = neigh["X_neighbors"].to(device).to(X_train.dtype)
            y_neigh = neigh["y_neighbors"].to(device).to(X_train.dtype)
            x_t_test = X_test[t]
            
            #et initial parameters from jumpgp_ld_wrapper
            X_neigh_reduced = torch.matmul(X_neigh, transform_matrix)  # (n, Q)
            X_t_test_reduced = torch.matmul(x_t_test, transform_matrix)  # (1, Q)
            mu_t, sig2_t, model_torch, _ = jumpgp_ld_wrapper(
                X_neigh_reduced, 
                y_neigh.view(-1, 1), 
                X_t_test_reduced.view(1, -1), 
                mode="CEM", 
                flag=False, 
                device=device
            )
            
            # Extract omega from logtheta as initial value
            omega = torch.exp(model_torch['logtheta'][:Q+1]).detach().clone().to(
                device=device, 
                dtype=X_train.dtype
            ).requires_grad_(True)
        else:
            omega = torch.randn(Q+1, device=device, dtype=X_train.dtype, requires_grad=True)
        
        u_params.append({
            'U_logit':     torch.zeros(1, device=device, dtype=X_train.dtype, requires_grad=True),
            'mu_u':        torch.zeros(m1, device=device, dtype=X_train.dtype, requires_grad=True),
            'Sigma_u':     torch.eye(m1, device=device, dtype=X_train.dtype, requires_grad=True),
            'sigma_noise': torch.tensor(0.5, device=device, dtype=X_train.dtype, requires_grad=True),
            'sigma_k':     torch.tensor(0.5, device=device, dtype=X_train.dtype, requires_grad=True),
            # 'omega':       torch.randn(Q+1, device=device, requires_grad=True),
            'omega':       omega,
        })
    # print("(1),(m1),(m1,m1),(0.5),(0.5),(Q+1)",u_params[0]['U_logit'].shape,u_params[0]['mu_u'].shape,u_params[0]['Sigma_u'].shape,u_params[0]['sigma_noise'].shape,u_params[0]['sigma_k'].shape,u_params[0]['omega'].shape)

    # 6. Initialize hyperparams
    hyperparams = {
        'Z':             Z,                      # (m2, Q)
        'X_test':        X_test,                 # (T, D)
        'lengthscales':  torch.full((Q,), 2.0, device=device, dtype=X_train.dtype, requires_grad=True),
        'var_w':         torch.tensor(2.0, device=device, dtype=X_train.dtype, requires_grad=True),
    }
    # print("(m2,Q),(T,D),(Q),(Q)",hyperparams['Z'].shape,hyperparams['X_test'].shape,hyperparams['lengthscales'].shape,hyperparams['var_w'].shape)
    
    return regions, V_params, u_params, hyperparams

def run_experiment(X_train, Y_train, X_test, Y_test, args, device):
    """
    Train & evaluate on a single train/test split with given args.Q, args.m2.
    Returns [rmse, mean_crps, runtime].
    """
    T, D = X_test.shape
    Q  = args.Q
    m1 = args.m1
    m2 = args.m2
    n  = args.n
    num_steps = args.num_steps
    MC_num    = args.MC_num

    start_time = time.time()
    
    # Initialize model parameters
    regions, V_params, u_params, hyperparams = initialize_model(
        X_train, Y_train, X_test, Y_test, args, device
    )

    # Train VI
    V_params, u_params, hyperparams = train_vi(
        regions=regions,
        V_params=V_params,
        u_params=u_params,
        hyperparams=hyperparams,
        lr=args.lr,
        num_steps=num_steps,
        log_interval=50
    )

    # Predict
    mu_pred, var_pred = predict_vi(
        regions, V_params, hyperparams, M=MC_num
    )

    # Compute metrics
    sigmas = torch.sqrt(var_pred)
    rmse, mean_crps = compute_metrics(mu_pred, sigmas, Y_test)
    run_time = time.time() - start_time

    return [rmse, mean_crps, run_time]

from sklearn.model_selection import ShuffleSplit
def cv_score_for_params(X, Y, args, device, Q, m2):
    """
    Perform KFold CV on (X,Y) for given (Q, m2).
    Returns average mean-CRPS across folds.
    """
    # kf = KFold(n_splits=args.cv_splits, shuffle=True, random_state=2025)
    ss = ShuffleSplit(
        n_splits=args.cv_splits,
        test_size=0.05,       # 验证集占比
        random_state=2025
    )
    scores = []

    args.MC_num = 3
    for train_idx, val_idx in ss.split(X):
        X_tr, Y_tr = X[train_idx], Y[train_idx]
        X_va, Y_va = X[val_idx],   Y[val_idx]

        # Move to device
        X_tr_d = X_tr.to(device);   Y_tr_d = Y_tr.to(device)
        X_va_d = X_va.to(device);   Y_va_d = Y_va.to(device)

        # Override args
        args.Q  = Q
        args.m2 = m2

        # Run experiment on this split
        rmse, crps, _ = run_experiment(
            X_tr_d, Y_tr_d, X_va_d, Y_va_d, args, device
        )
        scores.append(rmse.detach().cpu().item())

    return float(np.mean(scores))

def main():
    args = parse_args()

    MC_num = args.MC_num

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load full dataset
    dataset = load_dataset(args.folder_name)
    X_train = dataset["X_train"]  # keep on CPU for CV
    Y_train = dataset["Y_train"]
    X_test  = dataset["X_test"].to(device)
    Y_test  = dataset["Y_test"].to(device)

    if args.use_cv:
        # args.MC_num = 3
        # Cross-validation to select Q and m2
        print("Starting CV over Q and m2...")
        best_score = float('inf')
        best_params = {'Q': None, 'm2': None}

        for Q in args.Q_list:
            for m2 in args.m2_list:
                score = cv_score_for_params(X_train, Y_train, args, device, Q, m2)
                print(f"CV (Q={Q}, m2={m2}) → mean-RMSE = {score:.4f}")
                if score < best_score:
                    best_score = score
                    best_params = {'Q': Q, 'm2': m2}

        print(f"Best hyperparams: Q={best_params['Q']}, m2={best_params['m2']} with RMSE={best_score:.4f}")

        # Re-run on full training set + true test set
        args.Q  = best_params['Q']
        args.m2 = best_params['m2']

    args.MC_num = MC_num
    print("Training final model with selected hyperparameters...")
    rmse, mean_crps, run_time = run_experiment(
        X_train.to(device), Y_train.to(device), X_test, Y_test, args, device
    )
    print(f"Final results → RMSE: {rmse:.4f}, mean-CRPS: {mean_crps:.4f}, runtime: {run_time:.1f}s")
    return [rmse, mean_crps, run_time]

if __name__ == "__main__":
    main()
