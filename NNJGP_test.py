import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import time

import torch
from JumpGP_test import *
from NNJGP import *

def parse_args():
    parser = argparse.ArgumentParser(description='Test NN-JGP model')
    parser.add_argument('--folder_name', type=str, required=True,
                        help='Folder containing dataset.pkl')
    parser.add_argument('--Q', type=int, default=2,
                        help='Latent dimensionality for the feature network')
    parser.add_argument('--m1', type=int, default=5,
                        help='Number of inducing points per region')
    parser.add_argument('--n', type=int, default=100,
                        help='Number of neighbors per region')
    parser.add_argument('--num_steps', type=int, default=200,
                        help='Number of ELBO training steps')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) Load data
    dataset = load_dataset(args.folder_name)
    X_train = dataset["X_train"].to(device)
    Y_train = dataset["Y_train"].to(device)
    X_test  = dataset["X_test"].to(device)
    Y_test  = dataset["Y_test"].to(device)

    # 3) Dimensions
    N_test, D = X_test.shape
    T = N_test
    Q = args.Q
    m1 = args.m1
    n  = args.n

    # 4) Build regions via neighborhoods
    neighborhoods = find_neighborhoods(
        X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=n
    )
    regions = []
    for i in range(T):
        nbr = neighborhoods[i]
        regions.append({
            'X':      nbr['X_neighbors'].to(device),   # [n, D]
            'y':      nbr['y_neighbors'].to(device),   # [n]
            'X_test': X_test[i],                       # [D]
            'C':      torch.randn(m1, Q, device=device)  # init inducing points in latent space
        })

    # 5) Instantiate feature network
    g_phi = FeatureNet(input_dim=D, latent_dim=Q, hidden_dims=(8,)).to(device)

    # 6) Initialize variational u_params
    u_params = []
    for _ in range(T):
        u_params.append({
            'm_u':     torch.zeros(m1, device=device, requires_grad=True),
            'L_u':     torch.eye(m1, device=device, requires_grad=True),
            'sigma_n': torch.tensor(0.5, device=device, requires_grad=True),
            'sigma_k': torch.tensor(1.0, device=device, requires_grad=True),
            'omega':   torch.randn(Q+1, device=device, requires_grad=True),
            'U_logit': torch.tensor(0.0, device=device, requires_grad=True),
        })

    # 7) (No extra hyperparams needed for NN-JGP)
    hyperparams = {}

    # 8) Quick ELBO before training
    elbo0 = compute_ELBO_nn_vectorized(regions, u_params, g_phi, hyperparams)
    print(f"Initial ELBO: {elbo0.item():.4f}")

    # 9) Train
    start_time = time.time()
    u_params, g_phi, hyperparams = train_nn(
        regions, u_params, g_phi, hyperparams,
        lr=args.lr, num_steps=args.num_steps, log_interval=50
    )
    

    # 10) ELBO after training
    # elbo1 = compute_ELBO_nn_vectorized(regions, u_params, g_phi, hyperparams)
    # print(f"ELBO after training: {elbo1.item():.4f}")

    # 11) Predict
    mu_pred, var_pred = predict_nn(regions, u_params, g_phi, hyperparams)
    sigmas = torch.sqrt(var_pred)
    rmse, mean_crps = compute_metrics(mu_pred, sigmas, Y_test)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f}s")
    print(f"Results [RMSE, mean-CRPS, train_time]: {[rmse, mean_crps, train_time]:}")
    return [rmse, mean_crps, train_time]

if __name__ == "__main__":
    main()
