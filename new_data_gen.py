import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from simulate_case_d_linear import *
import argparse
from data_generate import *

def to_tensor(x_np, y_np):
    return torch.from_numpy(x_np).float(), torch.from_numpy(y_np).float().squeeze(-1)

def sample_rff_params(d, D_rff, lengthscale, device):
    """
    Sample one set of RFF parameters (Omega, phases).
    Omega: [D_rff, d],   phases: [D_rff]
    """
    Omega  = torch.randn(D_rff, d, device=device) / lengthscale
    phases = 2 * torch.pi * torch.rand(D_rff, device=device)
    return Omega, phases

def make_rff_features(x, Omega, phases, kernel_var):
    """
    Given x: [N, d], Omega: [D_rff, d], phases: [D_rff],
    returns Phi: [N, D_rff].
    """
    proj  = x @ Omega.t() + phases    # [N, D_rff]
    scale = torch.sqrt(torch.tensor(2 * kernel_var / Omega.shape[0],
                                   device=x.device))
    return scale * torch.cos(proj)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic data for high-dimensional GP')
    parser.add_argument('--N', type=int, default=2000, help='Number of training points')
    parser.add_argument('--T', type=int, default=100, help='Number of test points')
    parser.add_argument('--D', type=int, default=10, help='Dimension of observation space')
    parser.add_argument('--latent_dim', type=int, default=2, help='Dimension of latent space')
    parser.add_argument('--Q', type=int, default=2, help='Number of rows in A(x) matrix')
    parser.add_argument('--caseno', type=int, default=4, help='Case number for boundary generation (4,5,6 for phantom boundary)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    return parser.parse_args()

def main():
    args = parse_args()
    # 1) simulate data
    d, sig, N, Nt, Nc = args.Q, 0.1, args.N, args.T, args.T
    x_np, y_np, xc_np, yc_np, xt_np, yt_np, _, _, _ = \
        simulate_case_d_linear(d, sig, N, Nt, Nc)

    # 2) to torch
    x_train, y_train = to_tensor(x_np, y_np)     # [N,d], [N,1]
    x_test,  y_test  = to_tensor(xt_np, yt_np)   # [Nt,d], [Nt,1]
    x_cand,  y_cand  = to_tensor(xc_np, yc_np)   # [Nc,d], [Nc,1]

    # 3) sample one set of RFF params
    D_rff, lengthscale, kernel_var = args.D, 1.0, 1.0
    device = x_train.device
    Omega, phases = sample_rff_params(d, D_rff, lengthscale, device)

    # 4) map all at once (or separately, but with same Omega/phases)
    Phi_train = make_rff_features(x_train, Omega, phases, kernel_var)
    Phi_test  = make_rff_features(x_test,  Omega, phases, kernel_var)
    Phi_cand  = make_rff_features(x_cand,  Omega, phases, kernel_var)

    data = {
        "X_train": Phi_train, "Y_train": y_train,
        "X_test":  Phi_test,  "Y_test":  y_test,
        "X_cand":  Phi_cand,  "Y_cand":  y_cand
    }

    folder_name = save_dataset(data, args)
    return folder_name


if __name__ == "__main__":
    main()
