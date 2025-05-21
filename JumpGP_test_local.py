import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
# 将 JumpGP_code_py 所在的目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import torch
import time
import math
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigs
import argparse
from utils1 import jumpgp_ld_wrapper
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Test JumpGP model')
    parser.add_argument('--folder_name', type=str, required=True, 
                      help='Folder name containing dataset.pkl')
    parser.add_argument('--M', type=int, default=100, 
                      help='Number of nearest neighbors')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to use (cpu/cuda)')
    parser.add_argument('--use_sir', action='store_true',
                      help='Whether to use SIR dimension reduction')
    parser.add_argument('--sir_H', type=int, default=10,
                      help='Number of slices for SIR')
    parser.add_argument('--sir_K', type=int, default=2,
                      help='Number of components to keep in SIR')
    return parser.parse_args()


def apply_sir_reduction(X_train, Y_train, X_test, args):
    """Apply SIR dimension reduction to the data within a neighborhood"""
    # 转换为numpy数组进行SIR处理
    X_train_np = X_train.cpu().numpy()
    Y_train_np = Y_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    
    # 初始化并训练SIR模型
    sir = SIR(H=args.sir_H, K=args.sir_K)
    sir.fit(X_train_np, Y_train_np)
    
    # 获取变换矩阵并应用降维
    transform_matrix = sir.transform()
    transform_matrix = np.real(transform_matrix)
    
    X_train_reduced = np.matmul(X_train_np, transform_matrix)
    X_test_reduced = np.matmul(X_test_np, transform_matrix)
    
    # 转回PyTorch张量
    return (torch.from_numpy(X_train_reduced).to(X_train.device).to(torch.float64), 
            torch.from_numpy(X_test_reduced).to(X_test.device).to(torch.float64))


def load_dataset(folder_name):
    """Load dataset from pickle file"""
    dataset_path = os.path.join(folder_name, 'dataset.pkl')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


class SIR:
    def __init__(self, H, K=None, estdim=False, threshold=0.9, Cn=1):
        self.H = H
        self.K = K
        self.estdim = estdim
        self.threshold = threshold
        self.Cn = Cn

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.mean_x = np.mean(X, axis=0)
        self.sigma_x = np.cov(X, rowvar=False)

        Z = (X - self.mean_x) @ fractional_matrix_power(self.sigma_x, -0.5)
        n, p = Z.shape

        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.shape[0] != n or Y.shape[1] != 1:
            raise ValueError("X and Y must have matching samples, and Y must be 1D.")

        W = np.ones((n, 1)) / n

        base_count = n // self.H
        extra = n % self.H
        counts = np.array([base_count + (1 if i < extra else 0) for i in range(self.H)])
        cum_counts = np.cumsum(counts)

        data = np.hstack((Z, Y, W))
        data = data[np.argsort(data[:, p], axis=0)]
        z, y, w = data[:, :p], data[:, p:p+1], data[:, p+1:p+2]

        muh = np.zeros((self.H, p))
        wh  = np.zeros((self.H, 1))
        slice_idx = 0
        for i in range(n):
            if i >= cum_counts[slice_idx]:
                slice_idx += 1
            muh[slice_idx] += z[i]
            wh[slice_idx]  += w[i]
        muh = muh / (wh * n)

        M = np.zeros((p, p))
        for h in range(self.H):
            M += wh[h] * np.outer(muh[h], muh[h])
        self.M = M

        if self.estdim:
            eigvals, eigvecs = np.linalg.eigh(M)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            cumvar = np.cumsum(eigvals) / np.sum(eigvals)
            K_auto = int(np.searchsorted(cumvar, self.threshold) + 1)
            self.K = K_auto
            self.D = eigvals[:self.K]
            self.V = eigvecs[:, :self.K]
        else:
            if self.K is None:
                raise ValueError("K must be specified when estdim=False.")
            D, V = eigs(A=M, k=self.K, which='LM')
            self.D = np.real(D)
            self.V = np.real(V)

        return self.V, self.K, self.M, self.D

    def transform(self):
        hatbeta = fractional_matrix_power(self.sigma_x, -0.5) @ self.V
        return hatbeta


def find_neighborhoods(X_test, X_train, Y_train, M):
    """Find M nearest neighbors for each test point"""
    T = X_test.shape[0]
    dists = torch.cdist(X_test, X_train)
    _, indices = torch.sort(dists, dim=1)
    indices = indices[:, :M]
    neighborhoods = []
    for t in range(T):
        idx = indices[t]
        neighborhoods.append({
            "X_neighbors": X_train[idx],
            "y_neighbors": Y_train[idx],
            "indices": idx
        })
    return neighborhoods


def compute_metrics(predictions, sigmas, Y_test):
    """Compute RMSE and mean CRPS for Gaussian predictive distributions."""
    rmse = torch.sqrt(torch.mean((predictions - Y_test)**2))
    z = (Y_test - predictions) / sigmas
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    pdf = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    crps = sigmas * ( z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi) )
    mean_crps = torch.mean(crps)
    return rmse, mean_crps


def evaluate_jumpgp(X_test, Y_test, neighborhoods, device, args, std=None):
    """Evaluate JumpGP model on test data, with optional per-neighborhood SIR."""
    T = X_test.shape[0]
    jump_gp_results = []
    jump_gp_res_sig = []
    start_time = time.time()

    for t in tqdm(range(T)):
        neigh = neighborhoods[t]
        X_neigh = neigh["X_neighbors"]
        y_neigh = neigh["y_neighbors"]
        x_t_test = X_test[t].unsqueeze(0)

        # 在每个邻域里单独做 SIR 降维
        if args.use_sir:
            X_neigh, x_t_test = apply_sir_reduction(
                X_neigh, y_neigh.view(-1,1), x_t_test, args
            )

        mu_t, sig2_t, _, _ = jumpgp_ld_wrapper(
            X_neigh,
            y_neigh.view(-1, 1),
            x_t_test,
            mode="CEM",
            flag=False,
            device=device
        )

        jump_gp_results.append(mu_t)
        jump_gp_res_sig.append(torch.sqrt(sig2_t))

    run_time = time.time() - start_time
    predictions = torch.tensor([x.detach().item() for x in jump_gp_results])
    sigmas = torch.tensor([s.detach().item() for s in jump_gp_res_sig])

    if std is not None:
        predictions = predictions * std[1] + std[0]
        sigmas = sigmas * std[1]

    rmse, mean_crps = compute_metrics(predictions, sigmas, Y_test)
    return [rmse, mean_crps, run_time]


def main():
    args = parse_args()
    device = torch.device(args.device)
    dataset = load_dataset(args.folder_name)
    X_train = dataset["X_train"]
    Y_train = dataset["Y_train"]
    X_test = dataset["X_test"]
    Y_test = dataset["Y_test"]

    # 找到邻域
    neighborhoods = find_neighborhoods(X_test, X_train, Y_train, args.M)

    # 评估模型
    result = evaluate_jumpgp(X_test, Y_test, neighborhoods, device, args)

    # 打印结果
    print(f"RMSE: {result[0]:.4f}")
    print(f"CRPS: {result[1]:.4f}")
    print(f"Runtime: {result[2]:.2f} seconds")

    return result

if __name__ == "__main__":
    main()
