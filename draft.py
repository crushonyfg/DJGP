import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import math
import numpy as np
import argparse
import pickle
from datetime import datetime
from skimage import io, color, transform

def normalize_A(A):
    """
    对A矩阵进行正则化
    Input: A shape is (N, Q, D)
    Output: 正则化后的A
    """
    N, Q, D = A.shape
    A_normalized = A.clone()
    
    for i in range(N):
        # 方法1：Frobenius范数归一化
        # A_i = A[i]  # shape: (Q, D)
        # frob_norm = torch.norm(A_i, p='fro')
        # A_normalized[i] = A_i / frob_norm
        
        # 方法2：行归一化
        # for q in range(Q):
        #     row_norm = torch.norm(A[i, q], p=2)
        #     A_normalized[i, q] = A[i, q] / row_norm
        
        # 方法3：正交化 (使用QR分解)
        A_i = A[i]  # shape: (Q, D)
        Z, R = torch.linalg.qr(A_i.T)  # 对转置进行QR分解
        A_normalized[i] = Z[:, :Q].T  # 取前Q列的转置，确保是正交矩阵
        
    return A_normalized

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

N, D, T, Q = 2000, 20, 100, 2
# Generate input X_train and X_test
X_train = torch.randn(N, D, device=device)
X_test = torch.randn(T, D, device=device)
X_all = torch.cat([X_train, X_test], dim=0)
N_all = X_all.shape[0]

# Sample GP hyperparameters for A(x)
# gamma_dist = torch.distributions.Gamma(2.0, 1.0)
gamma_dist = torch.distributions.Gamma(4.0, 2.0)
sigma_a_A = gamma_dist.sample().to(device)
sigma_q_A = gamma_dist.sample((1,)).to(device)

# Compute GP covariance matrix K_A
diff = X_all.unsqueeze(1) - X_all.unsqueeze(0)
sqdist = torch.sum(diff**2, dim=2)
K_A = sigma_a_A**2 * torch.exp(-0.5 * sqdist / (sigma_q_A**2))
K_A = K_A + 1e-6 * torch.eye(N_all, device=device, dtype=K_A.dtype)

# Generate A for each (q, d)
A_all = torch.empty(N_all, Q, D, device=device)
mean_zero = torch.zeros(N_all, device=device)
for q in range(Q):
    for d in range(D):
        mvn = torch.distributions.MultivariateNormal(mean_zero, covariance_matrix=K_A)
        A_all[:, q, d] = mvn.sample()

# 对A进行正则化
A_all = normalize_A(A_all)

# Split A into train and test
A_train = A_all[:N]
A_test = A_all[N:]

# Compute latent variables z(x)
z_train = torch.bmm(A_train, X_train.unsqueeze(-1)).squeeze(-1)
z_test = torch.bmm(A_test, X_test.unsqueeze(-1)).squeeze(-1)