# python JumpGP_test.py --folder_name "2025_04_01_23" --M 100 --device cpu
# python JumpGP_test.py --folder_name "2025_04_01_23" --M 100 --device cpu --use_sir --sir_H 10 --sir_K 2
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
    """Apply SIR dimension reduction to the data"""
    # 转换为numpy数组进行SIR处理
    X_train_np = X_train.cpu().numpy()
    Y_train_np = Y_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    
    # 初始化并训练SIR模型
    sir = SIR(H=args.sir_H, K=args.sir_K)
    sir.fit(X_train_np, Y_train_np)
    
    # 获取变换矩阵并应用降维
    transform_matrix = sir.transform()
    # 确保变换矩阵是实数
    transform_matrix = np.real(transform_matrix)
    
    X_train_reduced = np.matmul(X_train_np, transform_matrix)
    X_test_reduced = np.matmul(X_test_np, transform_matrix)
    
    # 转回PyTorch张量，确保使用正确的数据类型
    return (torch.from_numpy(X_train_reduced).to(X_train.device).to(torch.float64), 
            torch.from_numpy(X_test_reduced).to(X_test.device).to(torch.float64))

def load_dataset(folder_name):
    """Load dataset from pickle file"""
    dataset_path = os.path.join(folder_name, 'dataset.pkl')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

class SIR:
    def __init__(self, H, K, estdim=0, Cn=1):
        self.H = H
        self.K = K
        self.estdim = estdim
        self.Cn = Cn

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.mean_x = np.mean(X, axis=0)  # 计算样本均值
        self.sigma_x = np.cov(X, rowvar=False)  # 计算样本协方差矩阵
        self.Z = np.matmul(X - np.tile(self.mean_x, (X.shape[0], 1)),
                           fractional_matrix_power(self.sigma_x, -0.5))  # 标准化后的数据阵
        n, p = self.Z.shape
        if self.Y.ndim == 1:  # 判断响应变量Y的维度
            self.Y = self.Y.reshape(-1, 1)
        ny, py = self.Y.shape
        W = np.ones((n, 1)) / n
        nw, pw = W.shape
        # 输入数据异常判断
        if n != ny:
            raise ValueError('X and Y must have the same number of samples')
        elif p == 1:
            raise ValueError('X must have at least 2 dimensions')
        elif py != 1:
            raise ValueError('Y must have only 1 dimension')
        # 将y分为H片，c为每个片的样本数
        c = np.ones((1, self.H)) * (n // self.H) + np.hstack(
            [np.ones(
                (1, n % self.H)), np.zeros((1, self.H - n % self.H))])
        cumc = np.cumsum(c)  # 计算切片累计和
        # 参照Y的取值从小到大进行排序
        temp = np.hstack((self.Z, self.Y, W))
        temp = temp[np.argsort(temp[:, p])]
        # 提取排序后的z,y,w
        z, y, w = temp[:, :p], temp[:, p:p + 1], temp[:, p + 1]
        muh = np.zeros((self.H, p))
        wh = np.zeros((self.H, 1))  # 每个切片的权重
        k = 0  # 初始化切片编号
        for i in range(n):
            if i >= cumc[k]:  # 如果超过了切片的边界，则换下一个切片
                k += 1
            muh[k, :] = muh[k, :] + z[i, :]  # 计算切片内自变量之和
            wh[k] = wh[k] + w[i]  # 计算每个切片包含Yi的概率
        # 计算每个切片的样本均值,将其作为切片内自变量的取值
        muh = muh / (np.tile(wh, (1, p)) * n)
        # 加权主成分分析
        self.M = np.zeros((p, p))  # 初始化切片 加权协方差矩阵
        for i in range(self.H):
            self.M = self.M + wh[i] * muh[i, :].reshape(-1, 1) * muh[i, :]
        if self.estdim == 0: # 一般情况
            self.D, self.V = eigs(A=self.M, k=self.K, which='LM')
        else:
            """ # 稀疏矩阵情况，待修正
            [V D] = np.linalg,eig(full(M))
            lambda = np.sort(abs(diag(D)),'descend')
            L = np.log(lambda+1) - lambda
            G = np.zeros((p,1))
            if Cn == 1
                Cn = n^(1/2)
            elif Cn == 2
            Cn = n^(1/3)
            elif Cn == 3:
                Cn = 0.5 * n^0.25
            for k in range(p):
                G(k) = n / 2 * sum(L(1:k)) / sum(L) - Cn * k * (k+1) / p
            maxG, K = np. max(G)
            V, D = eigs(M,K,'lm')
            """
            pass
        return self.V, self.K, self.M, self.D
    def transform(self):
        hatbeta = np.matmul(fractional_matrix_power(self.sigma_x, -0.5),self.V)
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
    """Compute RMSE and NLPD metrics"""
    # 计算RMSE
    rmse = torch.sqrt(torch.mean((predictions - Y_test)**2))
    
    # 计算NLPD
    nlpd = 0.5 * torch.log(2 * math.pi * sigmas**2) + \
           0.5 * ((Y_test - predictions)**2) / (sigmas**2)
    
    # 计算四分位数
    q75 = torch.quantile(nlpd, 0.75)
    q25 = torch.quantile(nlpd, 0.25)
    q50 = torch.quantile(nlpd, 0.50)
    
    return rmse, q25, q50, q75

def evaluate_jumpgp(X_test, Y_test, neighborhoods, device):
    """Evaluate JumpGP model on test data"""
    T = X_test.shape[0]
    jump_gp_results = []
    jump_gp_res_sig = []
    
    # 记录开始时间
    start_time = time.time()
    
    for t in range(T):
        neigh = neighborhoods[t]
        X_neigh = neigh["X_neighbors"]
        y_neigh = neigh["y_neighbors"]
        x_t_test = X_test[t]
        
        mu_t, sig2_t, _, _ = jumpgp_ld_wrapper(
            X_neigh, 
            y_neigh.view(-1, 1), 
            x_t_test.view(1, -1), 
            mode="CEM", 
            flag=False, 
            device=device
        )
        
        jump_gp_results.append(mu_t)
        jump_gp_res_sig.append(torch.sqrt(sig2_t))
    
    # 计算运行时间
    run_time = time.time() - start_time
    
    # 转换预测结果为tensor
    predictions = torch.tensor([x.detach().item() for x in jump_gp_results])
    sigmas = torch.tensor([s.detach().item() for s in jump_gp_res_sig])
    
    # 计算评估指标
    rmse, q25, q50, q75 = compute_metrics(predictions, sigmas, Y_test)
    
    return [rmse, q25, q50, q75, run_time]

def main():
    # 解析命令行参数
    args = parse_args()
    device = torch.device(args.device)
    
    # 加载数据集
    dataset = load_dataset(args.folder_name)
    X_train = dataset["X_train"]
    Y_train = dataset["Y_train"]
    X_test = dataset["X_test"]
    Y_test = dataset["Y_test"]

    # 如果使用SIR，先进行降维
    if args.use_sir:
        print("Applying SIR dimension reduction...")
        X_train, X_test = apply_sir_reduction(X_train, Y_train, X_test, args)
        print(f"Data dimensionality reduced to {X_train.shape[1]}")
    
    # 找到邻域
    neighborhoods = find_neighborhoods(X_test, X_train, Y_train, args.M)
    
    # 评估模型并获取结果
    result = evaluate_jumpgp(X_test, Y_test, neighborhoods, device)
    
    # 打印结果
    print(f"RMSE: {result[0]:.4f}")
    print(f"NLPD Q25: {result[1]:.4f}")
    print(f"NLPD Q50: {result[2]:.4f}")
    print(f"NLPD Q75: {result[3]:.4f}")
    print(f"Runtime: {result[4]:.2f} seconds")
    
    return result

if __name__ == "__main__":
    main()