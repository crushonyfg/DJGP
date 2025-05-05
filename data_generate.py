# python data_generate.py --N 1000 --T 200 --D 15 --caseno 5 --device cuda
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import math
import numpy as np
import argparse
import pickle
from datetime import datetime
from skimage import io, color, transform

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic data for high-dimensional GP')
    parser.add_argument('--N', type=int, default=5000, help='Number of training points')
    parser.add_argument('--T', type=int, default=100, help='Number of test points')
    parser.add_argument('--D', type=int, default=10, help='Dimension of observation space')
    parser.add_argument('--latent_dim', type=int, default=2, help='Dimension of latent space')
    parser.add_argument('--Q', type=int, default=2, help='Number of rows in A(x) matrix')
    parser.add_argument('--caseno', type=int, default=4, help='Case number for boundary generation (4,5,6 for phantom boundary)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    return parser.parse_args()

def generate_Y_from_image(z, caseno, noise_std=0.1):
    """
    根据图像边界生成目标值 Y。
    参数：
      z: 2D 隐变量，形状 (num, 2)
      caseno: 4,5,6 对应不同边界图像
      noise_std: 噪声标准差
    返回：
      Y: 目标值，形状 (num,)
    """
    # 读取边界图像（请修改为你实际的文件路径）
    image_file = {4: 'bound3.png',
                  5: 'bound1.png',
                  6: 'bound4.png'}[caseno]
    I = io.imread(image_file)
    # 转为灰度，并二值化：True 表示边界内部
    bw = color.rgb2gray(I) > 0.5
    # 调整尺寸：将图像尺寸调整为 (G, G)，这里 G 与希望的网格分辨率有关
    G = 100
    bw = transform.resize(bw, (G, G), anti_aliasing=False)
    # 将 bw 转为二值图
    bw = bw > 0.5
    # 构造网格：这里假设隐变量 z 的范围约在 [-3, 3]（可根据数据调整）
    gx = np.linspace(-3, 3, G)
    gy = np.linspace(-3, 3, G)
    # 注意：bw 的行对应 y 方向，列对应 x 方向
    Y_out = torch.empty(z.shape[0], device=z.device, dtype=z.dtype)
    z_np = z.cpu().numpy()  # 转为 numpy 便于索引
    for i in range(z.shape[0]):
        z0, z1 = z_np[i]
        # 找到最近邻网格索引
        idx0 = np.argmin(np.abs(gx - z0))
        idx1 = np.argmin(np.abs(gy - z1))
        # 如果该点在边界内部，则赋予高均值，否则低均值
        if bw[idx1, idx0]:
            Y_out[i] = 5.0 + noise_std * torch.randn(1, device=z.device)
        else:
            Y_out[i] = -5.0 + noise_std * torch.randn(1, device=z.device)
            # Y_out[i] = 0 + noise_std * torch.randn(1, device=z.device)
    return Y_out

def generate_Y(z, noise_std=0.1, caseno=None):
    """
    如果 caseno 为 4、5 或 6，则使用图像边界；否则，使用简单规则（如 z[0]+z[1]>0）。
    """
    if caseno in [4, 5, 6]:
        return generate_Y_from_image(z, caseno, noise_std)
    else:
        Y = torch.empty(z.shape[0], device=z.device, dtype=z.dtype)
        for i in range(z.shape[0]):
            if z[i, 0] + z[i, 1] > 0:
                Y[i] = 5.0 + noise_std * torch.randn(1, device=z.device)
            else:
                Y[i] = -5.0 + noise_std * torch.randn(1, device=z.device)
                # Y[i] = 0 + noise_std * torch.randn(1, device=z.device)
        return Y

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

def generate_data(args):
    """
    Generate synthetic data based on input arguments.
    """
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)

    # Generate input X_train and X_test
    X_train = torch.randn(args.N, args.D, device=device)
    X_test = torch.randn(args.T, args.D, device=device)
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
    A_all = torch.empty(N_all, args.Q, args.D, device=device)
    mean_zero = torch.zeros(N_all, device=device)
    for q in range(args.Q):
        for d in range(args.D):
            mvn = torch.distributions.MultivariateNormal(mean_zero, covariance_matrix=K_A)
            A_all[:, q, d] = mvn.sample()

    # 对A进行正则化
    A_all = normalize_A(A_all)

    # Split A into train and test
    A_train = A_all[:args.N]
    A_test = A_all[args.N:]

    # Compute latent variables z(x)
    z_train = torch.bmm(A_train, X_train.unsqueeze(-1)).squeeze(-1)
    z_test = torch.bmm(A_test, X_test.unsqueeze(-1)).squeeze(-1)

    # Generate target variables
    # Y_train = generate_Y(z_train, noise_std=0.1, caseno=args.caseno)
    # Y_test = generate_Y(z_test, noise_std=0.1, caseno=args.caseno)
    Y_train = generate_Y(z_train, noise_std=1, caseno=args.caseno)
    Y_test = generate_Y(z_test, noise_std=1, caseno=args.caseno)

    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_test": X_test, "Y_test": Y_test,
        "hyperparams": {"sigma_a_A": sigma_a_A.item(), "sigma_q_A": sigma_q_A.item()}
    }

import torch
import math

def random_fourier_features(X, M, sigma):
    """
    X: (N, D)
    M: 目标特征维度
    sigma: RBF 核宽度参数
    返回 Z: (N, M)
    """
    N, D = X.shape
    # W ~ N(0, I/σ²), b ~ Uniform(0, 2π)
    W = torch.randn(D, M, device=X.device) / sigma
    b = 2 * math.pi * torch.rand(M, device=X.device)
    Z = math.sqrt(2.0 / M) * torch.cos(X @ W + b)
    return Z

def generate_data1(args):
    """
    利用随机傅里叶特征（RFF）将低维 X 非线性映射到高维 Z，再生成 Y。
    新增 args.M 和 args.sigma 两个字段。
    """
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)

    # 1. 生成原始输入 X
    X_train = torch.randn(args.N, args.D, device=device)
    X_test  = torch.randn(args.T, args.D, device=device)
    X_all   = torch.cat([X_train, X_test], dim=0)  # (N+T, D)

    # 2. 随机傅里叶特征映射到高维 Z_all
    #    M: 高维特征数，sigma: 控制核宽度
    Z_all = random_fourier_features(X_all, M=2, sigma=1)  # (N+T, M)

    # 3. 切分 Z 为训练/测试
    Z_train = Z_all[:args.N]     # (N, M)
    Z_test  = Z_all[args.N:]     # (T, M)

    # 4. 根据高维特征 Z 生成目标 Y
    Y_train = generate_Y(Z_train, noise_std=1.0, caseno=args.caseno)
    Y_test  = generate_Y(Z_test,  noise_std=1.0, caseno=args.caseno)

    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_test":  X_test,  "Y_test":  Y_test,
        "hyperparams": {"M": 2, "sigma": 1}
    }


def save_dataset(data, args, folder_name=None):
    """
    Save the generated dataset and arguments to files.
    Args:
        data: Dictionary containing the dataset
        args: Namespace object containing command line arguments
        folder_name: Optional folder name for saving
    """
    if folder_name is None:
        folder_name = datetime.now().strftime("%Y_%m_%d_%H_%M")
    os.makedirs(folder_name, exist_ok=True)
    
    # 将 args 转换为字典并添加到 data 中
    args_dict = vars(args)
    data['args'] = args_dict
    
    # 保存数据集
    with open(os.path.join(folder_name, 'dataset.pkl'), 'wb') as f:
        pickle.dump(data, f)

    return folder_name

def main():
    args = parse_args()
    data = generate_data1(args)
    
    # Print shapes and hyperparameters
    print("X_train shape:", data["X_train"].shape)
    print("Y_train shape:", data["Y_train"].shape)
    print("X_test shape:", data["X_test"].shape)
    print("Y_test shape:", data["Y_test"].shape)
    # print("\nSampled A-GP hyperparameters:")
    # print(" sigma_a_A =", data["hyperparams"]["sigma_a_A"])
    # print(" sigma_q_A =", data["hyperparams"]["sigma_q_A"])
    print("\nArguments:")
    for k, v in vars(args).items():
        print(f" {k} = {v}")
    
    # Save the dataset with arguments
    folder_name = save_dataset(data, args)
    return folder_name

if __name__ == "__main__":
    main()