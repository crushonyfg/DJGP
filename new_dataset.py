# Appliances Energy Prediction
from ucimlrepo import fetch_ucirepo 
import numpy as np
import torch

import sys
import os

import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from DJGP_test import *
from argparse import Namespace
from JumpGP_test import *
from DeepGP_test import *

# 添加父级目录到系统路径
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_dir = os.getcwd()
# 添加父级目录到系统路径
sys.path.append(os.path.dirname(current_dir))

# 然后导入 JumpGP
from JumpGaussianProcess.jumpgp import JumpGP

# ... existing code ...

from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import numpy as np

import pickle
from datetime import datetime
import random

def sir_reduction(X, y, H=10, K=5):
    """
    执行 SIR 降维
    
    参数:
    X: 输入特征矩阵
    y: 目标变量（需要确保是1维数组）
    H: 分箱数量
    K: 目标维度
    """
    # 确保y是1维数组
    y = np.asarray(y).ravel()
    
    # 标准化特征
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # 处理可能的无穷大和NaN值
    X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 使用percentile进行分箱
    percentiles = np.linspace(0, 100, H+1)[1:-1]
    bin_edges = np.percentile(y, percentiles)
    y_binned = np.digitize(y, bin_edges)
    
    # 计算每个箱子的均值和比例
    means = np.zeros((H, X.shape[1]))
    props = np.zeros(H)
    for h in range(H):
        idx = (y_binned == h)
        if np.any(idx):  # 确保该箱子有数据
            means[h] = np.nan_to_num(X_std[idx].mean(axis=0))
            props[h] = idx.mean()
    
    # 标准化props以确保和为1
    props = props / np.sum(props)
    
    # 计算切片协方差矩阵
    m = np.zeros_like(means[0])
    V = np.zeros((X.shape[1], X.shape[1]))
    for h in range(H):
        if props[h] > 0:  # 只处理非空箱子
            diff = means[h] - m
            V += props[h] * np.outer(diff, diff)
    
    # 添加小的正则化项以确保数值稳定性
    V += 1e-8 * np.eye(V.shape[0])
    
    # 求解特征值问题
    try:
        eigvals, eigvecs = eigh(V)
    except np.linalg.LinAlgError:
        # 如果还是失败，使用SVD作为备选方案
        _, eigvecs = np.linalg.svd(V)[:2]
        eigvecs = eigvecs.T
    
    # 选择前K个特征向量
    selected_vecs = eigvecs[:, -K:]
    
    # 转换数据
    X_transformed = X_std @ selected_vecs
    
    return X_transformed, selected_vecs, scaler


# ... rest of your existing code ...
total_res = {} 
num_exp = 10 
dataset_candidates = ['Wine Quality', 'Parkinsons Telemonitoring', 'Appliances Energy Prediction']
# num_exp = 1 
# dataset_candidates = ['Wine Quality']
for dataset_name in dataset_candidates:
    total_res[dataset_name] = {}
    for seed in range(num_exp):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        if dataset_name == 'Appliances Energy Prediction':
            # fetch dataset 
            appliances_energy_prediction = fetch_ucirepo(id=374) 
            
            # data (as pandas dataframes) 
            X = appliances_energy_prediction.data.features 
            y = appliances_energy_prediction.data.targets 
            
            # metadata 
            print(appliances_energy_prediction.metadata) 
            
            # variable information 
            print(appliances_energy_prediction.variables) 

            X_np = X.iloc[:, 1:].to_numpy()
            y_np = y.to_numpy()

            # np.random.seed(seed)
            indices = np.random.permutation(X_np.shape[0])
            split = int(X_np.shape[0] * 0.97)

            train_idx, test_idx = indices[:split], indices[split:]
            X_train_np, X_test_np = X_np[train_idx], X_np[test_idx]
            y_train_np, y_test_np = y_np[train_idx], y_np[test_idx]

            X_train = torch.from_numpy(X_train_np).float()
            Y_train = torch.from_numpy(y_train_np).float()
            X_test  = torch.from_numpy(X_test_np).float()
            Y_test  = torch.from_numpy(y_test_np).float()

            # 输出形状验证
            print("X_train:", X_train.shape)
            print("y_train:", Y_train.shape)
            print("X_test: ", X_test.shape)
            print("y_test: ", Y_test.shape)

            K = 5
        elif dataset_name == 'Parkinsons Telemonitoring':
            # Parkinsons Telemonitoring
            # fetch dataset 
            parkinsons_telemonitoring = fetch_ucirepo(id=189) 
            
            # data (as pandas dataframes) 
            X = parkinsons_telemonitoring.data.features 
            y = parkinsons_telemonitoring.data.targets 
            
            # metadata 
            print(parkinsons_telemonitoring.metadata) 
            
            # variable information 
            print(parkinsons_telemonitoring.variables) 
            
            X_np = X.to_numpy()
            y_np = y.to_numpy()
            indices = np.random.permutation(X_np.shape[0])
            split = int(X_np.shape[0] * 0.9)

            train_idx, test_idx = indices[:split], indices[split:]
            X_train_np, X_test_np = X_np[train_idx,:-1], X_np[test_idx,:-1]
            y_train_np, y_test_np = y_np[train_idx, 0], y_np[test_idx, 0]

            X_train = torch.from_numpy(X_train_np).float()
            Y_train = torch.from_numpy(y_train_np).float()
            X_test  = torch.from_numpy(X_test_np).float()
            Y_test  = torch.from_numpy(y_test_np).float()

            # 输出形状验证
            print("X_train:", X_train.shape)
            print("y_train:", Y_train.shape)
            print("X_test: ", X_test.shape)
            print("y_test: ", Y_test.shape)
            K = 5
        elif dataset_name == 'Wine Quality':
            wine_quality = fetch_ucirepo(id=186) 
        
            # data (as pandas dataframes) 
            X = wine_quality.data.features 
            y = wine_quality.data.targets 
            
            # metadata 
            print(wine_quality.metadata) 
            
            # variable information 
            print(wine_quality.variables) 

            # np.random.seed(seed)
            X_np = X.to_numpy()
            y_np = y.to_numpy()
            indices = np.random.permutation(X_np.shape[0])
            split = int(X_np.shape[0] * 0.9)

            train_idx, test_idx = indices[:split], indices[split:]
            X_train_np, X_test_np = X_np[train_idx,:], X_np[test_idx,:]
            y_train_np, y_test_np = y_np[train_idx, 0], y_np[test_idx, 0]

            X_train = torch.from_numpy(X_train_np).float()
            Y_train = torch.from_numpy(y_train_np).float()
            X_test  = torch.from_numpy(X_test_np).float()
            Y_test  = torch.from_numpy(y_test_np).float()

            # 输出形状验证
            print("X_train:", X_train.shape)
            print("y_train:", Y_train.shape)
            print("X_test: ", X_test.shape)
            print("y_test: ", Y_test.shape)
            K = 3

        res = {}
        H = 10  # 分箱数量

        # 对训练集和测试集应用SIR
        start_time = time.time()
        X_train_sir, sir_vecs, scaler = sir_reduction(X_train_np, y_train_np, H=H, K=K)
        X_test_sir = scaler.transform(X_test_np) @ sir_vecs

        JGP = JumpGP(X_train_sir, y_train_np, X_test_sir, L=1, M=20, mode='CEM', bVerbose=False)
        JGP.fit()
        rmse, mean_crps = JGP.metrics(y_test_np)
        print(f"RMSE: {rmse}, Mean CRPS: {mean_crps}")
        end_time = time.time()
        run_time = end_time - start_time

        res['jgp_sir'] = [rmse, mean_crps, run_time]

        start_time = time.time()
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train_np)
        X_test_std = scaler.transform(X_test_np)

        # 2. PCA降维
        n_components = K  # 降到5维
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_std)
        X_test_pca = pca.transform(X_test_std)

        JGP = JumpGP(X_train_pca, y_train_np, X_test_pca, L=1, M=20, mode='CEM', bVerbose=False)
        JGP.fit()
        rmse, mean_crps = JGP.metrics(y_test_np)
        print(f"RMSE: {rmse}, Mean CRPS: {mean_crps}")
        end_time = time.time()
        run_time = end_time - start_time

        res['jgp_pca'] = [rmse, mean_crps, run_time]

        start_time = time.time()
        JGP = JumpGP(X_train_np, y_train_np, X_test_np, L=1, M=20, mode='CEM', bVerbose=False)
        JGP.fit()
        rmse, mean_crps = JGP.metrics(y_test_np)
        print(f"RMSE: {rmse}, Mean CRPS: {mean_crps}")
        end_time = time.time()
        run_time = end_time - start_time

        res['jgp_raw'] = [rmse, mean_crps, run_time]

        args = Namespace(
            epochs=100,           # 训练轮数
            num_epochs=300,
            batch_size=1024,
            hidden_dim=2,
            lr=0.01,
            print_freq=10,        # 每多少个 epoch 打印一次信息
            # save_path="./model",  # 模型保存路径（可选）
            Q = K,
            m1 = 2,
            m2 = 40,
            n = 20,
            num_steps = 500,
            MC_num = 3,
            eval=True,            # 是否评估模型性能
            patience=10,          # 提前停止的耐心轮数
            clip_grad=1.0,        # 梯度裁剪（可选）
            # 如果你的 train_model 用了更多字段，也可以在这里继续加
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train = X_train.float().to(device)
        Y_train = Y_train.float().squeeze(-1).to(device)
        X_test = X_test.float().to(device)
        Y_test = Y_test.float().squeeze(-1).to(device)

        N_test, D = X_test.shape
        T         = N_test
        Q         = args.Q
        m1        = args.m1
        m2        = args.m2
        n         = args.n
        num_steps = args.num_steps
        MC_num    = args.MC_num

        start_time = time.time()
        # 4. Build neighborhoods (on CPU) then move to GPU
        neighborhoods = find_neighborhoods(
            X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=n
        )
        regions = []
        start_time = time.time()
        for i in range(T):
            X_nb = neighborhoods[i]['X_neighbors'].to(device)  # (n, D)
            y_nb = neighborhoods[i]['y_neighbors'].to(device)  # (n,)
            regions.append({
                'X': X_nb,
                'y': y_nb,
                'C': torch.randn(m1, Q, device=device)       # random init
            })

        # 5. Initialize V_params on GPU
        V_params = {
            'mu_V':    torch.randn(m2, Q, D, device=device, requires_grad=True),
            'sigma_V': torch.rand( m2, Q, D, device=device, requires_grad=True),
        }

        # 6. Initialize u_params on GPU
        u_params = []
        for _ in range(T):
            u_params.append({
                'U_logit':    torch.zeros(1, device=device, requires_grad=True),
                'mu_u':       torch.randn(m1, device=device, requires_grad=True),
                'Sigma_u':    torch.eye(m1, device=device, requires_grad=True),
                'sigma_noise':torch.tensor(0.5, device=device, requires_grad=True),
                'sigma_k':torch.tensor(0.5, device=device, requires_grad=True),
                'omega':      torch.randn(Q+1, device=device, requires_grad=True),
            })

        # 7. Initialize hyperparams on GPU
        X_train_mean = X_train.mean(dim=0)
        X_train_std  = X_train.std(dim=0)
        Z = X_train_mean + torch.randn(m2, D, device=device) * X_train_std

        hyperparams = {
            'Z':            Z,                      # (m2, D)
            'X_test':       X_test,                 # (T, D)
            'lengthscales':torch.rand(Q, device=device, requires_grad=True),
            'var_w':        torch.tensor(1.0, device=device, requires_grad=True),
        }

        print("Everything set!")

        # 8. Compute ELBO, backprop, train, predict
        # L = compute_ELBO(regions, V_params, u_params, hyperparams)
        # print("ELBO L =", L.item())
        # L.backward()
        # print("Gradients OK")

        V_params, u_params, hyperparams = train_vi(
            regions=regions,
            V_params=V_params,
            u_params=u_params,
            hyperparams=hyperparams,
            lr=args.lr,
            num_steps=num_steps,
            log_interval=50
        )
        print("train OK")

        mu_pred, var_pred = predict_vi(
            regions, V_params, hyperparams, M=MC_num
        )
        print("Prediction OK")
        # print("mu_pred:", mu_pred.shape)
        # print("var_pred:", var_pred.shape)

        # 9. Compute metrics & runtime
        run_time = time.time() - start_time
        # rmse, q25, q50, q75 = compute_metrics(mu_pred, var_pred, Y_test)
        # print("Results [rmse, 25%, 50%, 75%, runtime]:")
        # print([rmse, q25, q50, q75, run_time])

        # return [rmse, q25, q50, q75, run_time]
        sigmas = torch.sqrt(var_pred)
        rmse, mean_crps = compute_metrics(mu_pred, sigmas, Y_test)
        print(f"Results [rmse, mean crps, runtime]:{[rmse, mean_crps]}")
        end_time = time.time()
        run_time = end_time - start_time

        res['lmjgp'] = [rmse, mean_crps, run_time]

        args = Namespace(
            epochs=100,           # 训练轮数
            num_epochs=300,
            batch_size=128,
            hidden_dim=10,
            lr=0.01,
            print_freq=10,        # 每多少个 epoch 打印一次信息
            # save_path="./model",  # 模型保存路径（可选）
            eval=True,            # 是否评估模型性能
            patience=30,          # 提前停止的耐心轮数
            clip_grad=1.0,        # 梯度裁剪（可选）
            # 如果你的 train_model 用了更多字段，也可以在这里继续加
        )

        # X_train = X_train.float().to(device)
        # Y_train = Y_train.float().squeeze(-1).to(device)
        # X_test = X_test.float().to(device)
        # Y_test = Y_test.float().squeeze(-1).to(device)

        # 创建数据加载器
        batch_size, hidden_dim, lr = 128, 2, 0.01
        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # 初始化模型
        model = DeepGP(X_train.shape, hidden_dim).to(device)

        # 初始化优化器和损失函数
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr)
        mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, X_train.shape[-2]))

        # 训练模型
        final_metrics = train_model(model, train_loader, test_loader, optimizer, mll, args)


        res['djgp'] = final_metrics
        print(res)

        total_res[dataset_name][seed] = res
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'total_res_{timestamp}.pkl'

        # 保存数据
        with open(filename, 'wb') as f:
            pickle.dump(total_res, f)


filename = 'total_res_finalversion.pkl'

# 保存数据
with open(filename, 'wb') as f:
    pickle.dump(total_res, f)









    

    

