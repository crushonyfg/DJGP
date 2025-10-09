import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import pandas as pd

from ucimlrepo import fetch_ucirepo 
import numpy as np
import torch

import sys
import os

import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pickle
from datetime import datetime
import random

# 假设 X_train, Y_train 已定义为 PyTorch Tensor
def dataset_analysis(X_train, Y_train, X_test, Y_test):
    X_tr = X_train.numpy()
    y_tr = Y_train.numpy()
    X_va = X_test.numpy()
    y_va = Y_test.numpy()

    # 1. 用高斯过程回归拟合残差，估计噪声方差
    # kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    # gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    # gpr.fit(X_np, y_np)
    # y_pred, _ = gpr.predict(X_np, return_std=True)

    # residuals = y_np - y_pred
    # sigma2_hat = np.var(residuals)
    # signal_var = np.var(y_pred)
    # snr = signal_var / sigma2_hat

    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(X_tr, y_tr)

    # 2. 在验证集上预测并获取预测标准差
    y_pred_va, _ = gpr.predict(X_va, return_std=True)

    # 3. 噪声方差和信号方差
    sigma2_hat = gpr.kernel_.k2.noise_level
    signal_var = np.var(y_pred_va)
    snr = signal_var / sigma2_hat

    # 2. kNN 图上的局部梯度幅度
    nbrs = NearestNeighbors(n_neighbors=6).fit(X_np)
    _, indices = nbrs.kneighbors(X_np)
    grads = []
    for i in range(len(X_np)):
        for j in indices[i][1:]:
            dist = np.linalg.norm(X_np[i] - X_np[j])
            if dist > 0:
                grads.append(abs(y_np[i] - y_np[j]) / dist)
    grads = np.array(grads)
    G = grads.mean()
    L = grads.max()

    # 3. 二阶差分总变分 TV2
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_np).flatten()
    order = np.argsort(pc1)
    y_sorted = y_np[order]
    tv2 = np.sum(np.abs((y_sorted[2:] - y_sorted[1:-1]) - (y_sorted[1:-1] - y_sorted[:-2])))

    # 整理并打印
    metrics = pd.DataFrame({
        'sigma2_hat': [sigma2_hat],
        'snr': [snr],
        'avg_grad (G)': [G],
        'max_grad (L)': [L],
        'tv2': [tv2]
    })

    print(metrics)
    return metrics

dataset_candidates = ['Wine Quality', 'Parkinsons Telemonitoring', 'Appliances Energy Prediction']
analysis_res = {}
num_exp = 1 
# dataset_candidates = ['Wine Quality']
for dataset_name in dataset_candidates:
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
            split = int(X_np.shape[0] * 0.3)
            test_split = int(X_np.shape[0] * 0.4)

            train_idx, test_idx = indices[:split], indices[split:test_split]
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

        metrics = dataset_analysis(X_train, Y_train, X_test, Y_test)
        analysis_res[dataset_name] = metrics

import pickle
with open('UCI_dataset_analysis_res.pkl', 'wb') as f:
    pickle.dump(analysis_res, f)