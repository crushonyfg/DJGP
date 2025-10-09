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
from AE import AE_dim

# 添加父级目录到系统路径
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_dir = os.getcwd()
# 添加父级目录到系统路径
sys.path.append(os.path.dirname(current_dir))

# 然后导入 JumpGP
from JumpGaussianProcess.jumpgp import JumpGP

use_jgp_sir = False
use_jgp_pca = False
use_jgp = False
use_lmjgp = True
use_dgp = False

from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import numpy as np

import pickle
from datetime import datetime
import random

total_res = {} 
Erosion_res = {}
num_exp = 10 
encoder_structures = [[64,32,5], [32,5], [16,5]]
encoder_idx = 1
encoder_layers = encoder_structures[encoder_idx]
dataset_candidates = ['Wine Quality', 'Parkinsons Telemonitoring', 'Appliances Energy Prediction']
# dataset_candidates = ['Wine Quality']
store_file = f'exp_data/AE_UCIdataset/encoder_{encoder_idx}_new'
os.makedirs(store_file, exist_ok=True)
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

        use_ae = True
        if use_ae:
            encoder_layers[-1] = K
            start_time = time.time()
            Z_train, Z_test, model = AE_dim(
                X_train, X_test,
                latent_dim=K,                 # or直接给 encoder_layers=[64,32,5]
                encoder_layers=encoder_layers,          # 若为 None,默认用 [64,32,latent_dim]
                epochs=100,
                patience=10,
                batch_size=128,
                val_ratio=0.1,
                lr=1e-3,
                weight_decay=1e-5,
                dropout=0.0,
                use_tqdm=True,                # 关掉进度条:False
                seed=42
            )
            Z_train_np, Z_test_np = Z_train.cpu().numpy(), Z_test.cpu().numpy()

            JGP = JumpGP(Z_train_np, y_train_np, Z_test_np, L=1, M=20, mode='CEM', bVerbose=False)
            JGP.fit()
            rmse, mean_crps = JGP.metrics(y_test_np)
            print(f"RMSE: {rmse}, Mean CRPS: {mean_crps}")
            end_time = time.time()
            run_time = end_time - start_time

            res['jgp_ae'] = [rmse, mean_crps, run_time]

        print(res)

        total_res[dataset_name][seed] = res
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'AE_UCIdataset_res_{dataset_name}_{timestamp}.pkl'

        # 保存数据
        # with open(filename, 'wb') as f:
        #     pickle.dump(total_res, f)

filename = f'{store_file}/AE_UCIdataset.pkl'

# 保存数据
with open(filename, 'wb') as f:
    pickle.dump(total_res, f)









    

    

