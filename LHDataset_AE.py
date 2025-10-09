import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from importlib import import_module
import sys
import torch   # 用于检测 GPU tensor

from ucimlrepo import fetch_ucirepo 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
from datetime import datetime
import random

from DJGP_test_validation import *
from argparse import Namespace
from JumpGP_test import *
from DeepGP_test import *
from AE import AE_dim

# 添加父级目录到系统路径
current_dir = os.getcwd()
sys.path.append(os.path.dirname(current_dir))

# 然后导入 JumpGP
from JumpGaussianProcess.jumpgp import JumpGP

use_lmjgp = True
motivation = 'see_K_influence'


def save_results(results, filename='results.pkl'):
    """保存结果到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")


def run_with_args(module, args_dict):
    """使用模拟命令行参数运行模块的 main()"""
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        for key, value in args_dict.items():
            if isinstance(value, bool):
                if value:
                    sys.argv.append(f'--{key}')
            elif isinstance(value, list):
                sys.argv.append(f'--{key}')
                for item in value:
                    sys.argv.append(str(item))
            else:
                sys.argv.append(f'--{key}')
                sys.argv.append(str(value))
        return module.main()
    finally:
        sys.argv = original_argv


def plot_metrics_boxplot(results: dict, output_file: str):
    """
    对每个方法在 RMSE 和 CRPS 上绘制 boxplot
    """
    metrics = ['rmse', 'crps']
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes = axes.ravel()

    colors = {
        'JumpGP':    'blue',
        'JumpGPsirGlobal': 'red',
        'DeepGP':    'green',
        'LMJGP':     'cyan',
        'GPsir':     'purple',
        'GP':        'orange',
        'NNJGP':     'magenta',
        'BNNJGP':    'brown',
    }
    default_color = 'gray'

    metric_by_method = {}
    for run_res in results.values():
        for method, vals in run_res.items():
            base = method.split('_')[0]
            if base == 'DJGP':
                base = 'LMJGP'
            raw_rmse, raw_crps = vals[0], vals[1]
            rmse = raw_rmse.detach().cpu().item() if isinstance(raw_rmse, torch.Tensor) else float(raw_rmse)
            crps = raw_crps.detach().cpu().item() if isinstance(raw_crps, torch.Tensor) else float(raw_crps)
            metric_by_method.setdefault(base, []).append((rmse, crps))

    labels = sorted(metric_by_method.keys())

    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = [[entry[i] for entry in metric_by_method[base]] for base in labels]
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch, base in zip(bp['boxes'], labels):
            patch.set_facecolor('gray')
            patch.set_edgecolor('black')

        ax.set_title(f'{metric.upper()} by Method')
        ax.set_ylabel(metric.upper())
        ax.grid(True, axis='y')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved boxplot to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='DeepMahalaJumpGP实验参数')
    parser.add_argument('--L', type=int, default=10, help='不同数据集生成次数')
    parser.add_argument('--T', type=int, default=1, help='DeepGP/DJGP/SIR 循环次数')
    parser.add_argument('--T_param', type=int, default=512, help='时间序列参数')
    parser.add_argument('--caseno', type=int, default=5, help='案例编号')
    parser.add_argument('--M', type=int, default=25, help='M参数')
    parser.add_argument('--K', type=int, default=5, help='K参数')
    parser.add_argument('--use_cv', action='store_true', help='是否使用交叉验证')
    parser.add_argument('--noise_var', type=float, default=4.0, help='噪声方差')
    parser.add_argument('--use_module', nargs='+', default=["LMJGP"], help='要使用的模块列表')
    
    args = parser.parse_args()
    L, T, T_param, caseno, M, K, use_cv, noise_var, use_module = (
        args.L, args.T, args.T_param, args.caseno, args.M, args.K, args.use_cv, args.noise_var, args.use_module
    )

    if args.use_cv: 
        DJGP_test = import_module('DJGP_CV')
    else:
        DJGP_test = import_module('DJGP_test')
    
    JumpGP_test = import_module('JumpGP_test_CV')
    JumpGP_test_local = import_module('JumpGP_test_local')
    DeepGP_test = import_module('DeepGP_test')
    NNJGP_test = import_module('NNJGP_test')  
    BNNJGP_test = import_module('BNNJGP')
    SIR_GP = import_module('SIR_GP')

    results = {}
    store_file = f'exp_data_ae'
    os.makedirs(store_file, exist_ok=True)

    for N, D, Q, M in [(1000, 20, 2, 25), (1000, 20, 4, 25), (1000, 30, 5, 25), (2000, 50, 7, 35)]:
        m1, m2 = 4, 60
        K = Q
        results[f'{N}N_{D}D_{Q}Q_{M}M'] = {}
        for method in ["random projection", "rff", "autoencoder", "polynomial"]:
            results[f'{N}N_{D}D_{Q}Q_{M}M'][method] = []
            for noise_var in [4]:
                print(f"\n开始处理参数组合: D={D}, Q={Q}, N={N}, K={K}, M={M}, m1={m1}")
                
                data_generate = import_module('new_highdata_gen')
                
                
                filename = f'{store_file}/results_method={method}.pkl'
                caseno = 4 if Q==2 else 0
                for i in range(L):
                    print(f"\nProcessing iteration {i+1}/{L} for D={D}, Q={Q}, N={N}, method={method}, noise_var={noise_var}")
                    args_data = {'N': N, 'Nt': T_param, 'H': D, 'd': Q, 'device': 'cpu', 'noise_var': noise_var, 'methods': method, "caseno": caseno, "seed": i}
                    folder_name = run_with_args(data_generate, args_data)
                    
                    print(f"Data generated in folder: {folder_name}")
                    
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    dataset = load_dataset(folder_name)  # 确保有定义
                    X_train, Y_train = dataset["X_train"].to(device), dataset["Y_train"].to(device)
                    X_test, Y_test = dataset["X_test"].to(device), dataset["Y_test"].to(device)

                    encoder_layers = [32, K]
                    start_time = time.time()
                    Z_train, Z_test, model = AE_dim(
                        X_train, X_test,
                        latent_dim=K,
                        encoder_layers=encoder_layers,
                        epochs=100,
                        patience=10,
                        batch_size=128,
                        val_ratio=0.1,
                        lr=1e-3,
                        weight_decay=1e-5,
                        dropout=0.0,
                        use_tqdm=True,
                        seed=42
                    )
                    Z_train_np, Z_test_np = Z_train.cpu().numpy(), Z_test.cpu().numpy()
                    y_train_np, y_test_np = Y_train.cpu().numpy(), Y_test.cpu().numpy()

                    JGP = JumpGP(Z_train_np, y_train_np, Z_test_np, L=1, M=20, mode='CEM', bVerbose=False)
                    JGP.fit()
                    rmse, mean_crps = JGP.metrics(y_test_np)
                    print(f"RMSE: {rmse}, Mean CRPS: {mean_crps}")
                    end_time = time.time()
                    run_time = end_time - start_time

                    results[f'{N}N_{D}D_{Q}Q_{M}M'][method].append([rmse, mean_crps, run_time])                  

    filename = f"{store_file}/results.pkl"
    save_results(results, filename)
    print("\n所有实验完成!")


if __name__ == "__main__":
    main()
