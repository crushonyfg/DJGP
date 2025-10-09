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
import numpy as np
import torch

import sys
import os

import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from DJGP_test_validation import *
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

use_lmjgp = True

motivation = 'see_K_influence'

from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import numpy as np

import pickle
from datetime import datetime
import random

def save_results(results, filename='results.pkl'):
    """保存结果到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")

import matplotlib.pyplot as plt
import torch  # 确保能检测到 torch.Tensor

def plot_metrics(results):
    """绘制 RMSE 和 CRPS vs Runtime，标注 folder index，legend 去重只保留 base 方法名"""
    # 只要两个指标
    metrics = ['rmse', 'crps']
    # 一行两列
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes = axes.ravel()
    
    colors = {
        'JumpGP':     'blue',
        'JumpGPsirGlobal':  'red',
        'DeepGP':     'green',
        'DJGP':       'cyan',
        'JumpGPsirLocal':      'purple',
        'GP':         'orange',
        'NNJGP':       'magenta',   # 新增
        'BNNJGP':      'brown',
    }
    markers = {
        'JumpGP':     'o',
        'JumpGPsirGlobal':  's',
        'DeepGP':     '^',
        'DJGP':       '*',
        'JumpGPsirLocal':      'D',
        'GP':         'v',
        'NNJGP':       'X',         # 新增
        'BNNJGP':      'o',
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for folder_idx, (folder_key, res_i) in enumerate(results.items(), start=1):
            for method, values in res_i.items():
                # 假设 values = [rmse, crps, run_time]
                raw_x = values[2]  # run_time
                raw_y = values[idx]  # rmse 或 crps
                
                x = raw_x.detach().cpu().item() if isinstance(raw_x, torch.Tensor) else float(raw_x)
                y = raw_y.detach().cpu().item() if isinstance(raw_y, torch.Tensor) else float(raw_y)
                
                base_label = method.split('_')[0]
                ax.scatter(
                    x, y,
                    label=base_label,
                    color=colors.get(base_label, 'black'),
                    marker=markers.get(base_label, 'x')
                )
                ax.annotate(
                    str(folder_idx),
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=9
                )
        
        ax.set_xlabel('Runtime (s)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Runtime')
        ax.grid(True)
        
        # 去重 legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(
            unique.values(),
            unique.keys(),
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Plot saved as metrics_comparison.png")



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
                # 处理列表参数，将每个元素作为单独的参数
                sys.argv.append(f'--{key}')
                for item in value:
                    sys.argv.append(str(item))
            else:
                sys.argv.append(f'--{key}')
                sys.argv.append(str(value))
        return module.main()
    finally:
        sys.argv = original_argv

def plot_metrics_with_mean(results: dict, output_file: str):
    """
    Plot RMSE and CRPS vs runtime, and on the RMSE subplot draw
    horizontal lines showing each method’s mean RMSE across runs.
    """
    metrics = ['rmse', 'crps']
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes = axes.ravel()

    colors = {
        'JumpGP':     'blue',
        'JumpGPsirGlobal':  'red',
        'DeepGP':     'green',
        'DJGP':       'cyan',
        'JumpGPsirLocal':      'purple',
        'GP':         'orange',
        'NNJGP':       'magenta',   # 新增
        'BNNJGP':      'brown',
    }
    markers = {
        'JumpGP':     'o',
        'JumpGPsirGlobal':  's',
        'DeepGP':     '^',
        'DJGP':       '*',
        'JumpGPsirLocal':      'D',
        'GP':         'v',
        'NNJGP':       'X',         # 新增
        'BNNJGP':      'o',
    }

    # 1) Compute mean RMSE per base method
    rmse_by_method = {}
    for run_res in results.values():
        for method, vals in run_res.items():
            base = method.split('_')[0]
            raw_rmse = vals[0]
            rmse = (raw_rmse.detach().cpu().item()
                    if isinstance(raw_rmse, torch.Tensor)
                    else float(raw_rmse))
            rmse_by_method.setdefault(base, []).append(rmse)
    mean_rmse = {base: np.mean(vals) for base, vals in rmse_by_method.items()}

    # 2) Plot scatter for each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for run_idx, run_res in enumerate(results.values(), start=1):
            for method, vals in run_res.items():
                base = method.split('_')[0]
                raw_time = vals[2]  # runtime is at index 2
                raw_score = vals[i] # rmse at 0, crps at 1
                x = (raw_time.detach().cpu().item()
                     if isinstance(raw_time, torch.Tensor)
                     else float(raw_time))
                y = (raw_score.detach().cpu().item()
                     if isinstance(raw_score, torch.Tensor)
                     else float(raw_score))
                ax.scatter(
                    x, y,
                    color=colors.get(base, 'black'),
                    marker=markers.get(base, 'x'),
                    label=base
                )
                ax.annotate(
                    str(run_idx),
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8
                )

        ax.set_xlabel('Runtime (s)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Runtime')
        ax.grid(True)

        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(
            unique.values(),
            unique.keys(),
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

        # 3) On RMSE subplot, draw horizontal mean lines
        if metric == 'rmse':
            for base, m in mean_rmse.items():
                ax.axhline(
                    y=m,
                    color=colors.get(base, 'black'),
                    linestyle='-',
                    linewidth=1.5,
                    label=f'{base} mean'
                )
            # Update legend to include mean lines
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(
                unique.values(),
                unique.keys(),
                bbox_to_anchor=(1.05, 1),
                loc='upper left'
            )

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plot with mean lines to {output_file}")

def plot_metrics_boxplot(results: dict, output_file: str):
    """
    对每个方法在 RMSE 和 CRPS 上绘制 boxplot，方法动态识别，
    不再假定事先知道有哪些 base。
    """
    metrics = ['rmse', 'crps']
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes = axes.ravel()

    # 预定义常见方法的颜色，未知方法使用 'gray'
    colors = {
        'JumpGP':    'blue',
        'JumpGPsirGlobal': 'red',
        'DeepGP':    'green',
        'LMJGP':      'cyan',
        'GPsir':     'purple',
        'GP':        'orange',
        'NNJGP':     'magenta',
        'BNNJGP':    'brown',
    }
    default_color = 'gray'

    # 动态收集每个 base 方法在所有 runs 上的 (rmse, crps)
    metric_by_method = {}
    for run_res in results.values():
        for method, vals in run_res.items():
            base = method.split('_')[0]
            # 将 DJGP 替换为 LMJGP
            if base == 'DJGP':
                base = 'LMJGP'
            # 提取指标值
            raw_rmse, raw_crps = vals[0], vals[1]
            rmse = (raw_rmse.detach().cpu().item()
                    if isinstance(raw_rmse, torch.Tensor)
                    else float(raw_rmse))
            crps = (raw_crps.detach().cpu().item()
                    if isinstance(raw_crps, torch.Tensor)
                    else float(raw_crps))
            metric_by_method.setdefault(base, []).append((rmse, crps))

    # 准备排序后的标签列表，保证图中方法顺序一致
    labels = sorted(metric_by_method.keys())

    # 对每个指标画 boxplot
    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = [
            [entry[i] for entry in metric_by_method[base]]
            for base in labels
        ]

        # 绘制 boxplot
        bp = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False
        )
        # 根据方法名上色
        for patch, base in zip(bp['boxes'], labels):
            color = colors.get(base, default_color)
            # patch.set_facecolor(color)
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
    # 参数解析
    parser = argparse.ArgumentParser(description='DeepMahalaJumpGP实验参数')
    parser.add_argument('--L', type=int, default=10, help='不同数据集生成次数')
    parser.add_argument('--T', type=int, default=1, help='DeepGP/DJGP/SIR 循环次数')
    parser.add_argument('--T_param', type=int, default=512, help='时间序列参数')
    parser.add_argument('--caseno', type=int, default=5, help='案例编号')
    parser.add_argument('--M', type=int, default=25, help='M参数')
    parser.add_argument('--K', type=int, default=5, help='K参数')
    parser.add_argument('--use_cv', action='store_true', help='是否使用交叉验证')
    parser.add_argument('--noise_var', type=float, default=4.0, help='噪声方差')
    # parser.add_argument('--use_module', nargs='+', default=["LMJGP", "JGP", "DGP"], help='要使用的模块列表')
    parser.add_argument('--use_module', nargs='+', default=["LMJGP"], help='要使用的模块列表')
    
    args = parser.parse_args()
    
    # 实验设置
    L = args.L
    T = args.T
    T_param = args.T_param
    caseno = args.caseno
    M = args.M
    K = args.K
    use_cv = args.use_cv
    noise_var = args.noise_var
    use_module = args.use_module
    # use_module = ["LMJGP"]

    # 动态导入模块
    if args.use_cv: 
        DJGP_test = import_module('DJGP_CV')
    else:
        DJGP_test = import_module('DJGP_test')   # 新增 DJGP
    
    JumpGP_test = import_module('JumpGP_test_CV')
    JumpGP_test_local = import_module('JumpGP_test_local')
    DeepGP_test = import_module('DeepGP_test')
    NNJGP_test = import_module('NNJGP_test')  
    BNNJGP_test = import_module('BNNJGP')
    SIR_GP = import_module('SIR_GP')

    # 参数组合循环
    # for D in [20, 30, 50, 100]:
    N = 3000
    # Q = 4
    # for D in [30, 50, 100]:
    #     for Q in [3, 5, 7]:
    #         for N in [1000, 3000, 5000]:
    # for Q, K in ((q, k) for q in [3, 5, 7] for k in [2, 3, 5, 7]):
    # for M, m1 in ((m, m1) for m in [15, 25, 35, 45] for m1 in [2, 4, 6]):
    # for N, Q in ((n, q) for n in [3000, 5000, 10000] for q in [4, 6]):
    for Q in [35]:
        # M, m1 = 35, 4
        M, Q = 35, 5
        m1, m2 = 4, 60
        D = 30
        # Q = 5
        K = 5
        store_file = f'exp_data/LH_hypertune_batch_{M}M_{m1}m1_{m2}m2_{Q}Q_{N}N_{D}D_{K}K'
        os.makedirs(store_file, exist_ok=True)
        # for method in ["random projection", "rff", "autoencoder", "polynomial"]:
        for method in ["rff"]:
            # for noise_var in [1, 4, 9]:
            for noise_var in [4]:
                print(f"\n开始处理参数组合: D={D}, Q={Q}, N={N}, K={K}, M={M}, m1={m1}")
                
                # 根据Q值动态导入数据生成模块
                if Q > 2:
                    data_generate = import_module('new_highdata_gen')
                else:
                    data_generate = import_module('data_generate')
                
                results = {}
                filename = f'{store_file}/results_method={method}.pkl'
                
                for i in range(L):
                    print(f"\nProcessing iteration {i+1}/{L} for D={D}, Q={Q}, N={N}, method={method}, noise_var={noise_var}")
                    args_data = {
                            'N': N,
                            'Nt': T_param,
                            'H': D,
                            'd': Q,
                            'device': 'cpu',
                            'noise_var': noise_var,
                            'methods': method
                        }
                    folder_name = run_with_args(data_generate, args_data)
                    print(f"Data generated in folder: {folder_name}")

                    res = {}
                    
                    # batch_size = 128
                    m1 = 4
                    # for m2 in [60]:
                    m2 = 60
                    # for lr in [0.001, 0.01, 0.05]:
                    for batch_size in [512, 256, 128, 64, 32]:
                        lr = 0.1
                        n = 35
                        # batch_size = 64
                        args = Namespace(
                            epochs=100,           # 训练轮数
                            num_epochs=300,
                            batch_size=batch_size,
                            hidden_dim=2,
                            lr=lr,
                            print_freq=10,        # 每多少个 epoch 打印一次信息
                            # save_path="./model",  # 模型保存路径（可选）
                            Q = K,
                            m1 = m1,
                            m2 = m2,
                            n = n,
                            num_steps = 200,
                            MC_num = 3,
                            eval=True,            # 是否评估模型性能
                            patience=10,          # 提前停止的耐心轮数
                            clip_grad=1.0,        # 梯度裁剪（可选）
                            use_batch = True
                        )
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                        dataset = load_dataset(folder_name)
                        X_train = dataset["X_train"].to(device)   # (N_train, D)
                        Y_train = dataset["Y_train"].to(device)
                        X_test  = dataset["X_test"].to(device)    # (N_test, D)
                        Y_test  = dataset["Y_test"].to(device)

                        # X_val = X_test
                        # Y_val = Y_test

                        N_test, D = X_test.shape
                        T         = N_test
                        # Q         = args.Q
                        # m1        = args.m1
                        # m2        = args.m2
                        # n         = args.n
                        # num_steps = args.num_steps
                        # MC_num    = args.MC_num
                        MC_num, num_steps = 5, 200

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
                        
                        if batch_size==T_param:
                            M, K, m1, m2 = 35, 5, 4, 60
                            args_djgp = {
                            'folder_name': folder_name,
                            'num_steps': 200,
                            'n': M,
                            'Q': K,
                            'MC_num': 5,
                            'm2': m2,
                            'm1': m1,
                            'lr': 0.01
                            }
                            res[batch_size] = run_with_args(DJGP_test, args_djgp)
                            print(f"full batch completed, rmse: {res[batch_size][0]}, mean_crps: {res[batch_size][1]}, run_time: {res[batch_size][2]}")
                        else:
                            V_params, u_params, hyperparams, training_log, (best_val_V, best_val_u, best_val_hyp) = train_vi_minibatch(
                                regions=regions,
                                V_params=V_params,
                                u_params=u_params,
                                hyperparams=hyperparams,
                                lr=args.lr,
                                num_steps=num_steps,
                                log_interval=50,
                                batch_size=batch_size,
                                # regions_data=regions_data
                            )
                            print("train OK")
                            print(training_log)

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

                            # mu_pred_val, var_pred_val = predict_vi(
                            #     regions, best_val_V, best_val_hyp, M=MC_num
                            # )

                            # sigmas_val = torch.sqrt(var_pred_val)
                            # rmse_val, mean_crps_val = compute_metrics(mu_pred_val, sigmas_val, Y_test)
                            # print(f"Results [rmse, mean crps, runtime]:{[rmse_val, mean_crps_val]}")

                            # res['lmjgp-val'] = [rmse_val, mean_crps_val, run_time]
                            # print(f"comparison between train and use_val: train: {[rmse, mean_crps]}, use_val: {[rmse_val, mean_crps_val]}")

                            # res['lmjgp'] = [rmse, mean_crps, run_time]
                            # lmjgp_res.append({'m1': m1, 'm2': m2, 'rmse': rmse, 'mean_crps': mean_crps, 'run_time': run_time, 'n': n, 'batch_size': batch_size, 'lr': lr, 'training_log': training_log, 'res_val': res['lmjgp-val']})

                            res[batch_size] = [rmse, mean_crps, run_time] 
                            print(f"batch_size: {batch_size}, rmse: {rmse}, mean_crps: {mean_crps}, run_time: {run_time}")                   
                    

                    folder_key = str(i+1)
                    print(f"Storing results under key: {folder_key}")
                    results[folder_key] = res

                # 保存当前参数组合的结果
                save_results(results, filename)
                # plot_metrics_boxplot(results, f'{store_file}/metrics_boxplot_D={D}_Q={Q}_N={N}_method={method}.png')
                print(f"参数组合 D={D}, Q={Q}, N={N}, batch_size={batch_size}, method={method} 实验完成")

    print("\n所有实验完成!")

if __name__ == "__main__":
    main()# Appliances Energy Prediction






    

    

