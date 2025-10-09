import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from importlib import import_module
import sys
import torch   # 用于检测 GPU tensor

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
    # 实验设置
    L = 10   # 不同数据集生成次数
    T = 1   # DeepGP/DJGP/SIR 循环次数
    N = 1000
    T_param = 100
    D = 20
    caseno = 5
    M = 25
    Q = 5
    K = 3
    use_cv = False
    noise_var = 4
    # use_cv = True
    
    results = {}

    # 动态导入模块
    if Q>2:
    # data_generate = import_module('data_generate')
        data_generate = import_module('new_highdata_gen')
    else:
        data_generate = import_module('data_generate')
    JumpGP_test   = import_module('JumpGP_test_CV')
    JumpGP_test_local   = import_module('JumpGP_test_local')
    DeepGP_test   = import_module('DeepGP_test')
    if use_cv: 
        DJGP_test     = import_module('DJGP_CV')
    else:
        DJGP_test     = import_module('DJGP_test')   # 新增 DJGP
    # DJGP_test     = import_module('DJGP_CV')
    NNJGP_test    = import_module('NNJGP_test')  
    BNNJGP_test   = import_module('BNNJGP')
    SIR_GP        = import_module('SIR_GP')

    for i in range(L):
        print(f"\nProcessing iteration {i+1}/{L}")
        res_i = {}
        
        # 1. 生成数据
        if Q==2:
            args_data = {
                'N': N,
                'T': T_param,
                'D': D,
                'caseno': caseno,
                'device': 'cpu',
                'latent_dim': 2,
                'ins_dim': True,
                'Q': Q
            }
            if Q==D:
                args_data['ins_dim'] = False
        else:
            args_data = {
                'N': N,
                'Nt': T_param,
                'H': D,
                'd': Q,
                'device': 'cpu',
                'noise_var': noise_var
            }
        folder_name = run_with_args(data_generate, args_data)
        print(f"Data generated in folder: {folder_name}")
        
        # 3. JumpGP + SIR
        args_jump_sir = {
            'folder_name': folder_name,
            'M': M,
            'device': 'cpu',
            'use_sir': True,
            'use_cv': True,
            'sir_H': D,
            'sir_K': K
        }
        res_i['JumpGPsirGlobal'] = run_with_args(JumpGP_test, args_jump_sir)
        print("JumpGP with SIR completed")

        # 4. DeepGP, DJGP, SIR_GP 多次迭代
        for j in range(T):
            print(f"  Iteration {j+1}/{T} for {folder_name}")
            
            # DeepGP
            args_deep = {
                'folder_name': folder_name,
                'hidden_dim': K,
                'num_epochs': 500,
                'patience': 5,
                'batch_size': 1024,
                'lr': 0.01
            }
            res_i[f'DeepGP_{j}'] = run_with_args(DeepGP_test, args_deep)
            print(f"    DeepGP iteration {j+1} completed")
            
            # DJGP
            args_djgp = {
                'folder_name': folder_name,
                'num_steps': 600,
                'n': M,
                'Q': K,
                # 'MC_num': 5,
                'MC_num': 5,
                'm2': 20,
                'm1': 5,
                'lr': 0.01
            }
            res_i[f'DJGP_{j}'] = run_with_args(DJGP_test, args_djgp)
            print(f"    DJGP iteration {j+1} completed")


        folder_key = str(i+1)
        print(f"Storing results under key: {folder_key}")
        results[folder_key] = res_i

    # 保存并绘图
    save_results(results)
    plot_metrics_boxplot(results, 'metrics_with_boxplot.png')
    print("\nExperiment completed successfully.")

if __name__ == "__main__":
    main()
