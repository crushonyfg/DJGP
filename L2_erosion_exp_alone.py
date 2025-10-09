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
    parser.add_argument('--T_param', type=int, default=100, help='时间序列参数')
    parser.add_argument('--caseno', type=int, default=5, help='案例编号')
    parser.add_argument('--M', type=int, default=25, help='M参数')
    parser.add_argument('--K', type=int, default=3, help='K参数')
    parser.add_argument('--use_cv', action='store_true', help='是否使用交叉验证')
    parser.add_argument('--noise_var', type=float, default=4.0, help='噪声方差')
    parser.add_argument('--use_module', nargs='+', default=["LMJGP", "JGP", "DGP"], help='要使用的模块列表')
    
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
    N = 1000
    Q = 2
    use_old = False
    caseno = 4
    # for D in [30, 50, 100]:
    #     for Q in [3, 5, 7]:
    #         for N in [1000, 3000, 5000]:
    for D in [20]:
        store_file = f'exp_data/Phantom expansion methods_{caseno}caseno_{N}N_{D}D'
        os.makedirs(store_file, exist_ok=True)
        for method in ["random projection", "rff", "autoencoder", "polynomial"]:
        # for method in ["autoencoder"]:
            # for noise_var in [1, 4, 9]:
            for noise_var in [4]:
                print(f"\n开始处理参数组合: D={D}, Q={Q}, N={N}")
                
                # 根据Q值动态导入数据生成模块
                if not use_old:
                    data_generate = import_module('new_highdata_gen')
                else:
                    data_generate = import_module('data_generate')
                
                results = {}
                filename = f'{store_file}/results_D={D}_Q={Q}_M={M}_K={K}_N={N}_noise_var={noise_var}_method={method}.pkl'
                
                for i in range(L):
                    print(f"\nProcessing iteration {i+1}/{L} for D={D}, Q={Q}, N={N}, method={method}, noise_var={noise_var}")
                    res_i = {}
                    
                    # 1. 生成数据
                    # if Q == 2:
                    if use_old:
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
                        if Q == D:
                            args_data['ins_dim'] = False
                    else:
                        args_data = {
                            'N': N,
                            'Nt': T_param,
                            'H': D,
                            'd': Q,
                            'device': 'cpu',
                            'noise_var': noise_var,
                            'methods': method,
                            'caseno': caseno
                        }
                    folder_name = run_with_args(data_generate, args_data)
                    print(f"Data generated in folder: {folder_name}")
                    
                    # 3. JumpGP + SIR
                    if "JGP" not in use_module and i > 0:
                        continue
                    else:
                        args_jump_sir = {
                            'folder_name': folder_name,
                            'M': M,
                            'device': 'cpu',
                            'use_sir': True,
                            # 'use_cv': True,
                            "use_cv": False,
                            'sir_H': D,
                            'sir_K': K,
                            'return_dim': "True",
                            'sir_K_list': [5]
                        }
                        res_jgp, dim = run_with_args(JumpGP_test, args_jump_sir)
                        res_i['JGP_sir'] = res_jgp
                        print(f"JumpGP with SIR completed, choose {dim} as the dimension of the latent space")

                    # 4. DeepGP, DJGP, SIR_GP 多次迭代
                    for j in range(T):
                        print(f"  Iteration {j+1}/{T} for {folder_name}")
                        
                        # DeepGP
                        if "DGP" in use_module:
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
                        if "LMJGP" in use_module:
                            args_djgp = {
                                'folder_name': folder_name,
                                'num_steps': 600,
                                'n': M,
                                'Q': K,
                                'MC_num': 5,
                                'm2': 40,
                                'm1': 5,
                                'lr': 0.01
                            }
                            res_i[f'DJGP_{j}'] = run_with_args(DJGP_test, args_djgp)
                            print(f"    DJGP iteration {j+1} completed")

                    folder_key = str(i+1)
                    print(f"Storing results under key: {folder_key}")
                    results[folder_key] = res_i

                # 保存当前参数组合的结果
                save_results(results, filename)
                plot_metrics_boxplot(results, f'{store_file}/metrics_boxplot_D={D}_Q={Q}_N={N}_method={method}.png')
                print(f"参数组合 D={D}, Q={Q}, N={N}, method={method} 实验完成")

    print("\n所有实验完成!")

if __name__ == "__main__":
    main()
