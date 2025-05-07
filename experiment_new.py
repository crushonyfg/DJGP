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

# def plot_metrics(results):
#     """绘制散点图并在每个点上标注 folder index"""
#     metrics = ['rmse', 'q25', 'q50', 'q75']
#     fig, axes = plt.subplots(2, 2, figsize=(15, 15))
#     axes = axes.ravel()
    
#     colors = {
#         'JumpGP':     'blue',
#         'JumpGP_sir': 'red',
#         'DeepGP':     'green',
#         'DJGP':       'cyan',
#         'GP_sir':     'purple',
#         'GP':         'orange'
#     }
#     markers = {
#         'JumpGP':     'o',
#         'JumpGP_sir': 's',
#         'DeepGP':     '^',
#         'DJGP':       '*',
#         'GP_sir':     'D',
#         'GP':         'v'
#     }
    
#     for idx, metric in enumerate(metrics):
#         ax = axes[idx]
#         for folder_idx, (folder_key, res_i) in enumerate(results.items(), start=1):
#             for method, values in res_i.items():
#                 # 把可能在 GPU 上的 tensor 转成 float
#                 raw_x = values[4]
#                 x = raw_x.detach().cpu().item() if isinstance(raw_x, torch.Tensor) else float(raw_x)
#                 raw_y = values[idx]
#                 y = raw_y.detach().cpu().item() if isinstance(raw_y, torch.Tensor) else float(raw_y)
                
#                 # 提取基础方法名
#                 if '_' in method:
#                     parts = method.split('_')
#                     base_method = parts[0] + '_' + parts[1] if 'sir' in method.lower() else parts[0]
#                 else:
#                     base_method = method
                
#                 ax.scatter(
#                     x, y,
#                     label=f'{method}_{folder_key}',
#                     color=colors.get(base_method, 'black'),
#                     marker=markers.get(base_method, 'x')
#                 )
#                 # 在点上标注第几次 folder
#                 ax.annotate(
#                     str(folder_idx),
#                     (x, y),
#                     textcoords="offset points",
#                     xytext=(5, 5),
#                     fontsize=9
#                 )
        
#         ax.set_xlabel('Runtime (s)')
#         ax.set_ylabel(metric.upper())
#         ax.set_title(f'{metric.upper()} vs Runtime')
#         ax.grid(True)
#         ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     plt.tight_layout()
#     plt.savefig('metrics_comparison.png', bbox_inches='tight', dpi=300)
#     plt.close()
#     print("Plot saved as metrics_comparison.png")
# def plot_metrics(results):
#     """绘制散点图并在每个点上标注 folder index，且 legend 去重只保留 base 方法名"""
#     import torch  # 确保能检测到 torch.Tensor
#     metrics = ['rmse', 'q25', 'q50', 'q75']
#     fig, axes = plt.subplots(2, 2, figsize=(15, 15))
#     axes = axes.ravel()
    
#     colors = {
#         'JumpGP':     'blue',
#         'JumpGPsir': 'red',
#         'DeepGP':     'green',
#         'DJGP':       'cyan',
#         'GPsir':     'purple',
#         'GP':         'orange'
#     }
#     markers = {
#         'JumpGP':     'o',
#         'JumpGPsir': 's',
#         'DeepGP':     '^',
#         'DJGP':       '*',
#         'GPsir':     'D',
#         'GP':         'v'
#     }
    
#     for idx, metric in enumerate(metrics):
#         ax = axes[idx]
        
#         for folder_idx, (folder_key, res_i) in enumerate(results.items(), start=1):
#             for method, values in res_i.items():
#                 # 将可能在 GPU 上的 tensor 转成 float
#                 raw_x = values[4]
#                 x = raw_x.detach().cpu().item() if isinstance(raw_x, torch.Tensor) else float(raw_x)
#                 raw_y = values[idx]
#                 y = raw_y.detach().cpu().item() if isinstance(raw_y, torch.Tensor) else float(raw_y)
                
#                 # 只取 '_' 之前的 base label
#                 base_label = method.split('_')[0]
                
#                 ax.scatter(
#                     x, y,
#                     label=base_label,
#                     color=colors.get(base_label, 'black'),
#                     marker=markers.get(base_label, 'x')
#                 )
                
#                 # 在点上标注第几组数据（folder index）
#                 ax.annotate(
#                     str(folder_idx),
#                     (x, y),
#                     textcoords="offset points",
#                     xytext=(5, 5),
#                     fontsize=9
#                 )
        
#         ax.set_xlabel('Runtime (s)')
#         ax.set_ylabel(metric.upper())
#         ax.set_title(f'{metric.upper()} vs Runtime')
#         ax.grid(True)
        
#         # 去重 legend，只保留每个 base_label 的第一个 entry
#         handles, labels = ax.get_legend_handles_labels()
#         unique = dict(zip(labels, handles))
#         ax.legend(
#             unique.values(),
#             unique.keys(),
#             bbox_to_anchor=(1.05, 1),
#             loc='upper left'
#         )
    
#     plt.tight_layout()
#     plt.savefig('metrics_comparison.png', bbox_inches='tight', dpi=300)
#     plt.close()
#     print("Plot saved as metrics_comparison.png")

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
        'JumpGPsir':  'red',
        'DeepGP':     'green',
        'DJGP':       'cyan',
        'GPsir':      'purple',
        'GP':         'orange',
        'NNJGP':       'magenta',   # 新增
        'BNNJGP':      'brown',
    }
    markers = {
        'JumpGP':     'o',
        'JumpGPsir':  's',
        'DeepGP':     '^',
        'DJGP':       '*',
        'GPsir':      'D',
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

    # Color and marker map for each base method
    colors = {
        'JumpGP':    'blue',
        'JumpGPsir': 'red',
        'DeepGP':    'green',
        'DJGP':      'cyan',
        'GPsir':     'purple',
        'GP':        'orange',
        'NNJGP':     'magenta',
        'BNNJGP':    'brown',
    }
    markers = {
        'JumpGP':    'o',
        'JumpGPsir': 's',
        'DeepGP':    '^',
        'DJGP':      '*',
        'GPsir':     'D',
        'GP':        'v',
        'NNJGP':     'X',
        'BNNJGP':    'P',
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

def main():
    # 实验设置
    L = 20   # 不同数据集生成次数
    T = 1   # DeepGP/DJGP/SIR 循环次数
    N = 1500
    T_param = 100
    D = 20
    caseno = 5
    M = 25
    Q = 4
    K = Q
    use_cv = False
    
    results = {}

    # 动态导入模块
    if Q>2:
    # data_generate = import_module('data_generate')
        data_generate = import_module('new_data_gen')
    else:
        data_generate = import_module('data_generate')
    JumpGP_test   = import_module('JumpGP_test')
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
        args_data = {
            'N': N,
            'T': T_param,
            'D': D,
            'caseno': caseno,
            'device': 'cpu',
            'latent_dim': 2,
            'Q': Q
        }
        folder_name = run_with_args(data_generate, args_data)
        print(f"Data generated in folder: {folder_name}")
        
        # 2. JumpGP
        # args_jump = {
        #     'folder_name': folder_name,
        #     'M': M,
        #     'device': 'cpu'
        # }
        # res_i['JumpGP'] = run_with_args(JumpGP_test, args_jump)
        # print("JumpGP completed")
        
        # 3. JumpGP + SIR
        args_jump_sir = {
            'folder_name': folder_name,
            'M': M,
            'device': 'cpu',
            'use_sir': True,
            'sir_H': D,
            'sir_K': K
        }
        res_i['JumpGPsir'] = run_with_args(JumpGP_test, args_jump_sir)
        print("JumpGP with SIR completed")

        # 4. DeepGP, DJGP, SIR_GP 多次迭代
        for j in range(T):
            print(f"  Iteration {j+1}/{T} for {folder_name}")
            
            # DeepGP
            args_deep = {
                'folder_name': folder_name,
                'hidden_dim': K,
                'num_epochs': 300,
                'patience': 5,
                'batch_size': 1024,
                'lr': 0.01
            }
            res_i[f'DeepGP_{j}'] = run_with_args(DeepGP_test, args_deep)
            print(f"    DeepGP iteration {j+1} completed")
            
            # DJGP
            args_djgp = {
                'folder_name': folder_name,
                'num_steps': 300,
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

            # args_nnjgp = {
            #     'folder_name': folder_name,
            #     'num_steps': 300,
            #     'n': M,
            #     'Q': K,
            #     # 'MC_num': 5,
            #     'm1': 5,
            #     'lr': 0.01
            # }
            # res_i[f'NNJGP_{j}'] = run_with_args(NNJGP_test, args_nnjgp)
            # print(f"    NNJGP iteration {j+1} completed")

            args_bnnjgp = {
                'folder_name': folder_name,
                'steps': 300,
                'n': M,
                'Q': K,
                # 'MC_num': 5,
                'm1': 5,
                'lr': 0.01
            }
            res_i[f'BNNJGP_{j}'] = run_with_args(BNNJGP_test, args_bnnjgp)
            print(f"    BNNJGP iteration {j+1} completed")
            
            # SIR_GP
            # args_sir = {
            #     'folder_name': folder_name,
            #     'use_sir': True,
            #     'sir_K': 2,
            #     'M': M,
            #     'device': 'cpu',
            #     'sir_H': D
            # }
            # res_i[f'GPsir_{j}'] = run_with_args(SIR_GP, args_sir)
            # print(f"    SIR_GP iteration {j+1} completed")

        # 使用循环下标作为 key，避免 folder_name 重复覆盖
        folder_key = str(i+1)
        print(f"Storing results under key: {folder_key}")
        results[folder_key] = res_i

    # 保存并绘图
    save_results(results)
    # plot_metrics(results)
    plot_metrics_with_mean(results, 'metrics_with_means.png')
    print("\nExperiment completed successfully.")

if __name__ == "__main__":
    main()

# import os
# import pickle
# import matplotlib.pyplot as plt
# import numpy as np
# import time
# import argparse
# from importlib import import_module
# import sys

# def save_results(results, filename='results.pkl'):
#     """保存结果到文件"""
#     with open(filename, 'wb') as f:
#         pickle.dump(results, f)
#     print(f"Results saved to {filename}")

# # def plot_metrics(results):
# #     """绘制散点图并在每个点上标注 folder index"""
# #     metrics = ['rmse', 'q25', 'q50', 'q75']
# #     fig, axes = plt.subplots(2, 2, figsize=(15, 15))
# #     axes = axes.ravel()
    
# #     # 颜色和标记映射
# #     colors = {
# #         'JumpGP':      'blue',
# #         'JumpGP_sir':  'red',
# #         'DeepGP':      'green',
# #         'DJGP':        'cyan',    # 新增 DJGP
# #         'GP_sir':      'purple',
# #         'GP':          'orange'
# #     }
# #     markers = {
# #         'JumpGP':      'o',
# #         'JumpGP_sir':  's',
# #         'DeepGP':      '^',
# #         'DJGP':        '*',       # 新增 DJGP
# #         'GP_sir':      'D',
# #         'GP':          'v'
# #     }
    
# #     # results: 保持插入顺序，所以 enumerate 可以获取 folder index
# #     for idx, metric in enumerate(metrics):
# #         ax = axes[idx]
# #         for folder_idx, (folder_name, res_i) in enumerate(results.items(), start=1):
# #             for method, values in res_i.items():
# #                 # 提取基础方法名
# #                 if '_' in method:
# #                     parts = method.split('_')
# #                     if 'sir' in method.lower():
# #                         base_method = parts[0] + '_' + parts[1]
# #                     else:
# #                         base_method = parts[0]
# #                 else:
# #                     base_method = method
                
# #                 x = values[4]  # runtime
# #                 y = values[idx]  # rmse/q25/...
                
# #                 ax.scatter(x, y,
# #                            label=f'{method}_{folder_name}',
# #                            color=colors.get(base_method, 'black'),
# #                            marker=markers.get(base_method, 'x'))
                
# #                 # 在点上标注 folder index
# #                 ax.annotate(str(folder_idx),
# #                             (x, y),
# #                             textcoords="offset points",
# #                             xytext=(5, 5),
# #                             fontsize=9)
        
# #         ax.set_xlabel('Runtime (s)')
# #         ax.set_ylabel(metric.upper())
# #         ax.set_title(f'{metric.upper()} vs Runtime')
# #         ax.grid(True)
# #         ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
# #     plt.tight_layout()
# #     plt.savefig('metrics_comparison.png', bbox_inches='tight', dpi=300)
# #     plt.close()
# #     print("Plot saved as metrics_comparison.png")

# import torch   # 新增
# # … 其它 import 保持不变 …

# def plot_metrics(results):
#     """绘制散点图并在每个点上标注 folder index"""
#     metrics = ['rmse', 'q25', 'q50', 'q75']
#     fig, axes = plt.subplots(2, 2, figsize=(15, 15))
#     axes = axes.ravel()
    
#     colors = {
#         'JumpGP':      'blue',
#         'JumpGP_sir':  'red',
#         'DeepGP':      'green',
#         'DJGP':        'cyan',
#         'GP_sir':      'purple',
#         'GP':          'orange'
#     }
#     markers = {
#         'JumpGP':      'o',
#         'JumpGP_sir':  's',
#         'DeepGP':      '^',
#         'DJGP':        '*',
#         'GP_sir':      'D',
#         'GP':          'v'
#     }
    
#     for idx, metric in enumerate(metrics):
#         ax = axes[idx]
#         for folder_idx, (folder_name, res_i) in enumerate(results.items(), start=1):
#             for method, values in res_i.items():
#                 # 先把 raw_x, raw_y 转成 Python float
#                 raw_x = values[4]
#                 if isinstance(raw_x, torch.Tensor):
#                     x = raw_x.detach().cpu().item()
#                 else:
#                     x = float(raw_x)
                
#                 raw_y = values[idx]
#                 if isinstance(raw_y, torch.Tensor):
#                     y = raw_y.detach().cpu().item()
#                 else:
#                     y = float(raw_y)
                
#                 # 提取基础方法名
#                 if '_' in method:
#                     parts = method.split('_')
#                     base_method = parts[0] + '_' + parts[1] if 'sir' in method.lower() else parts[0]
#                 else:
#                     base_method = method

#                 ax.scatter(x, y,
#                            label=f'{method}_{folder_name}',
#                            color=colors.get(base_method, 'black'),
#                            marker=markers.get(base_method, 'x'))
                
#                 # 标注 folder 序号
#                 ax.annotate(str(folder_idx),
#                             (x, y),
#                             textcoords="offset points",
#                             xytext=(5, 5),
#                             fontsize=9)

#         ax.set_xlabel('Runtime (s)')
#         ax.set_ylabel(metric.upper())
#         ax.set_title(f'{metric.upper()} vs Runtime')
#         ax.grid(True)
#         ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     plt.tight_layout()
#     plt.savefig('metrics_comparison.png', bbox_inches='tight', dpi=300)
#     plt.close()
#     print("Plot saved as metrics_comparison.png")


# def run_with_args(module, args_dict):
#     """使用模拟命令行参数运行模块的 main()"""
#     original_argv = sys.argv[:]
#     try:
#         sys.argv = [sys.argv[0]]
#         for key, value in args_dict.items():
#             if isinstance(value, bool):
#                 if value:
#                     sys.argv.append(f'--{key}')
#             else:
#                 sys.argv.append(f'--{key}')
#                 sys.argv.append(str(value))
#         return module.main()
#     finally:
#         sys.argv = original_argv

# def main():
#     L, T = 2, 2
#     N = 100
#     T_param = 10
#     D = 7
#     caseno = 5
#     M = 20
#     K = 2
#     results = {}

#     # 动态导入各方法模块
#     data_generate = import_module('data_generate')
#     JumpGP_test  = import_module('JumpGP_test')
#     DeepGP_test  = import_module('DeepGP_test')
#     DJGP_test    = import_module('DJGP_test')    # 新增 DJGP
#     SIR_GP       = import_module('SIR_GP')

#     for i in range(L):
#         print(f"\nProcessing iteration {i+1}/{L}")
#         res_i = {}
        
#         # 1. 生成数据
#         args_data = {
#             'N': N,
#             'T': T_param,
#             'D': D,
#             'caseno': caseno,
#             'device': 'cpu',
#             'latent_dim': K,
#             'Q': K
#         }
#         folder_name = run_with_args(data_generate, args_data)
#         print(f"Data generated in folder: {folder_name}")
        
#         # 2. JumpGP
#         args_jump = {
#             'folder_name': folder_name,
#             'M': M,
#             'device': 'cpu'
#         }
#         res_i['JumpGP'] = run_with_args(JumpGP_test, args_jump)
#         print("JumpGP completed")
        
#         # 3. JumpGP + SIR
#         args_jump_sir = {
#             'folder_name': folder_name,
#             'M': M,
#             'device': 'cpu',
#             'use_sir': True,
#             'sir_H': D,
#             'sir_K': K
#         }
#         res_i['JumpGP_sir'] = run_with_args(JumpGP_test, args_jump_sir)
#         print("JumpGP with SIR completed")

#         # 4. DeepGP, DJGP, SIR_GP 多次迭代
#         for j in range(T):
#             print(f"  Iteration {j+1}/{T} for {folder_name}")
            
#             # DeepGP
#             args_deep = {
#                 'folder_name': folder_name,
#                 'num_epochs': 200,
#                 'patience': 5,
#                 'batch_size': 1024,
#                 'lr': 0.01
#             }
#             res_i[f'DeepGP_{j}'] = run_with_args(DeepGP_test, args_deep)
#             print(f"    DeepGP iteration {j+1} completed")
            
#             # DJGP (与 DeepGP 参数相同)
#             args_djgp = {
#                 'folder_name': folder_name,
#                 'num_steps': 200,
#                 'n': M,
#                 'Q': K,
#                 'MC_num': 5,
#                 'lr': 0.01
#             }
#             res_i[f'DJGP_{j}'] = run_with_args(DJGP_test, args_djgp)
#             print(f"    DJGP iteration {j+1} completed")
            
#             # SIR_GP
#             args_sir = {
#                 'folder_name': folder_name,
#                 'use_sir': True,
#                 'sir_K': 2,
#                 'M': M,
#                 'device': 'cpu',
#                 'sir_H': D
#             }
#             res_i[f'GP_sir_{j}'] = run_with_args(SIR_GP, args_sir)
#             print(f"    SIR_GP iteration {j+1} completed")

#         results[folder_name] = res_i
#         print(f"Iteration {i+1} completed")

#     # 保存并绘图
#     save_results(results)
#     plot_metrics(results)
#     print("\nExperiment completed successfully.")

# if __name__ == "__main__":
#     main()
