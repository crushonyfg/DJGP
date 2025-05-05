import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from importlib import import_module
import sys

def save_results(results, filename='results.pkl'):
    """保存结果到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")

def plot_metrics(results):
    """绘制散点图"""
    metrics = ['rmse', 'q25', 'q50', 'q75']
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    # 修改颜色映射，为所有方法使用不同的颜色
    colors = {
        'JumpGP': 'blue',
        'JumpGP_sir': 'red',
        'DeepGP': 'green',
        'GP_sir': 'purple',
        'GP': 'orange'  # 添加GP的颜色
    }
    markers = {
        'JumpGP': 'o',
        'JumpGP_sir': 's',
        'DeepGP': '^',
        'GP_sir': 'D',
        'GP': 'v'  # 添加GP的标记
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for folder_name, res_i in results.items():
            for method, values in res_i.items():
                # 修改基础方法名的获取逻辑
                if '_' in method:
                    base_method = method.split('_')[0] + '_' + method.split('_')[1] if 'sir' in method.lower() else method.split('_')[0]
                else:
                    base_method = method
                
                ax.scatter(values[4], values[idx], 
                          label=f'{method}_{folder_name}',
                          color=colors[base_method],
                          marker=markers[base_method])
        
        ax.set_xlabel('Runtime (s)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Runtime')
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Plot saved as metrics_comparison.png")

def create_args(args_dict):
    """创建参数对象"""
    args = type('Args', (), {})()
    for key, value in args_dict.items():
        setattr(args, key, value)
    return args

def run_with_args(module, args_dict):
    """使用命令行参数运行模块"""
    # 保存原始命令行参数
    original_argv = sys.argv[:]
    
    try:
        # 构建新的命令行参数
        sys.argv = [sys.argv[0]]
        for key, value in args_dict.items():
            if isinstance(value, bool):
                if value:
                    sys.argv.append(f'--{key}')
            else:
                sys.argv.append(f'--{key}')
                sys.argv.append(str(value))
        
        # 运行模块的main函数
        result = module.main()
        return result
    
    finally:
        # 恢复原始命令行参数
        sys.argv = original_argv

def main():
    L, T = 3, 3
    N = 1000
    T_param = 500
    D = 15
    caseno = 5
    M = 100
    results = {}

    # 导入所需的模块
    data_generate = import_module('data_generate')
    JumpGP_test = import_module('JumpGP_test')
    DeepGP_test = import_module('DeepGP_test')
    SIR_GP = import_module('SIR_GP')

    for i in range(L):
        print(f"\nProcessing iteration {i+1}/{L}")
        res_i = {}
        
        # 生成数据
        args_data = {
            'N': N,
            'T': T_param,
            'D': D,
            'caseno': caseno,
            'device': 'cpu',
            'latent_dim': 2,
            'Q': 2
        }
        folder_name = run_with_args(data_generate, args_data)
        print(f"Data generated in folder: {folder_name}")
        
        # 运行 JumpGP
        args_jump = {
            'folder_name': folder_name,
            'M': M,
            'device': 'cpu'
        }
        res_i['JumpGP'] = run_with_args(JumpGP_test, args_jump)
        print("JumpGP completed")
        
        # 运行 JumpGP with SIR
        args_jump_sir = {
            'folder_name': folder_name,
            'M': M,
            'device': 'cpu',
            'use_sir': True,
            'sir_H': D,
            'sir_K': 2
        }
        res_i['JumpGP_sir'] = run_with_args(JumpGP_test, args_jump_sir)
        print("JumpGP with SIR completed")

        for j in range(T):
            print(f"Running iteration {j+1}/{T} for {folder_name}")
            
            # 运行 DeepGP
            args_deep = {
                'folder_name': folder_name,
                'num_epochs': 200,
                'patience': 5,
                'batch_size': 1024,
                'lr': 0.01
            }
            res_i[f'DeepGP_{j}'] = run_with_args(DeepGP_test, args_deep)
            print(f"DeepGP iteration {j+1} completed")
            
            # 运行 SIR_GP
            args_sir = {
                'folder_name': folder_name,
                'use_sir': True,
                'sir_K': 2,
                'M': M,
                'device': 'cpu',
                'sir_H': D
            }
            res_i[f'GP_sir_{j}'] = run_with_args(SIR_GP, args_sir)
            print(f"SIR_GP iteration {j+1} completed")

        results[folder_name] = res_i
        print(f"Iteration {i+1} completed")

    # 保存结果
    save_results(results)
    
    # 绘制散点图
    plot_metrics(results)
    print("\nExperiment completed successfully.")

if __name__ == "__main__":
    main()

    