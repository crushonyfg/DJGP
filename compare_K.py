# import os
# import pickle
# import matplotlib.pyplot as plt
# import numpy as np
# import argparse
# from importlib import import_module
# import sys
# import torch   # for tensor detection

# def save_results(results, filename='results_by_K.pkl'):
#     """保存按 K 值索引的结果字典"""
#     with open(filename, 'wb') as f:
#         pickle.dump(results, f)
#     print(f"Results saved to {filename}")


# def plot_metrics_by_K(results, K_list, output_file='metrics_by_K.png'):
#     """
#     根据 K 值绘制各方法的 RMSE 和 CRPS 曲线
#     X 轴为 K，Y 轴为对应指标
#     """
#     metrics = ['rmse', 'crps']
#     fig, axes = plt.subplots(1, 2, figsize=(15, 7))
#     axes = axes.ravel()

#     # 配置颜色和标记
#     colors = {
#         'JumpGP':            'blue',
#         'JumpGPsirGlobal':   'red',
#         'JumpGPsirLocal':    'purple',
#         'DeepGP':            'green',
#         'DJGP':              'cyan',
#         'NNJGP':             'magenta',
#         'BNNJGP':            'brown',
#         'GPsir':             'orange'
#     }
#     markers = {
#         'JumpGP':            'o',
#         'JumpGPsirGlobal':   's',
#         'JumpGPsirLocal':    'D',
#         'DeepGP':            '^',
#         'DJGP':              '*',
#         'NNJGP':             'X',
#         'BNNJGP':            'P',
#         'GPsir':             'v'
#     }

#     for idx, metric in enumerate(metrics):
#         ax = axes[idx]
#         for method in results[K_list[0]].keys():  # 假设每个 K 下的方法列表一致
#             y_vals = []
#             for K in K_list:
#                 vals = results[K][method]
#                 raw = vals[0] if metric == 'rmse' else vals[1]
#                 y = raw.detach().cpu().item() if isinstance(raw, torch.Tensor) else float(raw)
#                 y_vals.append(y)
#             ax.plot(
#                 K_list,
#                 y_vals,
#                 label=method,
#                 marker=markers.get(method, 'x'),
#                 color=colors.get(method, 'black')
#             )
#         ax.set_xlabel('K')
#         ax.set_ylabel(metric.upper())
#         ax.set_title(f'{metric.upper()} vs K')
#         ax.grid(True)
#         ax.legend(loc='best')

#     plt.tight_layout()
#     plt.savefig(output_file, bbox_inches='tight', dpi=300)
#     plt.close()
#     print(f"Plot saved to {output_file}")


# def run_with_args(module, args_dict):
#     """模拟命令行调用模块的 main 函数"""
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
#     # 1. 实验全局设置
#     N = 1000               # 数据点数
#     T_param = 100           # 时间长度
#     D = 20                  # 维度
#     caseno = 5              # case 编号
#     M = 25                  # Monte Carlo 样本数
#     use_cv = False

#     # 2. 指定一系列 K 值进行比较
#     K_list = [2, 3, 4]

#     # 3. 动态导入模块
#     data_generate = import_module('data_generate')
#     JumpGP_test        = import_module('JumpGP_test')
#     JumpGP_test_local  = import_module('JumpGP_test_local')
#     DeepGP_test        = import_module('DeepGP_test')
#     DJGP_test          = import_module('DJGP_CV') if use_cv else import_module('DJGP_test')
#     NNJGP_test         = import_module('NNJGP_test')
#     BNNJGP_test        = import_module('BNNJGP')
#     SIR_GP             = import_module('SIR_GP')

#     # 4. 生成一次数据
#     args_data = {
#         'N': N,
#         'T': T_param,
#         'D': D,
#         'caseno': caseno,
#         'device': 'cpu',
#         'latent_dim': 2,
#         'Q': 2
#     }
#     folder_name = run_with_args(data_generate, args_data)
#     print(f"Data generated in folder: {folder_name}")

#     # 5. 对每个 K 运行所有方法
#     results = {}
#     for K in K_list:
#         print(f"\nRunning methods with K = {K}")
#         res_i = {}

#         # JumpGP + SIR global
#         args_jump_sir = {
#             'folder_name': folder_name,
#             'M': M,
#             'device': 'cpu',
#             'use_sir': True,
#             'sir_H': D,
#             'sir_K': K
#         }
#         res_i['JumpGPsirGlobal'] = run_with_args(JumpGP_test, args_jump_sir)

#         # JumpGP + SIR local
#         args_jump_sir_loc = args_jump_sir.copy()
#         res_i['JumpGPsirLocal'] = run_with_args(JumpGP_test_local, args_jump_sir_loc)

#         # DeepGP
#         # args_deep = {
#         #     'folder_name': folder_name,
#         #     'hidden_dim': K,
#         #     'num_epochs': 300,
#         #     'patience': 5,
#         #     'batch_size': 1024,
#         #     'lr': 0.01
#         # }
#         # res_i['DeepGP'] = run_with_args(DeepGP_test, args_deep)

#         # DJGP
#         args_djgp = {
#             'folder_name': folder_name,
#             'num_steps': 500,
#             'n': M,
#             'Q': K,
#             'MC_num': 5,
#             'm2': 20,
#             'm1': 5,
#             'lr': 0.01
#         }
#         res_i['DJGP'] = run_with_args(DJGP_test, args_djgp)

#         # # NNJGP
#         # args_nnjgp = {
#         #     'folder_name': folder_name,
#         #     'num_steps': 300,
#         #     'n': M,
#         #     'Q': K,
#         #     'm1': 5,
#         #     'lr': 0.01
#         # }
#         # res_i['NNJGP'] = run_with_args(NNJGP_test, args_nnjgp)

#         # # BNNJGP
#         # args_bnnj = {
#         #     'folder_name': folder_name,
#         #     'steps': 300,
#         #     'n': M,
#         #     'Q': K,
#         #     'm1': 5,
#         #     'lr': 0.01
#         # }
#         # res_i['BNNJGP'] = run_with_args(BNNJGP_test, args_bnnj)

#         # # GP + SIR (baseline)
#         # args_sir = {
#         #     'folder_name': folder_name,
#         #     'use_sir': True,
#         #     'sir_K': K,
#         #     'M': M,
#         #     'device': 'cpu',
#         #     'sir_H': D
#         # }
#         # res_i['GPsir'] = run_with_args(SIR_GP, args_sir)

#         # 存储该 K 下的结果
#         results[K] = res_i

#     # 6. 保存结果并绘图
#     save_results(results)
#     plot_metrics_by_K(results, K_list)


# if __name__ == '__main__':
#     main()
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
from importlib import import_module
import sys
import torch   # for detecting torch.Tensor


def save_results(results, filename='results_by_K.pkl'):
    """保存按 K 值索引的平均结果字典"""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Average results saved to {filename}")


def plot_metrics_by_K(results, K_list, output_file='metrics_by_K.png'):
    """
    根据 K 值绘制各方法的平均 RMSE 和 CRPS 曲线
    X 轴为 K，Y 轴为对应指标
    """
    metrics = ['rmse', 'crps']
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes = axes.ravel()

    # 配置颜色和标记
    colors = {
        'JumpGPsrGlobal':   'red',
        'JumpGPsrLocal':    'purple',
        'DeepGP':           'green',
        'DJGP':             'cyan',
        'NNJGP':            'magenta',
        'BNNJGP':           'brown',
        'GPsir':            'orange'
    }
    markers = {
        'JumpGPsrGlobal':   's',
        'JumpGPsrLocal':    'D',
        'DeepGP':           '^',
        'DJGP':             '*',
        'NNJGP':            'X',
        'BNNJGP':           'P',
        'GPsir':            'v'
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for method in results[K_list[0]].keys():
            y_vals = []
            for K in K_list:
                vals = results[K][method]
                # vals is [mean_rmse, mean_crps]
                raw = vals[0] if metric == 'rmse' else vals[1]
                y_vals.append(raw)

            ax.plot(
                K_list,
                y_vals,
                label=method,
                marker=markers.get(method, 'x'),
                color=colors.get(method, 'black')
            )
        ax.set_xlabel('K')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Mean {metric.upper()} vs K')
        ax.grid(True)
        ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plot of mean metrics saved to {output_file}")


def run_with_args(module, args_dict):
    """模拟命令行调用模块的 main 函数"""
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


def main():
    # 实验参数
    N = 1000               # 数据点数
    T_param = 100          # 时间长度
    D = 20                 # 维度
    caseno = 5             # case 编号
    M = 25                 # Monte Carlo 样本数
    use_cv = False
    repeats = 5            # 重复生成数据集次数，用于平均

    # 要比较的 K 值列表
    K_list = [2, 4, 6]

    # 模块导入
    data_generate      = import_module('data_generate')
    JumpGP_test        = import_module('JumpGP_test')
    JumpGP_test_local  = import_module('JumpGP_test_local')
    DeepGP_test        = import_module('DeepGP_test')
    DJGP_test          = import_module('DJGP_CV') if use_cv else import_module('DJGP_test')
    NNJGP_test         = import_module('NNJGP_test')
    BNNJGP_test        = import_module('BNNJGP')
    SIR_GP             = import_module('SIR_GP')

    # 用于存储所有重复的结果: results[K][method] = list of [rmse, crps]
    results = {K: {} for K in K_list}

    for r in range(repeats):
        print(f"\nRepeat {r+1}/{repeats}: generating dataset")
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
        print(f"Data in folder: {folder_name}")

        # 每个 K 跑一次所有方法
        for K in K_list:
            print(f" Running methods with K = {K}")
            # JumpGP + SIR global
            args_jump_sir = {
                'folder_name': folder_name,
                'M': M,
                'device': 'cpu',
                'use_sir': True,
                'sir_H': D,
                'sir_K': K
            }
            res_g = run_with_args(JumpGP_test, args_jump_sir)

            # JumpGP + SIR local
            res_l = run_with_args(JumpGP_test_local, args_jump_sir)

            # DeepGP
            args_deep = {'folder_name': folder_name, 'hidden_dim': K,
                         'num_epochs': 300, 'patience': 5,
                         'batch_size': 1024, 'lr': 0.01}
            res_d = run_with_args(DeepGP_test, args_deep)

            # DJGP
            args_dj = {'folder_name': folder_name, 'num_steps': 300,
                       'n': M, 'Q': K, 'MC_num': 5,
                       'm2': 20, 'm1': 5, 'lr': 0.01}
            res_j = run_with_args(DJGP_test, args_dj)

            # # NNJGP
            # args_n = {'folder_name': folder_name, 'num_steps': 300,
            #           'n': M, 'Q': K, 'm1': 5, 'lr': 0.01}
            # res_n = run_with_args(NNJGP_test, args_n)

            # # BNNJGP
            # args_b = {'folder_name': folder_name, 'steps': 300,
            #           'n': M, 'Q': K, 'm1': 5, 'lr': 0.01}
            # res_b = run_with_args(BNNJGP_test, args_b)

            # # GP + SIR baseline
            # args_s = {
            #     'folder_name': folder_name,
            #     'use_sir': True, 'sir_K': K,
            #     'M': M, 'device': 'cpu', 'sir_H': D
            # }
            # res_s = run_with_args(SIR_GP, args_s)

            # 收集
            for method, res in [('JumpGPsrGlobal', res_g),
                                 ('JumpGPsrLocal',  res_l),
                                 ('DeepGP',         res_d),
                                 ('DJGP',           res_j)]:
                                #  ('NNJGP',          res_n),
                                #  ('BNNJGP',         res_b),
                                #  ('GPsir',          res_s)]:
                # 拆出数值
                rmse = res[0].detach().cpu().item() if isinstance(res[0], torch.Tensor) else float(res[0])
                crps = res[1].detach().cpu().item() if isinstance(res[1], torch.Tensor) else float(res[1])
                results[K].setdefault(method, []).append((rmse, crps))

    # 计算平均
    avg_results = {}
    for K in K_list:
        avg_results[K] = {}
        for method, runs in results[K].items():
            rmses = [r[0] for r in runs]
            crpss = [r[1] for r in runs]
            avg_results[K][method] = [np.mean(rmses), np.mean(crpss)]

    # 保存与绘图
    save_results(avg_results)
    plot_metrics_by_K(avg_results, K_list)


if __name__ == '__main__':
    main()
