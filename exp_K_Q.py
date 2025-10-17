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

from active_djgp_acquisition import *

def main(Q=5):
    # 实验设置
    L = 10   # 不同数据集生成次数
    T = 1   # DeepGP/DJGP/SIR 循环次数
    N = 1500
    T_param = 100
    D = 30
    caseno = 5
    M = 35
    # Q = 5
    K = 3
    use_cv = False
    # use_cv = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {}

    # 动态导入模块
    data_generate = import_module('new_highdata_gen')
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
                'device': device,
                'seed': i
            }
        folder_name = run_with_args(data_generate, args_data)
        print(f"Data generated in folder: {folder_name}")
        
        dataset = load_dataset(folder_name)
        X_train = dataset["X_train"].to(device)   # (N_train, D)
        Y_train = dataset["Y_train"].to(device)
        X_test  = dataset["X_test"].to(device)    # (N_test, D)
        Y_test  = dataset["Y_test"].to(device)
        
        for LQ in [2,3,5,7]:
            args = DJGPArgs(
                Q=LQ, m1=4, m2=40, n=M,
                num_steps=300, MC_num=5, lr=1e-2,
                use_batch=False
            )
            res = djgp_fit_predict(
                X_train=X_train, Y_train=Y_train,
                X_test=X_test, Y_test=Y_test,
                args=args,need_predict=True,device=device
            )
            print(f"{LQ} exp {i+1} done, the result is {res['rmse']} and {res['crps']}")
            results[(Q,i,LQ)] = [res['rmse'],res['crps']]
    # 保存并绘图
    save_results(results,f"results_K_Q_{Q}.pkl")
    print("\nExperiment completed successfully.")

if __name__ == "__main__":
    for Q in [3,5,7]:
        main(Q)