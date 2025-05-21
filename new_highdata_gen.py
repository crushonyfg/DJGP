import torch
import argparse
from new_highdata_gen_utils import generate_data
from data_generate import save_dataset  

def sample_rff_params(d, D_rff, lengthscale, device):
    """
    采样一组 RFF 参数 Omega 和 phases
    Omega: [D_rff, d], phases: [D_rff]
    """
    Omega  = torch.randn(D_rff, d, device=device) / lengthscale
    phases = 2 * torch.pi * torch.rand(D_rff, device=device)
    return Omega, phases

def make_rff_features(x, Omega, phases, kernel_var):
    """
    x: [N, d]
    Omega: [D_rff, d]
    phases: [D_rff]
    返回 Phi: [N, D_rff]
    """
    # x @ Omega.T -> [N, D_rff]
    proj  = x @ Omega.t() + phases.unsqueeze(0)
    scale = torch.sqrt(torch.tensor(2.0 * kernel_var / Omega.shape[0], device=x.device))
    return scale * torch.cos(proj)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--d',        type=int,   default=3,   help='原始输入维度')
    p.add_argument('--N',        type=int,   default=500, help='训练集大小')
    p.add_argument('--Nt',       type=int,   default=200, help='测试集大小')
    p.add_argument('--H',        type=int,   default=20,  help='RFF 映射到的高维度')
    p.add_argument('--lengthscale', type=float, default=0.5, help='RFF lengthscale')
    p.add_argument('--kernel_var',  type=float, default=9, help='RFF kernel variance')
    p.add_argument('--device',   type=str,   default='cpu')
    p.add_argument('--ins_dim', type=bool, default=True, help='是否将输入维度扩展为高维')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # 1) 生成原始数据
    x_train, y_train, x_test, y_test = generate_data(
        d   = args.d,
        N   = args.N,
        Nt  = args.Nt,
        device = args.device
    )
    print(f"原始 X_train: {x_train.shape}, X_test: {x_test.shape}")

    if args.ins_dim:    
            # 2) 采样 RFF 参数
        Omega, phases = sample_rff_params(
            d        = args.d,
            D_rff    = args.H,
            lengthscale = args.lengthscale,
            device   = device
        )

        # 3) 映射到高维特征空间
        Phi_train = make_rff_features(x_train, Omega, phases, args.kernel_var)
        Phi_test  = make_rff_features(x_test,  Omega, phases, args.kernel_var)
        print(f"RFF 映射后 Phi_train: {Phi_train.shape}, Phi_test: {Phi_test.shape}")

        data = {
            'X_train': Phi_train,
            'Y_train'    : y_train,
            'X_test' : Phi_test,
            'Y_test'     : y_test,
            'Omega'      : Omega,
            'phases'     : phases,
        }
    else:
        data = {
            'X_train': x_train,
            'Y_train'    : y_train,
            'X_test' : x_test,
            'Y_test'     : y_test,
        }
    folder_name = save_dataset(data, args)
    return folder_name

if __name__ == '__main__':
    main()
