import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_highdata_gen_utils import generate_data
from new_highdata_gen import sample_rff_params, make_rff_features, autoencoder_transform

class Args:
    """简单的参数类，用于模拟 argparse 的结果"""
    def __init__(self):
        self.d = 3
        self.N = 500
        self.Nt = 100
        self.H = 20
        self.lengthscale = 0.5
        self.kernel_var = 9
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ins_dim = True
        self.noise_var = 4
        
def test_method_shapes():
    """测试不同 methods 参数时 X_train 的形状"""
    args = Args()
    device = torch.device(args.device)
    
    # 生成原始数据
    x_train, y_train, x_test, y_test = generate_data(
        d=args.d,
        N=args.N,
        Nt=args.Nt,
        device=args.device,
        noise_var=args.noise_var
    )
    
    print(f"Original data shapes:")
    print(f"  x_train: {x_train.shape}")
    print(f"  x_test: {x_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
    print("="*50)
    
    # 测试不同的 methods
    methods = ["rff", "random projection", "polynomial", "autoencoder"]
    
    for method in methods:
        print(f"\nTesting method: {method}")
        print("-" * 30)
        
        try:
            if method == "rff":
                # RFF 方法
                Omega, phases = sample_rff_params(
                    d=args.d,
                    D_rff=args.H,
                    lengthscale=args.lengthscale,
                    device=device
                )
                Phi_train = make_rff_features(x_train, Omega, phases, args.kernel_var)
                Phi_test = make_rff_features(x_test, Omega, phases, args.kernel_var)
                
                print(f"  RFF features shape:")
                print(f"    Phi_train: {Phi_train.shape}")
                print(f"    Phi_test: {Phi_test.shape}")
                
            elif method == "random projection":
                # Random projection 方法
                Omega = torch.randn(args.H - args.d, args.d, device=device)
                projected_train = x_train @ Omega.t()
                projected_test = x_test @ Omega.t()
                Phi_train = torch.cat([x_train, projected_train], dim=1)
                Phi_test = torch.cat([x_test, projected_test], dim=1)
                
                print(f"  Random projection features shape:")
                print(f"    Original dim: {x_train.shape[1]}")
                print(f"    Projected dim: {projected_train.shape[1]}")
                print(f"    Total Phi_train: {Phi_train.shape}")
                print(f"    Total Phi_test: {Phi_test.shape}")
                
            elif method == "polynomial":
                # Polynomial 方法
                x_train_poly = x_train.unsqueeze(-1)
                x_test_poly = x_test.unsqueeze(-1)
                x_train_poly = torch.cat([x_train_poly, x_train_poly**2, x_train_poly**3], dim=-1)
                x_test_poly = torch.cat([x_test_poly, x_test_poly**2, x_test_poly**3], dim=-1)
                
                # 重新调整形状
                Phi_train = x_train_poly.view(x_train_poly.shape[0], -1)
                Phi_test = x_test_poly.view(x_test_poly.shape[0], -1)
                
                print(f"  Polynomial features shape:")
                print(f"    Intermediate shape: {x_train_poly.shape}")
                print(f"    Final Phi_train: {Phi_train.shape}")
                print(f"    Final Phi_test: {Phi_test.shape}")
                
            elif method == "autoencoder":
                # Autoencoder 方法
                print(f"  Training autoencoder...")
                Z_train, Z_test, model = autoencoder_transform(x_train, x_test, args.H)
                
                print(f"  Autoencoder features shape:")
                print(f"    Z_train: {Z_train.shape}")
                print(f"    Z_test: {Z_test.shape}")
                
        except Exception as e:
            print(f"  Error with {method}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "="*50)
    print("Shape testing completed!")

def test_with_different_parameters():
    """测试不同参数组合下的形状变化"""
    print("\nTesting with different parameters:")
    print("="*50)
    
    # 测试不同的 d 和 H 值
    test_configs = [
        {'d': 5, 'H': 30}
        # {'d': 3, 'H': 20},
        # {'d': 5, 'H': 15},
        # {'d': 5, 'H': 25},
    ]
    
    for config in test_configs:
        print(f"\nTesting with d={config['d']}, H={config['H']}")
        print("-" * 30)
        
        args = Args()
        args.d = config['d']
        args.H = config['H']
        device = torch.device(args.device)
        
        # 生成数据
        x_train, y_train, x_test, y_test = generate_data(
            d=args.d,
            N=args.N,
            Nt=args.Nt,
            device=args.device,
            noise_var=args.noise_var
        )
        
        print(f"  Original: {x_train.shape}")
        
        # 测试 random projection (修改后的版本)
        Omega = torch.randn(args.H - args.d, args.d, device=device)
        projected_train = x_train @ Omega.t()
        Phi_train = torch.cat([x_train, projected_train], dim=1)
        print(f"  Random projection: {Phi_train.shape}")
        
        # 测试 polynomial
        x_train_poly = x_train.unsqueeze(-1)
        x_train_poly = torch.cat([x_train_poly, x_train_poly**2, x_train_poly**3], dim=-1)
        Phi_train_poly = x_train_poly.view(x_train_poly.shape[0], -1)
        print(f"  Polynomial: {Phi_train_poly.shape}")

def test_simple_run():
    """简单的运行测试，直接调用 new_highdata_gen 的 main 函数"""
    print("\nDirect testing by calling main function:")
    print("="*50)
    
    import subprocess
    methods = ["rff", "random projection", "polynomial", "autoencoder"]
    
    for method in methods:
        print(f"\nTesting method: {method}")
        print("-" * 30)
        
        # 构建命令
        cmd = [
            'python', 'new_highdata_gen.py',
            '--methods', method,
            '--d', '5',
            '--H', '20',
            '--N', '100',  # 减少数据量加快测试
            '--Nt', '20'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
            print(f"  Output: {result.stdout}")
            if result.stderr:
                print(f"  Error: {result.stderr}")
        except Exception as e:
            print(f"  Error running {method}: {e}")

if __name__ == "__main__":
    # test_method_shapes()
    # test_with_different_parameters()
    test_simple_run()