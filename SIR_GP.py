# python SIR_GP.py --folder_name "2025_04_01_23" --D 10 --use_sir --sir_K 2
import os
import pickle
import time
import numpy as np
import torch
import argparse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error

def parse_args():
    parser = argparse.ArgumentParser(description='Test GP model')
    parser.add_argument('--folder_name', type=str, required=True, 
                      help='Folder name containing dataset.pkl')
    parser.add_argument('--M', type=int, default=100, 
                      help='Number of nearest neighbors')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to use (cpu/cuda)')
    parser.add_argument('--use_sir', action='store_true',
                      help='Whether to use SIR dimension reduction')
    parser.add_argument('--sir_H', type=int, default=10,
                      help='Number of slices for SIR')
    parser.add_argument('--sir_K', type=int, default=2,
                      help='Number of components to keep in SIR')
    parser.add_argument('--D', type=int, default=10,
                      help='Number of dimensions for RBF kernel')
    return parser.parse_args()

def train_and_evaluate(X_train, Y_train, X_test, Y_test, args):
    start_time = time.time()
    
    # Convert to numpy for sklearn compatibility
    X_train1 = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
    Y_train1 = Y_train.numpy() if isinstance(Y_train, torch.Tensor) else Y_train
    X_test1 = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test
    Y_test1 = Y_test.numpy() if isinstance(Y_test, torch.Tensor) else Y_test

    # Get actual data dimensions
    n_dimensions = X_train1.shape[1]
    
    # Initialize kernel with correct dimensions
    length_scales = np.ones(n_dimensions)  # 使用实际数据维度
    kernel = (C(1.0, (1e-3, 1e3)) * 
             RBF(length_scale=length_scales, length_scale_bounds=(1e-2, 1e2)) + 
             WhiteKernel(1e-5, (1e-10, 1e-2)))

    # Train GP model
    print("start training")
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
    gp.fit(X_train1, Y_train1)

    # Predict and evaluate
    Y_pred, std = gp.predict(X_test1, return_std=True)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(Y_test1, Y_pred))
    nlpd = 0.5 * np.log(2 * np.pi * (std**2)) + 0.5 * ((Y_test1 - Y_pred)**2) / (std**2)
    
    # Calculate percentiles
    nlpd_25 = np.percentile(nlpd, 25)
    nlpd_50 = np.percentile(nlpd, 50)
    nlpd_75 = np.percentile(nlpd, 75)
    
    run_time = time.time() - start_time
    
    return [rmse, nlpd_25, nlpd_50, nlpd_75, run_time]

def main():
    # Parse arguments
    args = parse_args()
    
    # Load dataset
    dataset_path = os.path.join(args.folder_name, 'dataset.pkl')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    X_train = dataset["X_train"]
    Y_train = dataset["Y_train"]
    X_test = dataset["X_test"]
    Y_test = dataset["Y_test"]
    
    # Apply SIR reduction if specified
    if args.use_sir:
        from JumpGP_test import apply_sir_reduction
        X_train, X_test = apply_sir_reduction(X_train, Y_train, X_test, args)
    
    # Train and evaluate
    results = train_and_evaluate(X_train, Y_train, X_test, Y_test, args)
    
    # Print results
    print(f"\nResults:")
    print(f"RMSE: {results[0]:.4f}")
    print(f"NLPD (25th percentile): {results[1]:.4f}")
    print(f"NLPD (50th percentile): {results[2]:.4f}")
    print(f"NLPD (75th percentile): {results[3]:.4f}")
    print(f"Runtime: {results[4]:.2f} seconds")
    
    return results

if __name__ == "__main__":
    main()