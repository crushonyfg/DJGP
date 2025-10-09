from scipy.io import loadmat

# 加载 .mat 文件
data = loadmat('OHTDataset.mat')

# 查看文件中有哪些变量
print(data.keys())

X_train = data['x']
Y_train = data['y']
X_test = data['xt']
Y_test = data['yt']

import numpy as np

use_subsample = False
prop = 4

if use_subsample:
    # 假设 X_test, Y_test 都是 numpy 数组
    n_test = X_test.shape[0]
    n_sub  = n_test // prop           # 向下取整得到 1/4 的样本数

    # 随机不放回抽取 n_sub 个索引
    idx = np.random.choice(n_test, size=n_sub, replace=False)

    # 取出子集
    X_test = X_test[idx]
    Y_test = Y_test[idx]


import torch
x_train = torch.tensor(X_train, dtype=torch.float32)
x_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(Y_train, dtype=torch.float32).squeeze(-1)
y_test = torch.tensor(Y_test, dtype=torch.float32).squeeze(-1)

from JumpGP_test import *

M = 20
neighborhoods = find_neighborhoods(x_test, x_train, y_train, M)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
result = evaluate_jumpgp(x_test, y_test, neighborhoods, device)
print(result)

# # 标准化输入特征
# X_mean = x_train.mean(0)
# X_std = x_train.std(0)
# x_train = (x_train - X_mean) / X_std
# x_test = (x_test - X_mean) / X_std

# # 标准化输出目标
# y_mean = y_train.mean()
# y_std = y_train.std()
# y_train = (y_train - y_mean) / y_std
# y_test = (y_test - y_mean) / y_std


from DJGP_test import *
from argparse import Namespace
args = Namespace(
    epochs=100,           # 训练轮数
    num_epochs=300,
    batch_size=1024,
    hidden_dim=2,
    lr=0.01,
    print_freq=10,        # 每多少个 epoch 打印一次信息
    # save_path="./model",  # 模型保存路径（可选）
    Q = 2,
    m1 = 2,
    m2 = 20,
    n = 20,
    num_steps = 500,
    MC_num = 3,
    eval=True,            # 是否评估模型性能
    patience=10,          # 提前停止的耐心轮数
    clip_grad=1.0,        # 梯度裁剪（可选）
    # 如果你的 train_model 用了更多字段，也可以在这里继续加
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = x_train.float().to(device)
Y_train = y_train.float().squeeze(-1).to(device)
X_test = x_test.float().to(device)
Y_test = y_test.float().squeeze(-1).to(device)

N_test, D = X_test.shape
T         = N_test
Q         = args.Q
m1        = args.m1
m2        = args.m2
n         = args.n
num_steps = args.num_steps
MC_num    = args.MC_num

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

V_params, u_params, hyperparams = train_vi(
    regions=regions,
    V_params=V_params,
    u_params=u_params,
    hyperparams=hyperparams,
    lr=args.lr,
    num_steps=num_steps,
    log_interval=50
)
print("train OK")

mu_pred, var_pred = predict_vi(
    regions, V_params, hyperparams, M=MC_num
)
print("Prediction OK")

# mu_pred = mu_pred * y_std + y_mean
# var_pred = var_pred * (y_std ** 2)

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
print(f"Results [rmse, mean crps, runtime]:{[rmse, mean_crps, run_time]}")


# from JumpGP_test import *

# M = args.n
# neighborhoods = find_neighborhoods(x_test, x_train, y_train, M)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# result = evaluate_jumpgp(x_test, y_test, neighborhoods, device)
# print(result)