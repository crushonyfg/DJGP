from sklearn.cluster import KMeans
import numpy as np
import torch
import time
from DJGP_test import *
# from DJGP_CV_modified import *
from argparse import Namespace

def select_representative_points(X_train, Y_train, M, random_state=0):
    """
    使用 KMeans 从 X_train 中选择最具代表性的 M 个点
    
    参数:
        X_train: ndarray, shape (N, d) 输入特征
        Y_train: ndarray, shape (N, 1) 目标变量
        M: int，代表性点的数量
        random_state: int，用于 KMeans 的随机种子

    返回:
        X_selected: ndarray, shape (M, d) 被选中的代表点
        Y_selected: ndarray, shape (M, 1) 对应的标签
        selected_indices: list, 在 X_train 中选中点的索引
    """
    kmeans = KMeans(n_clusters=M, random_state=random_state)
    kmeans.fit(X_train)
    
    # 找每个聚类中心最近的点
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    selected_indices = []
    for i in range(M):
        cluster_points_idx = np.where(labels == i)[0]
        cluster_points = X_train[cluster_points_idx]
        dists = np.linalg.norm(cluster_points - centers[i], axis=1)
        closest_idx_in_cluster = cluster_points_idx[np.argmin(dists)]
        selected_indices.append(closest_idx_in_cluster)
    
    X_selected = X_train[selected_indices]
    Y_selected = Y_train[selected_indices]
    
    return X_selected, Y_selected, selected_indices


import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
# models = ['JGP', 'DGP', 'LMJGP']
models = ['LMJGP', 'JGP']

from scipy.io import loadmat

# 加载 .mat 文件
data = loadmat('result_OHT_4d')

# 查看文件中有哪些变量
print(data.keys())

X_train = data['xc']
Y_train = data['yc']
X_test = data['xt']
Y_test = data['yt']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

use_subsample = True
if use_subsample:
    X_test, Y_test, selected_indices = select_representative_points(X_test, Y_test, 500)

x_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(Y_test, dtype=torch.float32).squeeze(-1)
x_test = x_test.float().to(device)
y_test = y_test.float().squeeze(-1).to(device)

import torch
from JumpGP_test import *
from DeepGP_test import *
from argparse import Namespace

results = []

# M_candidate = [100, 200, 500, 1000, 2000]
M_candidate = [200]
# M_candidate = [200, 500]
for M_selected in M_candidate:
    X_selected, Y_selected, selected_indices = select_representative_points(X_train, Y_train, M_selected)

    x_train = torch.tensor(X_selected, dtype=torch.float32)
    y_train = torch.tensor(Y_selected, dtype=torch.float32).squeeze(-1)
    x_train = x_train.float().to(device)
    y_train = y_train.float().squeeze(-1).to(device)

    res = {"M_selected": M_selected}
    if 'JGP' in models:
        res_jgp = []
        for M in [20]:
            neighborhoods = find_neighborhoods(x_test, x_train, y_train, M)
            result = evaluate_jumpgp(x_test, y_test, neighborhoods, device)
            print(result)
            res_jgp.append(result)
        res['JGP'] = res_jgp


    if 'DGP' in models:
        args = Namespace(
            epochs=100,           # 训练轮数
            num_epochs=300,
            batch_size=1024,
            hidden_dim=2,
            lr=0.01,
            print_freq=10,        # 每多少个 epoch 打印一次信息
            # save_path="./model",  # 模型保存路径（可选）
            eval=True,            # 是否评估模型性能
            patience=30,          # 提前停止的耐心轮数
            clip_grad=1.0,        # 梯度裁剪（可选）
            # 如果你的 train_model 用了更多字段，也可以在这里继续加
        )


        # 创建数据加载器
        batch_size, hidden_dim, lr = 128, 2, 0.01
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # 初始化模型
        model = DeepGP(x_train.shape, hidden_dim).to(device)

        # 初始化优化器和损失函数
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr)
        mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, x_train.shape[-2]))

        # 训练模型
        final_metrics = train_model(model, train_loader, test_loader, optimizer, mll, args)
        print(final_metrics)
        res['DGP'] = final_metrics

    if 'LMJGP' in models:
        args = Namespace(
            epochs=100,           # 训练轮数
            num_epochs=300,
            batch_size=1024,
            hidden_dim=2,
            lr=0.01,
            print_freq=10,        # 每多少个 epoch 打印一次信息
            # save_path="./model",  # 模型保存路径（可选）
            Q = 2,
            m1 = 3,
            m2 = 50,
            n = 20,
            num_steps = 300,
            MC_num = 1,
            eval=True,            # 是否评估模型性能
            patience=10,          # 提前停止的耐心轮数
            clip_grad=1.0,        # 梯度裁剪（可选）
            # better_init=False,    # 是否使用更好的初始化
            # 如果你的 train_model 用了更多字段，也可以在这里继续加
        )
        

        # N_test, D = x_test.shape
        # T         = N_test
        # Q         = args.Q
        # m1        = args.m1
        # m2        = args.m2
        # n         = args.n
        # num_steps = args.num_steps
        # MC_num    = args.MC_num

        # # 4. Build neighborhoods (on CPU) then move to GPU
        # neighborhoods = find_neighborhoods(
        #     x_test.cpu(), x_train.cpu(), y_train.cpu(), M=n
        # )
        # regions = []
        # start_time = time.time()
        # for i in range(T):
        #     X_nb = neighborhoods[i]['X_neighbors'].to(device)  # (n, D)
        #     y_nb = neighborhoods[i]['y_neighbors'].to(device)  # (n,)
        #     regions.append({
        #         'X': X_nb,
        #         'y': y_nb,
        #         'C': torch.randn(m1, Q, device=device)       # random init
        #     })

        # # 5. Initialize V_params on GPU
        # V_params = {
        #     'mu_V':    torch.randn(m2, Q, D, device=device, requires_grad=True),
        #     'sigma_V': torch.rand( m2, Q, D, device=device, requires_grad=True),
        # }

        # # 6. Initialize u_params on GPU
        # u_params = []
        # for _ in range(T):
        #     u_params.append({
        #         'U_logit':    torch.zeros(1, device=device, requires_grad=True),
        #         'mu_u':       torch.randn(m1, device=device, requires_grad=True),
        #         'Sigma_u':    torch.eye(m1, device=device, requires_grad=True),
        #         'sigma_noise':torch.tensor(0.5, device=device, requires_grad=True),
        #         'sigma_k':torch.tensor(0.5, device=device, requires_grad=True),
        #         'omega':      torch.randn(Q+1, device=device, requires_grad=True),
        #     })

        # # 7. Initialize hyperparams on GPU
        # X_train_mean = x_train.mean(dim=0)
        # X_train_std  = x_train.std(dim=0)
        # Z = X_train_mean + torch.randn(m2, D, device=device) * X_train_std

        # hyperparams = {
        #     'Z':            Z,                      # (m2, D)
        #     'X_test':       x_test,                 # (T, D)
        #     'lengthscales':torch.rand(Q, device=device, requires_grad=True),
        #     'var_w':        torch.tensor(1.0, device=device, requires_grad=True),
        # }

        # print("Everything set!")
        num_steps = args.num_steps
        MC_num    = args.MC_num

        start_time = time.time()
        
        # Initialize model parameters
        regions, V_params, u_params, hyperparams = initialize_model(
            x_train, y_train, x_test, y_test, args, device
        )

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
        res_lmjgp = [rmse, mean_crps, run_time]
        print(f"Results [rmse, mean crps, runtime]:{[rmse, mean_crps, run_time]}")
        res['LMJGP'] = res_lmjgp
    
    # res = {"M_selected": M_selected, "res_jgp": res_jgp, "res_dgp": final_metrics, "res_lmjgp": res_lmjgp}
    results.append(res)

import pickle

# 保存
with open("res_OHTDataset.pkl", "wb") as f:
    pickle.dump(results, f)

# # 加载
# with open("my_list.pkl", "rb") as f:
#     loaded_list = pickle.load(f)





