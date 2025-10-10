# ====== run_synth_and_plot_acq.py ======
import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# 按你的工程实际来改；默认假设与 active_djgp_acquisition.py 在同一目录
from active_djgp_acquisition import *
# 需要用到：DJGPArgs, djgp_fit_predict, compute_acquisition_all_with_JGP,
# select_topk_candidates, find_neighborhoods, evaluate_jumpgp, predict_vi

import sys
import os
sys.path.append("D:/new_windows/PhD/autumn2025/park/codes/DJGP/ActiveJGP")  # Replace with actual path to ActiveJGP folder
from ActiveLearner import ActiveLearner

# -------------------------------
# 1) 生成二维网格 Z（用于背景绘图）
# -------------------------------
def generate_Z_grid(n=150):
    z1 = torch.linspace(0, 1, n)
    z2 = torch.linspace(0, 1, n)
    Z1, Z2 = torch.meshgrid(z1, z2, indexing='ij')
    Z = torch.stack([Z1.flatten(), Z2.flatten()], dim=-1)  # [n*n, 2]
    return Z, Z1, Z2

# -------------------------------
# 2) 2D -> 4D 映射 g 及近似逆
# -------------------------------
def g(Z: torch.Tensor) -> torch.Tensor:
    """
    Global mapping: R^2 -> R^4
    这里保持与之前一致，便于在 4D 上定义边界：
      X1 = z1
      X2 = z2
      X3 = sin(2π z1) cos(2π z2)
      X4 = sqrt(z1^2 + z2^2)
    """
    z1, z2 = Z[:, 0], Z[:, 1]
    X1 = z1
    X2 = z2
    X3 = torch.sin(2 * math.pi * z1) * torch.cos(2 * math.pi * z2)
    X4 = torch.sqrt(z1 ** 2 + z2 ** 2 + 1e-12)
    return torch.stack([X1, X2, X3, X4], dim=-1)

def g_inv(X: torch.Tensor) -> torch.Tensor:
    # 前两个维度直接返回
    return X[:, :2]

# -------------------------------
# 3) 4D 决策边界 (隐式) 及其在 Z 上的残差
# -------------------------------
def boundary_residual_Z(Z: torch.Tensor, a: float = 0.2) -> torch.Tensor:
    """
    给定 Z（[N,2]），映射到 X=g(Z)，返回 4D 边界的残差：
      h_X(Z) = X4 - 1 / (a + |X1 - X2 - X3|)
    边界为 h_X(Z)=0。
    """
    X = g(Z)  # [N,4]
    num = X[:, 3]
    den = a + torch.abs(X[:, 0] - X[:, 1] - X[:, 2])
    return num - 1.0 / (den + 1e-12)

# -------------------------------
# 4) 合成目标 y(Z)：共享 GP + 沿 4D 边界的“连续 jump”
# -------------------------------
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def generate_y_from_Z(
    Z: torch.Tensor,
    gp_model: GaussianProcessRegressor | None = None,
    a: float = 0.2,
    # k_slope: float = 25.0,
    # jump_amp: float = 1.0,
    k_slope: float = 50.0,
    jump_amp: float = 2.0,
    update_posterior: bool = False
):
    """
    使用共享的 GP prior + 沿 4D 决策面的连续 jump。
    - 决策面： x4 = 1/(a + |x1 - x2 - x3|)  （在 4D X 空间上定义）
    - jump 由 sigma(k * h_X(Z)) 给出，h_X(Z)见上
    - 最终 y = y_gp + jump_amp * sigmoid(k * h_X(Z))

    返回:
      y: torch.Tensor [N]
      gp_model: 共享的 sklearn GP 模型（便于后续网格可视化使用）
    """
    if gp_model is None:
        kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=0.3)
        gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-4)

        # 用少量随机点先拟合一下 “prior driver”
        init_Z = np.random.rand(30, 2)
        init_y = np.sin(6 * init_Z[:, 0]) * np.cos(6 * init_Z[:, 1])
        gp_model.fit(init_Z, init_y)

    # 共享 GP prior 生成平滑项
    y_gp, _ = gp_model.predict(Z.detach().cpu().numpy(), return_std=True)
    y_gp = torch.tensor(y_gp, dtype=torch.float32, device=Z.device)

    # 4D 边界残差 (在 Z 上评估)
    hX = boundary_residual_Z(Z, a=a)  # [N]
    jump = torch.sigmoid(k_slope * hX) * jump_amp

    y = y_gp + jump

    if update_posterior:
        gp_model.fit(Z.detach().cpu().numpy(), y.detach().cpu().numpy())

    return y, gp_model

# -------------------------------
# 5) 在 Z 上可视化背景 y(Z) + 4D 决策边界
# -------------------------------
def plot_dataset_with_4D_boundary(
    Z_grid: torch.Tensor,
    y_grid: torch.Tensor,
    Z_train: torch.Tensor | None = None,
    title: str = "Dataset (gray=y(Z), blue=4D boundary projected to Z)"
):
    """
    在 2D Z 平面：
      - 灰度显示 y(Z)
      - 画出 4D 决策边界在 Z 上的等高线 (h_X(Z)=0)
      - 可选：画训练点（绿色）
    """
    G = Z_grid.shape[0]
    n = int(np.sqrt(G))
    Z1 = Z_grid[:, 0].view(n, n).cpu().numpy()
    Z2 = Z_grid[:, 1].view(n, n).cpu().numpy()
    Y  = y_grid.view(n, n).detach().cpu().numpy()

    # 边界残差
    h = boundary_residual_Z(Z_grid).view(n, n).detach().cpu().numpy()

    plt.figure(figsize=(7, 6))
    cs = plt.contourf(Z1, Z2, Y, levels=50, cmap='gray', alpha=0.95)
    plt.colorbar(cs, label='y(Z)')

    # 4D 决策边界在 Z 上的投影：h_X(Z)=0
    plt.contour(Z1, Z2, h, levels=[0.0], colors='blue', linewidths=2, linestyles='-')

    if Z_train is not None:
        plt.scatter(Z_train[:, 0].cpu().numpy(), Z_train[:, 1].cpu().numpy(),
                    c='limegreen', s=35, edgecolors='k', linewidths=0.5, label='train (Z)')

    plt.title(title)
    plt.xlabel('z1'); plt.ylabel('z2')
    if Z_train is not None:
        plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.show()

# -------------------------------
# 6) 在 Z 上叠加 acquisition（与你之前版本一致，只是边界换成 4D）
# -------------------------------
def plot_Z_with_acquisition(Z_grid, y_grid, Z_train, Z_cand, A, topk_idx, title="Acquisition on Z"):
    """
    - 背景：y(Z) 灰度
    - 蓝色线：4D 决策边界在 Z 上的投影 (h_X(Z)=0)
    - 绿色点：训练点
    - 红色点：候选点（颜色映射 acquisition 大小），Top-K 黄圈
    """
    G = Z_grid.shape[0]
    n = int(np.sqrt(G))
    Z1 = Z_grid[:, 0].view(n, n).cpu().numpy()
    Z2 = Z_grid[:, 1].view(n, n).cpu().numpy()
    Y  = y_grid.view(n, n).detach().cpu().numpy()
    h  = boundary_residual_Z(Z_grid).view(n, n).detach().cpu().numpy()

    plt.figure(figsize=(7, 6))
    cs = plt.contourf(Z1, Z2, Y, levels=50, cmap='gray', alpha=0.95)
    plt.colorbar(cs, label='y(Z)')
    plt.contour(Z1, Z2, h, levels=[0.0], colors='blue', linewidths=2)

    plt.scatter(Z_train[:, 0].cpu().numpy(), Z_train[:, 1].cpu().numpy(),
                c='limegreen', s=35, edgecolors='k', linewidths=0.5, label='train (Z)')

    A = np.asarray(A, dtype=float)
    sc = plt.scatter(Z_cand[:, 0].cpu().numpy(), Z_cand[:, 1].cpu().numpy(),
                     c=A, cmap='plasma', s=45, edgecolors='k', linewidths=0.4, label='candidates (Z)')
    plt.colorbar(sc, label='Acquisition A(x)')

    if topk_idx is not None and len(topk_idx) > 0:
        plt.scatter(Z_cand[topk_idx, 0].cpu().numpy(), Z_cand[topk_idx, 1].cpu().numpy(),
                    facecolors='none', edgecolors='yellow', s=160, linewidths=2.0, label='Top-K')

    plt.title(title)
    plt.xlabel('z1'); plt.ylabel('z2')
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.show()

def build_eval_regions(X_train, Y_train, X_eval, args, device):
    """
    用 (X_train, Y_train) 给评估集 X_eval 构建 regions（只需要邻域 & C 随机初始化）。
    返回：regions_eval (不改动 V_params/hyperparams)
    """
    T, D = X_eval.shape
    Q, m1, n = args.Q, args.m1, args.n

    neighborhoods = find_neighborhoods(
        X_eval.cpu(), X_train.cpu(), Y_train.cpu(), M=n
    )
    regions_eval = []
    for i in range(T):
        X_nb = neighborhoods[i]['X_neighbors'].to(device)   # (n, D)
        y_nb = neighborhoods[i]['y_neighbors'].to(device)   # (n,)
        regions_eval.append({
            'X': X_nb,
            'y': y_nb,
            'C': torch.randn(m1, Q, device=device)          # 评估用，随便初始化即可
        })
    return regions_eval

# -------------------------------
# 7) 主流程：先画数据集与4D边界，再按原流程做 AL
# -------------------------------
def main_synth_and_plot(seed=123, N_train=50, N_cand=200, N_test=200, K_top=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (a) 采样 Z 并生成 y（用 4D 边界产生 jump），再映射到 X（4D）
    Z_train = torch.rand(N_train, 2, device=device)
    Z_cand  = torch.rand(N_cand,  2, device=device)
    Z_test  = torch.rand(N_test,  2, device=device)

    y_train, gp_model = generate_y_from_Z(Z_train)          # 共享同一个 GP prior + 4D 跳变
    y_test,  _        = generate_y_from_Z(Z_test, gp_model)

    X_train = g(Z_train)
    X_cand  = g(Z_cand)
    X_test  = g(Z_test)

    # (a.1) 可视化合成数据与 4D 边界（先给你看效果）
    Z_grid, _, _ = generate_Z_grid(n=150)
    Z_grid = Z_grid.to(device)
    y_grid, _ = generate_y_from_Z(Z_grid, gp_model)
    plot_dataset_with_4D_boundary(
        Z_grid=Z_grid, y_grid=y_grid, Z_train=Z_train.detach().cpu(),
        title="Synth dataset with 4D boundary projected to Z"
    )

    # (b) 评估 JGP（独立测试集）
    neighborhoods = find_neighborhoods(X_test.detach().cpu(), X_train.detach().cpu(), y_train.detach().cpu(), M=25)
    rmse_jgp, crps_jgp, _ = evaluate_jumpgp(X_test, y_test, neighborhoods, device, std=None)
    print(f"[Baseline JGP] RMSE={float(rmse_jgp):.4f}, CRPS={float(crps_jgp):.4f}")

    # (c) 训练 DJGP，并把 X_cand 当作 X_test（用于后续 acquisition 的 q(W|x)）
    args = DJGPArgs(Q=2, m1=4, m2=20, n=25, num_steps=250, MC_num=6, lr=1e-2, use_batch=False)
    print("\n[Train DJGP] ...")
    res = djgp_fit_predict(
        X_train=X_train, Y_train=y_train,
        X_test=X_cand,   Y_test=None,
        args=args, device=device
    )

    regions     = res['regions']
    V_params    = res['V_params']
    hyperparams = res['hyperparams']
    jgp_args    = res['jgp_args']

    # (d) 在独立测试集上评估 DJGP
    regions_eval = build_eval_regions(X_train, y_train, X_test, args, device)
    hyper_eval   = dict(hyperparams)
    hyper_eval['X_test'] = X_test

    mu_eval, var_eval = predict_vi(regions_eval, V_params, hyper_eval, M=args.MC_num)
    rmse_dj, crps_dj = compute_metrics(mu_eval, torch.sqrt(var_eval.clamp_min(1e-12)), y_test)
    print(f"[Baseline DJGP] RMSE={float(rmse_dj):.4f}, CRPS={float(crps_dj):.4f}")

    # (e) 计算候选集 acquisition 并画在 Z 上
    print("\n[Compute acquisition on candidates] ...")
    A, parts = compute_acquisition_all_with_JGP(
        regions=regions, V_params=V_params, hyperparams=hyperparams,
        jgp_args=jgp_args, M=args.MC_num, lambda_=0.3
    )
    top_idx = select_topk_candidates(A, K=K_top)
    print(f"Top-{K_top} idx:", top_idx.tolist())
    print("Top-K scores:", A[top_idx])

    plot_Z_with_acquisition(
        Z_grid=Z_grid, y_grid=y_grid,
        Z_train=Z_train.detach().cpu(), Z_cand=Z_cand.detach().cpu(),
        A=A, topk_idx=top_idx,
        title=f"DJGP acquisition over Z  |  JGP RMSE={float(rmse_jgp):.3f} DJGP RMSE={float(rmse_dj):.3f}"
    )

# -------------------------------
# 8) 一个完整的 AL 循环（保持你之前的逻辑不变）
#    这里沿用你现成的 run_al_loop，只是数据生成换成新的 4D 边界版本
# -------------------------------
def run_al_loop(
    seed=123,
    N_train=50,
    N_cand=200,
    N_test=200,
    rounds=30,
    KNN=25,
    save_prefix="djgp_al_4Dboundary"
):
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 数据，用 4D 边界版本 ---
    Z_train = torch.rand(N_train, 2, device=device)
    Z_pool  = torch.rand(N_cand,  2, device=device)
    Z_test  = torch.rand(N_test,  2, device=device)

    y_train, gp_model = generate_y_from_Z(Z_train)
    y_pool, _         = generate_y_from_Z(Z_pool,  gp_model)
    y_test, _         = generate_y_from_Z(Z_test,  gp_model)

    X_train = g(Z_train)
    X_pool  = g(Z_pool)
    X_test  = g(Z_test)

    args = DJGPArgs(Q=2, m1=4, m2=20, n=KNN, num_steps=100, MC_num=8, lr=1e-2, use_batch=False)

    rmse_jgp_hist, crps_jgp_hist = [], []
    rmse_djg_hist, crps_djg_hist = [], []
    picked_Z_list = []

    cmap = plt.get_cmap("viridis")

    # Round-0 baselines
    neighborhoods0 = find_neighborhoods(X_test.detach().cpu(), X_train.detach().cpu(), y_train.detach().cpu(), M=KNN)
    rmse0_jgp, crps0_jgp, _ = evaluate_jumpgp(X_test, y_test, neighborhoods0, device, std=None)
    rmse_jgp_hist.append(float(rmse0_jgp))
    crps_jgp_hist.append(float(crps0_jgp))

    print("\n[Round 0] Train DJGP for baseline ...")
    res0 = djgp_fit_predict(X_train, y_train, X_pool, None, args=args, device=device)
    regions0, V_params0, hyper0 = res0['regions'], res0['V_params'], res0['hyperparams']

    regions_eval0 = build_eval_regions(X_train, y_train, X_test, args, device)
    hyper_eval0   = dict(hyper0)
    hyper_eval0['X_test'] = X_test
    mu0_dj, var0_dj = predict_vi(regions_eval0, V_params0, hyper_eval0, M=args.MC_num)
    rmse0_dj, crps0_dj = compute_metrics(mu0_dj, torch.sqrt(var0_dj.clamp_min(1e-12)), y_test)
    rmse_djg_hist.append(float(rmse0_dj))
    crps_djg_hist.append(float(crps0_dj))
    print(f"[Round 0] JGP: RMSE={rmse0_jgp:.4f} CRPS={crps0_jgp:.4f} | DJGP: RMSE={rmse0_dj:.4f} CRPS={crps0_dj:.4f}")

    for r in range(1, rounds + 1):
        print(f"\n[Round {r}] Train DJGP & compute acquisition on pool (|pool|={len(X_pool)})")
        res = djgp_fit_predict(X_train, y_train, X_pool, None, args=args, device=device)
        regions, V_params, hyperparams = res['regions'], res['V_params'], res['hyperparams']
        jgp_args = res['jgp_args']

        # DJGP eval
        regions_eval = build_eval_regions(X_train, y_train, X_test, args, device)
        hyper_eval   = dict(hyperparams)
        hyper_eval['X_test'] = X_test
        mu_eval, var_eval = predict_vi(regions_eval, V_params, hyper_eval, M=args.MC_num)
        rmse_dj, crps_dj = compute_metrics(mu_eval, torch.sqrt(var_eval.clamp_min(1e-12)), y_test)
        rmse_djg_hist.append(float(rmse_dj))
        crps_djg_hist.append(float(crps_dj))

        # JGP eval
        neighborhoods = find_neighborhoods(X_test.detach().cpu(), X_train.detach().cpu(), y_train.detach().cpu(), M=KNN)
        rmse_jgp, crps_jgp, _ = evaluate_jumpgp(X_test, y_test, neighborhoods, device, std=None)
        rmse_jgp_hist.append(float(rmse_jgp))
        crps_jgp_hist.append(float(crps_jgp))
        print(f"[Round {r}] JGP: RMSE={rmse_jgp:.4f} CRPS={crps_jgp:.4f} | DJGP: RMSE={rmse_dj:.4f} CRPS={crps_dj:.4f}")

        # Acquisition Top-1
        A, _ = compute_acquisition_all_with_JGP(regions, V_params, hyperparams, jgp_args=jgp_args, M=args.MC_num, lambda_=0.3)
        best_idx = int(select_topk_candidates(A, K=1)[0])

        # Query "oracle"
        x_new = X_pool[best_idx:best_idx+1]
        z_new = Z_pool[best_idx:best_idx+1]
        y_new = y_pool[best_idx:best_idx+1]

        X_train = torch.cat([X_train, x_new], dim=0)
        Z_train = torch.cat([Z_train, z_new], dim=0)
        y_train = torch.cat([y_train, y_new.view(-1)], dim=0)

        X_pool  = torch.cat([X_pool[:best_idx],  X_pool[best_idx+1:]],  dim=0)
        Z_pool  = torch.cat([Z_pool[:best_idx],  Z_pool[best_idx+1:]],  dim=0)
        y_pool  = torch.cat([y_pool[:best_idx],  y_pool[best_idx+1:]],  dim=0)

        picked_Z_list.append(z_new.squeeze(0).detach().cpu().numpy())
        if len(X_pool) == 0:
            print("[Early stop] pool exhausted.")
            break

    # --- 绘制 Z 背景 + 4D 边界 + 初始训练点 + 每轮新增点 ---
    print("\n[Plot] Z-plane with 4D boundary and added points ...")
    Z_grid, _, _ = generate_Z_grid(n=150)
    Z_grid = Z_grid.to(device)
    y_grid, _ = generate_y_from_Z(Z_grid, gp_model)

    def plot_Z_bg(ax):
        G = Z_grid.shape[0]
        n = int(np.sqrt(G))
        Z1 = Z_grid[:, 0].view(n, n).cpu().numpy()
        Z2 = Z_grid[:, 1].view(n, n).cpu().numpy()
        Y  = y_grid.view(n, n).detach().cpu().numpy()
        h  = boundary_residual_Z(Z_grid).view(n, n).detach().cpu().numpy()
        cs = ax.contourf(Z1, Z2, Y, levels=50, cmap='gray', alpha=0.95)
        ax.contour(Z1, Z2, h, levels=[0.0], colors='blue', linewidths=2)

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_Z_bg(ax)

    init_N = N_train
    ax.scatter(Z_train[:init_N, 0].cpu().numpy(), Z_train[:init_N, 1].cpu().numpy(),
               c='limegreen', s=35, edgecolors='k', linewidths=0.5, label='initial train (Z)')

    if picked_Z_list:
        picked_Z = np.stack(picked_Z_list, axis=0)
        R = picked_Z.shape[0]
        for t in range(R):
            color = cmap((t + 1) / max(R, 1))
            ax.scatter(picked_Z[t, 0], picked_Z[t, 1],
                       facecolors='none', edgecolors=color,
                       s=160, linewidths=2.0, label=None if t > 0 else 'added per round')

    ax.set_title("Active Learning on Z with 4D boundary")
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_Z_added_points.png", dpi=200)
    plt.close(fig)

    # --- RMSE 曲线 (JGP vs DJGP) ---
    rounds_axis = np.arange(0, len(rmse_jgp_hist))
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(rounds_axis, rmse_jgp_hist, marker='o', label='JGP RMSE')
    ax2.plot(rounds_axis, rmse_djg_hist, marker='s', label='DJGP RMSE')
    ax2.set_xlabel("AL Round")
    ax2.set_ylabel("RMSE on held-out test")
    ax2.set_title("RMSE vs Rounds")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(f"{save_prefix}_rmse_curves.png", dpi=200)
    plt.close(fig2)

    # --- 保存关键数组 ---
    np.savez(
        f"{save_prefix}_data.npz",
        X_train=X_train.detach().cpu().numpy(),
        Z_train=Z_train.detach().cpu().numpy(),
        X_test=X_test.detach().cpu().numpy(),
        Z_test=Z_test.detach().cpu().numpy(),
        y_train=y_train.detach().cpu().numpy(),
        y_test=y_test.detach().cpu().numpy(),
        picked_Z=np.stack(picked_Z_list) if picked_Z_list else np.zeros((0, 2)),
        rmse_jgp=np.array(rmse_jgp_hist),
        rmse_djgp=np.array(rmse_djg_hist),
        crps_jgp=np.array(crps_jgp_hist),
        crps_djgp=np.array(crps_djg_hist),
        seed=np.array([seed])
    )
    print(f"[Saved] {save_prefix}_Z_added_points.png, {save_prefix}_rmse_curves.png, {save_prefix}_data.npz")

def run_al4jgp(seed=123,
    N_train=50,
    N_cand=200,
    N_test=200,
    rounds=30,
    KNN=25,
    save_prefix="djgp_al4jgp_4Dboundary"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 数据，用 4D 边界版本 ---
    Z_train = torch.rand(N_train, 2, device=device)
    Z_pool  = torch.rand(N_cand,  2, device=device)
    Z_test  = torch.rand(N_test,  2, device=device)

    y_train, gp_model = generate_y_from_Z(Z_train)
    y_pool, _         = generate_y_from_Z(Z_pool,  gp_model)
    y_test, _         = generate_y_from_Z(Z_test,  gp_model)

    X_train = g(Z_train)
    X_pool  = g(Z_pool)
    X_test  = g(Z_test)

    y_train_np = y_train.detach().cpu().numpy()
    y_pool_np  = y_pool.detach().cpu().numpy()
    y_test_np  = y_test.detach().cpu().numpy()

    X_train_np = X_train.detach().cpu().numpy()
    X_pool_np  = X_pool.detach().cpu().numpy()
    X_test_np  = X_test.detach().cpu().numpy()

    learner = ActiveLearner(
        x=X_train_np,
        y=y_train_np,
        xc=X_pool_np,
        yc=y_pool_np,
        xt=X_test_np,
        yt=y_test_np,
        method_name='MAX_MSPE',  # Choose your method
        S=rounds,  # Number of iterations
        neighbor_num = KNN
    )

    # Run active learning
    results = learner.run()

    # Plot results
    import matplotlib.pyplot as plt
    plt.plot(results['rmse'])
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('Learning Progress')
    plt.show()



# -------------------------------
# 9) 运行入口
# -------------------------------
if __name__ == "__main__":
    # 先把“数据集 + 4D 决策边界”画出来让你检查
    # main_synth_and_plot(seed=123, N_train=50, N_cand=200, N_test=200, K_top=5)

    # 通过后，直接跑 AL 实验（保留原有逻辑）
    # 你也可以注释掉上面一行，只运行下面的主动学习循环
    run_al4jgp(
        seed=123,
        N_train=40, N_cand=200, N_test=200,
        rounds=50, KNN=25,
        save_prefix="djgp_al_4Dboundary_knn25"
    )
