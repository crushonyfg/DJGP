# ====== run_synth_and_plot_acq.py（或粘到 active_djgp_acquisition.py 末尾） ======
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# 如果这些函数在别的文件里，请按你的工程实际调整 import
# （下行假定与 active_djgp_acquisition.py 在同一目录）
# from active_djgp_acquisition import djgp_fit_predict, DJGPArgs, compute_acquisition_all_with_JGP, select_topk_candidates
from active_djgp_acquisition import *

# -------- 1) 生成二维网格 Z（仅用于背景绘图） --------
def generate_Z_grid(n=120):
    z1 = torch.linspace(0, 1, n)
    z2 = torch.linspace(0, 1, n)
    Z1, Z2 = torch.meshgrid(z1, z2, indexing='ij')
    Z = torch.stack([Z1.flatten(), Z2.flatten()], dim=-1)  # [n*n, 2]
    return Z, Z1, Z2

# -------- 2) 构造 jump-like + GP 的合成目标（与你之前一致）--------
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# ===== 新增：仅用于评估集的邻域构建（重用训练数据）=====
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


def generate_y_from_Z(Z, gp_model=None, update_posterior=False):
    """
    Z: [N,2] torch.tensor in [0,1]^2
    """
    if gp_model is None:
        kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=0.3)
        gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-4)
        init_Z = np.random.rand(20, 2)
        init_y = np.sin(6*init_Z[:,0]) * np.cos(6*init_Z[:,1])
        gp_model.fit(init_Z, init_y)

    z1, z2 = Z[:,0], Z[:,1]
    # sigmoidal boundary: h(z) = z1 - sigmoid(8*(z2-0.5))
    h = z1 - torch.sigmoid(8*(z2 - 0.5))
    boundary = torch.sigmoid(20 * h)           # 连续但陡峭的“跃迁”
    slope_effect = 1.0 * boundary

    # 共享 GP prior 的采样
    y_gp, _ = gp_model.predict(Z.numpy(), return_std=True)
    y_gp = torch.tensor(y_gp, dtype=torch.float32)

    y = y_gp + slope_effect

    if update_posterior:
        gp_model.fit(Z.numpy(), y.numpy())
    return y, gp_model

# -------- 3) 2D -> 4D 映射及近似逆（和你之前一致）--------
def g(Z):
    z1, z2 = Z[:,0], Z[:,1]
    X1 = z1
    X2 = z2
    X3 = torch.sin(2*np.pi*z1) * torch.cos(2*np.pi*z2)
    X4 = torch.sqrt(z1**2 + z2**2)
    return torch.stack([X1, X2, X3, X4], dim=-1)

def g_inv(X):
    return X[:, :2]

# -------- 4) 可视化：在 2D Z 上叠加 acquisition --------
def plot_Z_with_acquisition(Z_grid, y_grid, Z_train, Z_cand, A, topk_idx, title="Acquisition on Z"):
    """
    Z_grid: [G,2] (torch) 网格
    y_grid: [G]   (torch) 网格上的 y 值
    Z_train: [N,2] (torch)
    Z_cand:  [M,2] (torch)
    A:       [M]   (np.ndarray) 每个候选的 acquisition
    topk_idx:[K]   (np.ndarray) Top-K 的索引（相对于 Z_cand）
    """
    # 准备画布数据
    G = Z_grid.shape[0]
    n = int(np.sqrt(G))
    Z1 = Z_grid[:,0].reshape(n, n).numpy()
    Z2 = Z_grid[:,1].reshape(n, n).numpy()
    Y  = y_grid.reshape(n, n).numpy()

    plt.figure(figsize=(7,6))
    # 背景：y(Z) 灰度
    cs = plt.contourf(Z1, Z2, Y, levels=50, cmap='gray', alpha=0.95)
    cbar = plt.colorbar(cs, label='y(Z)')

    # 决策面：h(z)=0（与数据生成一致）
    h = Z1 - 1.0/(1.0 + np.exp(-8*(Z2 - 0.5)))
    plt.contour(Z1, Z2, h, levels=[0.0], colors='blue', linewidths=2)

    # 训练点：绿色
    plt.scatter(Z_train[:,0].numpy(), Z_train[:,1].numpy(),
                c='limegreen', s=35, edgecolors='k', linewidths=0.5, label='train (Z)')

    # 候选点：红色，并用颜色表示 acquisition 强弱
    A = np.asarray(A, dtype=float)
    sc = plt.scatter(Z_cand[:,0].numpy(), Z_cand[:,1].numpy(),
                     c=A, cmap='plasma', s=45, edgecolors='k', linewidths=0.4, label='candidates (Z)')
    cbar2 = plt.colorbar(sc, label='Acquisition A(x)')

    # Top-K：空心黄圈
    if topk_idx is not None and len(topk_idx) > 0:
        plt.scatter(Z_cand[topk_idx, 0].numpy(), Z_cand[topk_idx, 1].numpy(),
                    facecolors='none', edgecolors='yellow', s=160, linewidths=2.0, label='Top-K')

    plt.title(title)
    plt.xlabel('z1'); plt.ylabel('z2')
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.show()

# -------- 5) 主流程：合成→训练DJGP→acquisition→在Z上绘图 --------
def main_synth_and_plot(seed=123, N_train=50, N_cand=200, N_test=200, K_top=5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (a) 采样 Z 并生成 y，再映射到 X（4D）
    Z_train = torch.rand(N_train, 2)
    Z_cand  = torch.rand(N_cand,  2)
    Z_test  = torch.rand(N_test,  2)         # <- 独立测试集（用于评估）

    y_train, gp_model = generate_y_from_Z(Z_train)            # 共享同一个 GP prior
    y_test,  _        = generate_y_from_Z(Z_test, gp_model)   # 用同一个 gp_model 生成

    X_train = g(Z_train)
    X_cand  = g(Z_cand)     # 候选集（用于 acquisition）
    X_test  = g(Z_test)     # 评估集（用于 RMSE/CRPS）

    # (b) 训练 DJGP，并在候选 X_cand 上作为 X_test（为了后续 acquisition 的 q(W|x)）
    args = DJGPArgs(Q=2, m1=4, m2=20, n=25, num_steps=300, MC_num=5, lr=1e-2, use_batch=False)

    neighborhoods = find_neighborhoods(X_test, X_train, y_train, args.n)
    rmse, mean_crps, run_time = evaluate_jumpgp(X_test, y_test, neighborhoods, device, std=None)
    print(f"[Test] RMSE of JGP={float(rmse):.4f}, CRPS={float(mean_crps):.4f}")

    print("\n[Train DJGP] ...")
    res = djgp_fit_predict(
        X_train=X_train, Y_train=y_train,
        X_test=X_cand,   Y_test=None,          # 此处不评估，只训练 + 建立候选的 q(W|x)
        args=args, device=device
    )

    regions     = res['regions']
    V_params    = res['V_params']
    hyperparams = res['hyperparams']
    jgp_args    = res['jgp_args']

    # (b.1) 评估：在独立 X_test 上做预测并计算 RMSE/CRPS
    # 构建评估用的 regions（用训练集邻域），并临时替换 X_test
    print("\n[Evaluate on held-out test set] ...")
    regions_eval = build_eval_regions(X_train, y_train, X_test, args, device)
    hyper_eval   = dict(hyperparams)
    hyper_eval['X_test'] = X_test.to(device)

    # 用相同的 V_params（已训练的 q(V)）进行预测
    from check_VI_utils_gpu_acc_UZ_qumodified_cor import predict_vi
    mu_eval, var_eval = predict_vi(regions_eval, V_params, hyper_eval, M=args.MC_num)

    # 兼容你已导入的 compute_metrics，如上行冲突则直接用现成的 compute_metrics
    rmse, mean_crps = compute_metrics(mu_eval, torch.sqrt(var_eval.clamp_min(1e-12)), y_test.to(device))
    print(f"[Test] RMSE={float(rmse):.4f}, CRPS={float(mean_crps):.4f}")

    # (c) 计算候选集 acquisition（hyperparams['X_test'] 仍指向 X_cand，顺序一致）
    print("\n[Compute acquisition on candidates] ...")
    A, parts = compute_acquisition_all_with_JGP(
        regions=regions, V_params=V_params, hyperparams=hyperparams,
        jgp_args=jgp_args, M=args.MC_num, lambda_=0.3
    )
    if not np.all(np.isfinite(A)):
        bad = np.where(~np.isfinite(A))[0]
        print(f"[WARN] non-finite A at indices: {bad.tolist()}")

    # Top-K
    top_idx = select_topk_candidates(A, K=K_top)
    print(f"Top-{K_top} indices:", top_idx.tolist())
    print("Top-K A:", A[top_idx])

    # (d) 为背景绘图准备 y(Z) 网格
    Z_grid, Z1, Z2 = generate_Z_grid(n=150)
    y_grid, _ = generate_y_from_Z(Z_grid, gp_model)

    # (e) 在 2D Z 上绘图（背景: y(Z)，蓝色: 决策面；绿色: 训练点；红色: 候选点；颜色=acq；Top-5 黄圈）
    plot_Z_with_acquisition(
        Z_grid=Z_grid, y_grid=y_grid,
        Z_train=Z_train, Z_cand=Z_cand,
        A=A, topk_idx=top_idx,
        title=f"DJGP acquisition over latent Z  |  Test RMSE={float(rmse):.3f}, CRPS={float(mean_crps):.3f}"
    )

def run_al_loop(
    seed=123,
    N_train=50,
    N_cand=200,
    N_test=200,
    rounds=30,
    KNN=25,                 # 每个区域的邻域大小，与 DJGPArgs.n 保持一致
    save_prefix="al_run"
):
    """
    主动学习循环：每轮训练 DJGP、评估 JGP&DJGP、基于 acquisition 选 Top-1 加入训练集。
    最终绘图并保存数据与图像。
    """
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 1) 生成数据 ===
    Z_train = torch.rand(N_train, 2)
    Z_pool  = torch.rand(N_cand,  2)
    Z_test  = torch.rand(N_test,  2)

    y_train, gp_model = generate_y_from_Z(Z_train)           # 共享同一个 GP prior
    y_pool, _         = generate_y_from_Z(Z_pool, gp_model)  # 用同一个 gp_model 生成“oracle”标签
    y_test, _         = generate_y_from_Z(Z_test, gp_model)

    X_train = g(Z_train)
    X_pool  = g(Z_pool)
    X_test  = g(Z_test)

    # === 2) 配置 DJGP 参数 ===
    args = DJGPArgs(Q=2, m1=4, m2=20, n=KNN, num_steps=100, MC_num=8, lr=1e-2, use_batch=False)

    # 记录曲线与被选点
    rmse_jgp_hist, crps_jgp_hist = [], []
    rmse_djg_hist, crps_djg_hist = [], []
    picked_Z_list = []   # 每轮新增的 Z（用于画圈）
    picked_idx_list = [] # 对应在当时候选池的索引（仅记录，调试用）

    # 用 color 映射每轮新增点颜色（由浅到深）
    cmap = plt.get_cmap("viridis")

    # === 3) 先计算第 0 轮（AL 前）的基线 JGP & DJGP 表现 ===
    # 3.1 JGP（用当前训练集，对独立 X_test）
    neighborhoods0 = find_neighborhoods(X_test, X_train, y_train, args.n)
    rmse0_jgp, crps0_jgp, _ = evaluate_jumpgp(X_test, y_test, neighborhoods0, device, std=None)
    rmse_jgp_hist.append(float(rmse0_jgp))
    crps_jgp_hist.append(float(crps0_jgp))

    # 3.2 DJGP（训练一次，评估独立测试集）
    print("\n[Round 0] Train DJGP for baseline ...")
    res0 = djgp_fit_predict(
        X_train=X_train, Y_train=y_train,
        X_test=X_pool,   Y_test=None,     # 这里的 X_test 仅用于 acquisition 建模
        args=args, device=device
    )
    regions0, V_params0, hyper0 = res0['regions'], res0['V_params'], res0['hyperparams']

    regions_eval0 = build_eval_regions(X_train, y_train, X_test, args, device)
    hyper_eval0   = dict(hyper0)
    hyper_eval0['X_test'] = X_test.to(device)
    mu0_dj, var0_dj = predict_vi(regions_eval0, V_params0, hyper_eval0, M=args.MC_num)
    rmse0_dj, crps0_dj = compute_metrics(mu0_dj, torch.sqrt(var0_dj.clamp_min(1e-12)), y_test.to(device))
    rmse_djg_hist.append(float(rmse0_dj))
    crps_djg_hist.append(float(crps0_dj))
    print(f"[Round 0] JGP: RMSE={rmse0_jgp:.4f} CRPS={crps0_jgp:.4f} | DJGP: RMSE={rmse0_dj:.4f} CRPS={crps0_dj:.4f}")

    # === 4) 主动学习迭代 ===
    for r in range(1, rounds+1):
        print(f"\n[Round {r}] Training DJGP and computing acquisition on pool (|pool|={X_pool.shape[0]}) ...")
        # 4.1 用当前训练集训练 DJGP（X_test=当前候选池，便于后续 acquisition）
        res = djgp_fit_predict(
            X_train=X_train, Y_train=y_train,
            X_test=X_pool,   Y_test=None,
            args=args, device=device
        )
        regions, V_params, hyperparams = res['regions'], res['V_params'], res['hyperparams']
        jgp_args = res['jgp_args']

        # 4.2 评估 DJGP（独立测试集）
        regions_eval = build_eval_regions(X_train, y_train, X_test, args, device)
        hyper_eval   = dict(hyperparams)
        hyper_eval['X_test'] = X_test.to(device)
        mu_eval, var_eval = predict_vi(regions_eval, V_params, hyper_eval, M=args.MC_num)
        rmse_dj, crps_dj = compute_metrics(mu_eval, torch.sqrt(var_eval.clamp_min(1e-12)), y_test.to(device))
        rmse_djg_hist.append(float(rmse_dj))
        crps_djg_hist.append(float(crps_dj))

        # 4.3 评估 JGP（独立测试集，邻域来自当前训练集）
        neighborhoods = find_neighborhoods(X_test, X_train, y_train, args.n)
        rmse_jgp, crps_jgp, _ = evaluate_jumpgp(X_test, y_test, neighborhoods, device, std=None)
        rmse_jgp_hist.append(float(rmse_jgp))
        crps_jgp_hist.append(float(crps_jgp))

        print(f"[Round {r}]  JGP: RMSE={rmse_jgp:.4f} CRPS={crps_jgp:.4f} | DJGP: RMSE={rmse_dj:.4f} CRPS={crps_dj:.4f}")

        # 4.4 计算候选池 acquisition 并选 Top-1
        A, _ = compute_acquisition_all_with_JGP(
            regions=regions, V_params=V_params, hyperparams=hyperparams,
            jgp_args=jgp_args, M=args.MC_num, lambda_=0.3
        )
        if not np.all(np.isfinite(A)):
            bad = np.where(~np.isfinite(A))[0]
            print(f"[WARN] non-finite acquisition at indices: {bad.tolist()}")
        best_idx = int(select_topk_candidates(A, K=1)[0])

        # 4.5 “查询真值”（oracle）：把该点加入训练集，并从池移除
        x_new = X_pool[best_idx:best_idx+1, :]   # [1,D]
        z_new = Z_pool[best_idx:best_idx+1, :]
        y_new = y_pool[best_idx:best_idx+1]

        X_train = torch.cat([X_train, x_new], dim=0)
        Z_train = torch.cat([Z_train, z_new], dim=0)
        y_train = torch.cat([y_train, y_new.view(-1)], dim=0)

        X_pool  = torch.cat([X_pool[:best_idx],  X_pool[best_idx+1:]],  dim=0)
        Z_pool  = torch.cat([Z_pool[:best_idx],  Z_pool[best_idx+1:]],  dim=0)
        y_pool  = torch.cat([y_pool[:best_idx],  y_pool[best_idx+1:]],  dim=0)

        picked_Z_list.append(z_new.squeeze(0).cpu().numpy())
        picked_idx_list.append(best_idx)

        # 4.6 可选：早停条件（如果池空或 rounds 超过等）
        if X_pool.shape[0] == 0:
            print("[Early stop] pool exhausted.")
            break

    # === 5) 绘图：Z 空间背景 + 决策面 + 初始训练点 + 每轮新增点 ===
    print("\n[Plot] Latent Z with boundary, initial train, and added points ...")
    Z_grid, _, _ = generate_Z_grid(n=150)
    y_grid, _    = generate_y_from_Z(Z_grid, gp_model)

    # 背景图
    def plot_Z_background(ax):
        G = Z_grid.shape[0]
        n = int(np.sqrt(G))
        Z1 = Z_grid[:,0].reshape(n, n).numpy()
        Z2 = Z_grid[:,1].reshape(n, n).numpy()
        Y  = y_grid.reshape(n, n).numpy()
        cs = ax.contourf(Z1, Z2, Y, levels=50, cmap='gray', alpha=0.95)
        # 决策面
        h = Z1 - 1.0/(1.0 + np.exp(-8*(Z2 - 0.5)))
        ax.contour(Z1, Z2, h, levels=[0.0], colors='blue', linewidths=2)

    fig, ax = plt.subplots(figsize=(7,6))
    plot_Z_background(ax)
    # 初始训练点（绿色）
    init_N = N_train
    ax.scatter(Z_train[:init_N,0].cpu().numpy(), Z_train[:init_N,1].cpu().numpy(),
               c='limegreen', s=35, edgecolors='k', linewidths=0.5, label='initial train (Z)')

    # 每轮新增点：空心圆，颜色由浅到深（越后越深）
    if picked_Z_list:
        picked_Z = np.stack(picked_Z_list, axis=0)  # [R,2]
        R = picked_Z.shape[0]
        for t in range(R):
            color = cmap((t+1)/max(R,1))   # 0->浅 1->深
            ax.scatter(picked_Z[t,0], picked_Z[t,1],
                       facecolors='none', edgecolors=color,
                       s=160, linewidths=2.0, label=None if t>0 else 'added per round')

    ax.set_title("Active Learning on Latent Z (gray=y, blue=boundary)")
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_Z_added_points.png", dpi=200)
    plt.close(fig)

    # === 6) 绘制 RMSE 曲线（JGP vs DJGP） ===
    rounds_axis = np.arange(0, len(rmse_jgp_hist))  # 包含 Round 0
    fig2, ax2 = plt.subplots(figsize=(7,4))
    ax2.plot(rounds_axis, rmse_jgp_hist, marker='o', label='JGP RMSE')
    ax2.plot(rounds_axis, rmse_djg_hist, marker='s', label='DJGP RMSE')
    ax2.set_xlabel("AL Round")
    ax2.set_ylabel("RMSE on held-out test")
    ax2.set_title("RMSE vs Rounds (lower is better)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(f"{save_prefix}_rmse_curves.png", dpi=200)
    plt.close(fig2)

    # === 7) 保存关键数组（便于之后做 JGP 主动学习等） ===
    # 注意：此时 Z_train / X_train 是最终的（含新增点）；如果你想保存初始版本，可在前面另外备份
    np.savez(
        f"{save_prefix}_data.npz",
        X_train=X_train.cpu().numpy(),
        Z_train=Z_train.cpu().numpy(),
        X_test=X_test.cpu().numpy(),
        Z_test=Z_test.cpu().numpy(),
        y_train=y_train.cpu().numpy(),
        y_test=y_test.cpu().numpy(),
        picked_Z=np.stack(picked_Z_list) if picked_Z_list else np.zeros((0,2)),
        rmse_jgp=np.array(rmse_jgp_hist),
        rmse_djgp=np.array(rmse_djg_hist),
        crps_jgp=np.array(crps_jgp_hist),
        crps_djgp=np.array(crps_djg_hist),
        seed=np.array([seed])
    )
    print(f"[Saved] figures: {save_prefix}_Z_added_points.png, {save_prefix}_rmse_curves.png")
    print(f"[Saved] arrays:  {save_prefix}_data.npz")

    # 返回结果（可选）
    return {
        "rmse_jgp": rmse_jgp_hist,
        "rmse_djgp": rmse_djg_hist,
        "crps_jgp": crps_jgp_hist,
        "crps_djgp": crps_djg_hist,
        "picked_Z": picked_Z_list
    }


if __name__ == "__main__":
    out = run_al_loop(
        seed=123,
        N_train=50, N_cand=200, N_test=200,
        rounds=30, KNN=15,
        save_prefix="djgp_al_30r_n15"
    )

