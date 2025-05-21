import torch
import math

def se_kernel(x1, x2, theta1=9.0, theta2=200.0):
    """
    平方指数核：
      K[i,j] = θ1 * exp(-||x1[i] - x2[j]||^2 / θ2)
    """
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)      # [n1, n2, d]
    dist2 = diff.pow(2).sum(-1)                   # [n1, n2]
    return theta1 * torch.exp(-dist2 / theta2)

def compute_region(x, r):
    """
    根据 d+1 个 g_j 分区函数，为每个点分配 region 索引
    输入:
      x: [m, d]
      r: 长度 d 的 ±1 Tensor
    返回:
      region: [m] 整数索引
    """
    m, d = x.shape
    # g0
    g0 = x.pow(2).sum(1) - 0.4**2                 # [m]
    signs = [(g0 >= 0).long()]

    # g1..gd
    for j in range(d):
        gj = x.pow(2).sum(1) \
             - x[:, j].pow(2) \
             + (x[:, j] + r[j]*0.5).pow(2) \
             - 0.3**2
        signs.append((gj >= 0).long())

    region = torch.zeros(m, dtype=torch.long, device=x.device)
    for j, s in enumerate(signs):
        region += (s << j)
    return region

def safe_cholesky(K, initial_jitter=1e-6, max_tries=5):
    """
    带自适应抖动的 Cholesky 分解
    """
    jitter = initial_jitter
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(K + jitter * torch.eye(K.size(0), device=K.device))
        except RuntimeError:
            jitter *= 10
    # 最后一次尝试，不再捕获异常
    return torch.linalg.cholesky(K + jitter * torch.eye(K.size(0), device=K.device))

def generate_data(d, N, Nt,
                  theta1=9.0, theta2=200.0,
                  noise_var=4.0,
                  test_tol=0.05,
                  device='cpu'):
    """
    生成训练/测试数据：
      d  -- 输入维度
      N  -- 训练集大小
      Nt -- 测试集大小
    返回:
      x_train [N,d], y_train [N],
      x_test  [Nt,d], y_test  [Nt]
    """
    device = torch.device(device)
    # torch.manual_seed(0)

    # 随机切分参数 r_j ∈ {+1,-1}
    r = torch.randint(0, 2, (d,), device=device).float() * 2 - 1  # ±1

    # 1) 生成训练点
    x_train = (torch.rand(N, d, device=device) - 0.5)
    region_train = compute_region(x_train, r)      # [N]
    num_regions = 2 ** (d + 1)

    # 2) 为每个 region 随机选均值 μ_k
    signs = (torch.rand(num_regions, device=device) < 0.5).float() * 2 - 1
    mu = signs * (13.5 * torch.arange(num_regions, device=device).float())

    # 3) 在各 region 内生成 y_train
    y_train = torch.zeros(N, device=device)
    for k in region_train.unique():
        idx = (region_train == k).nonzero(as_tuple=False).squeeze(1)
        Xk = x_train[idx]                         # [n_k, d]
        K = se_kernel(Xk, Xk, theta1, theta2)
        # 加噪声方差 + 基础抖动
        K += noise_var * torch.eye(len(idx), device=device)
        L = safe_cholesky(K)                      # 自适应抖动
        eps = torch.randn(len(idx), device=device)
        f = L @ eps
        y_train[idx] = mu[k] + f

    # 4) 构造测试点：要求至少一个 g_j 的 |g_j(x)| <= test_tol
    x_test = []
    while len(x_test) < Nt:
        cand = (torch.rand(Nt, d, device=device) - 0.5)
        # 计算所有 g_j 的绝对值
        abs_g = []
        abs_g.append((cand.pow(2).sum(1) - 0.4**2).abs())
        for j in range(d):
            gj = cand.pow(2).sum(1) \
                 - cand[:, j].pow(2) \
                 + (cand[:, j] + r[j]*0.5).pow(2) \
                 - 0.3**2
            abs_g.append(gj.abs())
        abs_g = torch.stack(abs_g, dim=1)        # [Nt, d+1]
        mask = (abs_g <= test_tol).any(dim=1)
        selected = cand[mask]
        x_test.append(selected)
    x_test = torch.cat(x_test, dim=0)[:Nt]
    region_test = compute_region(x_test, r)

    # 5) 在测试点做无噪声 GP 采样
    y_test = torch.zeros(Nt, device=device)
    for k in region_test.unique():
        idx = (region_test == k).nonzero(as_tuple=False).squeeze(1)
        Xk = x_test[idx]
        K = se_kernel(Xk, Xk, theta1, theta2)
        L = safe_cholesky(K)                    # 无噪声，但也加点小抖动
        eps = torch.randn(len(idx), device=device)
        f = L @ eps
        y_test[idx] = mu[k] + f

    return x_train, y_train, x_test, y_test

# === 测试示例 ===
if __name__ == '__main__':
    d, N, Nt = 3, 500, 200
    xtr, ytr, xte, yte = generate_data(d, N, Nt)
    print(xtr.shape, ytr.shape, xte.shape, yte.shape)
