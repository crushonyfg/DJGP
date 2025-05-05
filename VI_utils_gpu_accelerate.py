import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
# 将 JumpGP_code_py 所在的目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import math
import torch.nn.functional as F
from utils1 import jumpgp_ld_wrapper
from tqdm import tqdm

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 预生成 Gauss–Hermite 节点和权重
_GH_POINTS = 20
_nodes_np, _weights_np = np.polynomial.hermite.hermgauss(_GH_POINTS)
_nodes   = torch.from_numpy(_nodes_np).to(device)
_weights = torch.from_numpy(_weights_np).to(device)
_factor  = 1.0 / math.sqrt(math.pi)
_LOG_2PI = math.log(2 * math.pi)

# ===== 基础函数 =====
def Phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

# ===== KL(q(V)||p(V)) =====
def kl_qp(Z: torch.Tensor,
          mu: torch.Tensor,
          sigma: torch.Tensor,
          lengthscales: torch.Tensor,
          var_w: torch.Tensor) -> torch.Tensor:
    m2, Q, D = mu.shape
    d2 = (Z.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)
    kl = torch.tensor(0.0, device=device)
    for q in range(Q):
        ell = lengthscales[q] if lengthscales.ndim > 0 else lengthscales
        wvar = var_w[q] if var_w.ndim > 0 else var_w
        Kq = wvar * torch.exp(-0.5 * d2 / (ell**2))
        L = torch.linalg.cholesky(Kq)
        Kq_inv = torch.cholesky_inverse(L)
        diag_inv = torch.diagonal(Kq_inv)
        logdet_Kq = 2.0 * torch.log(torch.diagonal(L)).sum()
        for d in range(D):
            mu_qd = mu[:, q, d]
            s2 = sigma[:, q, d].pow(2)
            trace_term = (diag_inv * s2).sum()
            quad_term = mu_qd @ (Kq_inv @ mu_qd)
            logdet_S = torch.log(s2 + 1e-12).sum()
            kl += 0.5 * (trace_term + quad_term - m2 + logdet_Kq - logdet_S)
    return kl

# ===== q(W) posterior =====
def qW_from_qV(X: torch.Tensor,
               Z: torch.Tensor,
               mu_V: torch.Tensor,
               sigma_V: torch.Tensor,
               lengthscales: torch.Tensor,
               var_w: torch.Tensor):
    T, D = X.shape
    m2, Q, _ = mu_V.shape
    ZZ2 = (Z.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)
    XZ2 = (X.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)
    mu_W    = torch.empty((T, Q, D), device=device)
    sigma_W = torch.empty((T, Q, D), device=device)
    for q in range(Q):
        l = lengthscales[q]
        Kzz = var_w * torch.exp(-0.5 * ZZ2 / (l**2))
        Kxz = var_w * torch.exp(-0.5 * XZ2 / (l**2))
        L = torch.linalg.cholesky(Kzz)
        Kzz_inv = torch.cholesky_inverse(L)
        A = Kxz @ Kzz_inv
        var_prior = var_w
        diag_cross = (A * Kxz).sum(dim=1)
        U = Kzz_inv @ Kxz.t()
        for d in range(D):
            mu_W[:, q, d] = A @ mu_V[:, q, d]
            s2 = sigma_V[:, q, d].pow(2).unsqueeze(1)
            diag3 = (U.pow(2) * s2).sum(dim=0)
            sigma_W[:, q, d] = torch.sqrt(var_prior - diag_cross + diag3 + 1e-12)
    cov_W = torch.diag_embed(sigma_W.pow(2))
    return mu_W, cov_W

# ===== 批量计算 Kfu =====
def expected_Kfu(mu_W, cov_W, X, C, sigma_f):
    # mu_W [T,Q,D], cov_W diag->[T,Q,D], X [T,n,D], C [T,m1,Q]
    T, n, D = X.shape
    m1, Q = C.shape[1], C.shape[2]
    Sigma = cov_W  # diag embed already taken
    s = torch.einsum('tnd,tqde,tne->tnq', X, Sigma, X)
    den = torch.sqrt(s + 1.0)
    mu_proj = torch.einsum('tqd,tnd->tnq', mu_W, X)
    diff = mu_proj.unsqueeze(2) - C.unsqueeze(1)
    exp_term = torch.exp(-0.5 * diff**2 / (s.unsqueeze(2)+1.0))
    num = exp_term.prod(dim=-1)
    den_prod = den.prod(dim=-1, keepdim=True)
    return sigma_f.view(-1,1,1) * num/den_prod

# ===== 批量计算 KufKfu =====
def expected_KufKfu(mu_W, cov_W, X, C, sigma_f):
    """
    Compute E_q[K_{u,f} K_{f,u}] for all regions and data points in batch.
    Args:
      mu_W:    [T,Q,D]
      cov_W:   [T,Q,D,D] diagonal covariance
      X:       [T,n,D]
      C:       [T,m1,Q]
      sigma_f: [T]
    Returns:
      Tensor of shape [T,n,m1,m1]
    """
    # Shapes
    T, n, D = X.shape
    m1, Q = C.shape[1], C.shape[2]
    # 1) s[t,i,q] = x_{t,i}^T Sigma_{t,q} x_{t,i}
    s = torch.einsum('tnd,tqde,tne->tnq', X, cov_W, X)         # [T,n,Q]
    # 2) mu_proj[t,i,q] = mu_{t,q}^T x_{t,i}
    mu_proj = torch.einsum('tqd,tnd->tnq', mu_W, X)           # [T,n,Q]
    # 3) midpoints of inducing locations
    mid = 0.5 * (C.unsqueeze(2) + C.unsqueeze(1))             # [T,m1,m1,Q]
    # 4) diff for each t,i,l,l',q
    diff = mu_proj.unsqueeze(2).unsqueeze(3) - mid.unsqueeze(1) # [T,n,m1,m1,Q]
    # 5) exponent term
    denom = s + 1.0                                           # [T,n,Q]
    exp_term = torch.exp(-0.5 * diff**2 / denom.unsqueeze(2).unsqueeze(2)) # [T,n,m1,m1,Q]
    num = exp_term.prod(dim=-1)                              # [T,n,m1,m1]
    # 6) prior cross-kernel
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)    # [T,m1,m1]
    prior = torch.exp(-0.25 * d2)                           # [T,m1,m1]
    # 7) denominator prod over q: prod_q sqrt(s+1)
    den_prod = torch.prod(torch.sqrt(denom), dim=-1)        # [T,n]
    # 8) combine
    return (sigma_f.view(T,1,1,1)**2) * prior.unsqueeze(1) * (num / den_prod.unsqueeze(2).unsqueeze(3))

# ===== 批量 Logistic 期望 =====
def expected_log_sigmoid_gh_batch(omega, mu_W, cov_W, X):
    T, n, D = X.shape
    Q = mu_W.shape[1]
    # 1) projection of means: [T,n,Q]
    mu_proj = torch.einsum('tqd,tnd->tnq', mu_W, X)
    # 2) mean of z: omega_0 + sum_q omega_q * mu_proj
    mu_z = omega[:, 0].unsqueeze(1) + (omega[:, 1:].unsqueeze(1) * mu_proj).sum(dim=2)
    # 3) variance term s: [T,n,Q]
    s = torch.einsum('tnd,tqde,tne->tnq', X, cov_W, X)
    # 4) tau^2: sum_q omega_q^2 * s
    tau2 = (omega[:, 1:].unsqueeze(1).pow(2) * s).sum(dim=2)
    tau_z = torch.sqrt(tau2 + 1e-12)
    # 5) compute z at GH nodes: [T,n,P]
    z = mu_z.unsqueeze(2) + math.sqrt(2) * tau_z.unsqueeze(2) * _nodes.view(1,1,-1)
    # use F.logsigmoid
    log_sig = F.logsigmoid(z)
    log_one_minus = F.logsigmoid(-z)
    # 6) weighted sum
    E1 = _factor * (_weights.view(1,1,-1) * log_sig).sum(dim=-1)
    E2 = _factor * (_weights.view(1,1,-1) * log_one_minus).sum(dim=-1)
    return E1, E2

# ===== 批量 ELBO =====
def compute_ELBO(regions, V_params, u_params, hyperparams, ell=3):
    T = len(regions)
    # 堆叠 regions
    X = torch.stack([r['X'] for r in regions],0)  # [T,n,D]
    y = torch.stack([r['y'] for r in regions],0)  # [T,n]
    C = torch.stack([r['C'] for r in regions],0)  # [T,m1,Q]
    Uconst = torch.tensor([r['U'] for r in regions],device=device)

    # 提取 hyperparams
    Z = hyperparams['Z']            # [m2,D]
    lengthscales = hyperparams['lengthscales']
    var_w = hyperparams['var_w']
    X_test = hyperparams['X_test']  # [T,D]
    mu_V    = V_params['mu_V']      # [m2,Q,D]
    sigma_V = V_params['sigma_V']   # [m2,Q,D]

    KL_V = kl_qp(Z, mu_V, sigma_V, lengthscales, var_w)
    mu_W, cov_W = qW_from_qV(X_test, Z, mu_V, sigma_V, lengthscales, var_w)

    sigma_noise = torch.stack([u['sigma_noise'] for u in u_params],0) # [T]
    omega       = torch.stack([u['omega'] for u in u_params],0)       # [T,Q+1]
    mu_u        = torch.stack([u['mu_u'] for u in u_params],0)       # [T,m1]
    Sigma_u     = torch.stack([u['Sigma_u'] for u in u_params],0)    # [T,m1,m1]

    # Kuu
    m1 = C.shape[1]
    d2 = (C.unsqueeze(2)-C.unsqueeze(1)).pow(2).sum(-1)
    I = torch.eye(m1,device=device).unsqueeze(0)
    Kuu = torch.exp(-0.5*d2) + sigma_noise.view(-1,1,1)*I
    Luu = torch.linalg.cholesky(Kuu)
    Kuu_inv = torch.cholesky_inverse(Luu)

    # Kfu, KufKfu
    Kfu    = expected_Kfu(mu_W, cov_W, X, C, sigma_noise)
    KufKfu = expected_KufKfu(mu_W, cov_W, X, C, sigma_noise)

    # V1, E_fu, T3
    V1 = sigma_noise.view(-1,1)**2 - torch.einsum('tij,tnji->tn',Kuu_inv,KufKfu)
    v  = torch.einsum('tij,tj->ti',Kuu_inv,mu_u)
    E_fu  = torch.einsum('tni,ti->tn',Kfu,v)
    T3 = torch.einsum('tij,tnjk,tkm,tmi->tn',Kuu_inv,KufKfu,Kuu_inv,Sigma_u)

    quad = (y.pow(2) - 2*y*E_fu + (V1+T3)) / (2*sigma_noise.view(-1,1)**2)
    elog_sig, elog_one_minus = expected_log_sigmoid_gh_batch(omega,mu_W,cov_W,X)
    T1 = -0.5*_LOG_2PI - torch.log(sigma_noise.view(-1,1)**2)/2 + elog_sig - quad
    T2 = torch.log(Uconst.view(-1,1)) + elog_one_minus

    region_elbo = torch.logsumexp(torch.stack([T1,T2],0),dim=0).sum(dim=-1)
    return -KL_V + region_elbo.sum()

# ===== 训练和预测 =====
def train_vi(regions,
             V_params,
             u_params,
             hyperparams,
             lr=1e-3,
             num_steps=1000,
             log_interval=100):
    params = [V_params['mu_V'], V_params['sigma_V']]
    for u in u_params:
        params += [u['mu_u'], u['Sigma_u'], u['sigma_noise'], u['omega']]
    params += [hyperparams['lengthscales'], hyperparams['var_w']]
    opt = torch.optim.Adam(params, lr=lr)
    for step in range(1, num_steps+1):
        opt.zero_grad()
        elbo = compute_ELBO(regions, V_params, u_params, hyperparams)
        loss = -elbo
        loss.backward()
        opt.step()
        with torch.no_grad():
            V_params['sigma_V'].clamp_(min=1e-6)
            hyperparams['var_w'].clamp_(min=1e-6)
            hyperparams['lengthscales'].clamp_(min=1e-6)
            for u in u_params:
                u['sigma_noise'].clamp_(min=1e-6)
        if step % log_interval==0 or step==1:
            print(f"Step {step}/{num_steps}, ELBO={elbo.item():.4f}")
    return V_params, u_params, hyperparams


def predict_vi(regions,
               V_params,
               hyperparams,
               M=100):
    """
    Predict test outputs using M samples from variational W posterior, with NaN handling.
    Returns:
      mu_pred: [T] tensor
      var_pred: [T] tensor
    """
    T = len(regions)
    X_test = hyperparams['X_test']      # [T,D]
    mu_V = V_params['mu_V']             # [m2,Q,D]
    sigma_V = V_params['sigma_V']       # [m2,Q,D]
    # 1) q(W) posterior
    mu_W, cov_W = qW_from_qV(
        X_test,
        hyperparams['Z'],
        mu_V,
        sigma_V,
        hyperparams['lengthscales'],
        hyperparams['var_w']
    )  # mu_W [T,Q,D], cov_W [T,Q,D,D]
    sigma_W = torch.sqrt(cov_W.diagonal(dim1=2, dim2=3))  # [T,Q,D]
    # 2) Sample W: [M,T,Q,D]
    eps = torch.randn((M, T, mu_W.shape[1], mu_W.shape[2]), device=device)
    W_samples = mu_W.unsqueeze(0) + eps * sigma_W.unsqueeze(0)
    # 3) Allocate predictions
    mu_s = torch.zeros((M, T), device=device)
    var_s = torch.zeros((M, T), device=device)
    # 4) Loop over samples and regions
    for m in range(M):
        Wm = W_samples[m]  # [T,Q,D]
        for j, r in tqdm(enumerate(regions)):
            Xj = r['X']  # [n_j,D]
            yj = r['y']  # [n_j]
            Wj = Wm[j]   # [Q,D]
            # Project training and test
            Xn = Xj @ Wj.T               # [n_j,Q]
            yn = yj.view(-1,1)           # [n_j,1]
            xt = (hyperparams['X_test'][j].view(1,-1) @ Wj.T)  # [1,Q]
            # GP prediction
            mu_t, sig2_t, _, _ = jumpgp_ld_wrapper(
                Xn, yn, xt,
                mode="CEM", flag=False, device=device
            )
            mu_s[m, j] = mu_t.view(-1)
            var_s[m, j]= sig2_t.view(-1)
    # 5) Aggregate and handle NaNs
    mu_pred = mu_s.mean(dim=0)
    var_pred = var_s.mean(dim=0)
    mu_pred = torch.nan_to_num(mu_pred, nan=0.0)
    var_pred = torch.nan_to_num(var_pred, nan=1e-6)
    return mu_pred, var_pred

import torch

def predict_vi_analytic(regions, V_params, u_params, hyperparams):
    """
    Analytic VI prediction for each region's single test point x_t.

    Returns:
      mu_pred: [T] tensor of predictive means
      sigma2:  [T] tensor of predictive variances
    """
    device = hyperparams['Z'].device
    T = len(regions)

    # ——— 1. 堆叠 region 相关量 ———
    C           = torch.stack([r['C'] for r in regions], dim=0)      # [T,m1,Q]
    mu_u        = torch.stack([u['mu_u'] for u in u_params], dim=0) # [T,m1]
    Sigma_u     = torch.stack([u['Sigma_u'] for u in u_params], dim=0) # [T,m1,m1]
    sigma_noise = torch.stack([u['sigma_noise'] for u in u_params], dim=0) # [T]

    # ——— 2. Unpack global variational params ———
    Z            = hyperparams['Z']          # [m2,D]
    lengthscales = hyperparams['lengthscales']
    var_w        = hyperparams['var_w']
    X_test       = hyperparams['X_test']     # [T,D]
    mu_V         = V_params['mu_V']          # [m2,Q,D]
    sigma_V      = V_params['sigma_V']       # [m2,Q,D]

    # ——— 3. q(W) 后验在测试点的均值与二阶矩 ——批量版本———
    mu_W, cov_W = qW_from_qV(
        X_test, Z, mu_V, sigma_V, lengthscales, var_w
    )  # mu_W:[T,Q,D], cov_W diag->[T,Q,D,D]

    # 令 X_test_n=[T,1,D] 调用 batch 期望
    X_test_n = X_test.unsqueeze(1)  # [T,1,D]
    m_W = expected_Kfu(mu_W, cov_W, X_test_n, C, sigma_noise).squeeze(1)      # [T,m1]
    S_W = expected_KufKfu(mu_W, cov_W, X_test_n, C, sigma_noise).squeeze(1)   # [T,m1,m1]

    # ——— 4. 构造 K_uu 及其逆 ———
    m1 = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)  # [T,m1,m1]
    I = torch.eye(m1, device=device).unsqueeze(0)
    Kuu = torch.exp(-0.5 * d2) + sigma_noise.view(-1,1,1) * I  # [T,m1,m1]
    Luu = torch.linalg.cholesky(Kuu)                          # [T,m1,m1]
    Kuu_inv = torch.cholesky_inverse(Luu)                     # [T,m1,m1]

    # ——— 5. 计算 a = Kuu_inv @ mu_u ———
    a = torch.einsum('tij,tj->ti', Kuu_inv, mu_u)             # [T,m1]

    # ——— 6. 均值 μ_t = m_W ⋅ a ———
    mu_pred = (m_W * a).sum(dim=1)                            # [T]

    # ——— 7. 方差各部分 ———
    # V1 = σ_n^2 - trace(Kuu_inv @ S_W)
    V1 = sigma_noise**2 - torch.einsum('tij,tij->t', Kuu_inv, S_W)  # [T]

    # T3 = trace(Kuu_inv @ S_W @ Kuu_inv @ Sigma_u)
    T3 = torch.einsum('tij,tjk,tkl,tli->t',
                     Kuu_inv, S_W, Kuu_inv, Sigma_u)               # [T]

    # V_W = aᵀ (S_W - m_W m_Wᵀ) a
    diff = S_W - m_W.unsqueeze(2) * m_W.unsqueeze(1)             # [T,m1,m1]
    V_W = torch.einsum('ti,tij,tj->t', a, diff, a)               # [T]

    # 最终时刻方差
    sigma2 = V1 + T3 + V_W                                        # [T]

    return mu_pred, sigma2


# ===== 示例 =====
if __name__=="__main__":
    # 构造示例数据
    T, n, m1, m2, Q, D = 2, 5, 3, 4, 2, 3
    regions, u_params = [], []
    for _ in range(T):
        regions.append({
            'X': torch.randn(n,D,device=device),
            'y': torch.randn(n,device=device),
            'U': 1.0,
            # 'U': torch.tensor(0.5,device=device,requires_grad=True)
            'C': torch.randn(m1,Q,device=device)
        })
        u_params.append({
            'mu_u': torch.randn(m1,device=device,requires_grad=True),
            'Sigma_u':torch.eye(m1,device=device,requires_grad=True),
            'sigma_noise':torch.tensor(0.5,device=device,requires_grad=True),
            'omega':torch.randn(Q+1,device=device,requires_grad=True)
        })
    V_params = {
        'mu_V':torch.randn(m2,Q,D,device=device,requires_grad=True),
        'sigma_V':torch.rand(m2,Q,D,device=device,requires_grad=True)
    }
    hyperparams = {
        'Z': torch.randn(m2,D,device=device),
        'X_test': torch.randn(T,D,device=device),
        'lengthscales': torch.rand(Q,device=device,requires_grad=True),
        'var_w': torch.tensor(1.0,device=device,requires_grad=True)
    }
    V_params, u_params, hyperparams = train_vi(regions, V_params, u_params, hyperparams,
                                              lr=1e-3, num_steps=10, log_interval=5)
    # mu_p, var_p = predict_vi(regions, V_params, hyperparams, M=10)
    mu_p, var_p = predict_vi_analytic(regions, V_params, u_params, hyperparams)
    print("mu_pred=", mu_p)
    print("var_pred=", var_p)
