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

# 选择 GPU，如果不可用则回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

def eta(vartheta: torch.Tensor, tau: torch.Tensor, ell: int) -> torch.Tensor:
    prefactor = tau * math.sqrt(2 / math.pi) * torch.exp(-0.5 * (vartheta / tau) ** 2)
    linear = vartheta * Phi(vartheta / tau)
    series = torch.zeros_like(vartheta)
    for k in range(1, 2 * ell):
        sign = -1.0 if (k - 1) % 2 else 1.0
        exp1 = torch.exp(k * vartheta + 0.5 * k * k * tau * tau)
        phi1 = Phi(-vartheta / tau - k * tau)
        exp2 = torch.exp(-k * vartheta + 0.5 * k * k * tau * tau)
        phi2 = Phi(vartheta / tau - k * tau)
        series += (sign / k) * (exp1 * phi1 + exp2 * phi2)
    return prefactor + linear + series

def expected_log_sigmoid_gh(omega: torch.Tensor,
                            mu_W: torch.Tensor,
                            Sigma_W: torch.Tensor,
                            x: torch.Tensor,
                            n_points: int = 20):
    Q, D = mu_W.shape
    mu_proj = torch.einsum('qd,d->q', mu_W, x)
    mu_z = omega[0] + (omega[1:] * mu_proj).sum()
    s = torch.einsum('d,qde,e->q', x, Sigma_W, x)
    tau2 = (omega[1:]**2 * s).sum()
    tau_z = torch.sqrt(tau2 + 1e-12)
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(n_points)
    nodes = torch.from_numpy(nodes_np).to(mu_z)
    weights = torch.from_numpy(weights_np).to(mu_z)
    z = mu_z + math.sqrt(2.0) * tau_z * nodes
    log_sig = F.logsigmoid(z)
    log_one_minus = F.logsigmoid(-z)
    factor = 1.0 / math.sqrt(math.pi)
    E_log_sig = factor * (weights * log_sig).sum()
    E_log_one_minus = factor * (weights * log_one_minus).sum()
    return E_log_sig, E_log_one_minus

def kl_qp(Z: torch.Tensor,
          mu: torch.Tensor,
          sigma: torch.Tensor,
          lengthscales: torch.Tensor,
          var_w: torch.Tensor) -> torch.Tensor:
    m2, Q, D = mu.shape
    d2 = (Z.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(dim=2)
    kl_total = torch.tensor(0., device=Z.device)
    for q in range(Q):
        ell_q = lengthscales[q] if lengthscales.ndim > 0 else lengthscales
        wvar = var_w[q] if var_w.ndim > 0 else var_w
        Kq = wvar * torch.exp(-0.5 * d2 / (ell_q**2))
        L = torch.linalg.cholesky(Kq)
        Kq_inv = torch.cholesky_inverse(L)
        Kinv_diag = torch.diagonal(Kq_inv)
        logdet_Kq = 2.0 * torch.log(torch.diagonal(L)).sum()
        for d in range(D):
            mu_qd = mu[:, q, d]
            s2_qd = sigma[:, q, d].pow(2)
            trace_term = (Kinv_diag * s2_qd).sum()
            quad_term = mu_qd @ (Kq_inv @ mu_qd)
            logdet_Sigma = torch.log(s2_qd + 1e-12).sum()
            kl_qd = 0.5 * (
                trace_term + quad_term - m2 + logdet_Kq - logdet_Sigma
            )
            kl_total += kl_qd
    return kl_total

def qW_from_qV(X: torch.Tensor,
               Z: torch.Tensor,
               mu_V: torch.Tensor,
               sigma_V: torch.Tensor,
               lengthscales: torch.Tensor,
               var_w: torch.Tensor):
    T, D = X.shape
    m2, Q, Dv = mu_V.shape
    assert Dv == D
    ZZ2 = (Z.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)
    XZ2 = (X.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)
    mu_W = torch.empty((T, Q, D), device=X.device)
    sigma_W = torch.empty((T, Q, D), device=X.device)
    for q in range(Q):
        l = lengthscales[q]
        Kzz = var_w * torch.exp(-0.5 * ZZ2 / (l**2))
        Kxz = var_w * torch.exp(-0.5 * XZ2 / (l**2))
        L = torch.linalg.cholesky(Kzz)
        Kzz_inv = torch.cholesky_inverse(L)
        A = Kxz.matmul(Kzz_inv)
        var_prior = var_w * torch.ones(T, device=X.device)
        diag_cross = (A * Kxz).sum(dim=1)
        U = Kzz_inv.matmul(Kxz.t())
        for d in range(D):
            mu_W[:, q, d] = A.matmul(mu_V[:, q, d])
            s2 = sigma_V[:, q, d].pow(2).unsqueeze(1)
            diag3 = (U.pow(2) * s2).sum(dim=0)
            sigma_W[:, q, d] = torch.sqrt(var_prior - diag_cross + diag3 + 1e-12)
    cov_W = torch.diag_embed(sigma_W.pow(2))
    return mu_W, cov_W

def expected_Kfu(mu: torch.Tensor,
                 Sigma: torch.Tensor,
                 X: torch.Tensor,
                 Z: torch.Tensor,
                 sigma_f: torch.Tensor) -> torch.Tensor:
    n, D = X.shape
    m1, K = Z.shape
    s = torch.einsum('nd,kde,ne->nk', X, Sigma, X)
    den = torch.sqrt(s + 1.0)
    mu_proj = X @ mu.T
    diff = mu_proj.unsqueeze(1) - Z.unsqueeze(0)
    s_plus1 = (s + 1.0).unsqueeze(1)
    exp_term = torch.exp(-0.5 * diff**2 / s_plus1)
    num = torch.prod(exp_term, dim=2)
    den_prod = torch.prod(den, dim=1, keepdim=True)
    return sigma_f * (num / den_prod)

def expected_KufKfu(mu: torch.Tensor,
                    Sigma: torch.Tensor,
                    x: torch.Tensor,
                    Z: torch.Tensor,
                    sigma_f: torch.Tensor) -> torch.Tensor:
    m1, K = Z.shape
    s = torch.einsum('d,kde,e->k', x, Sigma, x)
    den_k = torch.sqrt(s + 1.0)
    den_prod = torch.prod(den_k)
    mu_proj = mu @ x
    Zl = Z.unsqueeze(1)
    Zl2 = Z.unsqueeze(0)
    mid = 0.5 * (Zl + Zl2)
    diff = mu_proj.unsqueeze(0).unsqueeze(0) - mid
    s_plus1 = (s + 1.0).unsqueeze(0).unsqueeze(0)
    exp_term = torch.exp(-0.5 * diff**2 / s_plus1)
    num = torch.prod(exp_term, dim=2)
    dist = torch.cdist(Z, Z, p=2)
    prior = torch.exp(-0.25 * dist.pow(2))
    return sigma_f**2 * prior * (num / den_prod)

def kl_q_u(Z: torch.Tensor,
           mu: torch.Tensor,
           Sigma: torch.Tensor,
           u_var: torch.Tensor) -> torch.Tensor:
    """
    Compute KL(q(u) || p(u)) for a GP prior p(u)=N(0, K) with
    K[i,j] = u_var * exp(-0.5 * ||Z[i] - Z[j]||^2)
    and q(u)=N(mu, Sigma).
    
    Args:
      Z:     [m1, Q] tensor of inducing locations (requires_grad if Z is a parameter)
      mu:    [m1] tensor of variational means (requires_grad=True)
      Sigma: [m1, m1] tensor of variational covariance (requires_grad=True)
      u_var: scalar tensor for prior variance (requires_grad=True)
    
    Returns:
      Scalar tensor: KL(q(u) || p(u))
    """
    m1, Q = Z.shape
    device, dtype = Z.device, Z.dtype
    
    # Pairwise squared distances [m1, m1]
    d2 = (Z.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(dim=2)
    
    # Prior covariance K = u_var * exp(-0.5 * d2)
    K = u_var * torch.exp(-0.5 * d2)
    
    # Cholesky of K
    L = torch.linalg.cholesky(K)
    
    # K^{-1} via Cholesky inverse
    K_inv = torch.cholesky_inverse(L)
    
    # logdet K = 2 * sum(log(diag(L)))
    logdet_K = 2.0 * torch.log(torch.diagonal(L)).sum()
    
    # trace term: tr(K^{-1} Sigma)
    trace_term = torch.tensordot(K_inv, Sigma, dims=2)
    
    # quadratic term: mu^T K^{-1} mu
    quad_term = mu @ (K_inv @ mu)
    
    # logdet Sigma
    sign, logdet_Sigma = torch.linalg.slogdet(Sigma)
    # assume sign > 0
    
    # KL divergence
    kl = 0.5 * (trace_term + quad_term - m1 + logdet_K - logdet_Sigma)
    return kl

def compute_ELBO(regions, V_params, u_params, hyperparams, ell=3):
    Z = hyperparams['Z']
    lengthscales = hyperparams['lengthscales']
    var_w = hyperparams['var_w']
    X_test = hyperparams['X_test']
    mu_V = V_params['mu_V']
    sigma_V = V_params['sigma_V']
    KL_V = kl_qp(Z, mu_V, sigma_V, lengthscales, var_w)
    ELBO = -KL_V
    mu_W, cov_W = qW_from_qV(X_test, Z, mu_V, sigma_V, lengthscales, var_w)
    for j, reg in enumerate(regions):
        X = reg['X']
        y = reg['y']
        Uconst = reg['U']
        Cj = reg['C']
        mu_u = u_params[j]['mu_u']
        Sigma_u = u_params[j]['Sigma_u']
        sigma_noise = u_params[j]['sigma_noise']
        omega_j = u_params[j]['omega']
        mu_Wj = mu_W[j]
        cov_Wj = cov_W[j]
        ELBO -= kl_q_u(Cj, mu_u, Sigma_u, sigma_noise)
        m1 = Cj.size(0)
        d2 = (Cj.unsqueeze(1)-Cj.unsqueeze(0)).pow(2).sum(dim=2)
        Kuu = torch.exp(-0.5 * d2) + sigma_noise * torch.eye(m1, device=Cj.device)
        Luu = torch.linalg.cholesky(Kuu)
        Kuu_inv = torch.cholesky_inverse(Luu)
        Kfu = expected_Kfu(mu_Wj, cov_Wj, X, Cj, sigma_noise)
        n = X.shape[0]
        for i in range(n):
            Kfu_i = Kfu[i]
            KufKfu_i = expected_KufKfu(mu_Wj, cov_Wj, X[i], Cj, sigma_noise)
            V1_i = sigma_noise**2 - torch.trace(Kuu_inv @ KufKfu_i)
            v = Kuu_inv @ mu_u
            E_fu = Kfu_i @ v
            term_S = torch.trace(Kuu_inv @ KufKfu_i @ Kuu_inv @ Sigma_u)
            T3_i = term_S
            Var_f_i = V1_i + T3_i
            sq_err = y[i]**2 - 2*y[i]*E_fu + Var_f_i
            quad = sq_err / (2 * sigma_noise**2)
            elog_sig, elog_one_minus = expected_log_sigmoid_gh(omega_j, mu_Wj, cov_Wj, X[i])
            T1 = -0.5*math.log(2*math.pi*sigma_noise**2) + elog_sig - quad
            T2 = math.log(Uconst) + elog_one_minus
            ELBO += torch.logsumexp(torch.stack([T1, T2]), dim=0)
    return ELBO

def train_vi(regions,
             V_params,
             u_params,
             hyperparams,
             lr: float = 1e-3,
             num_steps: int = 1000,
             log_interval: int = 100):
    optim_params = [V_params['mu_V'], V_params['sigma_V']]
    for u in u_params:
        optim_params += [u['mu_u'], u['Sigma_u'], u['sigma_noise'], u['omega']]
    optim_params += [hyperparams['lengthscales'], hyperparams['var_w']]
    optimizer = torch.optim.Adam(optim_params, lr=lr)
    for step in range(1, num_steps+1):
        optimizer.zero_grad()
        elbo = compute_ELBO(regions, V_params, u_params, hyperparams)
        loss = -elbo
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            V_params['sigma_V'].clamp_(min=1e-6)
            for u in u_params:
                u['sigma_noise'].clamp_(min=1e-6)
        if step % log_interval == 0 or step == 1:
            print(f"Step {step}/{num_steps}, ELBO = {elbo.item():.4f}")
    return V_params, u_params, hyperparams

def predict_vi(regions,
               V_params,
               hyperparams,
               M: int = 100,
               device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Z = hyperparams['Z']
    lengthscales = hyperparams['lengthscales']
    var_w = hyperparams['var_w']
    X_test = hyperparams['X_test']
    mu_V = V_params['mu_V']
    sigma_V = V_params['sigma_V']
    mu_W, cov_W = qW_from_qV(X_test, Z, mu_V, sigma_V, lengthscales, var_w)
    sigma_W = torch.sqrt(cov_W.diagonal(dim1=2, dim2=3))
    M_shape = (M, *mu_W.shape)
    eps = torch.randn(M_shape, device=device)
    W_samples = mu_W.unsqueeze(0) + eps * sigma_W.unsqueeze(0)
    T, Q, D = mu_W.shape
    mu_samples = torch.zeros((M, T), device=device)
    var_samples = torch.zeros((M, T), device=device)
    for m in range(M):
        W_m = W_samples[m]
        for j, reg in enumerate(regions):
            X_j = reg['X']
            y_j = reg['y']
            W_j = W_m[j]
            X_neigh = X_j.matmul(W_j.T)
            y_neigh = y_j.view(-1,1)
            x_t = X_test[j]
            x_t_proj = W_j.matmul(x_t).view(1,-1)
            mu_t, sig2_t, _, _ = jumpgp_ld_wrapper(
                X_neigh, y_neigh, x_t_proj,
                mode="CEM", flag=False, device=device
            )
            mu_samples[m,j] = mu_t.view(-1)
            var_samples[m,j] = sig2_t.view(-1)
    mu_pred = mu_samples.mean(dim=0)
    var_pred = var_samples.mean(dim=0)
    return mu_pred, var_pred

# === 示例运行 ===
if __name__ == "__main__":
    # Toy dimensions
    T, n, m1, m2, Q, D = 2, 5, 3, 4, 2, 3
    # 准备数据并搬到 GPU
    regions = []
    for _ in range(T):
        regions.append({
            'X': torch.randn(n, D, device=device),
            'y': torch.randn(n, device=device),
            'U': 1.0,
            'C': torch.randn(m1, Q, device=device)
        })
    # V 参数
    V_params = {
        'mu_V': torch.randn(m2, Q, D, device=device, requires_grad=True),
        'sigma_V': torch.rand(m2, Q, D, device=device, requires_grad=True)
    }
    # u 参数
    u_params = []
    for _ in range(T):
        u_params.append({
            'mu_u': torch.randn(m1, device=device, requires_grad=True),
            'Sigma_u': torch.eye(m1, device=device, requires_grad=True),
            'sigma_noise': torch.tensor(0.5, device=device, requires_grad=True),
            'omega': torch.randn(Q+1, device=device, requires_grad=True)
        })
    # 超参数
    hyperparams = {
        'Z': torch.randn(m2, D, device=device),
        'X_test': torch.randn(T, D, device=device),
        'lengthscales': torch.rand(Q, device=device, requires_grad=True),
        'var_w': torch.tensor(1.0, device=device, requires_grad=True)
    }
    # 训练和预测
    V_params, u_params, hyperparams = train_vi(regions, V_params, u_params, hyperparams, lr=1e-3, num_steps=10, log_interval=10)
    mu_pred, var_pred = predict_vi(regions, V_params, hyperparams, M=10, device=device)
    print("GPU device used:", device)
    print("mu_pred =", mu_pred)
    print("var_pred =", var_pred)
