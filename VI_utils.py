import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
# 将 JumpGP_code_py 所在的目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import math

from utils1 import jumpgp_ld_wrapper

def Phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

def eta(vartheta: torch.Tensor, tau: torch.Tensor, ell: int) -> torch.Tensor:
    prefactor = tau * math.sqrt(2 / math.pi) * torch.exp(-0.5 * (vartheta / tau) ** 2)
    linear   = vartheta * Phi(vartheta / tau)
    series   = torch.zeros_like(vartheta)
    for k in range(1, 2 * ell):
        sign  = -1.0 if (k - 1) % 2 else 1.0
        exp1  = torch.exp(k * vartheta + 0.5 * k * k * tau * tau)
        phi1  = Phi(-vartheta / tau - k * tau)
        exp2  = torch.exp(-k * vartheta + 0.5 * k * k * tau * tau)
        phi2  = Phi( vartheta / tau - k * tau)
        series += (sign / k) * (exp1 * phi1 + exp2 * phi2)
    return prefactor + linear + series

def expected_log_sigmoid(omega: torch.Tensor,
                         mu_W: torch.Tensor,
                         sigma_W: torch.Tensor,
                         x: torch.Tensor,
                         ell: int = 3) -> torch.Tensor:
    """
    Approximates E_q[log sigmoid(ωᵀ [1, W x])] under
    W ~ N(mu_W, sigma_W²) with independent entries.
    """
    # mu_W: [Q, D], sigma_W: [Q, D], omega: [Q+1], x: [D]

    # 1. Compute mean μ_z = ω₀ + Σ_q ω_{q+1} · (μ_{q,:} · x)
    Wx_mean = mu_W.matmul(x)          # → [Q]
    mu_z    = omega[0] + (omega[1:] * Wx_mean).sum()

    # 2. Compute variance τ_z² = Σ_{q,d} (ω_{q+1} x_d)² σ_{q,d}²
    w_coef  = omega[1:].unsqueeze(1)  # → [Q,1]
    x_vec   = x.unsqueeze(0)          # → [1,D]
    tau2    = ((w_coef * x_vec)**2 * sigma_W**2).sum()
    tau_z   = torch.sqrt(tau2 + 1e-12)

    # 3. E[log σ(z)] = -ηₗ(-μ_z, τ_z)
    return -eta(-mu_z, tau_z, ell), -eta(mu_z, tau_z, ell)

# def expected_log_sigmoid_gh(omega: torch.Tensor,
    #                         mu_W: torch.Tensor,
    #                         sigma_W: torch.Tensor,
    #                         x: torch.Tensor,
    #                         n_points: int = 20) -> torch.Tensor:
    # """
    # Gaussian–Hermite quadrature
    # bound = expected_log_sigmoid_gh(omega, mu_W, sigma_W, x)
    # E_q[log σ(z)] ≈ 1/√π Σ_i w_i * log σ(μ + √2 τ x_i)
    # """
    # # compute μ_z, τ_z
    # Wx_mean = mu_W.matmul(x)
    # mu_z = omega[0] + (omega[1:] * Wx_mean).sum()
    # w_coef = omega[1:].unsqueeze(1)
    # tau2 = ((w_coef * x.unsqueeze(0))**2 * sigma_W**2).sum()
    # tau_z = torch.sqrt(tau2 + 1e-12)

    # # GH nodes & weights
    # nodes, weights = np.polynomial.hermite.hermgauss(n_points)
    # nodes  = torch.from_numpy(nodes).to(mu_z)
    # weights= torch.from_numpy(weights).to(mu_z)

    # # evaluate
    # z = mu_z + math.sqrt(2.0) * tau_z * nodes
    # log_sigs = torch.log(torch.sigmoid(z))
    # return (weights * log_sigs).sum() / math.sqrt(math.pi)

import torch.nn.functional as F

def expected_log_sigmoid_gh(omega: torch.Tensor,
                            mu_W: torch.Tensor,
                            Sigma_W: torch.Tensor,
                            x: torch.Tensor,
                            n_points: int = 20):
    """
    Returns (E[log σ(z)], E[log(1-σ(z))]) via Gauss–Hermite,
    where z = ω₀ + Σ_q ω_q w_q^T x and q(w_q)=N(mu_W[q], Sigma_W[q]).
    用 Gauss–Hermite 求积近似
      E_q[log σ(z)],  E_q[log(1-σ(z))]
    其中 z = ω₀ + Σ_{q=1}^Q ω_q w_q^T x，
    q(w_q)=N(mu_W[q], Sigma_W[q]).
    Args:
      omega:     [Q+1]
      mu_W:      [Q, D]
      Sigma_W:   [Q, D, D]
      x:         [D]
      n_points:  GH 节点数
    Returns:
      (E_log_sigmoid, E_log_one_minus_sigmoid)
    """
    Q, D = mu_W.shape
    # 1) μ_z = ω₀ + Σ_q ω_q · (μ_q^T x)
    mu_proj = torch.einsum('qd,d->q', mu_W, x)       # [Q]
    mu_z = omega[0] + (omega[1:] * mu_proj).sum()    # scalar
    # 2) τ_z² = Σ_q ω_q² · (xᵀ Σ_q x)
    s = torch.einsum('d,qde,e->q', x, Sigma_W, x)     # [Q]
    tau2 = (omega[1:]**2 * s).sum()
    tau_z = torch.sqrt(tau2 + 1e-12)
    # 3) GH nodes & weights
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(n_points)
    nodes   = torch.from_numpy(nodes_np).to(mu_z)    # [n_points]
    weights = torch.from_numpy(weights_np).to(mu_z)  # [n_points]
    # 4) z values at nodes
    z = mu_z + math.sqrt(2.0) * tau_z * nodes        # [n_points]
    # 5) stable log‐sigmoid evaluations
    log_sig       = F.logsigmoid(z)                  # log σ(z)
    log_one_minus = F.logsigmoid(-z)                 # log(1−σ(z)) = logsigmoid(−z)
    # 6) weighted sum / √π
    factor = 1.0 / math.sqrt(math.pi)
    E_log_sig       = factor * (weights * log_sig).sum()
    E_log_one_minus = factor * (weights * log_one_minus).sum()
    return E_log_sig, E_log_one_minus


# def expected_log_one_minus_sigmoid_gh(omega: torch.Tensor,
#                                       mu_W: torch.Tensor,
#                                       sigma_W: torch.Tensor,
#                                       x: torch.Tensor,
#                                       n_points: int = 20) -> torch.Tensor:
#     """
#     E_q[log (1 - σ(z))] ≈ 1/√π Σ_i w_i * log(1 - σ(μ + √2 τ x_i))
#     """
#     # compute μ_z, τ_z
#     Wx_mean = mu_W.matmul(x)
#     mu_z = omega[0] + (omega[1:] * Wx_mean).sum()
#     w_coef = omega[1:].unsqueeze(1)
#     tau2 = ((w_coef * x.unsqueeze(0))**2 * sigma_W**2).sum()
#     tau_z = torch.sqrt(tau2 + 1e-12)

#     # GH nodes & weights
#     nodes, weights = np.polynomial.hermite.hermgauss(n_points)
#     nodes  = torch.from_numpy(nodes).to(mu_z)
#     weights= torch.from_numpy(weights).to(mu_z)

#     # evaluate
#     z = mu_z + math.sqrt(2.0) * tau_z * nodes
#     log_one_minus = torch.log1p(-torch.sigmoid(z))
#     return (weights * log_one_minus).sum() / math.sqrt(math.pi)


def kl_qp(Z: torch.Tensor,
          mu: torch.Tensor,
          sigma: torch.Tensor,
          lengthscales: torch.Tensor,
          var_w: torch.Tensor) -> torch.Tensor:
    """
    Compute KL(q(V) || p(V)) with gradient support for var_w and lengthscales.

    Args:
      Z:             [m2, D] inducing inputs
      mu:            [m2, Q, D] variational means
      sigma:         [m2, Q, D] variational stddevs (>0)
      lengthscales:  [Q] or [] tensor (requires_grad=True)
      var_w:         [] or [Q] tensor of variances (requires_grad=True)

    Returns:
      Scalar tensor: KL(q(V) || p(V))
    """
    m2, Q, D = mu.shape
    device = Z.device

    # pairwise squared distances [m2, m2]
    d2 = (Z.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(dim=2)

    kl_total = torch.tensor(0., device=device)

    for q in range(Q):
        # pick hyperparameters for this q
        ell_q = lengthscales[q] if lengthscales.ndim > 0 else lengthscales
        wvar  = var_w[q]        if var_w.ndim > 0 else var_w

        # prior covariance K_q [m2, m2]
        Kq = wvar * torch.exp(-0.5 * d2 / (ell_q**2))

        # Cholesky of Kq
        L = torch.linalg.cholesky(Kq)

        # log det Kq = 2 sum log diag(L)
        logdet_Kq = 2.0 * torch.log(torch.diagonal(L)).sum()

        # inverse of Kq via Cholesky, for trace and quadratic
        Kq_inv = torch.cholesky_inverse(L)

        # precompute diag(Kq_inv)
        Kinv_diag = torch.diagonal(Kq_inv)

        for d in range(D):
            mu_qd = mu[:, q, d]
            s2_qd = sigma[:, q, d].pow(2)

            # trace term: tr(Kq^{-1} Σ_qd)
            trace_term = (Kinv_diag * s2_qd).sum()

            # quadratic term: mu_qd^T Kq^{-1} mu_qd
            quad_term = mu_qd @ (Kq_inv @ mu_qd)

            # log det Σ_qd (diag) = sum log s2_qd
            logdet_Sigma = torch.log(s2_qd + 1e-12).sum()

            # KL for this (q,d)
            kl_qd = 0.5 * (
                trace_term
                + quad_term
                - m2
                + logdet_Kq
                - logdet_Sigma
            )
            kl_total += kl_qd

    return kl_total

# Example usage with gradients:
# m2, Q, D = 50, 3, 5
# Z = torch.randn(m2, D)
# mu = torch.randn(m2, Q, D, requires_grad=True)
# sigma = torch.rand(m2, Q, D, requires_grad=True)
# lengthscales = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# var_w = torch.tensor(1.5, requires_grad=True)
# kl = kl_qp(Z, mu, sigma, lengthscales, var_w)
# kl.backward()

def qW_from_qV(X: torch.Tensor,
               Z: torch.Tensor,
               mu_V: torch.Tensor,
               sigma_V: torch.Tensor,
               lengthscales: torch.Tensor,
               var_w: torch.Tensor):
    """
    Compute q(W) posterior mean & stddev given q(V) = N(mu_V, sigma_V^2)
    with scalar var_w.

    Args:
      X:            [T, D]    new input locations for W
      Z:            [m2, D]   inducing inputs for V
      mu_V:         [m2, Q, D] variational means for V
      sigma_V:      [m2, Q, D] variational stddevs for V
      lengthscales: [Q]        lengthscale l_q (requires_grad=True)
      var_w:        scalar     prior variance σ_w^2 (requires_grad=True)

    Returns:
      mu_W:    [T, Q, D] posterior mean for W
      cov_W: [T, Q, D, D] posterior stddev for W (diagonal only)
    """
    T, D = X.shape
    m2, Q, Dv = mu_V.shape
    assert Dv == D, f"Dimension mismatch: V has D={Dv}, X has D={D}"
    device, dtype = X.device, X.dtype

    # Precompute squared dists
    ZZ2 = (Z.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)  # [m2, m2]
    XZ2 = (X.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)  # [T,  m2]

    mu_W    = torch.empty((T, Q, D), device=device, dtype=dtype)
    sigma_W = torch.empty((T, Q, D), device=device, dtype=dtype)

    for q in range(Q):
        l = lengthscales[q]
        # Prior kernel matrices with scalar var_w
        Kzz = var_w * torch.exp(-0.5 * ZZ2 / (l**2))  # [m2, m2]
        Kxz = var_w * torch.exp(-0.5 * XZ2 / (l**2))  # [T,  m2]

        # Cholesky and inverse
        L     = torch.linalg.cholesky(Kzz)            # [m2, m2]
        Kzz_inv = torch.cholesky_inverse(L)           # [m2, m2]

        # Predictive weights A = Kxz @ Kzz_inv
        A = Kxz.matmul(Kzz_inv)                       # [T, m2]

        # Var_prior diagonal = var_w
        var_prior = var_w * torch.ones(T, device=device, dtype=dtype)

        # Correction from inducing: diag(A Kzx) = sum over i A[:,i] * Kxz[:,i]
        diag_cross = (A * Kxz).sum(dim=1)             # [T]

        # Term for variability from q(V)
        U = Kzz_inv.matmul(Kxz.t())                   # [m2, T]

        for d in range(D):
            # Posterior mean: μ = A @ μ_V[:,q,d]
            mu_W[:, q, d] = A.matmul(mu_V[:, q, d])

            # Variance injection: diag3 = sum_i U[i,j]^2 * σ_V[i,q,d]^2
            s2 = sigma_V[:, q, d].pow(2).unsqueeze(1)  # [m2,1]
            diag3 = (U.pow(2) * s2).sum(dim=0)         # [T]

            # Posterior stddev = sqrt(var_prior - diag_cross + diag3)
            sigma_W[:, q, d] = torch.sqrt(var_prior - diag_cross + diag3 + 1e-12)

    cov_W = torch.diag_embed(sigma_W.pow(2))  # [T, Q, D, D]

    return mu_W, cov_W

# Example usage:
# T, D, m2, Q = 10, 5, 20, 3
# X = torch.randn(T, D)
# Z = torch.randn(m2, D)
# mu_V = torch.randn(m2, Q, D, requires_grad=True)
# sigma_V = torch.rand(m2, Q, D, requires_grad=True)
# lengthscales = torch.rand(Q, requires_grad=True)
# var_w = torch.tensor(1.5, requires_grad=True)  # scalar
# mu_W, cov_W = qW_from_qV(X, Z, mu_V, sigma_V, lengthscales, var_w)
# (mu_W.sum() + sigma_W.sum()).backward()  # gradients flow into mu_V, sigma_V, lengthscales, var_w


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

# Example usage:
# m1, Q = 20, 5
# Z = torch.randn(m1, Q, requires_grad=False)
# mu = torch.randn(m1, requires_grad=True)
# # To ensure Sigma is PSD, parametrize via lower-triangular L and build Sigma = L L^T
# L_param = torch.eye(m1, requires_grad=True)
# Sigma = L_param @ L_param.T
# u_var = torch.tensor(1.0, requires_grad=True)
# kl_value = kl_q_u(Z, mu, Sigma, u_var)
# kl_value.backward()

def expected_Kfu(mu: torch.Tensor,
                 Sigma: torch.Tensor,
                 X: torch.Tensor,
                 Z: torch.Tensor,
                 sigma_f: torch.Tensor) -> torch.Tensor:
    """
    Compute E_q(W)[K_{f,u}] of shape [n, m1].
    Args:
      mu:      [K, D] mean of each weight dimension
      Sigma:   [K, D, D] covariance of each weight dimension
      X:       [n, D] data inputs
      Z:       [m1, K] inducing locations per dimension
      sigma_f: scalar amplitude
    Returns:
      [n, m1] tensor of E[k_fu(x_i, z_l)]
    """
    n, D = X.shape
    m1, K = Z.shape
    
    # 1. compute s = X Σ_k X^T for each k -> shape [n, K]
    #    using einsum for batch quadratic form
    s = torch.einsum('nd,kde,ne->nk', X, Sigma, X)  # [n, K]
    den = torch.sqrt(s + 1.0)                      # [n, K]
    
    # 2. compute mu_proj = X @ mu^T -> [n, K]
    mu_proj = X @ mu.T                             # [n, K]
    
    # 3. broadcast to compute diff for each l: shape [n, m1, K]
    diff = mu_proj.unsqueeze(1) - Z.unsqueeze(0)    # [n, m1, K]
    
    # 4. exponent term exp(- diff^2 / (2*(s+1))) broadcast s -> [n,1,K]
    s_plus1 = (s + 1.0).unsqueeze(1)                # [n, 1, K]
    exp_term = torch.exp(-0.5 * diff**2 / s_plus1)  # [n, m1, K]
    
    # 5. numerator = prod over K; denominator = prod over K of den
    num = torch.prod(exp_term, dim=2)               # [n, m1]
    den_prod = torch.prod(den, dim=1, keepdim=True) # [n, 1]
    
    # 6. final expectation
    return sigma_f * (num / den_prod)               # [n, m1]

# n, m, Q, D = 20, 5, 3, 5
# Z = torch.randn(m, Q, requires_grad=False)
# X = torch.randn(n, D, requires_grad=False)
# mu = torch.randn(Q, D, requires_grad=True)
# # To ensure Sigma is PSD, parametrize via lower-triangular L and build Sigma = L L^T
# A = torch.randn(Q, D, D)
# W = A.matmul(A.transpose(-1, -2))   # 形状 (Q, D, D)
# I = torch.eye(D, device=W.device).unsqueeze(0)  # 形状 (1, D, D)
# Sigma = W + 0.1 * I
# sigma_f = torch.tensor(1.0, requires_grad=True)
# Kfu = expected_Kfu(mu, Sigma, X, Z, sigma_f)
# # Kfu.backward()
# Kfu.shape


def expected_KufKfu(mu: torch.Tensor,
                    Sigma: torch.Tensor,
                    x: torch.Tensor,
                    Z: torch.Tensor,
                    sigma_f: torch.Tensor) -> torch.Tensor:
    """
    Compute E_q[ K_{u,f} K_{f,u} ] for a single x, shape [m1, m1].
    Args:
      mu:      [K, D]
      Sigma:   [K, D, D]
      x:       [D] single data point
      Z:       [m1, K]
      sigma_f: scalar amplitude
    Returns:
      [m1, m1] tensor
    """
    m1, K = Z.shape
    
    # 1. compute s_k = x^T Σ_k x  -> [K]
    s = torch.einsum('d,kde,e->k', x, Sigma, x)      # [K]
    den_k = torch.sqrt(s + 1.0)                     # [K]
    den_prod = torch.prod(den_k)                    # scalar
    
    # 2. mu projection [K]
    mu_proj = (mu @ x)                              # [K]
    
    # 3. pairwise avg of Z: M[l,l',k] = (Z[l,k] + Z[l',k])/2 -> [m1, m1, K]
    Zl = Z.unsqueeze(1)                             # [m1,1,K]
    Zl2 = Z.unsqueeze(0)                            # [1,m1,K]
    mid = 0.5 * (Zl + Zl2)                           # [m1,m1,K]
    
    # 4. diff for each l,l',k: [m1,m1,K]
    diff = mu_proj.unsqueeze(0).unsqueeze(0) - mid   # [m1,m1,K]
    
    # 5. exp_term = exp(- diff^2/(2*(s+1))) broadcast s -> [1,1,K]
    s_plus1 = (s + 1.0).unsqueeze(0).unsqueeze(0)    # [1,1,K]
    exp_term = torch.exp(-0.5 * diff**2 / s_plus1)   # [m1,m1,K]
    num = torch.prod(exp_term, dim=2)                # [m1,m1]
    
    # 6. prior cross-kernel exp(-1/4 * ||z_l - z_l'||^2) 
    #    compute squared distances between Z rows
    #    use cdist: Z (m1,K)
    dist = torch.cdist(Z, Z, p=2)                    # [m1, m1]
    sqd = dist**2                                   # [m1, m1]
    prior = torch.exp(-0.25 * sqd)                  # [m1, m1]
    
    # 7. final expectation: sigma_f^2 * prior * num / den_prod
    return sigma_f**2 * prior * (num / den_prod)     # [m1, m1]

# KufKfu = expected_KufKfu(mu, Sigma, X[0], Z, sigma_f)
# # Kfu.backward()
# KufKfu.shape

# === Main ELBO computation ===

def compute_ELBO(regions, V_params, u_params, hyperparams, ell=3):
    """
    Full ELBO including the Gaussian quadratic term.
    """
    # Unpack hyperparameters
    Z              = hyperparams['Z']
    lengthscales   = hyperparams['lengthscales']
    var_w          = hyperparams['var_w']
    # sigma_f_list   = hyperparams['sigma_f']   # list of length T
    X_test         = hyperparams['X_test']
    mu_V = V_params['mu_V']
    sigma_V = V_params['sigma_V']

    # Global KL for V
    KL_V = kl_qp(Z,
                 mu_V,
                 sigma_V,
                 lengthscales,
                 var_w)
    ELBO = -KL_V

    mu_W, cov_W = qW_from_qV(X_test, Z, mu_V, sigma_V, lengthscales, var_w)

    # Loop over regions
    for j, reg in enumerate(regions):
        X       = reg['X']           # [n, D]
        y       = reg['y']           # [n]
        Uconst  = reg['U']           # scalar
        Cj      = reg['C']           # [m1, Q]

        mu_u        = u_params[j]['mu_u']         # [m1]
        Sigma_u     = u_params[j]['Sigma_u']      # [m1, m1]
        sigma_noise = u_params[j]['sigma_noise']  # scalar
        omega_j     = u_params[j]['omega']        # [Q+1]
        mu_Wj   = mu_W[j]
        cov_Wj = cov_W[j]

        # KL for u_j
        ELBO -= kl_q_u(Cj, mu_u, Sigma_u, sigma_noise)

        # Precompute K_uu and its inverse
        m1 = Cj.size(0)
        d2 = (Cj.unsqueeze(1)-Cj.unsqueeze(0)).pow(2).sum(dim=2)  # [m1,m1]
        Kuu = torch.exp(-0.5 * d2) + sigma_noise * torch.eye(m1, device=Cj.device)
        Luu = torch.linalg.cholesky(Kuu)
        Kuu_inv = torch.cholesky_inverse(Luu)

        # Compute expected cross‐covariances
        # Kfu: [n, m1]
        Kfu = expected_Kfu(mu_Wj,
                           cov_Wj,
                           X, Cj, sigma_noise)
        

        # # Precompute first part of variance
        # Sf2 = sigma_noise**2
        # V1 = Sf2 - torch.sum(Kfu @ Kuu_inv * Kfu, dim=1)  # [n]

        # Loop over data points
        n = X.shape[0]
        for i in range(n):
            Kfu_i      = Kfu[i]               # [m1]
            KufKfu_i   = expected_KufKfu(mu_Wj, cov_Wj, X[i], Cj, sigma_noise)       # [m1,m1]
        
            # 1) V1
            V1_i = sigma_noise**2 - torch.trace(Kuu_inv @ KufKfu_i)
        
            # 2) 均值
            v     = Kuu_inv @ mu_u             # [m1]
            E_fu  = Kfu_i @ v                  # scalar
        
            # 3) 最后一项 T3
            # term_mu = v @ (KufKfu_i @ v)       # μ_u^T Kuu⁻¹ M Kuu⁻¹ μ_u
            term_S  = torch.trace(Kuu_inv @ KufKfu_i @ Kuu_inv @ Sigma_u)
            T3_i    = term_S        # scalar
        
            # 4) Var[f_i]
            Var_f_i = V1_i + T3_i
        
            # 5) E[(y-f)^2]
            # sq_err = y[i]**2 - 2*y[i]*E_fu + (Var_f_i + E_fu**2)
            sq_err = y[i]**2 - 2*y[i]*E_fu + Var_f_i
        
            # 6) 二次项
            quad = sq_err / (2 * sigma_noise**2)
            elog_sig, elog_one_minus = expected_log_sigmoid_gh(
                omega_j, mu_Wj, cov_Wj, X[i]
            )
        
            # 把 quad 插回 T1
            T1 = (
                -0.5*math.log(2*math.pi*sigma_noise**2)
                + elog_sig    # 事先调用 expected_log_sigmoid_gh 得到的
                - quad
            )
            T2 = math.log(Uconst) + elog_one_minus
        
            ELBO += torch.logsumexp(torch.stack([T1, T2]), dim=0)
        # for i in range(X.size(0)):
        #     # 1. GP mean and variance contributions
        #     E_fu = Kfu[i] @ (Kuu_inv @ mu_u)                                  # E[f]
        #     KufKfu = expected_KufKfu(mu_Wj, cov_Wj, X[i], Z, sigma_noise)
        #     V2   = Kfu[i] @ (Kuu_inv @ (Sigma_u @ (Kuu_inv @ Kfu[i])))       # var from q(u)
        #     Var_f_i = V1[i] + V2

        #     # 2. Quadratic term E[(y-f)^2]
        #     sq_err = y[i]**2 - 2*y[i]*E_fu + (Var_f_i + E_fu**2)
        #     quad   = sq_err / (2 * sigma_noise**2)

        #     # 3. Logistic expectations
        #     elog_sig, elog_one_minus = expected_log_sigmoid(
        #         omega_j, mu_u.unsqueeze(1)*0, Sigma_u.unsqueeze(0)*0, X[i], ell
        #     )

        #     # 4. Combine T1 and T2
        #     T1 = -0.5*math.log(2*math.pi*sigma_noise**2) + elog_sig - quad
        #     T2 = math.log(Uconst) + elog_one_minus

        #     ELBO += torch.logsumexp(torch.stack([T1, T2]), dim=0)

    return ELBO

import torch
from torch.optim import Adam

def train_vi(regions,
             V_params,
             u_params,
             hyperparams,
             lr: float = 1e-3,
             num_steps: int = 1000,
             log_interval: int = 100):
    """
    Train variational parameters by maximizing the ELBO via gradient ascent.

    Args:
      regions: list of region dicts, each with keys 'X', 'y', 'U', 'C'
      V_params: dict with 'mu_V' ([m2, Q, D], requires_grad) and 'sigma_V' ([m2, Q, D], requires_grad)
      u_params: list of dicts per region, each with 'mu_u' ([m1], requires_grad), 'Sigma_u' ([m1, m1], requires_grad),
                'sigma_noise' (scalar, requires_grad), 'omega' ([Q+1], requires_grad)
      hyperparams: dict with 'Z', 'X_test', 'lengthscales' ([Q], requires_grad), 'var_w' (scalar or [Q], requires_grad)
      lr: learning rate for Adam optimizer
      num_steps: number of training iterations
      log_interval: steps between logging ELBO

    Returns:
      Trained V_params, u_params, and hyperparams.
    """
    # Collect all torch.Tensor parameters to optimize
    optim_params = []
    # V variational parameters
    optim_params.append(V_params['mu_V'])
    optim_params.append(V_params['sigma_V'])
    # u variational parameters per region
    for u in u_params:
        optim_params.append(u['mu_u'])
        optim_params.append(u['Sigma_u'])
        optim_params.append(u['sigma_noise'])
        optim_params.append(u['omega'])
    # Hyperparameters (optional: if you wish to learn these)
    optim_params.append(hyperparams['lengthscales'])
    optim_params.append(hyperparams['var_w'])

    # Set up optimizer
    optimizer = Adam(optim_params, lr=lr)

    # Training loop
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        # Compute ELBO (scalar torch.Tensor)
        elbo = compute_ELBO(regions, V_params, u_params, hyperparams)
        # We maximize ELBO => minimize negative
        loss = -elbo
        loss.backward()
        optimizer.step()

        # Optional: enforce positivity for stddevs
        with torch.no_grad():
            V_params['sigma_V'].clamp_(min=1e-6)
            for u in u_params:
                u['Sigma_u'].data += 0  # assume PSD param
                u['sigma_noise'].clamp_(min=1e-6)

        # Logging
        if step % log_interval == 0 or step == 1:
            print(f"Step {step}/{num_steps}, ELBO = {elbo.item():.4f}")

    return V_params, u_params, hyperparams


def predict_vi(regions,
               V_params,
               hyperparams,
               M: int = 100,
               device=None):
    """
    Predict test outputs for each region using M samples from the variational posterior of W.

    Args:
      regions: list of dicts (keys 'X',[n,D], 'y') for each of T regions
      V_params: dict with 'mu_V' ([m2,Q,D]) and 'sigma_V' ([m2,Q,D])
      hyperparams: dict with 'Z' ([m2,D]), 'X_test' ([T,D]),
                   'lengthscales' ([Q]), 'var_w'
      M: number of posterior samples of W
      device: torch.device (defaults to mu_W device)

    Returns:
      mu_pred: [T] tensor of predictive means
      var_pred: [T] tensor of predictive variances
    """
    # Unpack
    Z = hyperparams['Z']
    lengthscales = hyperparams['lengthscales']
    var_w = hyperparams['var_w']
    X_test = hyperparams['X_test']
    mu_V = V_params['mu_V']
    sigma_V = V_params['sigma_V']

    # Posterior of W: mean and covariance
    mu_W, cov_W = qW_from_qV(X_test, Z, mu_V, sigma_V, lengthscales, var_w)
    # extract stddev from diagonal
    sigma_W = torch.sqrt(cov_W.diagonal(dim1=2, dim2=3))  # [T,Q,D]

    # Sample W: [M, T, Q, D]
    device = device or mu_W.device
    M_shape = (M, *mu_W.shape)
    eps = torch.randn(M_shape, device=device)
    W_samples = mu_W.unsqueeze(0) + eps * sigma_W.unsqueeze(0)

    T, Q, D = mu_W.shape
    mu_samples = torch.zeros((M, T), device=device)
    var_samples = torch.zeros((M, T), device=device)

    for m in range(M):
        W_m = W_samples[m]  # [T,Q,D]
        for j, reg in enumerate(regions):
            X_j = reg['X']         # [n_j, D]
            y_j = reg['y']         # [n_j]
            W_j = W_m[j]           # [Q, D]
            # Project
            X_neigh = X_j.matmul(W_j.T)         # [n_j, Q]
            y_neigh = y_j.view(-1, 1)           # [n_j, 1]
            x_t = X_test[j]                     # [D]
            x_t_proj = W_j.matmul(x_t).view(1, -1)  # [1, Q]
            # GP prediction via jumpgp_ld_wrapper
            mu_t, sig2_t, _, _ = jumpgp_ld_wrapper(
                X_neigh, y_neigh, x_t_proj,
                mode="CEM", flag=False, device=device
            )
            mu_samples[m, j] = mu_t.view(-1)
            var_samples[m, j] = sig2_t.view(-1)

    # Aggregate over samples
    mu_pred = mu_samples.mean(dim=0)
    var_pred = var_samples.mean(dim=0)
    return mu_pred, var_pred



# === Example check ===

if __name__ == "__main__":
    # Toy dimensions
    T, n, m1, m2, Q, D = 2, 5, 3, 4, 2, 3

    # Regions setup
    regions = [{
        'X': torch.randn(n, D),
        'y': torch.randn(n),
        'U': 1.0,
        'C': torch.randn(m1, Q)
    } for _ in range(T)]

    # V parameters
    V_params = {
        'mu_V': torch.randn(m2, Q, D, requires_grad=True),
        'sigma_V': torch.rand(m2, Q, D, requires_grad=True)
    }

    # u parameters per region
    u_params = []
    for _ in range(T):
        u_params.append({
            'mu_u': torch.randn(m1, requires_grad=True),
            'Sigma_u': torch.eye(m1, requires_grad=True),
            'sigma_noise': torch.tensor(0.5, requires_grad=True),
            'omega': torch.randn(Q+1, requires_grad=True)
        })

    # Hyperparameters
    hyperparams = {
        'Z': torch.randn(m2, D),
        'X_test': torch.randn(T, D),
        'lengthscales': torch.rand(Q, requires_grad=True),
        'var_w': torch.tensor(1.0, requires_grad=True),
        # 'sigma_f': [1.0 for _ in range(T)]
    }

    L = compute_ELBO(regions, V_params, u_params, hyperparams)
    print("ELBO L =", L.item())
    L.backward()
    print("Gradients OK")
    V_params, u_params, hyperparams = train_vi(regions=regions, V_params=V_params, u_params=u_params, hyperparams=hyperparams, lr=1e-3, num_steps=10, log_interval=10)
    print("train OK")
    mu_pred, var_pred = predict_vi(regions, V_params, hyperparams, M=10)
    print("Prediction OK")



