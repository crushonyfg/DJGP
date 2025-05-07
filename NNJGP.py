import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------
# 1. Feature network g𝜙
# -------------------------
class FeatureNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims=(64,64)):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        layers.append(nn.Linear(d, latent_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D] → z: [..., Q]
        return self.net(x)

# -------------------------
# 2. RBF kernel in latent space
# -------------------------
# def rbf_kernel(z1: torch.Tensor,
#                z2: torch.Tensor,
#                lengthscale: torch.Tensor,
#                sigma_f: torch.Tensor) -> torch.Tensor:
#     # z1: [*, Q], z2: [*, Q] or [M, Q]
#     # returns kernel matrix [..., M]
#     diff = z1.unsqueeze(-2) - z2.unsqueeze(-3)    # [..., M, Q]
#     d2 = (diff**2).sum(-1)                        # [..., M]
#     return sigma_f**2 * torch.exp(-0.5 * d2 / lengthscale**2)
def rbf_kernel(z1: torch.Tensor,
               z2: torch.Tensor) -> torch.Tensor:
    # z1: [*, Q], z2: [*, Q] or [M, Q]
    # returns kernel matrix [..., M]
    diff = z1.unsqueeze(-2) - z2.unsqueeze(-3)    # [..., M, Q]
    d2 = (diff**2).sum(-1)                        # [..., M]
    return torch.exp(-0.5 * d2)

# -------------------------
# 3. ELBO with NN mapping
# -------------------------
def compute_ELBO_nn(regions,
                    u_params,
                    g_phi: FeatureNet,
                    hyperparams,
                    prior_phi=None):
    """
    regions: list of T dicts with
        'X': [n_j, D] training inputs
        'y': [n_j]    training targets
        'C': [m1, Q]  inducing inputs in latent space
    u_params: list of T dicts with
        'm_u':   [m1]      variational mean
        'S_u':   [m1,m1]   variational covariance
        'sigma_n': scalar noise std
        'omega': [Q+1]     outlier-logit weights
        'U_logit': scalar logit for uniform-outlier
    g_phi: FeatureNet mapping R^D → R^Q
    hyperparams: dict with
        'lengthscale': [Q] or scalar
        'sigma_f':     scalar GP output scale
    prior_phi: if given, a function that returns
        KL[q(phi)||p(phi)] (e.g. weight decay or BNN KL)
    """
    T = len(regions)
    device = next(g_phi.parameters()).device
    lengthscale = hyperparams['lengthscale']
    sigma_f     = hyperparams['sigma_f']

    total_elbo = 0.0
    for j in range(T):
        X_j = regions[j]['X']                  # [n, D]
        y_j = regions[j]['y']                  # [n]
        C_j = regions[j]['C']                  # [m1, Q]

        # 1) map inputs to latent
        Z_j = g_phi(X_j)                       # [n, Q]

        # 2) compute kernel matrices
        K_uu = rbf_kernel(C_j, C_j, lengthscale, sigma_f)  # [m1,m1]
        K_fu = rbf_kernel(Z_j, C_j, lengthscale, sigma_f)  # [n,m1]
        # for second moment, since mapping is deterministic:
        K_ufK_fu = K_fu.t()[..., None] * K_fu[None, ...]   # [m1,n,m1] → we will use trace identities

        # 3) Cholesky & inverse of K_uu + σ_n^2 I
        sigma_n = u_params[j]['sigma_n']
        jitter = 1e-6
        L = torch.linalg.cholesky(K_uu + (sigma_n**2 + jitter) * torch.eye(K_uu.size(0), device=device))
        Kuu_inv = torch.cholesky_inverse(L)

        # 4) ELBO term (I): expected log-lik
        m_u = u_params[j]['m_u']              # [m1]
        S_u = u_params[j]['S_u']              # [m1,m1]
        # predictive mean E[f] = K_fu @ Kuu_inv @ m_u
        A = Kuu_inv @ m_u                     # [m1]
        E_f = K_fu @ A                        # [n]
        # second moment E[f f]
        # Var[f] = σ_n^2 I - diag(K_fu @ Kuu_inv @ K_fu.T) + diag(K_fu @ Kuu_inv @ S_u @ Kuu_inv @ K_fu.T)
        diag1 = (K_fu @ Kuu_inv * K_fu).sum(dim=1)     # [n]
        temp = Kuu_inv @ S_u @ Kuu_inv
        diag2 = (K_fu @ temp * K_fu).sum(dim=1)        # [n]
        Var_f = sigma_n**2 - diag1 + diag2             # [n]
        # compute E[(y - f)^2] = (y - E_f)^2 + Var_f
        sq_err = (y_j - E_f)**2 + Var_f                # [n]
        I_term = -0.5 * torch.log(2*math.pi*sigma_n**2) \
                 -0.5 * sq_err / sigma_n**2
        # handle outlier indicator
        omega = u_params[j]['omega']                  # [Q+1]
        v_logit = u_params[j]['U_logit']              # scalar
        # compute π = sigmoid( [1, Z_j] @ omega )
        ones = torch.ones(Z_j.size(0),1, device=device)
        z_feat = torch.cat([ones, Z_j], dim=1)        # [n, Q+1]
        logit = (z_feat @ omega)                      # [n]
        logp_inlier = -0.5*torch.log(2*math.pi*sigma_n**2) \
                      -0.5*sq_err / sigma_n**2 + F.logsigmoid(logit)
        logp_outlier= v_logit + F.logsigmoid(-logit)  # uniform density U absorbed into v_logit
        # ELBO contribution per point: logsumexp([inlier, outlier])
        point_elbo = torch.logsumexp(torch.stack([logp_inlier, logp_outlier], dim=0), dim=0)
        I_II = point_elbo.sum()

        # 5) KL(q(u)||p(u)) term (III)
        # p(u)=N(0,K_uu), q(u)=N(m_u,S_u)
        trace_term = torch.trace(Kuu_inv @ S_u)
        quad_term  = m_u @ (Kuu_inv @ m_u)
        logdet_q   = torch.logdet(S_u + jitter*torch.eye(S_u.size(0),device=device))
        logdet_p   = 2.0 * torch.log(torch.diagonal(L)).sum()
        kl_u = 0.5*(trace_term + quad_term - m_u.numel() + logdet_p - logdet_q)

        total_elbo += I_II - kl_u

    # 6) optional KL for φ
    if prior_phi is not None:
        total_elbo -= prior_phi(g_phi)

    return total_elbo

import torch
import torch.nn.functional as F
import math

# def compute_ELBO_nn_vectorized(regions, u_params, g_phi, hyperparams, prior_phi=None):
#     """
#     Vectorized ELBO computation across T regions (no explicit Python loop).
    
#     regions: list of T dicts with 'X':[n,D], 'y':[n], 'C':[m1,Q]
#     u_params: list of T dicts with 'm_u':[m1], 'S_u':[m1,m1],
#               'sigma_n':scalar, 'omega':[Q+1], 'U_logit':scalar
#     g_phi: FeatureNet, maps [*,D] -> [*,Q]
#     hyperparams: dict with 'lengthscale', 'sigma_f'
#     prior_phi: optional function(g_phi) -> KL term
#     """
#     device = next(g_phi.parameters()).device
    
#     # Stack region data
#     X = torch.stack([r['X'] for r in regions], dim=0)       # [T,n,D]
#     y = torch.stack([r['y'] for r in regions], dim=0)       # [T,n]
#     C = torch.stack([r['C'] for r in regions], dim=0)       # [T,m1,Q]
#     m_u = torch.stack([p['m_u'] for p in u_params], dim=0)  # [T,m1]
#     S_u = torch.stack([p['S_u'] for p in u_params], dim=0)  # [T,m1,m1]
#     sigma_n = torch.stack([p['sigma_n'] for p in u_params], dim=0).view(-1,1,1)  # [T,1,1]
#     omega = torch.stack([p['omega'] for p in u_params], dim=0)                  # [T,Q+1]
#     U_logit = torch.stack([p['U_logit'] for p in u_params], dim=0).view(-1,1)    # [T,1]

#     T, n, D = X.shape
#     _, m1, Q = C.shape

#     # 1) Latent mapping
#     Z = g_phi(X.view(T*n, D)).view(T, n, Q)  # [T,n,Q]
    
#     # 2) Kernel matrices
#     lengthscale = hyperparams['lengthscale']
#     sigma_f = hyperparams['sigma_f']
#     # K_uu: [T,m1,m1]
#     diff_uu = C.unsqueeze(2) - C.unsqueeze(1)  # [T,m1,m1,Q]
#     K_uu = sigma_f**2 * torch.exp(-0.5 * (diff_uu**2).sum(-1) / lengthscale**2)
#     # K_fu: [T,n,m1]
#     diff_fu = Z.unsqueeze(2) - C.unsqueeze(1)  # [T,n,m1,Q]
#     K_fu = sigma_f**2 * torch.exp(-0.5 * (diff_fu**2).sum(-1) / lengthscale**2)

#     # 3) Cholesky and inverse
#     jitter = 1e-6 * torch.eye(m1, device=device).unsqueeze(0)  # [1,m1,m1]
#     L = torch.linalg.cholesky(K_uu + sigma_n**2 * torch.eye(m1, device=device).unsqueeze(0) + jitter)
#     Kuu_inv = torch.cholesky_inverse(L)  # [T,m1,m1]

#     # 4) Predictive moments
#     A = torch.matmul(Kuu_inv, m_u.unsqueeze(-1))  # [T,m1,1]
#     E_f = torch.matmul(K_fu, A).squeeze(-1)       # [T,n]
    
#     C1 = torch.matmul(K_fu, Kuu_inv)              # [T,n,m1]
#     diag1 = (C1 * K_fu).sum(-1)                   # [T,n]
    
#     temp = torch.matmul(Kuu_inv, torch.matmul(S_u, Kuu_inv))  # [T,m1,m1]
#     diag2 = (torch.matmul(K_fu, temp) * K_fu).sum(-1)         # [T,n]
    
#     # Var_f = sigma_n.squeeze(-1)**2 - diag1 + diag2  # [T,n]
#     var_n2 = (sigma_n**2).squeeze(-1)
#     Var_f  = var_n2 - diag1 + diag2                # [T,n]
#     sq_err = (y - E_f)**2 + Var_f                               # [T,n]

#     # 5) ELBO term I+II via log-sum-exp
#     ones = torch.ones(T, n, 1, device=device)
#     z_feat = torch.cat([ones, Z], dim=2)  # [T,n,Q+1]
#     logit = torch.matmul(z_feat, omega.unsqueeze(-1)).squeeze(-1)  # [T,n]
    
#     # logp_inlier = -0.5*torch.log(2*math.pi*sigma_n.squeeze(-1).squeeze(-1)**2) \
#     #               -0.5*sq_err/sigma_n.squeeze(-1).squeeze(-1)**2 \
#     #               + F.logsigmoid(logit)
#     logp_inlier = -0.5*torch.log(2*math.pi*var_n2) -0.5*sq_err/var_n2 + F.logsigmoid(logit)
#     logp_outlier = U_logit + F.logsigmoid(-logit)  # [T,n]
    
#     point_elbo = torch.logsumexp(torch.stack([logp_inlier, logp_outlier], dim=0), dim=0)  # [T,n]
#     I_II = point_elbo.sum(dim=1)  # [T]

#     # 6) KL(q(u)||p(u)): [T]
#     trace_term = torch.einsum('tij,tji->t', Kuu_inv, S_u)
#     quad_term = (m_u.unsqueeze(1) @ torch.matmul(Kuu_inv, m_u.unsqueeze(-1))).squeeze(-1).squeeze(-1)  # [T]
#     logdet_q = torch.logdet(S_u + jitter)
#     logdet_p = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1)  # [T]
#     kl_u = 0.5 * (trace_term + quad_term - m1 + logdet_p - logdet_q)  # [T]

#     # Sum across regions
#     elbo = I_II.sum() - kl_u.sum()
#     if prior_phi is not None:
#         elbo -= prior_phi(g_phi)
#     return elbo

def compute_ELBO_nn_vectorized(regions, u_params, g_phi, hyperparams, prior_phi=None):
    """
    Vectorized ELBO across T regions, using separate sigma_k for kernel scaling
    and sigma_n for noise. Unit RBF prior (sigma_f=1, lengthscale=1).
    """
    device = next(g_phi.parameters()).device

    # 1) Stack region data
    X = torch.stack([r['X']    for r in regions], dim=0)        # [T,n,D]
    y = torch.stack([r['y']    for r in regions], dim=0)        # [T,n]
    C = torch.stack([r['C']    for r in regions], dim=0)        # [T,m1,Q]
    m_u = torch.stack([p['m_u'] for p in u_params], dim=0)      # [T,m1]
    # S_u = torch.stack([p['S_u'] for p in u_params], dim=0)      # [T,m1,m1]

    # stack the unconstrained lower‐triangular factors
    L_stack = torch.stack([p['L_u'] for p in u_params], dim=0)  # [T,m1,m1]
    # zero out any upper‐triangular part
    L_tril   = torch.tril(L_stack)    
    m1 = L_tril.shape[1]                          # [T,m1,m1]
    # build S_u = L_u @ L_u^T + jitter for each region
    jitter   = 1e-6 * torch.eye(m1, device=device).unsqueeze(0)  # [1,m1,m1]
    S_u      = L_tril @ L_tril.transpose(-1,-2) + jitter        # [T,m1,m1]


    sigma_k = torch.stack([p['sigma_k'] for p in u_params], dim=0).view(-1,1,1)  # [T,1,1]
    sigma_n = torch.stack([p['sigma_n'] for p in u_params], dim=0).view(-1,1,1)  # [T,1,1]
    omega   = torch.stack([p['omega']   for p in u_params], dim=0)               # [T,Q+1]
    U_logit = torch.stack([p['U_logit'] for p in u_params], dim=0).view(-1,1)     # [T,1]

    T, n, D  = X.shape
    _, m1, Q = C.shape

    # 2) Latent mapping
    Z = g_phi(X.view(T*n, D)).view(T, n, Q)  # [T,n,Q]

    # 3) Kernel matrices (unit RBF)
    diff_uu = C.unsqueeze(2) - C.unsqueeze(1)      # [T,m1,m1,Q]
    d2_uu   = (diff_uu**2).sum(-1)                # [T,m1,m1]
    K_uu    = torch.exp(-0.5 * d2_uu)             # [T,m1,m1]
    # K_uu   += sigma_n * torch.eye(m1, device=device).unsqueeze(0)

    diff_fu = Z.unsqueeze(2) - C.unsqueeze(1)      # [T,n,m1,Q]
    d2_fu   = (diff_fu**2).sum(-1)                 # [T,n,m1]
    K_fu    = torch.exp(-0.5 * d2_fu)              # [T,n,m1]
    K_fu    = K_fu * sigma_k                       # scale by sigma_k

    # 4) Cholesky & inverse
    jitter = 1e-6 * torch.eye(m1, device=device).unsqueeze(0)
    L      = torch.linalg.cholesky(K_uu + jitter)
    Kuu_inv = torch.cholesky_inverse(L)            # [T,m1,m1]

    # 5) Predictive moments
    A    = torch.matmul(Kuu_inv, m_u.unsqueeze(-1))  # [T,m1,1]
    E_f  = torch.matmul(K_fu, A).squeeze(-1)          # [T,n]

    C1   = torch.matmul(K_fu, Kuu_inv)                # [T,n,m1]
    diag1 = (C1 * K_fu).sum(-1)                       # [T,n]

    temp  = torch.matmul(Kuu_inv, torch.matmul(S_u, Kuu_inv))  # [T,m1,m1]
    diag2 = (torch.matmul(K_fu, temp) * K_fu).sum(-1)          # [T,n]

    # 6) Variance & squared error (use sigma_n^2 for noise var)
    var_noise = (sigma_n**2).squeeze(-1)            # [T,1]
    # var_noise = var_noise.clamp(min=1e-2)
    Var_f     = var_noise - diag1 + diag2          # [T,n]
    sq_err    = (y - E_f)**2 + Var_f           # [T,n]

    # 7) ELBO term (I+II) via log-sum-exp
    ones   = torch.ones(T, n, 1, device=device)
    z_feat = torch.cat([ones, Z], dim=2)             # [T,n,Q+1]
    logit  = torch.matmul(z_feat, omega.unsqueeze(-1)).squeeze(-1)  # [T,n]

    logp_inlier = -0.5*torch.log(2*math.pi*var_noise) \
                  -0.5*sq_err/var_noise \
                  + F.logsigmoid(logit)
    logp_outlier = U_logit + F.logsigmoid(-logit)    # [T,n]

    point_elbo = torch.logsumexp(torch.stack([logp_inlier, logp_outlier], dim=0), dim=0)  # [T,n]
    I_II       = point_elbo.sum(dim=1)              # [T]

    # 8) KL(q(u)||p(u)): [T]
    trace_term = torch.einsum('tij,tji->t', Kuu_inv, S_u)
    quad_term  = (m_u.unsqueeze(1) @ torch.matmul(Kuu_inv, m_u.unsqueeze(-1))).squeeze(-1).squeeze(-1)
    logdet_q   = torch.logdet(S_u + jitter)
    logdet_p   = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1)
    kl_u       = 0.5 * (trace_term + quad_term - m1 + logdet_p - logdet_q)  # [T]

    # 9) Sum over regions
    elbo = I_II.sum() - kl_u.sum()
    if prior_phi is not None:
        elbo -= prior_phi(g_phi)

    return elbo

import torch
import torch.nn.functional as F
import math

# def compute_ELBO_nn_vectorized(regions, u_params, g_phi, hyperparams, prior_phi=None):
#     """
#     Same as before, but with debug prints whenever an intermediate tensor contains NaNs.
#     """
#     device = next(g_phi.parameters()).device

#     def check_nan(name, tensor):
#         if torch.isnan(tensor).any():
#             # find the first few NaN indices (flattened)
#             idx = torch.nonzero(tensor != tensor, as_tuple=False)
#             print(f"*** NaN detected in {name} at positions {idx[:5].tolist()} (showing up to 5) ***")

#     # 1) Stack region data
#     X = torch.stack([r['X']    for r in regions], dim=0)        # [T,n,D]
#     y = torch.stack([r['y']    for r in regions], dim=0)        # [T,n]
#     C = torch.stack([r['C']    for r in regions], dim=0)        # [T,m1,Q]
#     m_u = torch.stack([p['m_u'] for p in u_params], dim=0)      # [T,m1]
#     S_u = torch.stack([p['S_u'] for p in u_params], dim=0)      # [T,m1,m1]
#     sigma_k = torch.stack([p['sigma_k'] for p in u_params], dim=0).view(-1,1,1)  # [T,1,1]
#     sigma_n = torch.stack([p['sigma_n'] for p in u_params], dim=0).view(-1,1,1)  # [T,1,1]
#     omega   = torch.stack([p['omega']   for p in u_params], dim=0)               # [T,Q+1]
#     U_logit = torch.stack([p['U_logit'] for p in u_params], dim=0).view(-1,1)     # [T,1]

#     T, n, D  = X.shape
#     _, m1, Q = C.shape

#     # 2) Latent mapping
#     Z = g_phi(X.view(T*n, D)).view(T, n, Q)  # [T,n,Q]
#     check_nan("Z", Z)

#     # 3) Kernel matrices (unit RBF)
#     diff_uu = C.unsqueeze(2) - C.unsqueeze(1)      # [T,m1,m1,Q]
#     d2_uu   = (diff_uu**2).sum(-1)                # [T,m1,m1]
#     K_uu    = torch.exp(-0.5 * d2_uu)             # [T,m1,m1]
#     check_nan("K_uu", K_uu)

#     diff_fu = Z.unsqueeze(2) - C.unsqueeze(1)      # [T,n,m1,Q]
#     d2_fu   = (diff_fu**2).sum(-1)                 # [T,n,m1]
#     K_fu    = torch.exp(-0.5 * d2_fu)              # [T,n,m1]
#     K_fu    = K_fu * sigma_k                       # scale by sigma_k
#     check_nan("K_fu", K_fu)

#     # 4) Cholesky & inverse
#     jitter = 1e-6 * torch.eye(m1, device=device).unsqueeze(0)
#     L      = torch.linalg.cholesky(K_uu + jitter)
#     Kuu_inv = torch.cholesky_inverse(L)            # [T,m1,m1]
#     check_nan("Kuu_inv", Kuu_inv)

#     # 5) Predictive moments
#     A    = torch.matmul(Kuu_inv, m_u.unsqueeze(-1))  # [T,m1,1]
#     E_f  = torch.matmul(K_fu, A).squeeze(-1)          # [T,n]
#     check_nan("E_f", E_f)

#     C1   = torch.matmul(K_fu, Kuu_inv)                # [T,n,m1]
#     diag1 = (C1 * K_fu).sum(-1)                       # [T,n]
#     check_nan("diag1", diag1)

#     temp  = torch.matmul(Kuu_inv, torch.matmul(S_u, Kuu_inv))  # [T,m1,m1]
#     diag2 = (torch.matmul(K_fu, temp) * K_fu).sum(-1)          # [T,n]
#     check_nan("diag2", diag2)

#     # 6) Variance & squared error
#     var_noise = (sigma_n**2).squeeze(-1)            # [T,1]
#     check_nan("var_noise", var_noise)
#     Var_f     = var_noise - diag1 + diag2          # [T,n]
#     check_nan("Var_f", Var_f)
#     sq_err    = (y - E_f)**2 + Var_f                # [T,n]
#     check_nan("sq_err", sq_err)

#     # 7) ELBO term (I+II)
#     ones   = torch.ones(T, n, 1, device=device)
#     z_feat = torch.cat([ones, Z], dim=2)             # [T,n,Q+1]
#     logit  = torch.matmul(z_feat, omega.unsqueeze(-1)).squeeze(-1)  # [T,n]
#     logp_inlier = -0.5*torch.log(2*math.pi*var_noise) \
#                   -0.5*sq_err/var_noise \
#                   + F.logsigmoid(logit)
#     check_nan("logp_inlier", logp_inlier)
#     logp_outlier = U_logit + F.logsigmoid(-logit)    # [T,n]
#     check_nan("logp_outlier", logp_outlier)

#     point_elbo = torch.logsumexp(
#         torch.stack([logp_inlier, logp_outlier], dim=0), dim=0
#     )  # [T,n]
#     check_nan("point_elbo", point_elbo)
#     I_II = point_elbo.sum(dim=1)              # [T]
#     check_nan("I_II", I_II)

#     # 8) KL(q(u)||p(u)): [T]
#     trace_term = torch.einsum('tij,tji->t', Kuu_inv, S_u)
#     quad_term  = (m_u.unsqueeze(1) @ torch.matmul(Kuu_inv, m_u.unsqueeze(-1))) \
#                  .squeeze(-1).squeeze(-1)
#     logdet_q   = torch.logdet(S_u + jitter)
#     logdet_p   = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1)
#     kl_u       = 0.5 * (trace_term + quad_term - m1 + logdet_p - logdet_q)  # [T]
#     check_nan("kl_u", kl_u)

#     # 9) Sum over regions
#     elbo = I_II.sum() - kl_u.sum()
#     check_nan("elbo", elbo)

#     if prior_phi is not None:
#         elbo_before = elbo.clone()
#         elbo = elbo - prior_phi(g_phi)
#         check_nan("elbo_after_prior_phi", elbo)

#     return elbo


from tqdm import tqdm

def train_nn(regions, u_params, g_phi, hyperparams,
             lr=1e-3, num_steps=1000, log_interval=100):
    """
    Train the NN-JGP model by maximizing ELBO (vectorized).
    Returns updated u_params, g_phi, hyperparams.
    """
    # Collect trainable parameters
    params = list(g_phi.parameters())
    # params += [hyperparams['lengthscale'], hyperparams['sigma_f']]
    for u in u_params:
        params += [u['U_logit'], u['omega'], u['m_u'], u['L_u'], u['sigma_n'], u['sigma_k']]
    opt = torch.optim.Adam(params, lr=lr)
    
    for step in range(1, num_steps+1):
        opt.zero_grad()
        # vectorized ELBO must be defined elsewhere
        elbo = compute_ELBO_nn_vectorized(regions, u_params, g_phi, hyperparams)
        (-elbo).backward()
        opt.step()
        
        # Clamp positive constraints
        with torch.no_grad():
            # hyperparams['lengthscale'].clamp_(min=1e-6)
            # hyperparams['sigma_f'].clamp_(min=1e-6)
            for u in u_params:
                u['sigma_n'].clamp_(min=1e-6)
                u['sigma_k'].clamp_(min=1e-6)
                # ensure Sigma_u remains positive definite on diagonal
                # u['S_u'].diagonal().clamp_(min=1e-6)
        
        if step % log_interval == 0 or step == 1:
            print(f"[Train] Step {step}/{num_steps}, ELBO={elbo.item():.4f}")
    
    return u_params, g_phi, hyperparams

from utils1 import jumpgp_ld_wrapper  # or wherever your wrapper lives

def predict_nn(regions, u_params, g_phi, hyperparams):
    """
    Analytic prediction using jumpgp_ld_wrapper on NN‐mapped inputs (no sampling).
    regions[j] must contain:
      'X':       Tensor [n_j, D]
      'y':       Tensor [n_j]
      'X_test':  Tensor [D]       # single test point per region
      'C':       Tensor [m1, Q]   # inducing inputs in latent space
    u_params[j] is ignored here (we assume standard GP prior on u)
    """
    device = next(g_phi.parameters()).device
    T = len(regions)
    mu_pred = torch.zeros(T, device=device)
    var_pred = torch.zeros(T, device=device)

    for j, r in tqdm(enumerate(regions)):
        # 1) unpack
        Xj      = r['X']            # [n_j, D]
        yj      = r['y']            # [n_j]
        X_testj = r['X_test']        # [D]
        Cj      = r['C']            # [m1, Q]

        # 2) NN‐map into latent space
        Zj = g_phi(Xj)               # [n_j, Q]
        zt = g_phi(X_testj.unsqueeze(0))  # [1, Q]

        # 3) call the same wrapper you used before
        #    here yn needs shape [n_j,1], xt shape [1,Q]
        yn = yj.view(-1, 1)
        mu_t, var_t, _, _ = jumpgp_ld_wrapper(
            Zj, yn, zt,
            mode="CEM",
            flag=False,
            device=device
        )

        # 4) collect
        mu_pred[j]  = mu_t.view(-1)
        var_pred[j] = var_t.view(-1)

    # nan‐safe
    mu_pred  = torch.nan_to_num(mu_pred,  nan=0.0)
    var_pred = torch.nan_to_num(var_pred, nan=1e-6)
    return mu_pred, var_pred



if __name__ == "__main__":
    import torch
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model dimensions
    T, n, D, Q, m1 = 3, 50, 10, 4, 20

    # 1) Instantiate feature network
    g_phi = FeatureNet(input_dim=D, latent_dim=Q).to(device)

    # 2) Create dummy regions with training & test data
    regions = []
    for _ in range(T):
        regions.append({
            'X':      torch.randn(n, D, device=device),
            'y':      torch.randn(n, device=device),
            'X_test': torch.randn(D, device=device),
            'C':      torch.randn(m1, Q, device=device),
        })

    # 3) Initialize variational parameters for each region
    u_params = []
    for _ in range(T):
        u_params.append({
            'm_u':     torch.zeros(m1,      device=device, requires_grad=True),
            'L_u':     torch.eye(m1,       device=device, requires_grad=True),
            'sigma_n': torch.tensor(0.5,    device=device, requires_grad=True),
            'sigma_k': torch.tensor(1.0,    device=device, requires_grad=True),  # kernel scale
            'omega':   torch.randn(Q+1,     device=device, requires_grad=True),
            'U_logit': torch.tensor(0.0,    device=device, requires_grad=True),
        })

    # 4) Dummy hyperparameters (not used in vectorized ELBO)
    hyperparams = {}

    # 5) Test compute_ELBO_nn_vectorized
    elbo_before = compute_ELBO_nn_vectorized(regions, u_params, g_phi, hyperparams)
    print("Initial ELBO:", elbo_before.item())

    # 6) Train the model
    u_params, g_phi, hyperparams = train_nn(
        regions, u_params, g_phi, hyperparams,
        lr=1e-3, num_steps=200, log_interval=50
    )

    # 7) Compute ELBO after training
    elbo_after = compute_ELBO_nn_vectorized(regions, u_params, g_phi, hyperparams)
    print("ELBO after training:", elbo_after.item())

    # 8) Predict
    mu_pred, var_pred = predict_nn(regions, u_params, g_phi, hyperparams)
    print("Predicted means:", mu_pred)
    print("Predicted variances:", var_pred)
