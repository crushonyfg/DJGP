import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import tqdm

from utils1 import jumpgp_ld_wrapper
from JumpGP_test import load_dataset, find_neighborhoods, compute_metrics
from NNJGP import predict_nn  # ensure this is on your PYTHONPATH

# -------------------------
# 1) Bayesian linear layer
# -------------------------
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_f, self.out_f = in_features, out_features
        self.prior_sigma = prior_sigma

        # variational params: mu and rho for weights and bias
        self.weight_mu   = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho  = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu     = nn.Parameter(torch.zeros(out_features))
        self.bias_rho    = nn.Parameter(torch.full((out_features,), -3.0))

    def forward(self, x, sample=True):
        w_sigma = F.softplus(self.weight_rho)
        b_sigma = F.softplus(self.bias_rho)

        if self.training or sample:
            eps_w = torch.randn_like(w_sigma)
            eps_b = torch.randn_like(b_sigma)
            w = self.weight_mu + w_sigma * eps_w
            b = self.bias_mu   + b_sigma * eps_b
        else:
            w, b = self.weight_mu, self.bias_mu

        return F.linear(x, w, b)

    def kl(self):
        pmu = self.prior_sigma
        w_mu, w_sig = self.weight_mu, F.softplus(self.weight_rho)
        b_mu, b_sig = self.bias_mu,   F.softplus(self.bias_rho)

        def kl_term(mu, sig):
            return (torch.log(pmu / sig)
                    + (sig**2 + mu**2) / (2 * pmu**2)
                    - 0.5).sum()

        return kl_term(w_mu, w_sig) + kl_term(b_mu, b_sig)


# -------------------------
# 2) Bayesian feature network
# -------------------------
class BayesianFeatureNet(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=(8,), prior_sigma=1.0):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [latent_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(BayesianLinear(dims[i], dims[i+1], prior_sigma))

    def forward(self, x, sample=True):
        out = x
        for layer in self.layers[:-1]:
            out = F.relu(layer(out, sample=sample))
        out = self.layers[-1](out, sample=sample)
        return out

    def kl(self):
        return sum(layer.kl() for layer in self.layers)


# -------------------------
# 3) RBF kernel
# -------------------------
def rbf_kernel(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    diff = z1.unsqueeze(-2) - z2.unsqueeze(-3)
    d2   = (diff**2).sum(-1)
    return torch.exp(-0.5 * d2)


# -------------------------
# 4) Vectorized ELBO + BNN KL
# -------------------------
def compute_ELBO_bnn(regions, u_params, g_phi, hyperparams, prior_phi=True):
    device = next(g_phi.parameters()).device

    # 1) Stack data
    X = torch.stack([r['X'] for r in regions], dim=0)      # [T,n,D]
    y = torch.stack([r['y'] for r in regions], dim=0)      # [T,n]
    C = torch.stack([r['C'] for r in regions], dim=0)      # [T,m1,Q]
    m_u = torch.stack([p['m_u'] for p in u_params], dim=0) # [T,m1]

    # build S_u from L_u
    L_stack = torch.stack([p['L_u'] for p in u_params], dim=0)  # [T,m1,m1]
    L_tril  = torch.tril(L_stack)
    jitter_S = 1e-6 * torch.eye(L_tril.size(1), device=device).unsqueeze(0)
    S_u = L_tril @ L_tril.transpose(-1, -2) + jitter_S          # [T,m1,m1]

    sigma_k = torch.stack([p['sigma_k'] for p in u_params], dim=0).view(-1,1,1)  # [T,1,1]
    sigma_n = torch.stack([p['sigma_n'] for p in u_params], dim=0).view(-1,1,1)  # [T,1,1]
    omega   = torch.stack([p['omega']   for p in u_params], dim=0)              # [T,Q+1]
    U_logit = torch.stack([p['U_logit'] for p in u_params], dim=0).view(-1,1)    # [T,1]

    T, n, D = X.shape
    _, m1, Q = C.shape

    # 2) sample-based latent mapping
    Z = g_phi(X.view(T*n, D), sample=True).view(T, n, Q)  # [T,n,Q]

    # 3) Kernel matrices
    diff_uu = C.unsqueeze(2) - C.unsqueeze(1)      # [T,m1,m1,Q]
    d2_uu   = (diff_uu**2).sum(-1)                 # [T,m1,m1]
    K_uu    = torch.exp(-0.5 * d2_uu)              # [T,m1,m1]

    diff_fu = Z.unsqueeze(2) - C.unsqueeze(1)      # [T,n,m1,Q]
    d2_fu   = (diff_fu**2).sum(-1)                 # [T,n,m1]
    K_fu    = torch.exp(-0.5 * d2_fu) * sigma_k    # [T,n,m1]

    # 4) Cholesky and inverse
    jitter_K = 1e-6 * torch.eye(m1, device=device).unsqueeze(0)
    Lm       = torch.linalg.cholesky(K_uu + jitter_K)   # [T,m1,m1]
    Kuu_inv  = torch.cholesky_inverse(Lm)               # [T,m1,m1]

    # 5) Predictive moments
    A     = Kuu_inv @ m_u.unsqueeze(-1)          # [T,m1,1]
    E_f   = (K_fu @ A).squeeze(-1)                # [T,n]

    C1    = K_fu @ Kuu_inv                        # [T,n,m1]
    diag1 = (C1 * K_fu).sum(-1)                   # [T,n]

    tmp   = Kuu_inv @ (S_u @ Kuu_inv)             # [T,m1,m1]
    diag2 = ( (K_fu @ tmp) * K_fu ).sum(-1)       # [T,n]

    var_noise = (sigma_n**2).squeeze(-1)          # [T,1]
    Var_f     = var_noise - diag1 + diag2        # [T,n]
    sq_err    = (y - E_f)**2 + Var_f              # [T,n]

    # 6) ELBO term I+II
    ones   = torch.ones(T, n, 1, device=device)
    z_feat = torch.cat([ones, Z], dim=2)          # [T,n,Q+1]
    logit  = (z_feat @ omega.unsqueeze(-1)).squeeze(-1)  # [T,n]

    logp_inlier  = -0.5 * torch.log(2*math.pi*var_noise) \
                   -0.5 * sq_err/var_noise \
                   + F.logsigmoid(logit)                     # [T,n]
    logp_outlier = U_logit + F.logsigmoid(-logit)           # [T,n]

    point_elbo = torch.logsumexp(torch.stack([logp_inlier, logp_outlier], dim=0), dim=0)  # [T,n]
    I_II       = point_elbo.sum(dim=1)                                                 # [T]

    # 7) KL(q(u)||p(u))
    trace_term = torch.einsum('tij,tji->t', Kuu_inv, S_u)
    quad_term  = (m_u.unsqueeze(1) @ (Kuu_inv @ m_u.unsqueeze(-1)))\
                  .squeeze(-1).squeeze(-1)                                             # [T]
    logdet_q   = torch.logdet(S_u + jitter_S)                                         # [T]
    logdet_p   = 2.0 * torch.log(torch.diagonal(Lm, dim1=1, dim2=2)).sum(-1)          # [T]
    kl_u       = 0.5 * (trace_term + quad_term - m1 + logdet_p - logdet_q)            # [T]

    elbo_jgp = I_II.sum() - kl_u.sum()

    # 8) subtract network KL
    if prior_phi:
        elbo = elbo_jgp/T - g_phi.kl()
    else:
        elbo = elbo_jgp/T

    return elbo


# -------------------------
# 5) Training with BNN
# -------------------------
def train_bnn(regions, u_params, g_phi, hyperparams,
              lr=1e-3, num_steps=1000, log_interval=100):
    params = list(g_phi.parameters())
    for u in u_params:
        params += [u['U_logit'], u['omega'],
                   u['m_u'], u['L_u'],
                   u['sigma_n'], u['sigma_k']]
    opt = optim.Adam(params, lr=lr)

    for step in range(1, num_steps+1):
        opt.zero_grad()
        elbo = compute_ELBO_bnn(regions, u_params, g_phi, hyperparams)
        (-elbo).backward()
        opt.step()
        # clamp for positivity
        with torch.no_grad():
            for u in u_params:
                u['sigma_n'].clamp_(min=1e-1)
                u['sigma_k'].clamp_(min=1e-1)
        if step % log_interval == 0 or step == 1:
            print(f"[Train] Step {step}/{num_steps}, ELBO={elbo.item():.3f}")

    return u_params, g_phi, hyperparams


# -------------------------
# 6) Prediction with MC
# -------------------------
def predict_bnn(regions, u_params, g_phi, hyperparams, M=10):
    device = next(g_phi.parameters()).device
    T = len(regions)
    mu_acc  = torch.zeros((M, T), device=device)
    var_acc = torch.zeros((M, T), device=device)

    for m in range(M):
        mu_s, var_s = predict_nn(regions, u_params, g_phi, hyperparams)
        mu_acc[m]  = mu_s
        var_acc[m] = var_s

    mu_pred  = mu_acc.mean(0)
    var_pred = var_acc.mean(0)
    return mu_pred, var_pred


# -------------------------
# 7) Main test script
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Test Bayesian NN-JGP')
    p.add_argument('--folder_name', required=True,
                   help='Folder containing dataset.pkl')
    p.add_argument('--Q',    type=int, default=2,
                   help='Latent dimension')
    p.add_argument('--m1',   type=int, default=5,
                   help='Inducing points per region')
    p.add_argument('--n',    type=int, default=100,
                   help='Neighbors per region')
    p.add_argument('--steps',type=int, default=200,
                   help='Training steps')
    p.add_argument('--lr',   type=float, default=1e-2,
                   help='Learning rate')
    p.add_argument('--M',    type=int, default=5,
                   help='MC samples for prediction')
    return p.parse_args()

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load & build regions
    dataset = load_dataset(args.folder_name)
    Xtrain, Ytrain = dataset["X_train"].to(device), dataset["Y_train"].to(device)
    Xtest,  Ytest  = dataset["X_test"].to(device),  dataset["Y_test"].to(device)

    neigh = find_neighborhoods(
        Xtest.cpu(), Xtrain.cpu(), Ytrain.cpu(), M=args.n
    )
    regions = []
    for i in range(len(neigh)):
        nb = neigh[i]
        regions.append({
            'X':      nb['X_neighbors'].to(device),
            'y':      nb['y_neighbors'].to(device),
            'X_test': Xtest[i],
            'C':      torch.randn(args.m1, args.Q, device=device)
        })

    # instantiate BNN feature net
    start_time = time.time()
    g_phi = BayesianFeatureNet(
        input_dim=Xtrain.shape[1],
        latent_dim=args.Q,
        hidden_dims=(64,),
        prior_sigma=1.0
    ).to(device)

    # initialize u_params
    u_params = []
    for _ in regions:
        u_params.append({
            'm_u':     torch.zeros(args.m1, device=device, requires_grad=True),
            'L_u':     torch.eye(args.m1, device=device, requires_grad=True),
            'sigma_n': torch.tensor(0.5, device=device, requires_grad=True),
            'sigma_k': torch.tensor(1.0, device=device, requires_grad=True),
            'omega':   torch.randn(args.Q+1, device=device, requires_grad=True),
            'U_logit': torch.tensor(0.0, device=device, requires_grad=True),
        })

    hyperparams = {}

    # initial ELBO
    # elbo0 = compute_ELBO_bnn(regions, u_params, g_phi, hyperparams)
    # print(f"Initial ELBO: {elbo0.item():.4f}")

    # train
    train_bnn(regions, u_params, g_phi, hyperparams,
              lr=args.lr, num_steps=args.steps, log_interval=50)

    # predict
    mu_pred, var_pred = predict_bnn(regions, u_params, g_phi, hyperparams, M=args.M)
    sigmas = torch.sqrt(var_pred)
    rmse, mean_crps = compute_metrics(mu_pred, sigmas, Ytest)
    train_time = time.time() - start_time
    print(f"Final [RMSE, CRPS]: {[rmse, mean_crps]}")
    return [rmse, mean_crps, train_time]

if __name__ == "__main__":
    main()
