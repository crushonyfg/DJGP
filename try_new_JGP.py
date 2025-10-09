# =========================
# 1) 模型组件
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BoundaryNet(nn.Module):
    """s_phi(x): 全局共享的边界网络（简单 MLP，可替换为 TCN/LSTM/Transformer）"""
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)   # (B,)

class RBFKernel(nn.Module):
    """可学习 RBF 核: k(x,x') = σ_f^2 * exp(-||x-x'||^2 / (2ℓ^2))"""
    def __init__(self, in_dim, log_lengthscale=0.0, log_sigma_f=0.0):
        super().__init__()
        self.log_ell = nn.Parameter(torch.tensor(float(log_lengthscale)))
        self.log_sf  = nn.Parameter(torch.tensor(float(log_sigma_f)))

    def forward(self, Xa, Xb):
        # Xa: (m,d), Xb: (n,d)
        ell2 = torch.exp(self.log_ell)*torch.exp(self.log_ell) + 1e-12
        sf2  = torch.exp(self.log_sf)*torch.exp(self.log_sf) + 1e-12
        # ||a-b||^2 = |a|^2 + |b|^2 - 2 a b^T
        a2 = (Xa**2).sum(-1, keepdim=True)   # (m,1)
        b2 = (Xb**2).sum(-1, keepdim=True).T # (1,n)
        dist2 = a2 + b2 - 2.0 * Xa @ Xb.T
        K = sf2 * torch.exp(-0.5 * dist2 / ell2)
        return K

def same_side_weight(sx, S, lam=5.0):
    """
    sx: s_phi(x*): (1,)   —— 单个查询
    S : s_phi(X):  (N,)   —— 全体样本
    return: w: (N,)
    """
    return torch.sigmoid(lam * (sx * S))

# =========================
# 2) 训练器（局部 GP NLL）
# =========================
class LocalGPTrainer:
    def __init__(self, X, Y, hidden=128, lam=5.0, topM=64, noise=1e-2, lr=1e-3):
        """
        X: (N,d), Y: (N,)
        """
        self.X = torch.as_tensor(X, dtype=torch.float32, device=device)
        self.Y = torch.as_tensor(Y, dtype=torch.float32, device=device)
        self.N, self.d = self.X.shape

        self.boundary = BoundaryNet(self.d, hidden).to(device)
        self.kernel   = RBFKernel(self.d).to(device)

        self.log_noise = nn.Parameter(torch.log(torch.tensor(noise)))
        self.lam = lam
        self.topM = topM

        self.opt = torch.optim.Adam(
            list(self.boundary.parameters()) +
            list(self.kernel.parameters()) +
            [self.log_noise], lr=lr
        )

    @torch.no_grad()
    def _topM_index(self, w, M):
        # 返回权重最大的 M 个索引
        M = min(M, w.numel())
        vals, idx = torch.topk(w, k=M, largest=True, sorted=False)
        return idx, vals

    def _gp_predict_one(self, x_star, S_all, return_var=False):
        """
        用“局部同侧加权”对单个 x* 做 GP 后验预测。
        x_star: (d,)
        S_all : s_phi(X): (N,)
        return: mu[, var]
        """
        # 1) s(x*), 权重 w(x*, X)
        sx = self.boundary(x_star.unsqueeze(0)).squeeze(0)     # scalar
        w  = same_side_weight(sx, S_all, lam=self.lam)         # (N,)

        # 2) 选 top-M 邻居
        idx, w_top = self._topM_index(w, self.topM)            # (M,)
        Xn = self.X[idx]                                       # (M,d)
        yn = self.Y[idx]                                       # (M,)

        # 3) 加权核
        K = self.kernel(Xn, Xn)                                # (M,M)
        wx = w_top                                             # (M,)
        W = torch.diag(wx)                                     # (M,M)
        Km = W @ K @ W                                         # (M,M)

        k_star = self.kernel(Xn, x_star.unsqueeze(0)).squeeze(1)  # (M,)
        k_star = wx * k_star                                   # (M,)

        # 4) 后验
        sn2 = torch.exp(self.log_noise)*torch.exp(self.log_noise) # σ_n^2
        A = Km + (sn2 + 1e-6)*torch.eye(Km.shape[0], device=device)

        L = torch.linalg.cholesky(A)                           # A = L L^T
        alpha = torch.cholesky_solve(yn.unsqueeze(1), L).squeeze(1)  # A^{-1} y

        mu = k_star @ alpha

        if return_var:
            v = torch.cholesky_solve(k_star.unsqueeze(1), L).squeeze(1)  # A^{-1} k*
            kxx = self.kernel(x_star.unsqueeze(0), x_star.unsqueeze(0)).squeeze()
            var = (kxx - (k_star @ v)).clamp_min(1e-10)
            return mu, var
        else:
            return mu

    def step(self, batch_idx, lam_bdry=0.0):
        """
        对一批查询点（样本自身）做 NLL 训练
        batch_idx: list/LongTensor of indices
        """
        self.opt.zero_grad()
        xB = self.X[batch_idx]              # (B,d)
        yB = self.Y[batch_idx]              # (B,)
        S_all = self.boundary(self.X)       # (N,)

        mus, vars_ = [], []
        for i in range(xB.shape[0]):
            mu_i, var_i = self._gp_predict_one(xB[i], S_all, return_var=True)
            mus.append(mu_i); vars_.append(var_i)
        mu = torch.stack(mus)               # (B,)
        var = torch.stack(vars_) + 1e-8     # (B,)

        # 负对数似然（高斯）
        nll = 0.5*torch.log(var) + 0.5*((yB - mu)**2)/var
        loss_like = nll.mean()

        # 简单边界正则（鼓励 |s| 稍大，远离不确定带）
        if lam_bdry > 0:
            reg = (1.0 / (1.0 + S_all.pow(2))).mean()  # 越靠近0惩罚越大
        else:
            reg = 0.0

        loss = loss_like + lam_bdry*reg
        loss.backward()
        self.opt.step()
        return float(loss.item()), float(loss_like.item()), float(reg if isinstance(reg,float) else reg)

    def train_epochs(self, epochs=50, batch_size=128, lam_bdry=0.0, lam_anneal=(1.0, 8.0)):
        """
        lam_anneal: (lam_start, lam_end) 训练中线性退火到更“硬”的边界
        """
        N = self.N
        lam0, lam1 = lam_anneal
        for ep in range(1, epochs+1):
            # 线性退火 λ
            self.lam = lam0 + (lam1 - lam0) * (ep-1) / max(1, epochs-1)

            perm = torch.randperm(N, device=device)
            tot_loss = 0.0
            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]
                loss, nll, reg = self.step(idx, lam_bdry=lam_bdry)
                tot_loss += loss*(len(idx)/N)
            print(f"[ep {ep:03d}] loss={tot_loss:.4f}  lambda={self.lam:.2f}  noise={torch.exp(self.log_noise).item():.4f}")

    # -----------------------
    # 推理：给 x, X, Y → ŷ
    # -----------------------
    @torch.no_grad()
    def predict(self, x_query, return_var=False):
        xq = torch.as_tensor(x_query, dtype=torch.float32, device=device)
        if xq.ndim == 1:
            S_all = self.boundary(self.X)
            out = self._gp_predict_one(xq, S_all, return_var=return_var)
            return tuple(o.detach().cpu().numpy() for o in out) if return_var else out.detach().cpu().numpy()
        else:
            S_all = self.boundary(self.X)
            mus, vars_ = [], []
            for i in range(xq.shape[0]):
                res = self._gp_predict_one(xq[i], S_all, return_var=return_var)
                if return_var: 
                    mu_i, var_i = res; mus.append(mu_i); vars_.append(var_i)
                else:
                    mus.append(res)
            mu = torch.stack(mus).detach().cpu().numpy()
            if return_var:
                var = torch.stack(vars_).detach().cpu().numpy()
                return mu, var
            return mu


# 生成一点玩具数据（两侧函数不同，模拟“跳变”）
import numpy as np
rng = np.random.default_rng(0)
N = 1200
X = rng.uniform(-3, 3, size=(N,1)).astype(np.float32)
Y = (np.where(X[:,0] < 0, np.sin(2*X[:,0]), 2+0.4*X[:,0]) 
     + 0.15*rng.normal(size=N)).astype(np.float32)

trainer = LocalGPTrainer(X, Y, hidden=64, lam=1.0, topM=64, noise=1e-2, lr=3e-3)
trainer.train_epochs(epochs=60, batch_size=256, lam_bdry=1e-3, lam_anneal=(1.0, 10.0))

# 推理：给 (x, X, Y) -> ŷ
x_test = np.linspace(-3,3,201, dtype=np.float32)[:,None]
y_pred, y_var = trainer.predict(x_test, return_var=True)

import numpy as np

# ---- 定义ground-truth函数（与数据生成一致）----
def f_true(x):
    x = np.asarray(x).reshape(-1)
    y = np.where(x < 0, np.sin(2*x), 2 + 0.4*x)
    return y

# ---- 从你的trainer里得到预测均值/方差 ----
# x_test 已有；若没有就解开下面两行
# x_test = np.linspace(-3, 3, 201, dtype=np.float32)[:, None]
# y_pred, y_var = trainer.predict(x_test, return_var=True)

y_true = f_true(x_test[:, 0])

# ---- RMSE ----
rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

# ---- CRPS（Gaussian的闭式公式）----
# CRPS(μ,σ;y) = σ * [ z*(2*Φ(z)-1) + 2*φ(z) - 1/√π ],  z=(y-μ)/σ
def gaussian_crps(mu, sigma, y):
    # 防止数值问题
    sigma = np.maximum(sigma, 1e-12)
    z = (y - mu) / sigma
    # Φ、φ实现
    Phi = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z ** 2)
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return crps

sigma_pred = np.sqrt(np.maximum(y_var, 0.0))
crps_vals = gaussian_crps(y_pred, sigma_pred, y_true)
crps_mean = float(np.mean(crps_vals))

print(f"RMSE = {rmse:.4f}")
print(f"Mean CRPS = {crps_mean:.4f}")

