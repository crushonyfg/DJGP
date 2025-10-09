import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import warnings

from JumpGP_LD import JumpGP_LD
from JumpGP_QD import JumpGP_QD

from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

def find_neighborhoods(X_test, X_train, Y_train, M):
    """Find M nearest neighbors for each test point"""
    T = X_test.shape[0]
    dists = np.sqrt(np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=2))
    indices = np.argsort(dists, axis=1)[:, :M]
    
    neighborhoods = []
    for t in range(T):
        idx = indices[t]
        neighborhoods.append({
            "X_neighbors": X_train[idx],
            "y_neighbors": Y_train[idx].reshape(-1,1),
            "indices": idx
        })
    return neighborhoods

def compute_metrics(predictions, sigmas, Y_test):
    """Compute RMSE and mean CRPS for Gaussian predictive distributions."""
    # compute RMSE
    rmse = np.sqrt(np.mean((predictions - Y_test)**2))
    
    # compute CRPS 
    # z = (y - μ) / σ
    z = (Y_test - predictions) / sigmas
    # standard normal CDF and PDF
    cdf = 0.5 * (1 + scipy.special.erf(z / np.sqrt(2)))
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    # CRPS closed-form formula
    crps = sigmas * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    
    # mean CRPS
    mean_crps = np.mean(crps)
    
    return rmse, mean_crps

class JumpGP:
    def __init__(self, x, y, xt, L=1, M=20, mode='CEM', bVerbose=False):
        self.x = x
        self.y = y
        self.xt = xt
        self.L = L
        self.M = M
        self.mode = mode
        self.bVerbose = bVerbose

        self.neighborhoods = find_neighborhoods(xt, x, y, M)
        self.jump_results = {}

    def fit(self):
        mu, sig2 = [], []
        models = []
        for t in tqdm(range(self.xt.shape[0])):
            x = self.neighborhoods[t]["X_neighbors"]
            y = self.neighborhoods[t]["y_neighbors"]
            xt = self.xt[t].reshape(1,-1)
            if self.L == 1:
                mu_t, sig2_t, model, h = JumpGP_LD(x, y, xt, self.mode, self.bVerbose)
            else:
                mu_t, sig2_t, model, h = JumpGP_QD(x, y, xt, self.mode, self.bVerbose)
            mu.append(mu_t)
            sig2.append(sig2_t)
            models.append(model)
        mu = np.array(mu)
        sig2 = np.array(sig2)
        self.jump_results = {"mu":mu, "sig2":sig2, "models":models}

        return self.jump_results
    
    def metrics(self, yt):
        mu = self.jump_results["mu"]
        sig2 = self.jump_results["sig2"]
        models = self.jump_results["models"]

        # ensure inputs are 1-dimensional arrays
        mu = np.asarray(mu).ravel()
        sig2 = np.asarray(sig2).ravel()
        yt = np.asarray(yt).ravel()

        # check if the dimensions match
        if not (mu.shape == sig2.shape == yt.shape):
            raise ValueError(f"Shape mismatch: mu {mu.shape}, sig2 {sig2.shape}, yt {yt.shape} should all be the same")
        
        # check if the arrays are 1-dimensional
        if len(mu.shape) != 1:
            raise ValueError(f"Arrays should be 1-dimensional, got shape {mu.shape}")

        sigmas = np.sqrt(sig2)
        rmse, mean_crps = compute_metrics(mu, sigmas, yt)

        return rmse, mean_crps
    
if __name__ == "__main__":
    x_train = np.random.rand(100, 2)
    y_train = np.random.rand(100)
    x_test = np.random.rand(20, 2)
    y_test = np.random.rand(20)
    JGP = JumpGP(x_train, y_train, x_test, L=2, M=20, mode='VEM', bVerbose=False)
    JGP.fit()
    rmse, mean_crps = JGP.metrics(y_test)
    print(f"RMSE: {rmse}, Mean CRPS: {mean_crps}")
    print("Successfully run the example!")
