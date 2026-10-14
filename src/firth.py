from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

import numpy as np
from scipy import stats

def firth_penalized_loglik(beta, X, y):
    z = X @ beta
    p = 1/(1+np.exp(-z))
    W = p*(1-p)
    XtWX = X.T @ (X * W[:,None])
    sign, logdet = np.linalg.slogdet(XtWX)
    ll = np.sum(y*np.log(p+1e-12) + (1-y)*np.log(1-p+1e-12))
    return ll + 0.5*logdet

def fit_firth(X, y, maxiter=200, tol=1e-8, ridge=1e-10):
    n, k = X.shape
    beta = np.zeros(k)
    for it in range(maxiter):
        z = np.clip(X @ beta, -30, 30)
        p = 1/(1+np.exp(-z))
        W = p*(1-p)
        W = np.clip(W, 1e-10, None)
        XtWX = X.T @ (X * W[:,None]) + ridge*np.eye(k)
        XtWX_inv = np.linalg.pinv(XtWX)
        Wsqrt = np.sqrt(W)
        Xw = X * Wsqrt[:,None]
        # diag(H) without forming the full n x n hat matrix
        h = np.einsum('ij,jk,ik->i', Xw, XtWX_inv, Xw)
        U = X.T @ (y - p + h*(0.5 - p))
        step = XtWX_inv @ U
        # damp large steps for stability
        stepnorm = np.max(np.abs(step))
        if stepnorm > 5:
            step = step * (5/stepnorm)
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta

def firth_plr_test(X_full, y, drop_idx):
    """PLR test for dropping column drop_idx from X_full."""
    beta_full = fit_firth(X_full, y)
    ll_full = firth_penalized_loglik(beta_full, X_full, y)
    cols = [i for i in range(X_full.shape[1]) if i != drop_idx]
    X_red = X_full[:, cols]
    beta_red = fit_firth(X_red, y)
    ll_red = firth_penalized_loglik(beta_red, X_red, y)
    lr = 2*(ll_full - ll_red)
    lr = max(lr, 0)
    p = stats.chi2.sf(lr, df=1)
    beta_target = beta_full[drop_idx]
    return np.exp(beta_target), p
