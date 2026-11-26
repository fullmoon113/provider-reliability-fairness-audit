from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = ROOT / 'output'
"""Boundary-aware inference for the true between-provider variance tau^2 of provider mean patient age (work order M7).
Model: A_i = mu + a_i + e_i, a_i ~ N(0, tau^2), e_i ~ N(0, sigma_w^2 / n_i) with sigma_w known (12.7275 years, from the individual ages).
Because the within-provider variance is known, this is the random-effects model for provider means; REML and likelihood
inference are computed directly.  Reported: Cochran's Q test of tau^2 = 0, REML estimate, profile-likelihood interval truncated at
zero, boundary likelihood-ratio test (chi-square mixture 0:1) and the implied reliability.
Usage: python src/variance_component_inference.py"""
import sys, io, numpy as np, pandas as pd
from scipy import stats, optimize
sys.path.insert(0, str(HERE))
from audit_common import load
buf = io.StringIO()
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); buf.write(s + '\n')
pf = load(); A = pf.AvgPatientAge.values; n = pf.TotalUniqueBene.values.astype(float); SW = 12.7275; v = SW**2 / n
def reml_ll(t2):
    w = 1 / (t2 + v); mu = (w * A).sum() / w.sum()
    return -0.5 * (np.log(t2 + v).sum() + np.log(w.sum()) * -1 * -1 + (w * (A - mu) ** 2).sum()) + 0.5 * 0  # up to constants
def ll(t2, reml=True):
    w = 1 / (t2 + v); mu = (w * A).sum() / w.sum()
    out = -0.5 * (np.log(t2 + v).sum() + (w * (A - mu) ** 2).sum())
    if reml: out -= 0.5 * np.log(w.sum())
    return out
grid = np.linspace(0, 6, 6001)
for reml in (True, False):
    L = np.array([ll(t, reml) for t in grid]); i = L.argmax(); t_hat = grid[i]
    lr0 = 2 * (L[i] - L[0]); p_b = 0.5 * stats.chi2.sf(lr0, 1) if lr0 > 0 else 1.0
    ok = grid[2 * (L[i] - L) <= stats.chi2.ppf(0.95, 1)]
    ok_b = grid[2 * (L[i] - L) <= stats.chi2.ppf(0.90, 1)]   # boundary-aware 95% one-sided style: mixture-quantile
    P(f"{'REML' if reml else 'ML  '}: tau^2 = {t_hat:.3f}  (tau = {np.sqrt(t_hat):.3f})  LR vs 0 = {lr0:.3f}  boundary LRT p = {p_b:.3f}  "
      f"profile 95% interval [{ok.min():.3f}, {ok.max():.3f}]  (boundary-adjusted 95%: [{ok_b.min():.3f}, {ok_b.max():.3f}])")
# Cochran's Q and DerSimonian-Laird, treating sigma_w as known
w = 1 / v; mu_w = (w * A).sum() / w.sum(); Q = (w * (A - mu_w) ** 2).sum(); k = len(A)
P(f"Cochran Q = {Q:.1f} on {k-1} df, p = {stats.chi2.sf(Q, k-1):.3f};  Q/(k-1) = {Q/(k-1):.4f}  (observed/expected variance ratio)")
dl = max(0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())); P(f"DerSimonian-Laird tau^2 = {dl:.3f}")
# reliability at the upper limit of the interval and share of providers below 0.7
L = np.array([ll(t, True) for t in grid]); ok = grid[2 * (L.max() - L) <= stats.chi2.ppf(0.95, 1)]
for t2 in (grid[L.argmax()], ok.max()):
    R = t2 / (t2 + v); P(f"tau^2 = {t2:.3f}: median R = {np.median(R):.3f}, share R<0.7 = {(R<0.7).mean()*100:.1f}%, providers with R>=0.7 = {(R>=0.7).sum()}")
# robustness to sigma_w: +-2%
for f in (0.98, 1.02):
    v2 = (SW * f) ** 2 / n; w2 = 1 / v2; mu2 = (w2 * A).sum() / w2.sum(); Q2 = (w2 * (A - mu2) ** 2).sum()
    P(f"sigma_w x {f}: Q = {Q2:.1f}, p = {stats.chi2.sf(Q2, k-1):.3f}")
(OUT / 'variance_component_inference_output.txt').write_text(buf.getvalue())
