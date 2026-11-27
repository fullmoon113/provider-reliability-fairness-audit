"""Proposition 1 prediction for a proportion-valued attribute: top quartile of the non-White share (Table 4, row 3).

A share is bounded, so the normal approximation of the age analysis is replaced by an exact mixture: the true share
p_i ~ Beta(a, b) with the mean of the provider-level shares and the REML variance tau^2 of Stage 0 (sigma_w^2 = pbar(1-pbar)); the measured share is Binomial(n_i, p_i) / n_i.  The partition is
G = 1{measured share >= its 75th percentile}, as in the audit.  pi(n) = Pr(G = 1 | n) is computed from the beta-binomial
and inserted in Proposition 1(i) with the fitted detection slope and the fraud-labeled providers' panel sizes.
Usage: python src/prop1_nonwhite.py   (writes output/prop1_nonwhite_output.txt)
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from audit_common import *
from scipy.stats import betabinom
from simparams import B_DET
pf = load(); n = pf.TotalUniqueBene.values.astype(int); m = pf.MinorityRatio.values
fraud = pf.Fraud.values == 1
z = (np.log1p(n) - np.log1p(n).mean()) / np.log1p(n).std(ddof=1)
h = 1 / (1 + np.exp(-(B_DET[0] + B_DET[1] * z)))
from stage0 import _reml
pbar = m.mean(); var_within = pbar * (1 - pbar)
tau2 = _reml(m, var_within / n)[0]
rel_med = np.median(tau2 / (tau2 + var_within / n))
k = pbar * (1 - pbar) / tau2 - 1; a, b = pbar * k, (1 - pbar) * k     # Beta(a, b) with variance tau2
q = np.quantile(m, .75)
need = np.ceil(q * n - 1e-9).astype(int)
pi = np.array([betabinom.sf(c - 1, ni, a, b) for ni, c in zip(n, need)])
w = fraud.astype(float); w /= w.sum()
pb = (w * pi).sum(); p1 = (w * h * pi).sum() / pb; p0 = (w * h * (1 - pi)).sum() / (1 - pb)
OR = (p1 / (1 - p1)) / (p0 / (1 - p0))
out = [f"non-White share: mean {pbar:.3f}, SD of measured share {m.std(ddof=1):.3f}, tau {np.sqrt(tau2):.3f}, "
       f"median reliability {rel_med:.2f}",
       f"top-quartile boundary (measured share) {q:.3f}; Pr(G=1) among labeled providers {pb:.3f}",
       f"Proposition 1 prediction among the 506 labeled providers: detection {p1:.3f} (group) vs {p0:.3f} (rest), "
       f"gap {p1 - p0:+.4f}, OR {OR:.3f}"]
print('\n'.join(out)); open(OUT / 'prop1_nonwhite_output.txt', 'w').write('\n'.join(out) + '\n')
