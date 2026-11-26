"""
Proposition 1 (partition-induced disparity under a null): analytic prediction and falsification tests.

Model of the null simulation (src/null_sim.py): A_i = mu + a_i + e_i, a_i ~ N(0, tau^2), e_i | n_i ~ N(0, sw^2/n_i);
detection D depends on panel size only, h(n) = logistic(g0 + g1 z(n)); fraud incidence logistic(f0 + f1 z(n)).
Among fraud-labeled providers the audit compares detection across G = 1{A in S}.  Prop. 1(i):
    Pr(D=1|G=1) - Pr(D=1|G=0) = Cov{h(N), pi(N)} / [pibar (1 - pibar)],   pi(n) = Pr(G=1 | n),
with expectations over the panel-size distribution of FRAUD-LABELLED providers.
Outputs: analytic gap and odds ratio for the outer-quartile, top-quartile and median partitions, for the empirical
fraud-labeled panel sizes (506 providers) and for the model-implied fraud weights over all 5,410 providers, at several tau;
and the observed audit for the median partition and the top-quartile partition.
Usage: python src/prop1_analytic.py
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from audit_common import *
from scipy.stats import norm
from scipy.optimize import brentq

pf = load(); n = pf.TotalUniqueBene.values.astype(float); logn = np.log1p(n)
z = (logn - logn.mean()) / logn.std(ddof=1)
SW = 12.7275; MU = pf.AvgPatientAge.mean(); import sys; sys.path.insert(0, str(HERE)); from simparams import B_DET, B_FRAUD
G0, G1 = B_DET; F0, F1 = B_FRAUD
logistic = lambda t: 1 / (1 + np.exp(-t))
h = logistic(G0 + G1 * z); pfr = logistic(F0 + F1 * z)
fraud = pf.Fraud.values == 1
lines = []
def emit(s=''): print(s); lines.append(s)

def cuts(tau):
    s = np.sqrt(tau ** 2 + SW ** 2 / n)
    f = lambda q, p: np.mean(norm.cdf((q - MU) / s)) - p
    return [brentq(f, MU - 40, MU + 40, args=(p,)) for p in (.25, .5, .75)], s

def prop1(tau, part, w):
    (q25, q50, q75), s = cuts(tau)
    if part == 'outer':  pi = norm.cdf((q25 - MU) / s) + 1 - norm.cdf((q75 - MU) / s)
    elif part == 'top':  pi = 1 - norm.cdf((q75 - MU) / s)
    else:                pi = 1 - norm.cdf((q50 - MU) / s)
    w = w / w.sum(); pb = (w * pi).sum()
    p1 = (w * h * pi).sum() / pb; p0 = (w * h * (1 - pi)).sum() / (1 - pb)
    cov = (w * h * pi).sum() - (w * h).sum() * pb
    gap = cov / (pb * (1 - pb))
    orr = (p1 / (1 - p1)) / (p0 / (1 - p0))
    return gap, orr, p1 - p0

emit("Analytic prediction from Proposition 1(i) (null: no age effect)")
emit(f"{'tau':>5} {'partition':>10} {'weights':>22} {'gap':>8} {'OR':>7}")
res = {}
for tau in (0.901, 1.65, 3.47, 4.85):
    for part in ('outer', 'top', 'median'):
        for wn, w in (('fraud-labeled (506)', fraud.astype(float)), ('model-implied', pfr)):
            gap, orr, chk = prop1(tau, part, w)
            assert abs(gap - chk) < 1e-9
            emit(f"{tau:5.3f} {part:>10} {wn:>22} {gap:8.4f} {orr:7.3f}")
            res[(tau, part, wn)] = (gap, orr)

# observed audits (detection among fraud-labeled providers, out-of-fold logistic regression)
emit("\nObserved audits among the 506 fraud-labeled providers (unadjusted OR [95% CI] p | adjusted OR [95% CI] p)")
A = pf.AvgPatientAge.values
parts = {'outer quartiles (Q1+Q4)': outer(A),
         'top quartile (>= Q3)': (A >= np.quantile(A, .75)).astype(int),
         'median split (above median)': (A > np.median(A)).astype(int)}
for strat in ('group_smote', 'class_weight'):
    emit(f"-- training strategy: {strat}" + (" (SMOTE stratified on the extreme-age partition)" if strat == 'group_smote' else ""))
    oof = oof_lr(pf, outer(A), strategy=strat)
    det = oof[fraud]; ln = logn[fraud]
    for name, g in parts.items():
        u, a = audit(det, g[fraud], ln)
        emit(f"  {name:30s} {u[0]:.3f} [{u[2]:.3f},{u[3]:.3f}] p={u[1]:.4f} | {a[0]:.3f} [{a[2]:.3f},{a[3]:.3f}] p={a[1]:.4f}")
open(OUT / 'prop1_analytic_output.txt', 'w').write('\n'.join(lines) + '\n')
