from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = ROOT / 'output'
"""Required reliability R*(eps) for an admissible misclassification rate eps, by partition type (Appendix B).
m(R) = share of units placed in the wrong group when a normally distributed attribute with reliability R is split at fixed quantiles
of the measured instead of the true value.  Also reports the audited population's expected misclassification M = mean_i m(R_i)
for the primary dataset."""
import sys, io, numpy as np
sys.path.insert(0, str(HERE))
from audit_common import load
buf = io.StringIO()
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); buf.write(s + '\n')
rng = np.random.default_rng(2026); N = 2_000_000
t = rng.standard_normal(N); e = rng.standard_normal(N)
def grp(x, kind):
    if kind == 'outer':
        q = np.quantile(x, [.25, .75]); return (x < q[0]) | (x > q[1])
    if kind == 'top':
        return x > np.quantile(x, .75)
    return x > np.quantile(x, .5)
kinds = ['outer', 'top', 'median']; Rg = np.round(np.arange(0, 1.0001, 0.01), 2)
gt = {k: grp(t, k) for k in kinds}
m = {k: [] for k in kinds}
for R in Rg:
    o = np.sqrt(R) * t + np.sqrt(1 - R) * e
    for k in kinds: m[k].append((gt[k] != grp(o, k)).mean())
m = {k: np.array(v) for k, v in m.items()}
P('R      ' + '  '.join(f'{k:>7s}' for k in kinds))
for R in (0.037, 0.111, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99):
    P(f'{R:<6} ' + '  '.join(f'{np.interp(R, Rg, m[k]) * 100:6.1f}%' for k in kinds))
P('\nR*(eps): smallest R with m(R) <= eps')
for eps in (0.30, 0.20, 0.15, 0.10, 0.05):
    P(f'eps {eps*100:4.0f}%: ' + '  '.join(f'{k}: {Rg[np.argmax(m[k] <= eps)]:.2f}' for k in kinds))
pf = load(); n = pf.TotalUniqueBene.values.astype(float); fm = pf.Fraud.values == 1
v = 12.7275 ** 2 / n
for t2 in (0.812,):
    R = t2 / (t2 + v)
    P(f'\nprimary dataset (tau^2 = {t2}): expected misclassification, outer split: all providers {np.interp(R, Rg, m["outer"]).mean()*100:.1f}%, '
      f'fraud-labelled {np.interp(R[fm], Rg, m["outer"]).mean()*100:.1f}%;  top-quartile split {np.interp(R, Rg, m["top"]).mean()*100:.1f}%; '
      f'share with R >= R*(10%) = {(R >= Rg[np.argmax(m["outer"] <= .10)]).mean()*100:.2f}%')
(OUT / 'rstar_table_output.txt').write_text(buf.getvalue())
