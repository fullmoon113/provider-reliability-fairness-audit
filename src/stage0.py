"""Stage 0 reliability check for an aggregate sensitive attribute (importable; paper Section III-E).

    from stage0 import stage0_reliability
    res = stage0_reliability(df, unit='n', attr='mean_age', partition='outer', sigma_w=12.7275, eps=0.10)
    print(res['summary'])

df         one row per unit (e.g. provider)
unit       column with the number of individuals aggregated in each unit (n_i)
attr       column with the unit's measured mean (or proportion) of the attribute
partition  'outer' (outer quartiles vs middle half), 'top' (top quartile) or 'median'
sigma_w    within-unit SD of the individual attribute; for a proportion p use sqrt(p(1-p)) (default: computed from the mean)
eps        tolerated expected misclassification of the audited population

Returns REML tau^2, Cochran Q with p-value, per-unit reliability R_i, expected misclassification M, the share of units with
R_i >= R*(eps) and the decision (partition on the observed attribute / report both estimates, adjust and shrink)."""
import numpy as np, pandas as pd
from scipy import stats, optimize

def _grp(x, kind):
    if kind == 'outer':
        q = np.quantile(x, [.25, .75]); return (x < q[0]) | (x > q[1])
    if kind == 'top':
        return x > np.quantile(x, .75)
    return x > np.quantile(x, .5)

def misclassification_curve(kind, N=400_000, seed=2026):
    rng = np.random.default_rng(seed); t = rng.standard_normal(N); e = rng.standard_normal(N)
    Rg = np.round(np.arange(0, 1.0001, 0.01), 2); gt = _grp(t, kind)
    m = np.array([(gt != _grp(np.sqrt(R) * t + np.sqrt(1 - R) * e, kind)).mean() for R in Rg])
    return Rg, m

def stage0_reliability(df, unit, attr, partition='outer', sigma_w=None, eps=0.10, weights=None):
    n = df[unit].to_numpy(float); a = df[attr].to_numpy(float)
    if sigma_w is None:
        p = np.average(a, weights=n); sigma_w = np.sqrt(p * (1 - p))
    v = sigma_w ** 2 / n
    def ll(t2):
        w = 1 / (t2 + v); mu = (w * a).sum() / w.sum()
        return -0.5 * (np.log(t2 + v).sum() + (w * (a - mu) ** 2).sum() + np.log(w.sum()))
    hi = max(np.var(a), 1e-9) * 2
    t2 = optimize.minimize_scalar(lambda x: -ll(x), bounds=(0, hi), method='bounded').x
    t2 = 0.0 if ll(0) >= ll(t2) else t2
    w = 1 / v; mu = (w * a).sum() / w.sum(); Q = (w * (a - mu) ** 2).sum(); k = len(a)
    R = t2 / (t2 + v) if t2 > 0 else np.zeros_like(v)
    Rg, m = misclassification_curve(partition)
    mi = np.interp(R, Rg, m); wt = np.ones(k) / k if weights is None else np.asarray(weights, float) / np.sum(weights)
    M = float((wt * mi).sum()); Rstar = float(Rg[np.argmax(m <= eps)]) if (m <= eps).any() else np.inf
    share = float(((R >= Rstar) * wt).sum())
    ok = M <= eps and t2 > 0
    decision = ('partition on the observed attribute' if ok else
                ('report unadjusted and size-adjusted estimates together; shrink the attribute (tau^2 > 0)' if t2 > 0 else
                 'decline to partition on this attribute (tau^2 = 0); name unit size directly'))
    summary = (f'tau^2 = {t2:.4g} (tau {np.sqrt(t2):.3g}); Cochran Q = {Q:.1f} on {k-1} df, p = {stats.chi2.sf(Q, k-1):.3g}; '
               f'median R = {np.median(R):.3f}; share R>=0.7 = {(R >= 0.7).mean()*100:.1f}%; expected misclassification M = {M*100:.1f}% '
               f'(tolerance {eps*100:.0f}%, R* = {Rstar:.2f}, share of units with R>=R* = {share*100:.1f}%) -> {decision}')
    return dict(tau2=t2, Q=Q, p_Q=stats.chi2.sf(Q, k - 1), R=R, M=M, R_star=Rstar, share_above_Rstar=share, decision=decision, summary=summary)

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_common import load
    d = load(); r = stage0_reliability(d, 'TotalUniqueBene', 'AvgPatientAge', 'outer', sigma_w=12.7275)
    print(r['summary'])
