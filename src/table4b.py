"""Table 4b (paper Section IV-D): detection (TPR), false-positive rate (FPR), flag rate and PPV by partition, with unadjusted and size-adjusted
odds ratios, the panel-size-quartile stratified difference, the two-tail rows of Table 4 and the TOST of the false-positive analysis.
Primary specification (class-weighted logistic regression, out-of-fold predictions). Usage: python src/table4b.py"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from audit_common import *
from scipy.stats import norm
pf = load(); y = pf.Fraud.values; n = pf.TotalUniqueBene.values.astype(float); A = pf.AvgPatientAge.values; logn_all = np.log1p(n)
pred = oof_lr(pf, outer(A), strategy='class_weight')
parts = {'Extreme age-mix (Q1+Q4)': outer(A), 'Top age quartile': (A >= np.quantile(A, .75)).astype(int),
         'Lowest age quartile vs middle half': None, 'Highest age quartile vs middle half': None}
q = np.quantile(A, [.25, .5, .75]); mid = (A > q[0]) & (A <= q[2])
lines = []
def emit(s=''): print(s); lines.append(s)
def rate(mask, g): return pred[mask & (g == 1)].mean() * 100, pred[mask & (g == 0)].mean() * 100
def strat_diff(mask, g):
    qs = np.quantile(n[mask], [.25, .5, .75]); b = np.digitize(n[mask], qs); d = pred[mask]; gg = g[mask]; w = []; diffs = []
    for k in range(4):
        m = b == k
        if (gg[m] == 1).sum() > 0 and (gg[m] == 0).sum() > 0: diffs.append(d[m & (gg == 1)].mean() - d[m & (gg == 0)].mean() if False else d[m][gg[m] == 1].mean() - d[m][gg[m] == 0].mean()); w.append(m.sum())
    return np.average(diffs, weights=w) * 100
emit('Partition | TPR g / rest | OR unadj [CI] | OR adj [CI] | stratified diff (pts) || FPR g / rest | OR unadj [CI] | OR adj [CI] || flag rate g / rest | PPV g / rest')
rows = {}
for name in ('Extreme age-mix (Q1+Q4)', 'Top age quartile', 'Lowest age quartile vs middle half', 'Highest age quartile vs middle half'):
    if name.startswith('Lowest'): sel = (A <= q[0]) | mid; g = (A <= q[0]).astype(int)
    elif name.startswith('Highest'): sel = (A > q[2]) | mid; g = (A > q[2]).astype(int)
    else: sel = np.ones(len(A), bool); g = parts[name]
    pos = (y == 1) & sel; neg = (y == 0) & sel
    tu, ta = logit_or(pred[pos], [g[pos]]), logit_or(pred[pos], [g[pos], logn_all[pos]])
    fu, fa = logit_or(pred[neg], [g[neg]]), logit_or(pred[neg], [g[neg], logn_all[neg]])
    t1, t0 = rate(pos, g); f1, f0 = rate(neg, g); fl1, fl0 = rate(sel, g)
    ppv1 = y[sel & (g == 1) & (pred == 1)].mean() * 100; ppv0 = y[sel & (g == 0) & (pred == 1)].mean() * 100
    sd_t = strat_diff(pos, g); sd_f = strat_diff(neg, g)
    emit(f'{name} | {t1:.1f}/{t0:.1f} | {tu[0]:.3f} [{tu[2]:.3f},{tu[3]:.3f}] p={tu[1]:.4f} | {ta[0]:.3f} [{ta[2]:.3f},{ta[3]:.3f}] p={ta[1]:.4f} | {sd_t:.1f} || '
         f'{f1:.1f}/{f0:.1f} | {fu[0]:.3f} [{fu[2]:.3f},{fu[3]:.3f}] | {fa[0]:.3f} [{fa[2]:.3f},{fa[3]:.3f}] p={fa[1]:.3f} | {sd_f:.1f} || {fl1:.1f}/{fl0:.1f} | {ppv1:.1f}/{ppv0:.1f}')
    rows[name] = dict(tpr=(t1, t0), tu=tu, ta=ta, fpr=(f1, f0), fu=fu, fa=fa, sd=(sd_t, sd_f), flag=(fl1, fl0), ppv=(ppv1, ppv0))
    if name.startswith('Extreme'):
        se = (np.log(fa[3]) - np.log(fa[2])) / (2 * 1.959964)
        pl = norm.sf((np.log(fa[0]) - np.log(0.8)) / se); pu = norm.sf((np.log(1.25) - np.log(fa[0])) / se)
        emit(f'TOST (FPR, size-adjusted OR {fa[0]:.3f}, SE {se:.3f}; margin 0.8-1.25): p_lower {pl:.4f}, p_upper {pu:.4f}, TOST p = {max(pl, pu):.4f}')
        se_t = ta[3] and (np.log(ta[3]) - np.log(ta[2])) / (2 * 1.959964)
        pl = norm.sf((np.log(ta[0]) - np.log(0.8)) / se_t); pu = norm.sf((np.log(1.25) - np.log(ta[0])) / se_t)
        emit(f'TOST (TPR, size-adjusted OR {ta[0]:.3f}, SE {se_t:.3f}): TOST p = {max(pl, pu):.4f}')
open(OUT / 'table4b_output.txt', 'w').write('\n'.join(lines) + '\n')
