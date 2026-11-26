from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = ROOT / 'output'
"""Numbers for supplement tables S1, S4, S5, S8, S10, S11 under the class-weighting primary specification."""
import sys, io, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(HERE))
from audit_common import *
from firth import fit_firth, firth_plr_test
buf = io.StringIO()
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); buf.write(s + '\n')
pf = load(); y = pf.Fraud.values; fm = y == 1
n = pf.TotalUniqueBene.values.astype(float); logn_all = np.log1p(n); logn = logn_all[fm]
ext = pf.IsExtremeAge.values; age = pf.AvgPatientAge.values
def fmt(r): return f'{r[0]:.3f} [{r[2]:.3f},{r[3]:.3f}] p={r[1]:.4f}'
P('== S1: strategies (extreme age-mix; unadj | adj)')
for strat, seeds in [('class_weight', [42]), ('group_smote', [42, 1, 7, 99, 2024]), ('pooled_smote', [42]), ('none', [42])]:
    for s in seeds:
        det = oof_lr(pf, ext, strategy=strat, seed=s)[fm]
        u, a = audit(det, ext[fm], logn)
        pp = sm.Logit(det, sm.add_constant(np.column_stack([ext[fm], logn]))).fit(disp=0).pvalues[2]
        P(f'{strat:13s} seed {s:5d} det {det.mean():.3f} | {fmt(u)} | {fmt(a)} | panel size p={pp:.2g}')
oof = oof_lr(pf, ext, strategy='class_weight'); det = oof[fm]
P('\n== S4: bootstrap of recall gap (class weighting), 3000 resamples')
rng = np.random.default_rng(42); gaps = []
idx = np.where(fm)[0]
for b in range(3000):
    s = rng.choice(len(pf), len(pf)); m = fm[s]
    e = ext[s][m]; o = oof[s][m]
    if e.sum() == 0 or (1 - e).sum() == 0: continue
    gaps.append(o[e == 0].mean() - o[e == 1].mean())
gaps = np.array(gaps); P(f'recall gap (mid - extreme) {oof[fm & (ext==0)].mean()-oof[fm & (ext==1)].mean():.3f}, 95% CI [{np.quantile(gaps,.025):.3f}, {np.quantile(gaps,.975):.3f}]')
P('\n== S5: OLS on predicted probability / percentile rank (class weighting)')
X = pf[FEATS].values; folds = pf.fold.values; prob = np.zeros(len(pf))
for f in range(5):
    tr, va = folds != f, folds == f
    sc = StandardScaler(); mdl = make_model('LR', 42, 'balanced').fit(sc.fit_transform(X[tr]), y[tr]); prob[va] = mdl.predict_proba(sc.transform(X[va]))[:, 1]
pr_ = prob[fm]; rk = stats.rankdata(pr_) / len(pr_)
for lab, v in [('probability', pr_), ('percentile rank', rk)]:
    ru = sm.OLS(v, sm.add_constant(ext[fm].astype(float))).fit(); ra = sm.OLS(v, sm.add_constant(np.column_stack([ext[fm], logn]))).fit()
    P(f'{lab}: unadj {ru.params[1]:.4f} p={ru.pvalues[1]:.2g} | adj {ra.params[1]:.4f} p={ra.pvalues[1]:.4f}')
P('\n== S8: functional form of panel size (adjusted OR, extreme age-mix)')
dfm = n[fm]; e = ext[fm].astype(float)
def spline(x, df=4):
    from patsy import dmatrix
    return np.asarray(dmatrix(f'cr(x, df={df})', {'x': x}, return_type='matrix'))[:, 1:]
dec = pd.qcut(pd.Series(dfm).rank(method='first'), 10, labels=False).values
forms = {'None (unadjusted)': None, 'log(1 + n), primary': [logn], 'n (raw)': [dfm], 'sqrt n': [np.sqrt(dfm)], 'Quadratic in log(1 + n)': [logn, logn ** 2],
         'Natural cubic spline, df = 4': [spline(logn)], 'Decile indicators (9)': [np.eye(10)[dec][:, 1:]], '1 / n': [1 / dfm]}
for lab, cov in forms.items():
    cols = [e] + ([np.atleast_2d(c.T).T if c.ndim == 1 else c for c in cov] if cov else [])
    Xm = sm.add_constant(np.column_stack(cols), has_constant='add')
    r = sm.Logit(det, Xm).fit(disp=0, maxiter=300); ci = np.exp(r.conf_int()[1])
    P(f'{lab:32s} {np.exp(r.params[1]):.3f} [{ci[0]:.3f},{ci[1]:.3f}] p={r.pvalues[1]:.4f}')
P('\n== S10 Panel A: ML vs Firth, three partitions')
parts = {'Extreme age-mix': ext, 'High-elderly (top 25%)': (age > np.quantile(age, .75)).astype(int), 'High-minority (top 25%)': (pf.MinorityRatio.values > np.quantile(pf.MinorityRatio.values, .75)).astype(int)}
for lab, g in parts.items():
    gf = g[fm].astype(float); out = []
    for cols in ([gf], [gf, logn]):
        Xm = np.column_stack([np.ones(fm.sum())] + cols); b = fit_firth(Xm, det)
        r = sm.Logit(det, Xm).fit(disp=0, maxiter=300)
        orf, pf_ = firth_plr_test(Xm, det, 1)
        out.append((np.exp(r.params[1]), r.pvalues[1], orf, pf_))
    P(f'{lab:26s} n={int(gf.sum())} | ML unadj {out[0][0]:.3f} ({out[0][1]:.4f}) Firth {out[0][2]:.3f} ({out[0][3]:.4f}) | ML adj {out[1][0]:.3f} ({out[1][1]:.4f}) Firth {out[1][2]:.3f} ({out[1][3]:.4f})')
P('\n== S11: selection-strength sensitivity (weights exp(-gamma z)), adjusted extreme age-mix effect')
z = (logn - logn.mean()) / logn.std(ddof=1)
for gam in [0, .25, .5, 1, 1.5, 2]:
    w = np.exp(-gam * z)
    Xm = sm.add_constant(np.column_stack([ext[fm], logn]))
    r = sm.GLM(det, Xm, family=sm.families.Binomial(), freq_weights=w / w.mean()).fit()
    # robust SE
    r2 = sm.GLM(det, Xm, family=sm.families.Binomial(), var_weights=w / w.mean()).fit(cov_type='HC0')
    P(f'gamma {gam:4.2f}: adj OR {np.exp(r2.params[1]):.3f} p={r2.pvalues[1]:.4f}')
def evalue(est, p0):
    rr = est / ((1 - p0) + p0 * est) if True else est
    rr = 1 / rr if rr < 1 else rr
    return rr + np.sqrt(rr * (rr - 1))
for lab, est, p0 in [('extreme (adj OR 1.018)', 1.018, .883), ('top quartile (adj OR 0.807)', 0.807, .871)]:
    P(f'E-value, {lab}: point {evalue(est, p0):.2f}')
(OUT / 'supplement_runs_output.txt').write_text(buf.getvalue())
