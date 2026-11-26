from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = ROOT / 'output'
"""Numbers quoted in Sections III-E, IV-C, IV-D, IV-F, V-D under the class-weighting primary specification."""
import sys, io, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, str(HERE))
from audit_common import *
from firth import fit_firth
buf = io.StringIO()
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); buf.write(s + '\n')
pf = load(); y = pf.Fraud.values; fm = y == 1
n = pf.TotalUniqueBene.values.astype(float); logn_all = np.log1p(n); logn = logn_all[fm]
age = pf.AvgPatientAge.values; ext = pf.IsExtremeAge.values
oof = oof_lr(pf, ext, strategy='class_weight'); det = oof[fm]
q = np.quantile(age, [.25, .5, .75]); qi = np.digitize(age, q[:3], right=True)   # 0..3 (quartile index)
P('== Quartile detection (fraud-labeled)')
for k in range(4):
    m = fm & (qi == k)
    P(f'Q{k+1}: n_fraud={m.sum()} det={oof[m].mean():.3f} median panel (fraud) {np.median(n[m]):.0f}; median panel (all) {np.median(n[qi==k]):.0f}')
a, b = fm & (qi == 0), fm & (qi == 3)
tab = [[oof[a].sum(), (1-oof[a]).sum()], [oof[b].sum(), (1-oof[b]).sum()]]
P('Q1 vs Q4 diff', oof[a].mean()-oof[b].mean(), 'Fisher p', stats.fisher_exact(tab)[1])
d = oof[a].mean()-oof[b].mean(); se = np.sqrt(oof[a].mean()*(1-oof[a].mean())/a.sum()+oof[b].mean()*(1-oof[b].mean())/b.sum())
P('  95% CI', d-1.96*se, d+1.96*se)
mid = fm & ((qi == 1) | (qi == 2))
P('median panel size, fraud-labeled: middle quartiles pooled', np.median(n[mid]), ' outer', np.median(n[fm & (ext == 1)]))
P('all providers: ext median', np.median(n[ext == 1]), 'mid', np.median(n[ext == 0]))
t, p = stats.ttest_ind(logn_all[ext == 1], logn_all[ext == 0], equal_var=True); P('t on log panel size', t, p)

P('\n== Firth / penalized LRT, row 1')
from firth import firth_plr_test
for lab, cols in [('unadj', [ext[fm]]), ('adj', [ext[fm], logn])]:
    X = np.column_stack([np.ones(fm.sum())]+[c.astype(float) for c in cols])
    orf, pp = firth_plr_test(X, det, 1)
    P(f'{lab}: Firth OR {orf:.3f}, penalized LRT p {pp:.4f}')
u = logit_or(det, [ext[fm]]); ad = logit_or(det, [ext[fm], logn])
P(f'ML: unadj {u[0]:.3f}  adj {ad[0]:.3f} [{ad[2]:.3f},{ad[3]:.3f}] p {ad[1]:.4f}')
se_a = (np.log(ad[3])-np.log(ad[2]))/(2*1.96); P('SE adj (log OR)', se_a, ' 80% power OR', np.exp(-2.80*se_a))
bt = np.log(ad[0]); z1 = (bt-np.log(.8))/se_a; z2 = (np.log(1.25)-bt)/se_a
P('TOST p (0.8-1.25)', max(stats.norm.sf(z1), stats.norm.sf(z2)))
P('panel size p in adjusted model', sm.Logit(det, sm.add_constant(np.column_stack([ext[fm], logn]))).fit(disp=0).pvalues[2])

P('\n== Restriction to larger providers')
for thr in [0, 10, 20, 30, 50, 100]:
    m = n[fm] > thr; r = logit_or(det[m], [ext[fm][m]]); P(f'>{thr}: n={m.sum()} OR {r[0]:.3f} p={r[1]:.4f}')

P('\n== Partitions and Holm (six): extreme, top quartile, minority, continuous age, lowest quartile, highest quartile')
top = (age >= np.quantile(age, .75)).astype(int); low = (age <= np.quantile(age, .25)).astype(int)
mino = (pf.MinorityRatio >= pf.MinorityRatio.quantile(.75)).astype(int).values
zage = (age-age.mean())/age.std(ddof=1)
res = []
midhalf = fm & (qi != 0) & (qi != 3)
for lab, mask, g in [('extreme', fm, ext), ('top', fm, top), ('minority', fm, mino), ('continuous', fm, zage),
                     ('lowest Q vs mid half', fm & ((qi == 0) | (qi != 3) & (qi != 0)), (qi == 0).astype(int)),
                     ('highest Q vs mid half', fm & (qi != 0), (qi == 3).astype(int))]:
    if lab.startswith('lowest'): mask = fm & (qi != 3); 
    if lab.startswith('highest'): mask = fm & (qi != 0)
    if lab.startswith('lowest') or lab.startswith('highest'):
        mask = fm & (qi != 3) if lab.startswith('lowest') else fm & (qi != 0)
        mask = mask & (((qi == 0) | (qi == 1) | (qi == 2)) if lab.startswith('lowest') else ((qi == 1) | (qi == 2) | (qi == 3)))
    ln = logn_all[mask]; dd = oof[mask]; gg = g[mask]
    r1 = logit_or(dd, [gg]); r2 = logit_or(dd, [gg, ln])
    res.append((lab, r1, r2))
    P(f'{lab:24s} unadj {r1[0]:.3f} [{r1[2]:.3f},{r1[3]:.3f}] p={r1[1]:.4f} | adj {r2[0]:.3f} [{r2[2]:.3f},{r2[3]:.3f}] p={r2[1]:.4f}')
pv = [r[1][1] for r in res]; hm = multipletests(pv, method='holm')[1]
P('Holm (six, unadjusted p):', dict(zip([r[0] for r in res], hm.round(4))))

P('\n== Functional forms (adjusted OR of extreme-age)')
x = n[fm]; forms = {'log1p': np.log1p(x), 'raw': x, 'sqrt': np.sqrt(x), 'inverse 1/n': 1/np.maximum(x, 1)}
lg = np.log1p(x); forms['quadratic in log'] = None
for k, v in forms.items():
    if v is None: r = logit_or(det, [ext[fm], lg, lg**2])
    else: r = logit_or(det, [ext[fm], v])
    P(f'{k:18s} adj OR {r[0]:.3f} p={r[1]:.4f}')
from patsy import dmatrix
Bs = np.asarray(dmatrix('cr(x, df=4) - 1', {'x': lg})); r = logit_or(det, [ext[fm]] + [Bs[:, i] for i in range(Bs.shape[1])]); P(f'natural spline        adj OR {r[0]:.3f} p={r[1]:.4f}')
dec = pd.qcut(pd.Series(x).rank(method='first'), 10, labels=False).values
r = logit_or(det, [ext[fm]] + [(dec == j).astype(float) for j in range(1, 10)]); P(f'decile indicators     adj OR {r[0]:.3f} p={r[1]:.4f}')

P('\n== Stratified gap and Mantel-Haenszel')
from statsmodels.stats.contingency_tables import StratifiedTable
for k, nm in [(4, 'quartiles'), (10, 'deciles')]:
    st = pd.qcut(pd.Series(x).rank(method='first'), k, labels=False).values
    tabs = []; gaps = []; ws = []
    for s in range(k):
        m = st == s; e = ext[fm][m]; dd = det[m]
        t_ = np.array([[(dd[e == 1] == 1).sum(), (dd[e == 1] == 0).sum()], [(dd[e == 0] == 1).sum(), (dd[e == 0] == 0).sum()]])+0.0
        if t_.sum(1).min() == 0: continue
        tabs.append(t_); gaps.append(dd[e == 0].mean()-dd[e == 1].mean()); ws.append(m.sum())
    P(f'{nm}: MH OR {StratifiedTable([t+0 for t in tabs]).oddsratio_pooled:.3f}  stratum-weighted gap {np.average(gaps, weights=ws):.3f}')
P('unstratified gap', det[ext[fm] == 0].mean()-det[ext[fm] == 1].mean())

P('\n== Thresholds (probability >= t)')
from sklearn.preprocessing import StandardScaler
X = pf[FEATS].values; folds = pf.fold.values; prob = np.zeros(len(pf))
for f in range(5):
    tr, va = folds != f, folds == f
    sc = StandardScaler(); m = make_model('LR', 42, 'balanced').fit(sc.fit_transform(X[tr]), y[tr]); prob[va] = m.predict_proba(sc.transform(X[va]))[:, 1]
for t in [.3, .4, .5, .6, .7, .8, .9]:
    dd = (prob[fm] >= t).astype(float); u = logit_or(dd, [ext[fm]]); a2 = logit_or(dd, [ext[fm], logn])
    P(f'thr {t}: det {dd.mean():.3f} unadj {u[0]:.3f} p={u[1]:.4f} | adj {a2[0]:.3f} p={a2[1]:.4f}')
pr_ = prob[fm]; rk = stats.rankdata(pr_)/len(pr_)
for lab, v in [('probability', pr_), ('percentile rank', rk)]:
    r1 = sm.OLS(v, sm.add_constant(ext[fm].astype(float))).fit(); r2 = sm.OLS(v, sm.add_constant(np.column_stack([ext[fm], logn]))).fit()
    P(f'{lab}: unadj p={r1.pvalues[1]:.4g}  adj p={r2.pvalues[1]:.4f}')
(OUT / 'cw_text_numbers_output.txt').write_text(buf.getvalue())
