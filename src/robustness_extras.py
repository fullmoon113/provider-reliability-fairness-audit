"""Additional robustness analyses of the primary-dataset audit (supplement Tables S5-S23).

(1) Repeated stratified cross-validation: the headline unadjusted and panel-size-adjusted odds ratios of the
    extreme-age and top-quartile partitions over 10 repeats of stratified five-fold CV (class-weighted logistic regression,
    detection threshold 0.5, detection among the 506 fraud-labeled providers).
(2) Flag-rule audit: the model flags the top 10% of providers by out-of-fold score (the operating point of the LEIE check);
    detection is the flag rate among fraud-labeled providers, the investigation burden the flag rate among the others.
    Logistic regression, random forest and XGBoost; extreme-age partition, unadjusted and panel-size-adjusted.
(3) Detection-threshold sweep for random forest and XGBoost (extends supplement Table S16).
Usage: python src/robustness_extras.py   (writes output/robustness_extras_output.txt)"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from audit_common import *
from sklearn.model_selection import StratifiedKFold
pf = load(); y = pf.Fraud.values; X = pf[FEATS].values; A = pf.AvgPatientAge.values
logn = np.log1p(pf.TotalUniqueBene.values.astype(float)); lab = y == 1
ext = outer(A); top = (A >= np.quantile(A, .75)).astype(int)
L = []
def P(s=''): print(s); L.append(s)

def oof_proba(model, folds, seed=42):
    out = np.zeros(len(pf))
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        sc = StandardScaler(); Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
        m = make_model(model, seed, 'balanced' if model != 'XGB' else None)
        if model == 'XGB': m.set_params(scale_pos_weight=(y[tr] == 0).sum() / (y[tr] == 1).sum())
        m.fit(Xtr, y[tr]); out[va] = m.predict_proba(Xva)[:, 1]
    return out
def fmt(r): return f"{r[0]:.3f} [{r[2]:.3f}, {r[3]:.3f}] p={r[1]:.4f}"

# (1) repeated stratified CV
P('== (1) Repeated stratified 5-fold CV, class-weighted LR, threshold 0.5 (10 repeats)')
rows = []
for rep in range(10):
    folds = np.zeros(len(pf), int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=1000 + rep).split(X, y)): folds[va] = k
    det = (oof_proba('LR', folds) >= .5)[lab]
    for nm, g in (('extreme', ext), ('top quartile', top)):
        u, a = audit(det, g[lab], logn[lab]); rows.append((nm, rep, u[0], u[1], a[0], a[1], det.mean()))
import pandas as pd
d = pd.DataFrame(rows, columns=['part', 'rep', 'uOR', 'up', 'aOR', 'ap', 'det'])
for nm, g in d.groupby('part'):
    P(f"{nm}: unadjusted OR mean {g.uOR.mean():.3f} (range {g.uOR.min():.3f}-{g.uOR.max():.3f}), p<0.05 in {(g.up<.05).sum()}/10; "
      f"adjusted OR mean {g.aOR.mean():.3f} (range {g.aOR.min():.3f}-{g.aOR.max():.3f}), p<0.05 in {(g.ap<.05).sum()}/10; detection rate {g.det.mean():.3f}")
d.to_csv(OUT / 'robustness_extras_repeated_cv.csv', index=False)

# (2) top-10% flag rule and (3) threshold sweep, reference folds
folds = pf.fold.values
P('\n== (2) Top-10% flag rule (flags 10% of all providers by out-of-fold score); extreme-age partition')
P(f"{'model':5s} {'detection (labeled)':>20s} {'OR unadj [CI]':>28s} {'OR adj [CI]':>28s} | {'flag rate (unlabeled)':>22s} {'OR unadj':>9s} {'OR adj [CI]':>26s}")
probs = {}
for mdl in ('LR', 'RF', 'XGB'):
    p = oof_proba(mdl, folds); probs[mdl] = p
    flag = p >= np.quantile(p, .90)
    u, a = audit(flag[lab], ext[lab], logn[lab]); un = ~lab
    u2, a2 = audit(flag[un], ext[un], logn[un])
    P(f"{mdl:5s} {flag[lab].mean():20.3f} {fmt(u):>28s} {fmt(a):>28s} | {flag[un].mean():22.3f} {u2[0]:9.3f} {fmt(a2):>26s}")
P('\n== (3) Detection-threshold sweep, class-weighted RF and XGBoost; extreme-age partition')
P(f"{'model':5s} {'thr':>4s} {'detection':>9s} {'OR unadj':>9s} {'p':>8s} {'OR adj':>8s} {'p':>8s}")
for mdl in ('RF', 'XGB'):
    for t in (.3, .4, .5, .6, .7, .8, .9):
        det = (probs[mdl] >= t)[lab]
        if det.min() == det.max(): continue
        u, a = audit(det, ext[lab], logn[lab])
        P(f"{mdl:5s} {t:4.1f} {det.mean():9.3f} {u[0]:9.3f} {u[1]:8.4f} {a[0]:8.3f} {a[1]:8.4f}")
open(OUT / 'robustness_extras_output.txt', 'w').write('\n'.join(L) + '\n')
