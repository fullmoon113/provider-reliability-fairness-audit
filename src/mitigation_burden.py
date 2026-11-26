"""Table S4 extension: group-wise recall, false-positive rate, flag rate and precision before and after
equalized-odds mitigation (class-weighted logistic regression; 30 prediction draws of the randomized classifier).
Usage: python src/mitigation_burden.py  -> output/mitigation_burden_output.txt"""
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from audit_common import load, FEATS, oof_lr
OUT = Path(__file__).resolve().parent.parent / 'output'
pf = load(); y = pf.Fraud.values; folds = pf.fold.values; X = pf[FEATS].values; ext = pf.IsExtremeAge.values
def rates(p):
    r = {}
    for g, lab in ((0, 'mid'), (1, 'ext')):
        m = ext == g; yy, pp = y[m], p[m]
        r[f'recall_{lab}'] = pp[yy == 1].mean(); r[f'fpr_{lab}'] = pp[yy == 0].mean(); r[f'flag_{lab}'] = pp.mean()
        r[f'ppv_{lab}'] = yy[pp == 1].mean() if pp.sum() else np.nan
    r['eod'] = equalized_odds_difference(y, p, sensitive_features=ext); r['dpd'] = demographic_parity_difference(y, p, sensitive_features=ext)
    r['f1'] = f1_score(y, p); r['flag_all'] = p.mean()
    return r
before = rates(oof_lr(pf, ext, strategy='class_weight').astype(bool))
draws = 30; mit = [np.zeros(len(pf), bool) for _ in range(draws)]
for f in range(5):
    tr, va = folds != f, folds == f
    sc = StandardScaler(); Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
    eg = ExponentiatedGradient(LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'), constraints=EqualizedOdds(), eps=0.02, max_iter=30)
    eg.fit(Xtr, y[tr], sensitive_features=ext[tr])
    for d in range(draws): mit[d][va] = eg._pmf_predict(Xva)[:, 1] > np.random.default_rng(d * 100 + f).random(va.sum())
rs = [rates(p) for p in mit]
lines = ['metric  before  after_mean  after_sd']
for k in before:
    a = np.array([r[k] for r in rs]); lines.append(f'{k:12s} {before[k]:.4f}  {np.nanmean(a):.4f}  {np.nanstd(a):.4f}')
s = '\n'.join(lines); print(s); (OUT / 'mitigation_burden_output.txt').write_text(s + '\n')
