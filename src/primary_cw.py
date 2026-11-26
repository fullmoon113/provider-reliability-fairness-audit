from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

"""
Primary specification with class weighting (decision D1): regenerates the numbers of Tables 1, 3, 4, 6, the
false-positive audit, the fairness-mitigation paragraph and Table 8 with the detection model trained WITHOUT resampling
(class_weight='balanced'), so that the model does not depend on any partition.  Group-wise SMOTE results are kept
as a sensitivity analysis (src/02_experiments.py, src/refit_sweep.py).
Usage: python src/primary_cw.py   -> output/primary_cw_output.txt
"""
import sys, io
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from sklearn.metrics import (f1_score, recall_score, precision_score, roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from audit_common import load, FEATS, make_model, oof_lr, logit_or, outer

buf = io.StringIO()
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); buf.write(s + '\n')

pf = load(); y = pf.Fraud.values; folds = pf.fold.values; X = pf[FEATS].values
n = pf.TotalUniqueBene.values.astype(float); age = pf.AvgPatientAge.values
fm = y == 1; logn = np.log1p(n[fm])
ext = pf.IsExtremeAge.values
top = (pf.AvgPatientAge >= pf.AvgPatientAge.quantile(.75)).astype(int).values
mino = (pf.MinorityRatio >= pf.MinorityRatio.quantile(.75)).astype(int).values

# ---------------- Table 1: performance, class weighting ----------------
P('== Table 1 (class weighting)')
for name in ['LR', 'RF', 'XGB']:
    f1s, aucs, prs, recs, precs = [], [], [], [], []
    for f in range(5):
        tr, va = folds != f, folds == f
        sc = StandardScaler(); Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
        m = make_model(name, 42, 'balanced' if name != 'XGB' else None)
        if name == 'XGB':
            m.set_params(scale_pos_weight=(y[tr] == 0).sum() / (y[tr] == 1).sum())
        m.fit(Xtr, y[tr]); p, pr = m.predict(Xva), m.predict_proba(Xva)[:, 1]
        f1s.append(f1_score(y[va], p)); aucs.append(roc_auc_score(y[va], pr)); prs.append(average_precision_score(y[va], pr))
        recs.append(recall_score(y[va], p)); precs.append(precision_score(y[va], p, zero_division=0))
    P(f'{name:4s} F1 {np.mean(f1s):.3f}+-{np.std(f1s):.3f}  P {np.mean(precs):.3f}+-{np.std(precs):.3f}  '
      f'R {np.mean(recs):.3f}+-{np.std(recs):.3f}  ROC {np.mean(aucs):.3f}+-{np.std(aucs):.3f}  PR {np.mean(prs):.3f}+-{np.std(prs):.3f}')

# ---------------- Table 3 and Table 6 ----------------
oof = {m: oof_lr(pf, ext, strategy='class_weight', model=m) for m in ['LR', 'RF', 'XGB']}
det = oof['LR'][fm]
P('\n== Table 3 (LR, class weighting): partition, n, unadj OR [CI] p, adj OR [CI] p')
for lab, g in [('Extreme age-mix (Q1+Q4)', ext), ('High-elderly (top 25%)', top), ('High-minority (top 25%)', mino)]:
    gg = g[fm]
    u = logit_or(det, [gg]); a = logit_or(det, [gg, logn])
    P(f'{lab:26s} n={int(gg.sum()):4d} unadj {u[0]:.3f} [{u[2]:.3f},{u[3]:.3f}] p={u[1]:.4f} | adj {a[0]:.3f} [{a[2]:.3f},{a[3]:.3f}] p={a[1]:.4f}')
    P(f'    detection rate group/rest: {det[gg==1].mean():.3f}/{det[gg==0].mean():.3f}')
P('\n== Table 6 (detection model; class weighting)')
for m in ['LR', 'RF', 'XGB']:
    d = oof[m][fm]; u = logit_or(d, [ext[fm]]); a = logit_or(d, [ext[fm], logn])
    P(f'{m:4s} det {d.mean()*100:.1f}% unadj {u[0]:.3f} [{u[2]:.3f},{u[3]:.3f}] p={u[1]:.4f} adj {a[0]:.3f} [{a[2]:.3f},{a[3]:.3f}] p={a[1]:.4f}')

# ---------------- Table 4: specificity ----------------
P('\n== Table 4 (LR, class weighting): control, arm1 d, arm2 stdOR, age p, Holm p')
ctrl = {'Panel size, log': np.log1p(n), 'Mean length of stay': pf.IP_AvgLOS.values, 'Chronic conditions': pf.AvgChronicCondCount.values,
        'Claims per beneficiary': pf.ClaimsPerBene.values, 'Inpatient beneficiaries': pf.IP_UniqueBene.values,
        'Outpatient reimbursement': pf.OP_AvgReimbursed.values, 'Deceased ratio': pf.DeceasedRatio.values, 'Female ratio': pf.FemaleRatio.values}
rows = []
for k, v in ctrl.items():
    vv = v[fm]
    r = logit_or(det, [ext[fm], vv])
    # arm 1: standardized mean difference, all providers (extreme minus mid-range)
    d1 = (v[ext == 1].mean() - v[ext == 0].mean()) / np.sqrt((v[ext == 1].var(ddof=1) + v[ext == 0].var(ddof=1)) / 2)
    z = (vv - vv.mean()) / vv.std()
    s = sm.Logit(det, sm.add_constant(z)).fit(disp=0)
    rows.append([k, d1, np.exp(s.params[1]), r[1]])
pv = np.array([r[3] for r in rows]); o = np.argsort(pv); holm = np.empty(len(pv)); run = 0
for rank, i in enumerate(o):
    run = max(run, min(1, pv[i] * (len(pv) - rank))); holm[i] = run
for r, h in zip(rows, holm):
    P(f'{r[0]:26s} d={r[1]:+.2f}  stdOR={r[2]:.2f}  p={r[3]:.4f}  Holm={h:.4f}')
others = [v[fm] for k, v in ctrl.items() if k != 'Panel size, log']
Zo = np.column_stack([(o_ - o_.mean()) / o_.std() for o_ in others])
j1 = logit_or(det, [ext[fm]] + [Zo[:, i] for i in range(Zo.shape[1])])
j2 = logit_or(det, [ext[fm], logn] + [Zo[:, i] for i in range(Zo.shape[1])])
P(f'Joint (7 alternatives): OR {j1[0]:.3f} [{j1[2]:.3f},{j1[3]:.3f}] p={j1[1]:.4f};  + panel size: OR {j2[0]:.3f} [{j2[2]:.3f},{j2[3]:.3f}] p={j2[1]:.4f}')

# ---------------- Flagging among non-fraudulent ----------------
nf = y == 0
fl = oof['LR'][nf]; lnn = np.log1p(n[nf])
u = logit_or(fl, [ext[nf]]); a = logit_or(fl, [ext[nf], lnn])
P(f'\n== Flagging among {int(nf.sum())} negatives: rates {fl[ext[nf]==1].mean()*100:.1f}%/{fl[ext[nf]==0].mean()*100:.1f}%  '
  f'unadj {u[0]:.3f} [{u[2]:.3f},{u[3]:.3f}] p={u[1]:.4g}  adj {a[0]:.3f} [{a[2]:.3f},{a[3]:.3f}] p={a[1]:.4f}')

# ---------------- Fairness metrics and mitigation ----------------
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
pred = oof['LR']
P('\n== Fairness metrics, class weighting')
P(f'EOD {equalized_odds_difference(y, pred, sensitive_features=ext):.4f}  DPD {demographic_parity_difference(y, pred, sensitive_features=ext):.4f}  '
  f'F1 {f1_score(y, pred):.3f}  recall mid {recall_score(y[ext==0], pred[ext==0]):.3f} extreme {recall_score(y[ext==1], pred[ext==1]):.3f}')
draws = 30; eods, f1m, rm, re = [], [], [], []
mit_oof = [np.zeros(len(pf)) for _ in range(draws)]
for f in range(5):
    tr, va = folds != f, folds == f
    sc = StandardScaler(); Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
    eg = ExponentiatedGradient(LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'),
                               constraints=EqualizedOdds(), eps=0.02, max_iter=30)
    eg.fit(Xtr, y[tr], sensitive_features=ext[tr])
    for d_ in range(draws):
        mit_oof[d_][va] = eg._pmf_predict(Xva)[:, 1] > np.random.default_rng(d_ * 100 + f).random(va.sum())
for d_ in range(draws):
    p_ = mit_oof[d_]
    eods.append(equalized_odds_difference(y, p_, sensitive_features=ext)); f1m.append(f1_score(y, p_))
    rm.append(recall_score(y[ext == 0], p_[ext == 0])); re.append(recall_score(y[ext == 1], p_[ext == 1]))
P(f'Mitigated (30 draws): EOD {np.mean(eods):.4f} (SD {np.std(eods):.4f}; falls in {sum(e < equalized_odds_difference(y, pred, sensitive_features=ext) for e in eods)}/30), '
  f'F1 {np.mean(f1m):.3f}, recall mid {np.mean(rm):.3f} extreme {np.mean(re):.3f}')

# ---------------- Table 8: shrinkage ----------------
st2 = 0.812; sw2 = 12.7275 ** 2
rel = st2 / (st2 + sw2 / n); shr = age.mean() + rel * (age - age.mean())
ext_s = outer(shr)
P(f'\n== Table 8 inputs: SD observed {age.std():.2f} -> shrunken {shr.std():.3f}; membership kept {(ext_s==ext).mean()*100:.1f}%')
gs = ext_s[fm]
P(f'extreme (shrunken) group: median panel {np.median(n[ext_s==1]):.0f} (was {np.median(n[ext==1]):.0f}); '
  f'share in top panel-size quartile {np.mean(n[ext_s==1] >= np.quantile(n, .75))*100:.1f}% (was {np.mean(n[ext==1] >= np.quantile(n, .75))*100:.1f}%); '
  f'fraud prevalence {y[ext_s==1].mean()*100:.1f}% (was {y[ext==1].mean()*100:.1f}%)')
u = logit_or(det, [gs]); a = logit_or(det, [gs, logn])
P(f'Row 2 (shrunken groups, same class-weighted model): unadj {u[0]:.3f} [{u[2]:.3f},{u[3]:.3f}] p={u[1]:.4f} | adj {a[0]:.3f} [{a[2]:.3f},{a[3]:.3f}] p={a[1]:.4f}')
oof_ref = oof_lr(pf, ext_s, strategy='group_smote')
u = logit_or(oof_ref[fm], [gs]); a = logit_or(oof_ref[fm], [gs, logn])
P(f'Row 3 (shrunken groups, model refitted with resampling stratified on them): unadj {u[0]:.3f} [{u[2]:.3f},{u[3]:.3f}] p={u[1]:.4f} | adj {a[0]:.3f} [{a[2]:.3f},{a[3]:.3f}] p={a[1]:.4f}')
oof_pool = oof_lr(pf, ext_s, strategy='pooled_smote')
u = logit_or(oof_pool[fm], [gs]); a = logit_or(oof_pool[fm], [gs, logn])
P(f'Row (pooled SMOTE): unadj {u[0]:.3f} [{u[2]:.3f},{u[3]:.3f}] p={u[1]:.4f} | adj {a[0]:.3f} [{a[2]:.3f},{a[3]:.3f}] p={a[1]:.4f}')
for st in [0.247, 0.812, 0.948, 1.97]:
    rl = st / (st + sw2 / n); sh = age.mean() + rl * (age - age.mean()); g_ = outer(sh)[fm]
    u = logit_or(det, [g_])
    P(f'  sigma2_true={st:5.3f}: fixed-model shrunken OR {u[0]:.3f} p={u[1]:.4f}')
(OUT / 'primary_cw_output.txt').write_text(buf.getvalue())
