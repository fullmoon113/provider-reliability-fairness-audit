"""
Ablation: remove the features that carry panel size and refit (reviewer item R2-5; paper limitation 2).
Feature sets: full (13 features) | no direct size (drops IP_UniqueBene, TotalUniqueBene, ClaimsPerBene)
              | no size-related (also drops the two attending-physician counts).
Models: logistic regression, random forest, XGBoost; training: group-wise SMOTE (primary) and class weighting.
Reports, among fraud-labelled providers: detection rate, slope of detection on log panel size (gamma1, per SD of log size),
extreme-age odds ratio unadjusted and adjusted for log panel size.
Usage: python src/ablation_size_features.py
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from audit_common import *
pf = load(); fm = pf.Fraud.values == 1; A = pf.AvgPatientAge.values; sens = outer(A)
logn = np.log1p(pf.TotalUniqueBene.values[fm]); zf = (logn - logn.mean()) / logn.std(ddof=1)
SETS = {'full (13)': FEATS,
        'no direct size (10)': [f for f in FEATS if f not in SIZE_FEATS],
        'no size-related (8)': [f for f in FEATS if f not in SIZE_FEATS + ['IP_UniqueAttPhysician', 'OP_UniqueAttPhysician']]}
lines = []
def emit(s=''): print(s, flush=True); lines.append(s)
emit(f"{'model':5} {'training':12} {'features':20} {'det%':>5} {'gamma1':>7} {'OR unadj [95% CI]':>24} {'p':>7} {'OR adj [95% CI]':>22} {'p':>6}")
for model in ('LR', 'RF', 'XGB'):
    for strat in ('group_smote', 'class_weight'):
        for fs, feats in SETS.items():
            oof = oof_lr(pf, sens, feats=feats, strategy=strat, model=model)
            det = oof[fm]
            u, a = audit(det, sens[fm], logn)
            g = logit_or(det, [zf])
            emit(f"{model:5} {strat:12} {fs:20} {det.mean()*100:5.1f} {np.log(g[0]):7.3f} "
                 f"{u[0]:7.3f} [{u[2]:.3f},{u[3]:.3f}] {u[1]:7.4f} {a[0]:7.3f} [{a[2]:.3f},{a[3]:.3f}] {a[1]:6.3f}")
open(OUT / 'ablation_size_features_output.txt', 'w').write('\n'.join(lines) + '\n')
