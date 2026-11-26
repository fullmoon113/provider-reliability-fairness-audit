"""The Stage 0 and Algorithm 1 table of the paper: Stage 0 and Algorithm 1 applied end to end to the primary dataset, plus the exact expected misclassification
numbers of Section IV-B.  Writes output/table12_output.txt.  Usage: python src/table12.py"""
import sys, io; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from stage0 import stage0_reliability, expected_misclassification
from audit_common import load, oof_lr, outer, OUT
buf = io.StringIO()
def P(*a):
    t = ' '.join(str(x) for x in a); print(t); buf.write(t + '\n')
d = load(); fm = d.Fraud.values == 1; SW = 12.7275
d['det'] = np.nan; d.loc[fm, 'det'] = oof_lr(d, outer(d.AvgPatientAge.values), strategy='class_weight')[fm]
n = d.TotalUniqueBene.values.astype(float)
for part, lab in (('outer', 'mean age, outer vs middle'), ('top', 'mean age, top quartile')):
    r = stage0_reliability(d, 'TotalUniqueBene', 'AvgPatientAge', part, sigma_w=SW, eps=0.10, weights='Fraud', detection='det')
    P('==', lab); P(r['summary'])
    for k in ('tau2', 'p_Q', 'p_LR', 'median_R', 'M', 'eta_G', 'eta_D', 'rho', 'phi', 'r_hat', 'predicted_OR', 'MDE', 'power_at_delta', 'branch', 'decision'):
        P(f'  {k}: {r[k]}')
    for k in ('unadjusted', 'adjusted', 'center', 'shrunken_adjusted', 'size_slope'):
        v = r[k]; P(f'  {k}: OR {v["OR"]:.3f} [{v["lo"]:.3f}, {v["hi"]:.3f}] p={v["p"]:.4f}')
    # exact M over all providers, over labeled providers, over providers with > 300 beneficiaries
    Mall, mis, lam = expected_misclassification(n, r['mu'], r['tau2'], SW, part, None, 400, 2026)
    Mlab, _, lam_lab = expected_misclassification(n, r['mu'], r['tau2'], SW, part, d.Fraud.values, 400, 2026)
    big = n > 300
    P(f'  exact M: all providers {Mall*100:.1f}%, fraud-labeled {Mlab*100:.1f}%, providers > 300 beneficiaries {mis[big].mean()*100:.1f}% (n={big.sum()}), lambda(all) {lam:.3f}')
# female share (Stage 0 only, fails to separate tau^2 from zero)
r = stage0_reliability(d, 'TotalUniqueBene', 'FemaleRatio', 'outer', eps=0.10, weights='Fraud', detection='det')
P('== share female, outer vs middle'); P(r['summary'])
for k in ('tau2', 'p_Q', 'p_LR', 'median_R', 'branch', 'decision'):
    P(f'  {k}: {r.get(k)}')
open(OUT / 'table12_output.txt', 'w').write(buf.getvalue())
