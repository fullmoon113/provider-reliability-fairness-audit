from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = ROOT / 'output'
"""Table 4 under the class-weighting primary specification. Skewed covariates enter on the log scale.
Arm 1: standardized mean difference (pooled SD) between extreme and mid-range age-mix among the 506 fraud-labeled providers.
Arm 2: odds ratio of detection per SD of the covariate (Firth), in a model that also contains the age-mix indicator."""
import sys, numpy as np, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, str(HERE))
from audit_common import *
from firth import fit_firth
pf = load(); fm = pf.Fraud.values == 1; ext = pf.IsExtremeAge.values[fm].astype(float)
det = oof_lr(pf, pf.IsExtremeAge.values, strategy='class_weight')[fm]
n = pf.TotalUniqueBene.values.astype(float)
L = np.log1p
ctrl = [('Panel size, log', L(n)), ('Mean length of stay', pf.IP_AvgLOS.values), ('Chronic conditions', pf.AvgChronicCondCount.values),
        ('Claims per beneficiary, log', L(pf.ClaimsPerBene.values)), ('Inpatient beneficiaries, log', L(pf.IP_UniqueBene.values)),
        ('Outpatient reimbursement, log', L(pf.OP_AvgReimbursed.values)), ('Deceased ratio', pf.DeceasedRatio.values),
        ('Female ratio', pf.FemaleRatio.values)]
def firth_p(X, y):
    b = fit_firth(X, y); p = 1/(1+np.exp(-np.clip(X@b, -30, 30))); W = p*(1-p)
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ (X*W[:, None]) + 1e-10*np.eye(X.shape[1]))))
    return b, 2*stats.norm.sf(np.abs(b/se))
rows = []
for k, v in ctrl:
    x = v[fm]; z = (x-x.mean())/x.std(ddof=1)
    a, b = x[ext == 1], x[ext == 0]
    d1 = (a.mean()-b.mean())/np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    r = sm.Logit(det, np.column_stack([np.ones(len(det)), ext, z])).fit(disp=0)
    bf, pf_ = firth_p(np.column_stack([np.ones(len(det)), ext, z]), det)
    rows.append((k, d1, np.exp(r.params[2]), r.pvalues[1], pf_[1], np.exp(bf[2])))
ml = [r[3] for r in rows]; fi = [r[4] for r in rows]
hm = multipletests(ml, method='holm')[1]; hf = multipletests(fi, method='holm')[1]
print('control | arm1 d | arm2 std OR (ML) | age p (ML) | Holm | age p (Firth) | Holm(Firth) | arm2 Firth')
for r, a, b in zip(rows, hm, hf):
    print(f'{r[0]:30s} {r[1]:+.2f} {r[2]:6.2f} {r[3]:.4f} {a:.4f} | {r[4]:.4f} {b:.4f} | {r[5]:.2f}')
Z = lambda v: (v[fm]-v[fm].mean())/v[fm].std(ddof=1)
alt = np.column_stack([Z(v) for k, v in ctrl[1:]])
j1 = logit_or(det, [ext]+[alt[:, i] for i in range(alt.shape[1])])
j2 = logit_or(det, [ext, L(n)[fm]]+[alt[:, i] for i in range(alt.shape[1])])
print(f'joint 7 alt: OR {j1[0]:.3f} [{j1[2]:.3f},{j1[3]:.3f}] p={j1[1]:.4f}; + panel size OR {j2[0]:.3f} [{j2[2]:.3f},{j2[3]:.3f}] p={j2[1]:.4f}')
