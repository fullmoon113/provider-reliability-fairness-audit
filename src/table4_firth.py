from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy import stats
sys.path.insert(0,str(HERE))
from firth import fit_firth

pf = pd.read_csv(str(DATA/'provider_features_FINAL.csv'))
F13 = ['IP_UniqueBene','IP_AvgReimbursed','IP_AvgLOS','IP_AvgDiagCodes','IP_UniqueAttPhysician',
 'OP_AvgReimbursed','OP_AvgDiagCodes','OP_UniqueAttPhysician','TotalUniqueBene',
 'FemaleRatio','AvgChronicCondCount','DeceasedRatio','ClaimsPerBene']
y = pf.Fraud.values; folds = pf.fold.values; sens = pf.IsExtremeAge.values
X = pf[F13].values

def smote_strat(Xa, ya, sa, seed=42):
    Xo, yo = [], []
    for g in np.unique(sa):
        m = sa == g; Xg, yg = Xa[m], ya[m]
        k = min(5, max(1, (yg == 1).sum() - 1))
        Xb, yb = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xg, yg)
        Xo.append(Xb); yo.append(yb)
    return np.vstack(Xo), np.concatenate(yo)

oof = np.zeros(len(pf))
for f in range(5):
    tr = folds != f; va = folds == f
    sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
    Xb, yb = smote_strat(Xtr, y[tr], sens[tr], seed=42)
    lr = LogisticRegression(max_iter=500, random_state=42).fit(Xb, yb)
    oof[va] = lr.predict(Xva)

fm = y == 1
det = oof[fm]
ext = sens[fm].astype(float)

covars = [
    ('Mean length of stay', 'IP_AvgLOS'),
    ('Chronic conditions', 'AvgChronicCondCount'),
    ('Claims per beneficiary', 'ClaimsPerBene'),
    ('Inpatient beneficiaries', 'IP_UniqueBene'),
    ('Outpatient reimbursement', 'OP_AvgReimbursed'),
    ('Deceased ratio', 'DeceasedRatio'),
    ('Female ratio', 'FemaleRatio'),
]

def firth_wald(X, y):
    beta = fit_firth(X, y)
    z = np.clip(X @ beta, -30, 30)
    p = 1/(1+np.exp(-z))
    W = p*(1-p)
    XtWX = X.T @ (X*W[:,None]) + 1e-10*np.eye(X.shape[1])
    cov = np.linalg.pinv(XtWX)
    se = np.sqrt(np.diag(cov))
    zval = beta/se
    pval = 2*stats.norm.sf(np.abs(zval))
    return beta, pval

print(f"{'Covariate':26s} {'ML OR':>7s} {'ML p':>9s}   {'Firth OR':>8s} {'Firth p':>9s}")
firth_ps = []
ml_ps = []
for label, col in covars:
    cv = pf[col].values.astype(float)[fm]
    cvz = (cv - cv.mean())/cv.std(ddof=1)
    Xd = np.column_stack([np.ones(len(det)), ext, cvz])
    r = sm.Logit(det, Xd).fit(disp=0)
    ml_or, ml_p = np.exp(r.params[1]), r.pvalues[1]
    beta_f, pval_f = firth_wald(Xd, det)
    firth_or, firth_p = np.exp(beta_f[1]), pval_f[1]
    firth_ps.append(firth_p); ml_ps.append(ml_p)
    print(f"{label:26s} {ml_or:7.3f} {ml_p:9.4f}   {firth_or:8.3f} {firth_p:9.4f}")

# panel size too
logn = np.log1p(pf.TotalUniqueBene.values.astype(float))
lognf = logn[fm]; lognf_z = (lognf-lognf.mean())/lognf.std(ddof=1)
Xd = np.column_stack([np.ones(len(det)), ext, lognf_z])
r = sm.Logit(det, Xd).fit(disp=0)
beta_f, pval_f = firth_wald(Xd, det)
print(f"{'Panel size, log':26s} {np.exp(r.params[1]):7.3f} {r.pvalues[1]:9.4f}   {np.exp(beta_f[1]):8.3f} {pval_f[1]:9.4f}")
firth_ps.append(pval_f[1])

print()
print("ML p range (7 alts):", min(ml_ps), max(ml_ps))
print("Firth p range (7 alts):", min(firth_ps[:-1]), max(firth_ps[:-1]))

# Holm correction on the 7 alternatives + panel size (8 tests), like Table 4
from statsmodels.stats.multitest import multipletests
all_p = ml_ps + [pval_f[1]]  # not used
rej, holm_ml, _, _ = multipletests(ml_ps + [r.pvalues[1]], method='holm')
print("Holm (ML, 8 tests incl panel size):", holm_ml)
rej2, holm_firth, _, _ = multipletests(firth_ps, method='holm')
print("Holm (Firth, 8 tests incl panel size):", holm_firth)
print("Max Holm p among the 7 alternatives (Firth):", max(holm_firth[:-1]))
