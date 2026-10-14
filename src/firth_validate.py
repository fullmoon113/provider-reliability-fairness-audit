from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
sys.path.insert(0,str(HERE))
from firth import fit_firth, firth_plr_test, firth_penalized_loglik

pf = pd.read_csv(str(DATA/'provider_features_FINAL.csv'))
F13 = ['IP_UniqueBene','IP_AvgReimbursed','IP_AvgLOS','IP_AvgDiagCodes','IP_UniqueAttPhysician',
 'OP_AvgReimbursed','OP_AvgDiagCodes','OP_UniqueAttPhysician','TotalUniqueBene',
 'FemaleRatio','AvgChronicCondCount','DeceasedRatio','ClaimsPerBene']
y = pf.Fraud.values; folds = pf.fold.values; sens = pf.IsExtremeAge.values
n = pf.TotalUniqueBene.values.astype(float)
X = pf[F13].values
logn = np.log1p(n)

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
lognf = logn[fm]
lognf_z = (lognf - lognf.mean())/lognf.std(ddof=1)

# Design matrix: [intercept, extreme_age, panel_size_z]
X_full = np.column_stack([np.ones(len(det)), ext, lognf_z])
orr, p = firth_plr_test(X_full, det, drop_idx=1)
print(f"Firth adjusted (for panel size), extreme-age: OR={orr:.3f}  PLR-p={p:.4f}   (paper's reported Firth p = 0.5179)")

# unadjusted (no panel size)
X_u = np.column_stack([np.ones(len(det)), ext])
orr_u, p_u = firth_plr_test(X_u, det, drop_idx=1)
print(f"Firth unadjusted, extreme-age: OR={orr_u:.3f}  PLR-p={p_u:.4f}   (paper's reported Firth unadj p = 0.0064, OR=0.460)")
