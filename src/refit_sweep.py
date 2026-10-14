from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

import pandas as pd, numpy as np, statsmodels.api as sm, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

pf = pd.read_csv(str(DATA/'provider_features_FINAL.csv'))
F13 = ['IP_UniqueBene','IP_AvgReimbursed','IP_AvgLOS','IP_AvgDiagCodes','IP_UniqueAttPhysician',
 'OP_AvgReimbursed','OP_AvgDiagCodes','OP_UniqueAttPhysician','TotalUniqueBene',
 'FemaleRatio','AvgChronicCondCount','DeceasedRatio','ClaimsPerBene']
y = pf.Fraud.values; folds = pf.fold.values
n = pf.TotalUniqueBene.values.astype(float); a = pf.AvgPatientAge.values
X = pf[F13].values
sw2 = 12.7275**2

def smote_strat(Xa, ya, sa, seed=42):
    Xo, yo = [], []
    for g in np.unique(sa):
        m = sa == g; Xg, yg = Xa[m], ya[m]
        k = min(5, max(1, (yg == 1).sum() - 1))
        Xb, yb = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xg, yg)
        Xo.append(Xb); yo.append(yb)
    return np.vstack(Xo), np.concatenate(yo)

def refit_with(ext, seed=42):
    """Refit detection model with resampling stratified on `ext`, evaluate OR on `ext`."""
    oof = np.zeros(len(pf))
    for f in range(5):
        tr = folds != f; va = folds == f
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        Xb, yb = smote_strat(Xtr, y[tr], ext[tr], seed=seed)
        lr = LogisticRegression(max_iter=500, random_state=seed).fit(Xb, yb)
        oof[va] = lr.predict(Xva)
    fm = y == 1
    r = sm.Logit(oof[fm], sm.add_constant(ext[fm].astype(float))).fit(disp=0)
    return np.exp(r.params[1]), r.pvalues[1]

print("Row 4 (shrunken attribute, model refitted with matching stratification)")
print("sweep over sigma^2_true, seed=42\n")
for st2 in [0.247, 0.5, 1.0, 1.97]:
    rel = st2 / (st2 + sw2 / n)
    shr = a.mean() + rel * (a - a.mean())
    q = np.quantile(shr, [.25, .75])
    ext = ((shr <= q[0]) | (shr > q[1])).astype(float)
    orr, p = refit_with(ext)
    print(f"  sigma^2_true={st2:5.3f}  OR={orr:.3f}  p={p:.4f}  {'*' if p<.05 else ''}")
