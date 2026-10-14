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
y = pf.Fraud.values; folds = pf.fold.values; sens = pf.IsExtremeAge.values
n = pf.TotalUniqueBene.values.astype(float)
X = pf[F13].values
logn = np.log1p(n)

def smote_strat(Xa, ya, sa, seed):
    Xo, yo = [], []
    for g in np.unique(sa):
        m = sa == g; Xg, yg = Xa[m], ya[m]
        k = min(5, max(1, (yg == 1).sum() - 1))
        Xb, yb = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xg, yg)
        Xo.append(Xb); yo.append(yb)
    return np.vstack(Xo), np.concatenate(yo)

def run(seed):
    oof = np.zeros(len(pf))
    for f in range(5):
        tr = folds != f; va = folds == f
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        Xb, yb = smote_strat(Xtr, y[tr], sens[tr], seed=seed)
        lr = LogisticRegression(max_iter=500, random_state=seed).fit(Xb, yb)
        oof[va] = lr.predict(Xva)
    fm = y == 1
    det = oof[fm]; ext = sens[fm].astype(float)
    lognf = logn[fm]

    Xu = sm.add_constant(ext)
    ru = sm.Logit(det, Xu).fit(disp=0)
    Xa_ = sm.add_constant(np.column_stack([ext, lognf]))
    ra = sm.Logit(det, Xa_).fit(disp=0)
    ppanel = ra.pvalues[2]
    return np.exp(ru.params[1]), ru.pvalues[1], np.exp(ra.params[1]), ra.pvalues[1], ppanel

print(f"{'Seed':6s} {'Unadj OR':>9s} {'Unadj p':>9s} {'Adj OR':>8s} {'Adj p':>9s} {'PanelSize p':>12s}")
for seed in [42, 1, 7, 99, 2024]:
    orU, pU, orA, pA, pP = run(seed)
    print(f"{seed:<6d} {orU:9.3f} {pU:9.4f} {orA:8.3f} {pA:9.4f} {pP:12.2e}")
