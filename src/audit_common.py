"""Shared helpers for the extended analyses (Prop. 1 check, ablation, permutation null, power study).
Reproduces the reference audit of the paper: logistic regression, out-of-fold predictions, detection among
fraud-labeled providers, extreme-age partition (outer quartiles) with and without log panel size."""
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

FEATS = ['IP_UniqueBene', 'IP_AvgReimbursed', 'IP_AvgLOS', 'IP_AvgDiagCodes',
         'IP_UniqueAttPhysician', 'OP_AvgReimbursed', 'OP_AvgDiagCodes',
         'OP_UniqueAttPhysician', 'TotalUniqueBene', 'FemaleRatio',
         'AvgChronicCondCount', 'DeceasedRatio', 'ClaimsPerBene']
SIZE_FEATS = ['IP_UniqueBene', 'TotalUniqueBene', 'ClaimsPerBene']   # features that carry panel size directly

def load():
    return pd.read_csv(DATA / 'provider_features.csv')

def outer(a):
    q = np.quantile(a, [.25, .75]); return ((a <= q[0]) | (a > q[1])).astype(int)

def stratified_smote(Xa, ya, sa, seed=42):
    Xo, yo, so = [], [], []
    for g in np.unique(sa):
        m = sa == g
        k = min(5, max(1, (ya[m] == 1).sum() - 1))
        Xb, yb = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xa[m], ya[m])
        Xo.append(Xb); yo.append(yb); so.append(np.full(len(yb), g))
    return np.vstack(Xo), np.concatenate(yo), np.concatenate(so)

def make_model(name, seed=42, cw=None):
    if name == 'LR':
        return LogisticRegression(max_iter=500, random_state=seed, class_weight=cw)
    if name == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, class_weight=cw)
    if name == 'XGB':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric='logloss',
                             random_state=seed, n_jobs=-1)
    raise ValueError(name)

def oof_lr(pf, sens, feats=FEATS, strategy='group_smote', seed=42, model='LR'):
    """Out-of-fold hard predictions of a logistic regression.
    strategy: group_smote (SMOTE within each sensitive group), pooled_smote, class_weight, none."""
    X = pf[feats].values; y = pf['Fraud'].values; folds = pf['fold'].values
    oof = np.zeros(len(pf))
    for f in range(5):
        tr, va = folds != f, folds == f
        sc = StandardScaler(); Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
        ytr = y[tr]; cw = None
        if strategy == 'group_smote':
            Xb, yb, _ = stratified_smote(Xtr, ytr, sens[tr], seed)
        elif strategy == 'pooled_smote':
            k = min(5, max(1, (ytr == 1).sum() - 1))
            Xb, yb = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xtr, ytr)
        elif strategy == 'class_weight':
            Xb, yb, cw = Xtr, ytr, 'balanced'
        else:
            Xb, yb = Xtr, ytr
        m = make_model(model, seed, cw)
        if model == 'XGB' and cw == 'balanced':
            m.set_params(scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1))
        m.fit(Xb, yb)
        oof[va] = m.predict(Xva)
    return oof

def logit_or(d, cols):
    X = sm.add_constant(np.column_stack(cols), has_constant='add')
    try:
        r = sm.Logit(d, X).fit(disp=0, maxiter=200)
        ci = np.exp(r.conf_int()[1])
        return np.exp(r.params[1]), r.pvalues[1], ci[0], ci[1]
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

def audit(det, g, logn):
    """Unadjusted and panel-size-adjusted odds ratio of detection for group indicator g (fraud-labeled providers)."""
    return logit_or(det, [g]), logit_or(det, [g, logn])
