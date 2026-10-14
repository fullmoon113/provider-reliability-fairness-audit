from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

import warnings; warnings.filterwarnings("ignore")
exec(open(HERE/'refit_sweep.py').read().split('print("Row 4')[0])
from sklearn.linear_model import LogisticRegression as LR
fm = y==1
logn = np.log1p(n)

def oof_smote(ext, seed=42):
    oof = np.zeros(len(pf))
    for f in range(5):
        tr = folds!=f; va = folds==f
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        Xb,yb = smote_strat(Xtr,y[tr],ext[tr],seed=seed)
        oof[va] = LR(max_iter=500,random_state=seed).fit(Xb,yb).predict(Xva)
    return oof

def oof_cw(seed=42):
    oof = np.zeros(len(pf))
    for f in range(5):
        tr = folds!=f; va = folds==f
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        oof[va] = LR(max_iter=500,random_state=seed,class_weight='balanced').fit(Xtr,y[tr]).predict(Xva)
    return oof

def fit(outcome, mask, ext, adj=False):
    cols=[ext[mask].astype(float)]
    if adj: cols.append(logn[mask])
    Xd = sm.add_constant(np.column_stack(cols))
    r = sm.Logit(outcome[mask], Xd).fit(disp=0)
    ci = np.exp(r.conf_int()[1])
    return np.exp(r.params[1]), ci[0], ci[1], r.pvalues[1]

fmt=lambda t: f"OR {t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}] p={t[3]:.4g}"
ext0 = pf.IsExtremeAge.values.astype(float)
rel = 0.247/(0.247+sw2/n); shr = a.mean()+rel*(a-a.mean()); q=np.quantile(shr,[.25,.75])
ext1 = ((shr<=q[0])|(shr>q[1])).astype(float)

oof0 = oof_smote(ext0)
print('=== check detection among fraud, observed ext (SMOTE):', fmt(fit(oof0,fm,ext0)), '| adj', fmt(fit(oof0,fm,ext0,True)))
neg = y==0
print('=== NEGATIVES n=',neg.sum(),' flagged rate: ext',oof0[neg&(ext0==1)].mean(),' mid',oof0[neg&(ext0==0)].mean(),' overall',oof0[neg].mean())
print('FPR unadj', fmt(fit(oof0,neg,ext0)))
print('FPR adj  ', fmt(fit(oof0,neg,ext0,True)))
# flag rate among all providers
allm = np.ones(len(pf),bool)
print('flag rate all unadj', fmt(fit(oof0,allm,ext0)), '| adj', fmt(fit(oof0,allm,ext0,True)))

oofc = oof_cw()
print('=== class weighting, observed ext, fraud:', fmt(fit(oofc,fm,ext0)),'| adj',fmt(fit(oofc,fm,ext0,True)))
print('    class weighting, FPR observed ext:', fmt(fit(oofc,neg,ext0)),'| adj',fmt(fit(oofc,neg,ext0,True)))
print('=== class weighting (partition independent), SHRUNKEN ext, fraud:', fmt(fit(oofc,fm,ext1)),'| adj',fmt(fit(oofc,fm,ext1,True)))
print('    class weighting, SHRUNKEN, FPR:', fmt(fit(oofc,neg,ext1)),'| adj',fmt(fit(oofc,neg,ext1,True)))
oof1 = oof_smote(ext1)
print('=== SMOTE refit on shrunken, fraud:', fmt(fit(oof1,fm,ext1)),'| adj panel size',fmt(fit(oof1,fm,ext1,True)))
print('    SMOTE refit shrunken, FPR:', fmt(fit(oof1,neg,ext1)),'| adj',fmt(fit(oof1,neg,ext1,True)))
# mismatched (row 2): model fit stratified on observed, evaluated on shrunken
print('=== row2 (oof0 evaluated on shrunken):', fmt(fit(oof0,fm,ext1)),'| adj',fmt(fit(oof0,fm,ext1,True)))
