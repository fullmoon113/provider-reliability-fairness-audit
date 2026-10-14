from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

import warnings; warnings.filterwarnings("ignore")
exec(open(HERE/'fpr_class_weighting.py').read().split("oof0 = oof_smote(ext0)")[0])
def oof_pooled(seed=42):
    oof=np.zeros(len(pf))
    for f in range(5):
        tr=folds!=f; va=folds==f
        sc=StandardScaler(); Xtr=sc.fit_transform(X[tr]); Xva=sc.transform(X[va])
        Xb,yb=SMOTE(random_state=seed,k_neighbors=5).fit_resample(Xtr,y[tr])
        oof[va]=LR(max_iter=500,random_state=seed).fit(Xb,yb).predict(Xva)
    return oof
def oof_none(seed=42):
    oof=np.zeros(len(pf))
    for f in range(5):
        tr=folds!=f; va=folds==f
        sc=StandardScaler(); Xtr=sc.fit_transform(X[tr]); Xva=sc.transform(X[va])
        oof[va]=LR(max_iter=500,random_state=seed).fit(Xtr,y[tr]).predict(Xva)
    return oof
for nm,o in (('pooled SMOTE',oof_pooled()),('no resampling',oof_none())):
    print(nm,'shrunken fraud',fmt(fit(o,fm,ext1)),'| adj',fmt(fit(o,fm,ext1,True)))
