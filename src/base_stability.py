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
pf=pd.read_csv(str(DATA/'provider_features_FINAL.csv'))
F13=['IP_UniqueBene','IP_AvgReimbursed','IP_AvgLOS','IP_AvgDiagCodes','IP_UniqueAttPhysician',
 'OP_AvgReimbursed','OP_AvgDiagCodes','OP_UniqueAttPhysician','TotalUniqueBene',
 'FemaleRatio','AvgChronicCondCount','DeceasedRatio','ClaimsPerBene']
y=pf.Fraud.values; folds=pf.fold.values; sens=pf.IsExtremeAge.values; X=pf[F13].values
def ss(Xa,ya,sa,seed):
    Xo,yo=[],[]
    for g in np.unique(sa):
        m=sa==g; Xg,yg=Xa[m],ya[m]; k=min(5,max(1,(yg==1).sum()-1))
        Xb,yb=SMOTE(random_state=seed,k_neighbors=k).fit_resample(Xg,yg)
        Xo.append(Xb); yo.append(yb)
    return np.vstack(Xo),np.concatenate(yo)
def run(mode,seed):
    o=np.zeros(len(pf))
    for f in range(5):
        tr=folds!=f; va=folds==f
        sc=StandardScaler(); Xtr=sc.fit_transform(X[tr]); Xva=sc.transform(X[va])
        if mode=='strat': Xb,yb=ss(Xtr,y[tr],sens[tr],seed); m=LogisticRegression(max_iter=500,random_state=seed).fit(Xb,yb)
        elif mode=='plain': Xb,yb=SMOTE(random_state=seed).fit_resample(Xtr,y[tr]); m=LogisticRegression(max_iter=500,random_state=seed).fit(Xb,yb)
        else: m=LogisticRegression(max_iter=3000,class_weight='balanced',random_state=seed).fit(Xtr,y[tr])
        o[va]=m.predict(Xva)
    fm=y==1
    r=sm.Logit(o[fm],sm.add_constant(sens[fm].astype(float))).fit(disp=0)
    return np.exp(r.params[1]), r.pvalues[1]
print("기본 결과(미보정 OR)의 리샘플링·시드 민감도\n")
print(f"{'방식':22s} " + " ".join(f"seed{s:<8}" for s in [42,1,7,123,2024]))
for mode,lbl in [('strat','층화 SMOTE (논문)'),('plain','일반 SMOTE'),('cw','class_weight')]:
    cells=[]
    for s in [42,1,7,123,2024]:
        o,p=run(mode,s); cells.append(f"{o:.3f}{'*' if p<.05 else ' '}")
    print(f"{lbl:22s} " + "    ".join(cells))
