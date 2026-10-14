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
F17=F13+['AvgPatientAge','MinorityRatio','IP_ClaimCount','OP_ClaimCount']
y=pf.Fraud.values; folds=pf.fold.values; sens=pf.IsExtremeAge.values
n=pf.TotalUniqueBene.values.astype(float); a=pf.AvgPatientAge.values
logn=np.log1p(n); sw2=12.7275**2

def smote_strat(Xa,ya,sa,seed=42):
    Xo,yo=[],[]
    for g in np.unique(sa):
        m=sa==g; Xg,yg=Xa[m],ya[m]
        k=min(5,max(1,(yg==1).sum()-1))
        Xb,yb=SMOTE(random_state=seed,k_neighbors=k).fit_resample(Xg,yg)
        Xo.append(Xb); yo.append(yb)
    return np.vstack(Xo),np.concatenate(yo)

def fit_oof(feats, mode):
    X=pf[feats].fillna(0).values; oof=np.zeros(len(pf))
    for f in range(5):
        tr=folds!=f; va=folds==f
        sc=StandardScaler(); Xtr=sc.fit_transform(X[tr]); Xva=sc.transform(X[va])
        if mode=='smote_strat':
            Xb,yb=smote_strat(Xtr,y[tr],sens[tr])
            lr=LogisticRegression(max_iter=500,random_state=42).fit(Xb,yb)
        elif mode=='smote_plain':
            Xb,yb=SMOTE(random_state=42).fit_resample(Xtr,y[tr])
            lr=LogisticRegression(max_iter=500,random_state=42).fit(Xb,yb)
        else:
            lr=LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42).fit(Xtr,y[tr])
        oof[va]=lr.predict(Xva)
    return oof

def sweep(oof):
    fm=y==1; det=oof[fm]; out={}
    r=sm.Logit(det,sm.add_constant(sens[fm].astype(float))).fit(disp=0)
    out['base']=(np.exp(r.params[1]), r.pvalues[1])
    for st2 in [0.247,0.5,1.0,1.97]:
        rel=st2/(st2+sw2/n); shr=a.mean()+rel*(a-a.mean())
        q=np.quantile(shr,[.25,.75]); ext=((shr<=q[0])|(shr>q[1])).astype(float)
        rr=sm.Logit(det,sm.add_constant(ext[fm])).fit(disp=0)
        out[st2]=(np.exp(rr.params[1]), rr.pvalues[1])
    return out

PIPES=[('논문본(13F+층화SMOTE)',F13,'smote_strat'),
       ('13F + 일반SMOTE',F13,'smote_plain'),
       ('13F + class_weight',F13,'cw'),
       ('17F + 층화SMOTE',F17,'smote_strat'),
       ('17F + class_weight',F17,'cw')]
print(f"{'파이프라인':24s} {'미보정OR':>9s} | " + " | ".join(f"σ²={s:<5}" for s in ['0.247','0.5','1.0','1.97']))
print("-"*94)
for lbl,fs,md in PIPES:
    o=sweep(fit_oof(fs,md))
    b=o['base']
    cells=" | ".join(f"{o[s][0]:5.2f}{'*' if o[s][1]<.05 else ' '}({o[s][1]:.3f})" for s in [0.247,0.5,1.0,1.97])
    print(f"{lbl:24s} {b[0]:6.3f}({b[1]:.4f}) | {cells}")
