from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score

# 논문용 폰트 크기 설정 (가독성 확보)
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 17,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13,
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans'
})

pf = pd.read_csv(DATA/"provider_features_FINAL.csv")
feats = ['IP_UniqueBene','IP_AvgReimbursed','IP_AvgLOS','IP_AvgDiagCodes','IP_UniqueAttPhysician',
         'OP_AvgReimbursed','OP_AvgDiagCodes','OP_UniqueAttPhysician','TotalUniqueBene',
         'FemaleRatio','AvgChronicCondCount','DeceasedRatio','ClaimsPerBene']
X=pf[feats].values; y=pf['Fraud'].values; folds=pf['fold'].values; sens=pf['IsExtremeAge'].values

def strat_smote(Xa,ya,sa,seed=42):
    Xo,yo=[],[]
    for g in np.unique(sa):
        m=sa==g; Xg,yg=Xa[m],ya[m]
        k=min(5,max(1,(yg==1).sum()-1))
        Xb,yb=SMOTE(random_state=seed,k_neighbors=k).fit_resample(Xg,yg)
        Xo.append(Xb); yo.append(yb)
    return np.vstack(Xo),np.concatenate(yo)

oof=np.zeros(len(pf))
for f in range(5):
    tr=folds!=f; va=folds==f
    sc=StandardScaler(); Xtr=sc.fit_transform(X[tr]); Xva=sc.transform(X[va])
    Xb,yb=strat_smote(Xtr,y[tr],sens[tr])
    lr=LogisticRegression(max_iter=500,random_state=42); lr.fit(Xb,yb)
    oof[va]=lr.predict(Xva)

# ===== Figure 1: 연령 사분위별 탐지율 + 청구량 (교란 시각화) =====
fig, axes = plt.subplots(1, 2, figsize=(14,5.5))
fm = y==1
q_order=['Q1','Q2','Q3','Q4']
recalls=[oof[fm][(pf.loc[fm,'age_quartile']==q).values].mean() for q in q_order]
volumes=[pf[pf['age_quartile']==q]['TotalUniqueBene'].median() for q in q_order]

ax=axes[0]
bars=ax.bar(q_order, recalls, color=['#c44e52','#4c72b0','#4c72b0','#c44e52'], edgecolor='black', linewidth=1.2)
ax.set_ylabel('Detection Rate (Recall)', fontweight='bold')
ax.set_xlabel('Patient Age Quartile of Provider', fontweight='bold')
ax.set_title('(a) Apparent Age-Based Disparity', fontweight='bold')
ax.set_ylim(0,1.0)
for b,v in zip(bars,recalls): ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.3f}', ha='center', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

ax=axes[1]
bars=ax.bar(q_order, volumes, color=['#c44e52','#4c72b0','#4c72b0','#c44e52'], edgecolor='black', linewidth=1.2)
ax.set_ylabel('Median Claim Volume\n(Unique Beneficiaries)', fontweight='bold')
ax.set_xlabel('Patient Age Quartile of Provider', fontweight='bold')
ax.set_title('(b) Confounding Factor: Provider Volume', fontweight='bold')
for b,v in zip(bars,volumes): ax.text(b.get_x()+b.get_width()/2, v+2, f'{v:.0f}', ha='center', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
# (superseded by make_figures_1_2.py) plt.savefig(...)
plt.close()
print("Fig1 저장 완료")

# ===== Figure 2: 통제변수별 나이효과 p-value (특이성) =====
det=oof[fm]; ext=sens[fm]
controls={'Claim Volume\n(log)':np.log1p(pf.loc[fm,'TotalUniqueBene'].values),
          'Chronic\nConditions':pf.loc[fm,'AvgChronicCondCount'].values,
          'Female\nRatio':pf.loc[fm,'FemaleRatio'].values,
          'Deceased\nRatio':pf.loc[fm,'DeceasedRatio'].values,
          'Claims per\nBeneficiary':pf.loc[fm,'ClaimsPerBene'].values,
          'Inpatient\nBeneficiaries':pf.loc[fm,'IP_UniqueBene'].values,
          'Avg Length\nof Stay':pf.loc[fm,'IP_AvgLOS'].values}
pvals=[]
for cn,cv in controls.items():
    m=sm.Logit(det,sm.add_constant(np.column_stack([ext,cv]))).fit(disp=0)
    pvals.append(m.pvalues[1])

fig,ax=plt.subplots(figsize=(11,6))
colors=['#c44e52' if p>=0.05 else '#4c72b0' for p in pvals]
bars=ax.bar(range(len(pvals)), pvals, color=colors, edgecolor='black', linewidth=1.2)
ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='Significance threshold (p=0.05)')
ax.set_xticks(range(len(controls))); ax.set_xticklabels(controls.keys(), fontsize=12)
ax.set_ylabel('p-value of Age Effect\n(after controlling for variable)', fontweight='bold')
ax.set_title('Only Claim Volume Eliminates the Age Effect', fontweight='bold', fontsize=17)
ax.legend(fontsize=13)
for b,p in zip(bars,pvals): ax.text(b.get_x()+b.get_width()/2, p+0.015, f'{p:.3f}', ha='center', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUT/'Fig3_specificity.png'), bbox_inches='tight')
plt.close()
print("Fig2 저장 완료")
