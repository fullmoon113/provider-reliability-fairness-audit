from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)


import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':14,'axes.labelsize':15,'axes.titlesize':16,
 'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':12.5,
 'figure.dpi':150,'savefig.dpi':300,'font.family':'DejaVu Sans'})
BINS=[0,25,50,100,200,500,1e9]; LAB=['\u226425','26-50','51-100','101-200','201-500','>500']
a=pd.read_csv(DATA/'cms_2018_2019_merged.csv.gz',low_memory=False); a.columns=['NPI','v1','v2','age1','age2','spec']
b=pd.read_csv(DATA/'cms_2021_2022_merged.csv.gz',low_memory=False)
b=b.rename(columns={'Tot_Benes_21':'v1','Tot_Benes_22':'v2','Bene_Avg_Age_21':'age1','Bene_Avg_Age_22':'age2'})
def sd_by(d):
    d=d.copy(); d['vb']=pd.cut(d[['v1','v2']].min(1),BINS,labels=LAB)
    return d.groupby('vb',observed=True).apply(lambda s:(s.age2-s.age1).std())
s1,s2=sd_by(a),sd_by(b)
fig,ax=plt.subplots(1,2,figsize=(13.5,5))
x=np.arange(len(LAB)); w=0.38
A=ax[0]
A.bar(x-w/2,[s1[l] for l in LAB],w,label='2018 \u2192 2019  (pre-pandemic)',color='#4c72b0',edgecolor='black',lw=1)
A.bar(x+w/2,[s2[l] for l in LAB],w,label='2021 \u2192 2022',color='#c44e52',edgecolor='black',lw=1)
A.set_xticks(x); A.set_xticklabels(LAB,rotation=18)
A.set_xlabel('Claim volume (unique beneficiaries)',fontweight='bold')
A.set_ylabel('SD of year-over-year change\nin mean patient age (years)',fontweight='bold')
A.set_title('(a) Instability falls fivefold with size,\nidentically in two independent periods',fontweight='bold')
A.legend(); A.grid(axis='y',alpha=.3)
for i,l in enumerate(LAB):
    A.text(i-w/2,s1[l]+.07,'%.2f'%s1[l],ha='center',fontsize=10.5)
    A.text(i+w/2,s2[l]+.07,'%.2f'%s2[l],ha='center',fontsize=10.5)
SP=['Nurse Practitioner','Internal Medicine','Family Practice','Physician Assistant',
    'Emergency Medicine','Anesthesiology','Certified Registered Nurse Anesthetist (CRNA)']
SH=['Nurse Pract.','Internal Med.','Family Pract.','Phys. Asst.','Emerg. Med.','Anesthesiol.','CRNA']
lo=[];hi=[]
for t in SP:
    s=b[b.Rndrng_Prvdr_Type_22==t]; v=s[['v1','v2']].min(1)
    l=s[v<=v.quantile(.25)]; h=s[v>=v.quantile(.75)]
    lo.append(np.corrcoef(l.age1,l.age2)[0,1]); hi.append(np.corrcoef(h.age1,h.age2)[0,1])
B=ax[1]; y=np.arange(len(SP))
B.barh(y-w/2,lo,w,label='Low claim volume (bottom quartile)',color='#c44e52',edgecolor='black',lw=1)
B.barh(y+w/2,hi,w,label='High claim volume (top quartile)',color='#4c72b0',edgecolor='black',lw=1)
B.set_yticks(y); B.set_yticklabels(SH,fontsize=12); B.set_ylim(len(SP)-0.5,-0.5)
B.set_xlabel('Year-over-year correlation of mean patient age',fontweight='bold')
B.set_xlim(0,1.02)
B.set_title('(b) The gradient holds within\nevery specialty examined',fontweight='bold')
B.legend(loc='upper center', bbox_to_anchor=(0.5,-0.13), ncol=2, frameon=False); B.grid(axis='x',alpha=.3)
plt.tight_layout(); plt.savefig(str(OUT/'Fig6_cms.png'),bbox_inches='tight')
print('saved')
