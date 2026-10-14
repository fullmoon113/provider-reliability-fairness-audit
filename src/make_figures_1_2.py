from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

import warnings; warnings.filterwarnings("ignore")
exec(open(HERE/'fpr_class_weighting.py').read().split("oof0 = oof_smote(ext0)")[0])
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
oof0=oof_smote(ext0)
q=['Q1','Q2','Q3','Q4']
rec=[oof0[fm][(pf.loc[fm,'age_quartile']==k).values].mean() for k in q]
vol=[pf[pf['age_quartile']==k]['TotalUniqueBene'].median() for k in q]
print([round(r,3) for r in rec], vol)
plt.rcParams.update({'font.size':14,'axes.labelsize':16,'axes.titlesize':17,'xtick.labelsize':14,'ytick.labelsize':14,'figure.dpi':150,'savefig.dpi':300,'font.family':'DejaVu Sans'})
fig,axes=plt.subplots(1,2,figsize=(14,5.5)); cols=['#c44e52','#4c72b0','#4c72b0','#c44e52']
ax=axes[0]; bars=ax.bar(q,rec,color=cols,edgecolor='black',linewidth=1.2)
ax.set_ylabel('Detection Rate (Recall)',fontweight='bold'); ax.set_xlabel('Patient Age Quartile of Provider',fontweight='bold')
ax.set_title('(a) Apparent Age-Based Disparity',fontweight='bold',pad=14); ax.set_ylim(0,1.12)
for b,v in zip(bars,rec): ax.text(b.get_x()+b.get_width()/2,v+0.02,f'{v:.3f}',ha='center',fontsize=13,fontweight='bold')
ax.grid(axis='y',alpha=0.3)
ax=axes[1]; bars=ax.bar(q,vol,color=cols,edgecolor='black',linewidth=1.2)
ax.set_ylabel('Median Panel Size\n(Unique Beneficiaries)',fontweight='bold'); ax.set_xlabel('Patient Age Quartile of Provider',fontweight='bold')
ax.set_title('(b) Confounding Factor: Provider Panel Size',fontweight='bold',pad=14); ax.set_ylim(0,max(vol)*1.22)
for b,v in zip(bars,vol): ax.text(b.get_x()+b.get_width()/2,v+1.2,f'{v:.0f}',ha='center',fontsize=13,fontweight='bold')
ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(str(OUT/'Fig2_confounding.png'),bbox_inches='tight'); plt.close()

# ---- DAG ----
plt.rcParams.update({'font.size':13})
fig,ax=plt.subplots(figsize=(11.5,6.8)); ax.set_xlim(0,11.5); ax.set_ylim(-0.3,7.2); ax.axis('off')
NODE={'Astar':(1.5,5.0),'V':(5.6,5.0),'Y':(9.6,5.0),'A':(1.5,2.7),'L':(9.6,2.7),'D':(5.6,0.85)}
LBL={'Astar':'A*','V':'V','Y':'Y','A':'A','L':'L','D':'D'}
SUB={'Astar':'true patient\nage mix','V':'panel size','Y':'true fraud','A':'measured\nage mix','L':'fraud label\n(analysis conditions on $L=1$)','D':'detected'}
COL={'Astar':'#dfe6f2','V':'#c44e52','Y':'#dfe6f2','A':'#4c72b0','L':'#f2d9a0','D':'#4c72b0'}; FG={'V':'white','A':'white','D':'white'}
ABOVE={'Astar','V','Y'}
for k,(x,y0) in NODE.items():
    if k=='L': ax.add_patch(Rectangle((x-.60,y0-.44),1.20,.88,fc=COL[k],ec='black',lw=2.4,zorder=3))
    else: ax.add_patch(Circle((x,y0),.52,fc=COL[k],ec='black',lw=1.8,zorder=3))
    ax.text(x,y0,LBL[k],ha='center',va='center',fontsize=20,fontweight='bold',style='italic',color=FG.get(k,'black'),zorder=4)
    if k in ABOVE: ax.text(x,y0+0.72,SUB[k],ha='center',va='bottom',fontsize=11.5,color='#333',zorder=4)
    else: ax.text(x,y0-(.92 if k=='L' else .88),SUB[k],ha='center',va='top',fontsize=11.5,color='#333',zorder=4)
def arrow(a,b,color='black',ls='-',lw=1.9):
    x1,y1=NODE[a]; x2,y2=NODE[b]
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=19,lw=lw,color=color,linestyle=ls,shrinkA=32,shrinkB=32,zorder=2))
arrow('Astar','A'); arrow('V','A',color='#c44e52',lw=2.6); arrow('V','D',color='#c44e52',lw=2.6)
arrow('V','L',color='#8c8c8c',ls=(0,(5,3))); arrow('Y','L'); arrow('A','D',color='#1f3864',ls=(0,(4,3)),lw=2.2)
bb=dict(fc='white',ec='none',pad=2.5)
ax.text(4.1,2.8,'measurement precision\n'+r'Var$(A \mid A^*, V) = \sigma_w^2 / V$',ha='center',va='center',fontsize=11.5,color='#c44e52',fontweight='bold',bbox=bb,zorder=5)
ax.text(6.05,2.95,'behavioural\nsignal',ha='left',fontsize=11.5,color='#c44e52',fontweight='bold',bbox=bb,zorder=5)
ax.text(7.85,4.30,'investigation\nintensity ?',ha='center',fontsize=11,color='#666',style='italic',bbox=bb,zorder=5)
ax.text(3.45,1.55,'effect under test',ha='center',va='center',fontsize=11.5,color='#1f3864',fontweight='bold',rotation=-27,bbox=bb,zorder=5)
ax.text(5.75,6.85,r'Backdoor path  $A \leftarrow V \rightarrow D$  is open unless $V$ is adjusted for',ha='center',fontsize=13.5,color='#c44e52',fontweight='bold')
plt.tight_layout(); plt.savefig(str(OUT/'Fig1_dag.png'),bbox_inches='tight'); plt.close()
