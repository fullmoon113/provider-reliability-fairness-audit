from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = ROOT / 'output'
"""Fig. 2: detection rate and median panel size by provider patient-age quartile (class-weighted logistic regression)."""
import sys, numpy as np
sys.path.insert(0, str(HERE))
from audit_common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
pf = load(); fm = pf.Fraud.values == 1; n = pf.TotalUniqueBene.values.astype(float); age = pf.AvgPatientAge.values
oof = oof_lr(pf, pf.IsExtremeAge.values, strategy='class_weight')
qi = np.digitize(age, np.quantile(age, [.25, .5, .75]), right=True)
rate = [oof[fm & (qi == k)].mean() for k in range(4)]; nf = [int((fm & (qi == k)).sum()) for k in range(4)]
med_f = [np.median(n[fm & (qi == k)]) for k in range(4)]; med_a = [np.median(n[qi == k]) for k in range(4)]
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'font.family': 'DejaVu Sans'})
col = ['#b5483a', '#3f6aa6', '#3f6aa6', '#b5483a']; lab = ['Q1\n(youngest)', 'Q2', 'Q3', 'Q4\n(oldest)']
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
b = ax[0].bar(range(4), rate, color=col, edgecolor='black', linewidth=.8)
for i, (v, k) in enumerate(zip(rate, nf)): ax[0].text(i, v+.015, f'{v:.3f}\n(n={k})', ha='center', va='bottom', fontsize=8)
ax[0].set_ylim(0, 1.18); ax[0].set_ylabel('Detection rate'); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(lab)
ax[0].set_xlabel('Provider mean patient age, quartile'); ax[0].text(.02, .97, '(a)', transform=ax[0].transAxes, va='top', fontweight='bold')
ax[1].bar(np.arange(4)-.2, med_f, .4, color=col, edgecolor='black', linewidth=.8, label='fraud-labelled providers')
ax[1].bar(np.arange(4)+.2, med_a, .4, color='white', edgecolor=col, linewidth=1.3, hatch='///', label='all providers')
for i in range(4):
    ax[1].text(i-.2, med_f[i]+3, f'{med_f[i]:.0f}', ha='center', fontsize=8); ax[1].text(i+.2, med_a[i]+3, f'{med_a[i]:.0f}', ha='center', fontsize=8)
ax[1].set_ylim(0, 300); ax[1].set_ylabel('Median panel size'); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(lab)
ax[1].set_xlabel('Provider mean patient age, quartile'); ax[1].text(.90, .97, '(b)', transform=ax[1].transAxes, va='top', fontweight='bold')
ax[1].legend(fontsize=7.5, loc='upper left', frameon=False, bbox_to_anchor=(0.08, .98))
plt.tight_layout(); plt.savefig(OUT/'fig2_v3.png', dpi=300, bbox_inches='tight')
print(rate, nf, med_f, med_a)
