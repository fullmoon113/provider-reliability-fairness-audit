from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = ROOT / 'output'; DATA = ROOT / 'data'
"""Fig. 4: null simulation (true age effect zero), three panels: (a) unadjusted OR, with the observed estimate, the analytic
prediction of Proposition 1 and, when available, the permutation null; (b) panel-size-adjusted OR; (c) significant-disparity rate
against the true between-provider SD."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r = pd.read_csv(DATA / 'null_sim_5000reps.csv'); A = json.load(open(OUT / 'sim5k_panelA_cw.json'))
A['A|0.9'] = dict(fpr_u=(r.up < .05).mean()*100, fpr_a=(r.ap < .05).mean()*100)
perm = OUT / 'permutation_null.csv'; perm = pd.read_csv(perm) if perm.exists() else None
OBS_U, OBS_A, PRED = 0.477, 1.018, 0.431
plt.rcParams.update({'font.size': 9})
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
a = ax[0]; bins = np.linspace(0.1, 1.6, 40)
a.hist(r.uOR, bins=bins, color='#b5483a', alpha=.85, edgecolor='black', linewidth=.4, label='Null simulation')
if perm is not None:
    a.hist(perm.uOR.clip(0.1, 1.6), bins=bins, histtype='step', color='black', lw=1.1, label='Re-linked ages')
a.axvline(OBS_U, color='#1f3864', lw=2, label='Observed'); a.axvline(PRED, color='#1f3864', lw=1.4, ls='--', label='Prop. 1')
a.axvline(1, color='grey', ls=':', lw=1); a.set_xlabel('Unadjusted odds ratio'); a.set_ylabel('Replications'); a.legend(fontsize=6.5, frameon=False, loc='upper right')
a.text(.02, .97, '(a)', transform=a.transAxes, va='top', fontweight='bold')
b = ax[1]; b.hist(r.aOR, bins=np.linspace(0.3, 2.5, 40), color='#3f6aa6', alpha=.85, edgecolor='black', linewidth=.4)
b.axvline(OBS_A, color='#1f3864', lw=2); b.axvline(1, color='grey', ls=':', lw=1); b.set_xlabel('Panel-size-adjusted odds ratio'); b.set_yticklabels([])
b.text(.02, .97, '(b)', transform=b.transAxes, va='top', fontweight='bold')
c = ax[2]; sds = sorted(float(k.split('|')[1]) for k in A if k.startswith('A|'))
u = [A[f'A|{s}']['fpr_u'] for s in sds]; ad = [A[f'A|{s}']['fpr_a'] for s in sds]
c.plot(sds, u, 'o-', color='#b5483a', ms=4, lw=1.5, label='Unadjusted'); c.plot(sds, ad, 's-', color='#3f6aa6', ms=4, lw=1.5, label='Adjusted')
c.axhline(5, color='grey', ls=':', lw=1); c.axvline(0.90, color='black', ls='--', lw=.8); c.annotate('primary dataset', xy=(0.9, 62), xytext=(2.2, 72), fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=.6, color='black'))
c.set_xlabel('True between-provider SD (years)'); c.set_ylabel('Significant at 0.05 (%)'); c.set_ylim(0, 102); c.legend(fontsize=6.5, frameon=False)
c.text(.45, .97, '(c)', transform=c.transAxes, va='top', fontweight='bold')
plt.tight_layout(w_pad=0.6); plt.savefig(OUT / 'fig3.png', dpi=300, bbox_inches='tight')
print('observed percentile', (r.uOR <= OBS_U).mean() * 100)
