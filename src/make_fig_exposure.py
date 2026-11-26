from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = ROOT / 'output'; DATA = ROOT / 'data'
"""New figure (M4): expected null odds ratio and significant-disparity rate of an unadjusted audit versus audit size,
by CMS specialty (detection slope gamma1 = 1.213 per SD of log panel size, the slope fitted to the primary dataset)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, str(HERE))
src = open(HERE / 'cms_exposure.py').read().split('GRID = ')[0]
src = src.replace("DATA = ROOT / 'data'", "DATA = ROOT / 'data'")
exec(src)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows = []
for name in ['All providers (pooled)'] + SPEC:
    g = d if name.startswith('All') else d[d.Rndrng_Prvdr_Type_22 == name]
    sw, t2, V, sh = decompose(g); n = g.Tot_Benes_22.values.astype(float); mu = g.Bene_Avg_Age_22.mean()
    out = prop1(n, np.sqrt(t2), sw, mu, 1.213)
    rows.append((name, out))
plt.rcParams.update({'font.size': 9})
fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.9))
ms = np.logspace(2.5, 5.2, 80)
sty = {'All providers (pooled)': ('k', '-', 2.2), SPEC[0]: ('#b5483a', '-', 1.4), SPEC[1]: ('#d98c3f', '-', 1.2), SPEC[2]: ('#3f6aa6', '--', 1.2),
       SPEC[3]: ('#6a9f58', '--', 1.2), SPEC[4]: ('#8a6bb0', ':', 1.4), SPEC[5]: ('#999999', ':', 1.4), SPEC[6]: ('#4c4c4c', '-.', 1.2), SPEC[7]: ('#2b8f9a', '-.', 1.2)}
short = {'All providers (pooled)': 'All providers', SPEC[0]: 'CRNA', SPEC[1]: 'Anesthesiology', SPEC[2]: 'Emergency Med.', SPEC[3]: 'Physician Asst.',
         SPEC[4]: 'Family Practice', SPEC[5]: 'Physical therapist', SPEC[6]: 'Internal Med.', SPEC[7]: 'Nurse Pract.'}
for name, o in rows:
    c, ls, lw = sty[name]; ax.plot(ms, [100*sig_rate(*o, m) for m in ms], color=c, ls=ls, lw=lw, label=short[name])
ax.axhline(5, color='grey', lw=.8); ax.set_xscale('log'); ax.set_ylim(0, 102)
ax.set_xlabel('Fraud-labeled providers audited, $m$'); ax.set_ylabel('Significant disparity under the null (%)')
ax.legend(fontsize=6.5, frameon=False, loc='upper left', ncol=1, handlelength=2.2)
plt.tight_layout(); plt.savefig(OUT / 'fig5.png', dpi=300, bbox_inches='tight')
print([(n_, round(np.exp(o[0]), 3)) for n_, o in rows])
