"""STEP 2 of the Medicare flag-rate audit (paper Section IV-G): observed associations between each group indicator and the flags, compared with the
predictions written by src/flag_rate_predict.py. Writes output/flag_rate_observed.csv, output/flag_rate_summary.txt and fig6.png (Fig. 6).
Usage: python src/flag_rate_observe.py CMS2022_EXTRACT.csv.gz UPDATED.csv"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import StratifiedTable
from flag_rate_common import *
d = load_cohort(sys.argv[1]); fl = pd.read_csv(OUT / 'flag_rate_flags.csv.gz'); pr = pd.read_csv(OUT / 'flag_rate_predictions.csv')
assert (fl.NPI.values == d.NPI.values).all()
d['y'] = fl.y.values; d['T1'] = fl.flag_T1.values; d['T2'] = fl.flag_T2.values
def fit(formula, data, term='G'):
    try:
        r = smf.glm(formula, data=data, family=sm.families.Binomial()).fit()
        b, se = float(r.params[term]), float(r.bse[term])
        if not np.isfinite(b) or abs(b) > 8 or se > 2: return np.nan, np.nan, np.nan   # separation / degenerate fit: reported as not estimable
        return b, se, float(r.pvalues[term])
    except Exception:
        return np.nan, np.nan, np.nan
rows = []
for s in top_specialties(d):
    g = d[(d.spec == s) & (d.y == 0)].copy(); allg = d[d.spec == s]
    for at in ATTRS:
        a_all = allg[at].values; q = np.quantile(a_all, [.25, .75]); med = np.median(a_all)
        g['G'] = ((g[at] <= q[0]) | (g[at] > q[1])).astype(int); g['Gc'] = (g[at] > med).astype(int)
        g['dec'] = pd.qcut(g.n, 10, labels=False, duplicates='drop')
        for det in ('T1', 'T2'):
            g['f'] = g[det]
            u = fit('f ~ G', g); a = fit('f ~ G + cr(logn, df=5)', g); c = fit('f ~ Gc', g, term='Gc')
            tabs = [np.array([[((x.G == 1) & (x.f == 1)).sum(), ((x.G == 1) & (x.f == 0)).sum()], [((x.G == 0) & (x.f == 1)).sum(), ((x.G == 0) & (x.f == 0)).sum()]]) + 0.5 for _, x in g.groupby('dec')]
            mh = np.log(StratifiedTable(tabs).oddsratio_pooled)
            p = pr[(pr.specialty == s) & (pr.attribute == at) & (pr.detector == det)].iloc[0]
            fails = bool(p.M > 0.10)
            hold = bool(fails and abs(u[0] - p.pred_logOR_P1) <= max(2 * u[1], 0.05) and abs(a[0]) < 0.05 and abs(c[0]) < 0.05)
            rows.append(dict(specialty=s, attribute=at, detector=det, m=len(g), M=p.M, fails_stage0=fails, pred_P1=p.pred_logOR_P1, pred_P2=p.pred_logOR_P2,
                             obs_unadj=u[0], se_unadj=u[1], p_unadj=u[2], obs_adj=a[0], se_adj=a[1], p_adj=a[2], obs_center=c[0], se_center=c[1], p_center=c[2],
                             obs_mh=mh, criterion_holds=hold))
    print(s, flush=True)
o = pd.DataFrame(rows); o.to_csv(OUT / 'flag_rate_observed.csv', index=False)
L = []
def emit(x=''): print(x); L.append(x)
emit('Flag-rate table: median over the eight specialties of log odds ratios of being flagged (outer vs middle split); P1 / P2 = predicted (constant mean / mean trend)')
for det in ('T1', 'T2'):
    t = o[o.detector == det]
    emit(f'\n-- detector {det}: flag rate {fl["flag_" + det][fl.y == 0].mean()*100:.1f}% of audited providers')
    r = t.groupby('attribute').agg(audits_failing=('fails_stage0', 'sum'), M=('M', 'median'), pred_P1=('pred_P1', 'median'), pred_P2=('pred_P2', 'median'), obs_unadj=('obs_unadj', 'median'),
                                   obs_adj=('obs_adj', 'median'), obs_center=('obs_center', 'median'), obs_mh=('obs_mh', 'median'),
                                   sig_unadj=('p_unadj', lambda x: (x < .05).sum()), sig_adj=('p_adj', lambda x: (x < .05).sum()), holds=('criterion_holds', 'sum')).round(3)
    emit(r.to_string())
    f = t[t.fails_stage0]
    emit(f'  failing audits: {len(f)}; significant unadjusted {(f.p_unadj < .05).sum()}, after adjustment {(f.p_adj < .05).sum()}; criterion holds in {f.criterion_holds.sum()}; '
         f'slope of observed on predicted (P1) log OR {np.polyfit(t.pred_P1, t.obs_unadj, 1)[0]:.2f}, correlation {np.corrcoef(t.pred_P1, t.obs_unadj)[0, 1]:.2f}; '
         f'(P2) slope {np.polyfit(t.pred_P2, t.obs_unadj, 1)[0]:.2f}, correlation {np.corrcoef(t.pred_P2, t.obs_unadj)[0, 1]:.2f}')
open(OUT / 'flag_rate_summary.txt', 'w').write('\n'.join(L) + '\n')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
for k, det in enumerate(('T1', 'T2')):
    t = o[o.detector == det]
    for fl_, lab, c in ((True, 'fails Stage 0 (M > 10%)', '#c0392b'), (False, 'passes Stage 0', '#2c6e9b')):
        x = t[t.fails_stage0 == fl_]; ax[k].scatter(x.pred_P1, x.obs_unadj, s=14, c=c, alpha=.8, label=lab)
    lim = [min(t.pred_P1.min(), t.obs_unadj.min()) - .03, max(t.pred_P1.max(), t.obs_unadj.max()) + .03]
    ax[k].plot(lim, lim, 'k--', lw=.8); ax[k].set_xlabel('Predicted log odds ratio (Proposition 1)'); ax[k].set_title(f'({"ab"[k]}) screening model {det}', fontsize=9)
ax[0].set_ylabel('Observed unadjusted log odds ratio'); ax[0].legend(fontsize=7, frameon=False, loc='upper left'); plt.tight_layout(); plt.savefig(ROOT / 'output' / 'fig6.png', dpi=300)
