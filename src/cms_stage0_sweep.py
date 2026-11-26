from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / 'output'

"""
Stage 0 sweep on CMS "Medicare Physician & Other Practitioners - by Provider" files.

For every provider-level aggregate attribute that an audit could partition on (mean age, age-band shares,
share female, race-category shares, share dual-eligible, mean risk score) and each of two years, the script reports
  - between-provider variance versus expected sampling variance, the true-variance estimate and the ratio,
  - per-provider reliability R_i = s2_true / (s2_true + sampling variance_i), its median and the share with R_i >= 0.7,
  - the expected share of providers that an outer-versus-middle (quartile) split puts in the wrong group, m(R_i),
  - all of the above by panel-size band and, within specialty, by band,
  - the year-to-year correlation of the provider value by band.
Counts of 1-10 beneficiaries are suppressed in the CMS files. A suppressed category count is set to 5.5 (midpoint of 1-10)
in the main run and to 0 in the sensitivity run (--zero), and the share of providers with a suppressed count is reported.

Usage:  python cms_stage0_sweep.py FILE_YEAR1_Prov.csv FILE_YEAR2_Prov.csv [--zero] [OUTDIR]
"""
import sys
import numpy as np, pandas as pd

args = [a for a in sys.argv[1:] if not a.startswith('--')]
ZERO = '--zero' in sys.argv
if len(args) < 2:
    sys.exit(__doc__)
OUTD = Path(args[2]) if len(args) > 2 else OUT
OUTD.mkdir(exist_ok=True, parents=True)
SW = 12.52          # CMS-estimated within-provider SD of beneficiary age (Section IV-E)
FILL = 0.0 if ZERO else 5.5
BANDS = [(11, 20), (21, 50), (51, 100), (101, 10 ** 9)]
bname = lambda lo, hi: f'{lo}-{hi}' if hi < 1e9 else f'{lo}+'

# m(R): share misclassified by an outer-vs-middle split, from simulation on a fine grid
rng = np.random.default_rng(1)
t = rng.standard_normal(1_000_000); e = rng.standard_normal(1_000_000)
qt = np.quantile(t, [.25, .75]); gt = (t < qt[0]) | (t > qt[1])
Rg = np.linspace(0, 1, 41); mg = []
for R in Rg:
    o = np.sqrt(R) * t + np.sqrt(1 - R) * e
    qo = np.quantile(o, [.25, .75]); go = (o < qo[0]) | (o > qo[1])
    mg.append((gt != go).mean())
mfun = lambda R: np.interp(R, Rg, mg)

cols = ['Rndrng_NPI', 'Rndrng_Prvdr_Type', 'Tot_Benes', 'Bene_Avg_Age', 'Bene_Age_LT_65_Cnt', 'Bene_Age_65_74_Cnt',
        'Bene_Age_75_84_Cnt', 'Bene_Age_GT_84_Cnt', 'Bene_Feml_Cnt', 'Bene_Male_Cnt', 'Bene_Race_Wht_Cnt',
        'Bene_Race_Black_Cnt', 'Bene_Race_API_Cnt', 'Bene_Race_Hspnc_Cnt', 'Bene_Race_NatInd_Cnt', 'Bene_Race_Othr_Cnt',
        'Bene_Dual_Cnt', 'Bene_Ndual_Cnt', 'Bene_Avg_Risk_Scre']

def load(path):
    d = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    d = d[d['Tot_Benes'] > 10].copy()
    o = pd.DataFrame({'NPI': d['Rndrng_NPI'], 'spec': d['Rndrng_Prvdr_Type'], 'n': d['Tot_Benes'].astype(float),
                      'age': d['Bene_Avg_Age'], 'risk': d['Bene_Avg_Risk_Scre']})
    supp = {}
    def share(num_cols, den_cols, name):
        num = d[num_cols].fillna(FILL).sum(axis=1)
        den = d[den_cols].fillna(FILL).sum(axis=1)
        o[name] = num / den.replace(0, np.nan)
    ageb = ['Bene_Age_LT_65_Cnt', 'Bene_Age_65_74_Cnt', 'Bene_Age_75_84_Cnt', 'Bene_Age_GT_84_Cnt']
    for c, nm in zip(ageb, ['age<65', 'age65-74', 'age75-84', 'age>84']):
        o[nm] = d[c].fillna(FILL) / d['Tot_Benes']; supp[nm] = d[c].isna().mean()
    o['female'] = d['Bene_Feml_Cnt'].fillna(FILL) / (d['Bene_Feml_Cnt'].fillna(FILL) + d['Bene_Male_Cnt'].fillna(FILL))
    race = ['Bene_Race_Wht_Cnt', 'Bene_Race_Black_Cnt', 'Bene_Race_API_Cnt', 'Bene_Race_Hspnc_Cnt',
            'Bene_Race_NatInd_Cnt', 'Bene_Race_Othr_Cnt']
    rt = d[race].fillna(FILL).sum(axis=1).replace(0, np.nan)
    for c, nm in zip(race[1:], ['Black', 'Asian/Pacific Is.', 'Hispanic', 'Native American', 'Other race']):
        o[nm] = d[c].fillna(FILL) / rt; supp[nm] = d[c].isna().mean()
    o['non-White'] = 1 - d['Bene_Race_Wht_Cnt'].fillna(FILL) / rt; supp['non-White'] = d['Bene_Race_Wht_Cnt'].isna().mean()  # blank = suppressed (1-10) or zero
    du = d['Bene_Dual_Cnt'].fillna(FILL); nd = d['Bene_Ndual_Cnt'].fillna(FILL)
    o['dual-eligible'] = du / (du + nd).replace(0, np.nan)
    return o.dropna(subset=['n', 'age']), supp

ATTR = [('mean age', 'age', 'mean'), ('age <65', 'age<65', 'p'), ('age 65-74', 'age65-74', 'p'),
        ('age 75-84', 'age75-84', 'p'), ('age >84', 'age>84', 'p'), ('female', 'female', 'p'),
        ('non-White (any)', 'non-White', 'p'), ('Black', 'Black', 'p'), ('Asian/Pacific Is.', 'Asian/Pacific Is.', 'p'),
        ('Hispanic', 'Hispanic', 'p'), ('Native American', 'Native American', 'p'), ('Other race', 'Other race', 'p'),
        ('dual-eligible', 'dual-eligible', 'p'), ('risk score', 'risk', 'risk')]

def reliability(v, n, kind, ref=None):
    """returns (s2_true, sampling variance per provider, R per provider) for one group of providers."""
    if kind == 'mean':
        samp = SW ** 2 / n
    elif kind == 'risk':
        return None
    else:
        p = np.clip(np.average(v, weights=n), 1e-6, 1 - 1e-6)
        samp = p * (1 - p) / n
    s2 = max(np.var(v, ddof=1) - samp.mean(), 0.0)
    R = s2 / (s2 + samp) if s2 > 0 else np.zeros_like(samp)
    return s2, samp, R

A, sa = load(args[0]); B, sb = load(args[1])
rows, band_rows, spec_rows = [], [], []
topspec = A['spec'].value_counts().head(8).index.tolist()
for yr, D, sp in (('year 1', A, sa), ('year 2', B, sb)):
    for name, tag, kind in ATTR:
        x = D[[tag, 'n', 'spec']].dropna(); n = x['n'].values; v = x[tag].values
        r = reliability(v, n, kind)
        if r is None:
            continue
        s2, samp, R = r
        rows.append(dict(attribute=name, year=yr, providers=len(x), suppressed_share=sp.get(tag, 0.0),
                         observed_var=np.var(v, ddof=1), mean_sampling_var=samp.mean(), ratio=np.var(v, ddof=1) / samp.mean(),
                         s2_true=s2, median_R=np.median(R), share_R_ge_0_7=(R >= .7).mean(), mean_m=mfun(R).mean()))
        for lo, hi in BANDS:
            m = (n >= lo) & (n <= hi)
            band_rows.append(dict(attribute=name, year=yr, band=bname(lo, hi), providers=int(m.sum()),
                                  median_R=np.median(R[m]), share_R_ge_0_7=(R[m] >= .7).mean(), mean_m=mfun(R[m]).mean()))
        # within specialty: variance decomposition done inside the specialty, by band
        for s in topspec:
            xs = x[x.spec == s]
            if len(xs) < 500: continue
            rs = reliability(xs[tag].values, xs['n'].values, kind)
            s2s, samps, Rs = rs
            nn = xs['n'].values
            for lo, hi in BANDS:
                m = (nn >= lo) & (nn <= hi)
                if m.sum() < 30: continue
                spec_rows.append(dict(attribute=name, year=yr, specialty=s, band=bname(lo, hi), providers=int(m.sum()),
                                      median_R=np.median(Rs[m]), share_R_ge_0_7=(Rs[m] >= .7).mean(), mean_m=mfun(Rs[m]).mean()))
res = pd.DataFrame(rows); br = pd.DataFrame(band_rows); sr = pd.DataFrame(spec_rows)

M = A.merge(B, on='NPI', suffixes=('_1', '_2'))
M['nmin'] = M[['n_1', 'n_2']].min(axis=1)
tr = []
for name, tag, kind in ATTR:
    r = {'attribute': name, 'all': M[[f'{tag}_1', f'{tag}_2']].corr().iloc[0, 1]}
    for lo, hi in BANDS:
        m = M[(M.nmin >= lo) & (M.nmin <= hi)]
        r[f'n {bname(lo, hi)}'] = m[[f'{tag}_1', f'{tag}_2']].corr().iloc[0, 1]
    tr.append(r)
tr = pd.DataFrame(tr)

# within-specialty test-retest: providers with the same specialty in both years; each year's value is centered at its
# specialty mean (n-weighted within year) before correlating, so differences between specialties do not count as stability.
W = M[M['spec_1'] == M['spec_2']].copy()
trw = []
for name, tag, kind in ATTR:
    c1, c2 = f'{tag}_1', f'{tag}_2'
    X = W[[c1, c2, 'spec_1', 'nmin']].dropna().copy()
    X['v1'] = X[c1] - X.groupby('spec_1')[c1].transform('mean')
    X['v2'] = X[c2] - X.groupby('spec_1')[c2].transform('mean')
    r = {'attribute': name, 'all': X[['v1', 'v2']].corr().iloc[0, 1]}
    for lo, hi in BANDS:
        m = X[(X.nmin >= lo) & (X.nmin <= hi)]
        r[f'n {bname(lo, hi)}'] = m[['v1', 'v2']].corr().iloc[0, 1]
    trw.append(r)
trw = pd.DataFrame(trw)

suf = '_zero' if ZERO else ''
res.to_csv(OUTD / f'cms_stage0_sweep{suf}.csv', index=False)
br.to_csv(OUTD / f'cms_stage0_by_band{suf}.csv', index=False)
sr.to_csv(OUTD / f'cms_stage0_by_specialty_band{suf}.csv', index=False)
tr.to_csv(OUTD / f'cms_stage0_testretest{suf}.csv', index=False)
trw.to_csv(OUTD / f'cms_stage0_testretest_within{suf}.csv', index=False)
pd.set_option('display.width', 250)
print('Overall (pooled across specialties)'); print(res.round(4).to_string(index=False))
print('\nBy panel-size band'); print(br.round(3).to_string(index=False))
print('\nTest-retest correlation by minimum panel size'); print(tr.round(3).to_string(index=False))
print('\nTest-retest correlation within specialty, by minimum panel size'); print(trw.round(3).to_string(index=False))
print('\nWithin-specialty (top specialties), by band'); print(sr.round(3).to_string(index=False))
