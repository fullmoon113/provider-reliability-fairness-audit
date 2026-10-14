from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

"""
Reliability of several provider-level demographic attributes in CMS data (reviewer item D-1).

Question: is the low reliability of provider mean patient age specific to age, or does it apply to the other
aggregate attributes a fairness audit could partition on (share female, share non-White, share dual-eligible,
mean HCC risk score)?

Input: two CMS "Medicare Physician & Other Practitioners - by Provider" files (names end in _Prov.csv), e.g. 2021 and 2022:
    python src/cms_multi_attribute.py MUP_PHY_R24_P05_V20_D21_Prov.csv MUP_PHY_R24_P05_V20_D22_Prov.csv
(download from data.cms.gov; the files are public and several hundred MB, so they are not in the repository).

For each attribute and each year file:
  proportions (female, non-White, dual): observed between-provider variance versus the expected binomial sampling variance
      mean[p(1-p)/n]; sigma2_true = max(observed - expected, 0); per-provider reliability R_i = sigma2_true/(sigma2_true + p(1-p)/n_i)
  mean age: expected variance with the within-provider SD of the main analysis (12.7275)
Across the two years, for ALL attributes including the risk score (which has no closed-form sampling variance):
  year-to-year correlation of the provider value, overall and by panel-size band (the test-retest analogue of Table S9).
Writes output/cms_multi_attribute.csv and prints a summary table.
"""
import sys
import numpy as np, pandas as pd

if len(sys.argv) < 3:
    sys.exit(__doc__)
SW = 12.7275
usecols = ['Rndrng_NPI', 'Tot_Benes', 'Bene_Avg_Age', 'Bene_Feml_Cnt', 'Bene_Male_Cnt', 'Bene_Race_Wht_Cnt',
           'Bene_Race_Black_Cnt', 'Bene_Race_API_Cnt', 'Bene_Race_Hspnc_Cnt', 'Bene_Race_NatInd_Cnt',
           'Bene_Race_Othr_Cnt', 'Bene_Dual_Cnt', 'Bene_Ndual_Cnt', 'Bene_Avg_Risk_Scre']

def load(path):
    d = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    d = d[d['Tot_Benes'] > 10].copy()                       # same restriction as the main CMS analysis
    n = d['Tot_Benes'].astype(float)
    out = pd.DataFrame({'NPI': d['Rndrng_NPI'], 'n': n, 'age': d['Bene_Avg_Age']})
    if 'Bene_Feml_Cnt' in d:
        out['female'] = d['Bene_Feml_Cnt'] / (d['Bene_Feml_Cnt'] + d['Bene_Male_Cnt'])
    race_cols = [c for c in d.columns if c.startswith('Bene_Race_')]
    if race_cols and 'Bene_Race_Wht_Cnt' in d:
        tot = d[race_cols].sum(axis=1).replace(0, np.nan)
        out['nonwhite'] = 1 - d['Bene_Race_Wht_Cnt'] / tot
    if 'Bene_Dual_Cnt' in d:
        out['dual'] = d['Bene_Dual_Cnt'] / (d['Bene_Dual_Cnt'] + d['Bene_Ndual_Cnt'])
    if 'Bene_Avg_Risk_Scre' in d:
        out['risk'] = d['Bene_Avg_Risk_Scre']
    return out.dropna(subset=['n', 'age'])

A, B = load(sys.argv[1]), load(sys.argv[2])
bandrows = []
rows = []
for name, tag in [('mean age', 'age'), ('share female', 'female'), ('share non-White', 'nonwhite'),
                  ('share dual-eligible', 'dual'), ('mean risk score', 'risk')]:
    for yr, D in (('year 1', A), ('year 2', B)):
        if tag not in D: continue
        x = D[[tag, 'n']].dropna(); n = x['n'].values; v = x[tag].values
        if tag == 'age':
            samp = SW ** 2 / n
        elif tag == 'risk':
            rows.append(dict(attribute=name, year=yr, n=len(x), observed_var=v.var(ddof=1))); continue
        else:
            p = np.clip(np.average(v, weights=n), 1e-6, 1 - 1e-6)
            samp = p * (1 - p) / n
        s2 = max(v.var(ddof=1) - samp.mean(), 0.0)
        R = s2 / (s2 + samp) if s2 > 0 else np.zeros_like(samp)
        for lo, hi in [(11, 20), (21, 50), (51, 100), (101, 10 ** 9)]:
            m = (n >= lo) & (n <= hi)
            bandrows.append(dict(attribute=name, year=yr, band=f'{lo}-{hi if hi < 1e9 else "+"}', providers=int(m.sum()),
                                 median_R=np.median(R[m]), share_R_ge_0_7=(R[m] >= .7).mean()))
        rows.append(dict(attribute=name, year=yr, n=len(x), observed_var=v.var(ddof=1), expected_sampling_var=samp.mean(),
                         ratio=v.var(ddof=1) / samp.mean(), sigma2_true=s2, median_R=np.median(R), share_R_ge_0_7=(R >= .7).mean()))
res = pd.DataFrame(rows)
print(res.round(4).to_string(index=False))
br = pd.DataFrame(bandrows)
print('\nReliability R_i by panel-size band (same sigma2_true for the whole file)')
print(br.round(3).to_string(index=False))
br.to_csv(OUT / 'cms_multi_attribute_by_band.csv', index=False)

# test-retest correlation by panel-size band
M = A.merge(B, on='NPI', suffixes=('_1', '_2'))
M['nmin'] = M[['n_1', 'n_2']].min(axis=1)
bands = [(11, 20), (21, 50), (51, 100), (101, 10 ** 9)]
tr = []
for name, tag in [('mean age', 'age'), ('share female', 'female'), ('share non-White', 'nonwhite'),
                  ('share dual-eligible', 'dual'), ('mean risk score', 'risk')]:
    if f'{tag}_1' not in M: continue
    r = {'attribute': name, 'all': M[[f'{tag}_1', f'{tag}_2']].corr().iloc[0, 1]}
    for lo, hi in bands:
        m = M[(M.nmin >= lo) & (M.nmin <= hi)]
        r[f'n {lo}-{hi if hi < 1e9 else "+"}'] = m[[f'{tag}_1', f'{tag}_2']].corr().iloc[0, 1]
    tr.append(r)
tr = pd.DataFrame(tr)
print('\nYear-to-year correlation of the provider value, by minimum panel size across the two years')
print(tr.round(3).to_string(index=False))
res.to_csv(OUT / 'cms_multi_attribute.csv', index=False)
tr.to_csv(OUT / 'cms_multi_attribute_testretest.csv', index=False)
