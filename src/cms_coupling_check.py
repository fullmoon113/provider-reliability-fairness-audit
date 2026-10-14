from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

"""
Sampling-induced vs substantive coupling between provider mean patient age and panel size (CMS 2021-2022).

Under sampling error alone the MEAN of the measured age is flat in panel size and only its spread falls like
sigma_w / sqrt(n). A mean that moves with size indicates substantive coupling.
"""
import numpy as np, pandas as pd

b = pd.read_csv(DATA / 'cms_2021_2022_merged.csv.gz', low_memory=False)
d = pd.DataFrame({'v': b.Tot_Benes_22.astype(float), 'age': b.Bene_Avg_Age_22.astype(float),
                  'sp': b.Rndrng_Prvdr_Type_22}).dropna()
d = d[d.v > 10]                                                   # providers with more than ten beneficiaries
d['dev'] = d.age - d.groupby('sp').age.transform('mean')          # remove specialty means
bins = [10, 25, 50, 100, 200, 500, 1e9]; lab = ['11-25', '26-50', '51-100', '101-200', '201-500', '>500']
d['band'] = pd.cut(d.v, bins, labels=lab)
SIGMA_W = 12.73                                                   # within-provider SD of patient age (years)

print(f"providers: {len(d):,}   corr(specialty-adjusted age, log panel size) = {np.corrcoef(d.dev, np.log(d.v))[0,1]:.3f}\n")
print("band       n        mean dev (yrs)  observed SD  sampling SD at median n")
for l in lab:
    s = d[d.band == l]
    print(f"{l:>8} {len(s):9,d}   {s.dev.mean():+8.2f}      {s.dev.std():7.2f}      {SIGMA_W/np.sqrt(s.v.median()):7.2f}")
