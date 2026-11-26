from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = ROOT / 'output'; RAW = ROOT / 'data' / 'raw'
"""Beneficiary-level tests of the true between-provider variance of patient age.
(1) One-way random-effects ANOVA of individual ages on provider (F test, ANOVA tau^2, intraclass correlation).
(2) Cluster bootstrap over providers: resample providers with replacement and recompute the precision-weighted REML tau^2
    (sigma_w = 12.7275 known), 1,000 replicates; percentile interval. The ANOVA treats provider-beneficiary pairs as independent although a
    beneficiary can appear at several providers, so its p-value is optimistic.
Writes output/beneficiary_level_variance_output.txt.  Needs data/raw/Train_*.csv."""
import sys, io, numpy as np, pandas as pd
from scipy import stats, optimize
sys.path.insert(0, str(HERE))
from audit_common import load
pf = load(); idx = pd.Series(np.arange(len(pf)), index=pf.Provider.values)
bene = pd.read_csv(RAW / 'Train_Beneficiarydata-1542865627584.csv', usecols=['BeneID', 'DOB'])
bene['Age'] = (pd.Timestamp('2009-12-31') - pd.to_datetime(bene['DOB'])).dt.days / 365.25
ip = pd.read_csv(RAW / 'Train_Inpatientdata-1542865627584.csv', usecols=['BeneID', 'Provider'])
op = pd.read_csv(RAW / 'Train_Outpatientdata-1542865627584.csv', usecols=['BeneID', 'Provider'])
pairs = pd.concat([ip, op]).drop_duplicates().merge(bene[['BeneID', 'Age']], on='BeneID', how='left').dropna(subset=['Age'])
pi = pairs.Provider.map(idx).values.astype(int); age = pairs.Age.values
order = np.argsort(pi, kind='stable'); pi, age = pi[order], age[order]
k = len(pf); n = np.bincount(pi, minlength=k).astype(float); N = n.sum()
starts = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(int)
buf = io.StringIO()
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); buf.write(s + '\n')
m = np.bincount(pi, weights=age, minlength=k) / n; gm = age.mean()
SSB = (n * (m - gm) ** 2).sum(); SSW = ((age - m[pi]) ** 2).sum()
MSB = SSB / (k - 1); MSW = SSW / (N - k); n0 = (N - (n ** 2).sum() / N) / (k - 1)
F = MSB / MSW; p = stats.f.sf(F, k - 1, N - k)
tau2_anova = (MSB - MSW) / n0
P(f'providers {k}, provider-beneficiary pairs {int(N)}; within-provider SD {np.sqrt(MSW):.4f}')
P(f'one-way ANOVA: F({k-1}, {int(N-k)}) = {F:.4f}, p = {p:.3g}; ANOVA tau^2 = {tau2_anova:.3f}; ICC = {tau2_anova/(tau2_anova+MSW):.5f}')
SW = 12.7275; grid = np.linspace(0, 4, 801)
def reml(mm):
    v = SW ** 2 / n; best = None
    L = []
    for t2 in grid:
        w = 1 / (t2 + v); mu = (w * mm).sum() / w.sum()
        L.append(-0.5 * (np.log(t2 + v).sum() + (w * (mm - mu) ** 2).sum() + np.log(w.sum())))
    return grid[int(np.argmax(L))]
def reml2(mm, nn):
    v = SW ** 2 / nn; L = []
    for t2 in grid:
        w = 1 / (t2 + v); mu = (w * mm).sum() / w.sum()
        L.append(-0.5 * (np.log(t2 + v).sum() + (w * (mm - mu) ** 2).sum() + np.log(w.sum())))
    return grid[int(np.argmax(L))]
rng = np.random.default_rng(7); B = 1000; est = []
for b in range(B):
    s_ = rng.integers(0, k, k); est.append(reml2(m[s_], n[s_]))
est = np.array(est)
P(f'cluster bootstrap over providers ({B}): REML tau^2 median {np.median(est):.3f}, 95% percentile interval [{np.quantile(est,.025):.3f}, {np.quantile(est,.975):.3f}]; share <= 0: {(est<=0).mean()*100:.1f}%')
(OUT / 'beneficiary_level_variance_output.txt').write_text(buf.getvalue())
