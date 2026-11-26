from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = ROOT / 'output'; RAW = ROOT / 'data' / 'raw'
"""Permutation null. Random re-linking of real patient ages to providers.
Each (beneficiary, provider) pair keeps its provider, so panel sizes and the overall age distribution are unchanged;
the ages are permuted across pairs, which removes every provider-level structure in patient age. For each permutation the
provider mean age, the outer-quartile and top-quartile partitions and the reference audit (class-weighted LR detections, fixed
out-of-fold predictions among the 506 fraud-labelled providers, with and without log panel size) are recomputed.
Writes output/permutation_null.csv and output/permutation_null_output.txt.
Usage: python src/permutation_null.py [REPS]   (needs data/raw/Train_*.csv)"""
import sys, io, numpy as np, pandas as pd
sys.path.insert(0, str(HERE))
from audit_common import load, oof_lr, logit_or, FEATS
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
pf = load(); prov = pf.Provider.values
bene = pd.read_csv(RAW / 'Train_Beneficiarydata-1542865627584.csv', usecols=['BeneID', 'DOB'])
bene['Age'] = (pd.Timestamp('2009-12-31') - pd.to_datetime(bene['DOB'])).dt.days / 365.25
ip = pd.read_csv(RAW / 'Train_Inpatientdata-1542865627584.csv', usecols=['BeneID', 'Provider'])
op = pd.read_csv(RAW / 'Train_Outpatientdata-1542865627584.csv', usecols=['BeneID', 'Provider'])
pairs = pd.concat([ip, op]).drop_duplicates().merge(bene[['BeneID', 'Age']], on='BeneID', how='left')
idx = pd.Series(np.arange(len(pf)), index=prov)
pi = pairs.Provider.map(idx).values; age = pairs.Age.values
ok = ~np.isnan(age) & ~np.isnan(pi.astype(float)); pi = pi[ok].astype(int); age = age[ok]
n = np.bincount(pi, minlength=len(pf)).astype(float)
obs = np.bincount(pi, weights=age, minlength=len(pf)) / n
print('check: max |mean age - stored| =', np.abs(obs - pf.AvgPatientAge.values).max(), '; panel size match =', (n == pf.TotalUniqueBene.values).mean())
fm = pf.Fraud.values == 1
det = oof_lr(pf, pf.IsExtremeAge.values, strategy='class_weight')[fm]
logn = np.log1p(n)[fm]
def parts(m):
    q = np.quantile(m, [.25, .75]); return ((m <= q[0]) | (m > q[1])).astype(int), (m > q[1]).astype(int)
g0, t0 = parts(obs)
print('observed (recomputed):', logit_or(det, [g0[fm]])[:2], 'adj', logit_or(det, [g0[fm], logn])[:2])
rng = np.random.default_rng(20261126); rows = []
for r in range(REPS):
    a = rng.permutation(age); m = np.bincount(pi, weights=a, minlength=len(pf)) / n
    g, t = parts(m)
    u = logit_or(det, [g[fm]]); ad = logit_or(det, [g[fm], logn]); ut = logit_or(det, [t[fm]])
    rows.append(dict(uOR=u[0], up=u[1], aOR=ad[0], ap=ad[1], tOR=ut[0], tp=ut[1]))
d = pd.DataFrame(rows); d.to_csv(OUT / 'permutation_null.csv', index=False)
OBS_U, OBS_A, OBS_T = 0.477, 1.018, 0.444
buf = io.StringIO()
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); buf.write(s + '\n')
P(f'{REPS} permutations; reference audit on 506 fraud-labelled providers (class-weighted LR detections)')
P(f'outer partition: median uOR {d.uOR.median():.3f} [{d.uOR.quantile(.025):.3f}, {d.uOR.quantile(.975):.3f}]; significant {(d.up<.05).mean()*100:.1f}%; adjusted significant {(d.ap<.05).mean()*100:.1f}%; median adj OR {d.aOR.median():.3f}')
P(f'observed 0.477 percentile of permutation uOR: {(d.uOR <= OBS_U).mean()*100:.1f}')
P(f'top quartile: median OR {d.tOR.median():.3f} [{d.tOR.quantile(.025):.3f}, {d.tOR.quantile(.975):.3f}]; significant {(d.tp<.05).mean()*100:.1f}%; observed 0.444 percentile {(d.tOR <= OBS_T).mean()*100:.1f}')
(OUT / 'permutation_null_output.txt').write_text(buf.getvalue())
