from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

"""
Stage 0: reliability of the sensitive attribute (provider mean patient age).

Observed between-provider variance = true variance + mean sampling variance (sigma_w^2 / n_i).
Reliability_i = sigma2_true / (sigma2_true + sigma_w^2 / n_i)   (Adams et al., 2010)
"""
import numpy as np, pandas as pd

pf = pd.read_csv(DATA / 'provider_features_FINAL.csv')
n = pf.TotalUniqueBene.values.astype(float)
a = pf.AvgPatientAge.values
SW = 12.7275                                   # within-provider SD of patient age (years)

var_obs = a.var(ddof=1)
var_samp = np.mean(SW**2 / n)
var_true = var_obs - var_samp
rel = var_true / (var_true + SW**2 / n)

print(f"providers                         {len(pf)}")
print(f"observed between-provider variance {var_obs:.2f}")
print(f"mean sampling variance             {var_samp:.2f}  ({100*var_samp/var_obs:.1f}% of observed)")
print(f"true between-provider variance     {var_true:.3f}  (SD {np.sqrt(max(var_true,0)):.2f} years)")
print(f"reliability: median {np.median(rel):.3f}, max {rel.max():.3f}, share >= 0.7: {(rel>=0.7).sum()} providers ({100*(rel>=0.7).mean():.2f}%)")
# Table 2: reliability by panel-size band
print("\npanel size band   providers  median n  median reliability  >=0.7")
bands = [(1,10),(11,30),(31,100),(101,300),(301,1e9)]
for lo,hi in bands:
    m = (n>=lo)&(n<=hi)
    lab = f"{lo}-{int(hi)}" if hi<1e9 else f">{lo-1}"
    print(f"{lab:>15}  {m.sum():9d}  {np.median(n[m]):8.0f}  {np.median(rel[m]):18.3f}  {(rel[m]>=0.7).sum():5d}")
