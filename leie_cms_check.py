"""
LEIE-labelled check on CMS data (supplementary; reviewer item 1b).

Labels: providers in the CMS 2021-22 panel (data/cms_2021_2022_merged.csv.gz) whose NPI appears in the
OIG List of Excluded Individuals/Entities (LEIE) with an exclusion date AFTER the claims window
(on or after 2023-01-01), i.e. providers later excluded.

Download the LEIE file yourself (it is public, about 15 MB):
    https://oig.hhs.gov/exclusions/exclusions_list.asp  ->  "LEIE Downloadable Databases", UPDATED.csv
and pass its path:
    python src/leie_cms_check.py path/to/UPDATED.csv

Stages (same definitions as the main analysis):
  0  reliability of provider mean patient age (observed vs expected variance, per-provider reliability)
  1  detection disparity: 5-fold out-of-fold flags; among LEIE-positive providers, detection (recall)
     for the extreme-age partition (outer quartiles of mean age) versus the middle half
  2  same model adding log panel size as a covariate (Firth is not needed here; plain logistic fit)
  3  specificity: panel size versus provider-type share as the adjusting variable
  5  robustness to the partition (outer deciles) and to the exclusion-date cut
Caveat to report with the result: the label itself depends on unit size (large billers are more often
investigated), so this check cannot separate detection bias from label bias.
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

leie_path = sys.argv[1]
CUT = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else pd.Timestamp("2023-01-01")

cms = pd.read_csv("data/cms_2021_2022_merged.csv.gz")
cms = cms.rename(columns={"Rndrng_NPI": "NPI"})
cms["benes"] = cms[["Tot_Benes_21", "Tot_Benes_22"]].mean(axis=1)
cms["age"] = cms[["Bene_Avg_Age_21", "Bene_Avg_Age_22"]].mean(axis=1)
cms = cms.dropna(subset=["benes", "age"]).query("benes > 10").copy()
cms["logV"] = np.log(cms["benes"])

le = pd.read_csv(leie_path, dtype=str)
le = le[le["NPI"].notna() & (le["NPI"] != "0000000000")].copy()
le["NPI"] = le["NPI"].astype(np.int64)
le["d"] = pd.to_datetime(le["EXCLDATE"], format="%Y%m%d", errors="coerce")
excl = set(le.loc[le["d"] >= CUT, "NPI"])
cms["y"] = cms["NPI"].isin(excl).astype(int)
print(f"providers {len(cms):,}; LEIE-positive after {CUT.date()}: {cms['y'].sum():,}")
if cms["y"].sum() < 50:
    sys.exit("too few positives for a stable analysis")

# Stage 0
sig_w2 = 12.7275 ** 2  # within-provider age variance reference from the main analysis
exp_var = np.mean(sig_w2 / cms["benes"])
print(f"Stage 0: observed var {cms['age'].var():.2f}, expected sampling var {exp_var:.2f}, "
      f"ratio {cms['age'].var() / exp_var:.2f}")

# features: provider type (top 20), age, panel size (as in the main model's inputs)
types = cms["Rndrng_Prvdr_Type_22"].fillna(cms["Rndrng_Prvdr_Type_21"])
top = types.value_counts().index[:20]
X = pd.get_dummies(types.where(types.isin(top), "other")).astype(float)
X["age"] = cms["age"].values
X["logV"] = cms["logV"].values
y = cms["y"].values

def oof_flags(X, y, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = HistGradientBoostingClassifier(random_state=seed, max_iter=100).fit(X.iloc[tr], y[tr])
        p[va] = m.predict_proba(X.iloc[va])[:, 1]
    thr = np.quantile(p, 1 - y.mean())  # flag the same share as the base rate
    return p >= thr

def disparity(flag, lo, hi, label):
    pos = y == 1
    ext = ((cms["age"] <= lo) | (cms["age"] >= hi)).values.astype(float)
    d = flag[pos].astype(float)
    e, v = ext[pos], cms["logV"].values[pos]
    f1 = sm.Logit(d, sm.add_constant(e)).fit(disp=0)
    f2 = sm.Logit(d, sm.add_constant(np.column_stack([e, v]))).fit(disp=0)
    for nm, f in [("unadjusted", f1), ("adj. panel size", f2)]:
        ci = np.exp(f.conf_int()[1])
        print(f"  {label:<18} {nm:<16} OR {np.exp(f.params[1]):.3f} [{ci[0]:.3f}, {ci[1]:.3f}] p={f.pvalues[1]:.3f}")
    return f1, f2

flag = oof_flags(X, y)
print(f"Stage 1-2 recall overall {flag[y == 1].mean():.3f}")
q = cms["age"].quantile([.1, .25, .75, .9]).values
disparity(flag, q[1], q[2], "outer quartiles")
disparity(flag, q[0], q[3], "outer deciles")

# Stage 3 specificity: share of the provider's type among positives is a non-panel-size control
tshare = types.map(types.value_counts(normalize=True)).values
pos = y == 1
ext = ((cms["age"] <= q[1]) | (cms["age"] >= q[2])).values.astype(float)
f3 = sm.Logit(flag[pos].astype(float), sm.add_constant(np.column_stack([ext[pos], np.log(tshare[pos])]))).fit(disp=0)
print(f"Stage 3: adjusting for provider-type share instead: OR {np.exp(f3.params[1]):.3f}, p={f3.pvalues[1]:.3f}")

# Stage 5: alternative exclusion-date cut
for alt in ["2024-01-01"]:
    ex2 = set(le.loc[le["d"] >= pd.Timestamp(alt), "NPI"])
    print(f"Stage 5: cut {alt}: positives {cms['NPI'].isin(ex2).sum():,} (rerun with the date as 2nd argument)")
