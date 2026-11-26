"""
LEIE-labeled check on CMS data (supplementary).

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
import sys, os, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

leie_path = sys.argv[1]
CUT = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else pd.Timestamp("2023-01-01")
FLAG = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10  # share of providers flagged by the screening rule

cms = pd.read_csv("data/cms_2021_2022_merged.csv.gz")
cms = cms.rename(columns={"Rndrng_NPI": "NPI"})
cms["benes"] = cms[["Tot_Benes_21", "Tot_Benes_22"]].mean(axis=1)
cms["age"] = cms[["Bene_Avg_Age_21", "Bene_Avg_Age_22"]].mean(axis=1)
cms = cms.dropna(subset=["benes", "age"]).query("benes > 10").copy()
cms["logV"] = np.log(cms["benes"])

le = pd.read_csv(leie_path, dtype=str)
le["NPI"] = pd.to_numeric(le["NPI"], errors="coerce").fillna(0).astype(np.int64)
le = le[le["NPI"] > 0].copy()  # LEIE codes a missing NPI as 0
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
X["logV"] = cms["logV"].values
# the sensitive attribute (mean patient age) is NOT a model input, as in the main analysis; the second run adds it back
X_with_age = X.assign(age=cms["age"].values)
y = cms["y"].values

SEED = int(os.environ.get('LEIE_SEED', 42)); SEED_ONLY = os.environ.get('LEIE_SEED_ONLY') == '1'   # seed-only mode: one age-blind fit, one JSON line

def oof_flags(X, y, seed=SEED):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = HistGradientBoostingClassifier(random_state=seed, max_iter=100, class_weight="balanced").fit(X.iloc[tr], y[tr])
        p[va] = m.predict_proba(X.iloc[va])[:, 1]
    thr = np.quantile(p, 1 - FLAG)  # screening rule: flag the top FLAG share of providers by score
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

q = cms["age"].quantile([.1, .25, .75, .9]).values
if SEED_ONLY:
    flag = oof_flags(X, y); pos = y == 1
    ext = ((cms["age"] <= q[1]) | (cms["age"] >= q[2])).values.astype(float)
    zV = ((cms["logV"] - cms["logV"].mean()) / cms["logV"].std()).values
    f1 = sm.Logit(flag[pos].astype(float), sm.add_constant(ext[pos])).fit(disp=0)
    f2 = sm.Logit(flag[pos].astype(float), sm.add_constant(np.column_stack([ext[pos], cms["logV"].values[pos]]))).fit(disp=0)
    fs = sm.Logit(flag[pos].astype(float), sm.add_constant(zV[pos])).fit(disp=0)
    print(json.dumps(dict(seed=SEED, recall=float(flag[pos].mean()), or_unadj=float(np.exp(f1.params[1])), or_adj=float(np.exp(f2.params[1])), slope=float(fs.params[1]))))
    sys.exit(0)
for name, XX in [("age excluded from model", X), ("age included in model", X_with_age)]:
    flag = oof_flags(XX, y)
    print(f"Stage 1-2 [{name}] recall overall {flag[y == 1].mean():.3f}")
    disparity(flag, q[1], q[2], "outer quartiles")
    disparity(flag, q[0], q[3], "outer deciles")
flag = oof_flags(X, y)  # Stage 3 below uses the age-excluded model

# size-detection slope of the screening rule among excluded providers, per SD of log panel size (all providers)
zV = ((cms["logV"] - cms["logV"].mean()) / cms["logV"].std()).values
fs = sm.Logit(flag[y == 1].astype(float), sm.add_constant(zV[y == 1])).fit(disp=0)
cs = fs.conf_int()[1]
print(f"Size-detection slope of the top-{FLAG:.0%} rule among excluded providers: {fs.params[1]:.3f} per SD of log panel size "
      f"[{cs[0]:.3f}, {cs[1]:.3f}], p={fs.pvalues[1]:.4f}; recall {flag[y == 1].mean():.3f}")

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

# Proposition 1 prediction for the extreme-age (outer quartile) split among the excluded providers, normal approximation:
# A | n ~ N(mu, tau2 + sigma_w^2 / n) with tau2 = observed variance - mean sampling variance; h(n) = logistic(a + b z) fitted above.
from scipy.stats import norm
tau2 = cms["age"].var() - exp_var; mu = cms["age"].mean(); lo, hi = q[1], q[2]
a_, b_ = fs.params
zp = zV[y == 1]; npos = cms["benes"].values[y == 1]; sn = np.sqrt(tau2 + sig_w2 / npos)
pi = norm.cdf((lo - mu) / sn) + 1 - norm.cdf((hi - mu) / sn); h = 1 / (1 + np.exp(-(a_ + b_ * zp)))
p1 = (h * pi).sum() / pi.sum(); p0 = (h * (1 - pi)).sum() / (1 - pi).sum()
print(f"Prop. 1 prediction for the outer-quartile split among excluded providers: OR {(p1 / (1 - p1)) / (p0 / (1 - p0)):.3f} (tau2 {tau2:.2f})")
