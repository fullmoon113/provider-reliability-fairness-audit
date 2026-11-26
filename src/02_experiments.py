"""
[02] Baseline experiments with group-wise SMOTE: model performance, unadjusted and adjusted odds ratios, specificity, seeds, Fairlearn mitigation
Run from the repository root: python src/02_experiments.py
Input: data/provider_features.csv
Output: output/Table1_model_performance.csv, output/Table2_confounding_analysis.csv

Packages: see requirements.txt. This script uses the group-wise SMOTE specification; the primary specification (class weighting) is run by src/primary_cw.py.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             roc_auc_score, average_precision_score)
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

pf = pd.read_csv(DATA / "provider_features.csv")

# The 13 features retained after removing near-duplicate aggregates
# (the claim counts and totals are near-duplicates of beneficiary counts and reimbursement
#  (r = 0.93 to 0.99) and destabilize post-hoc attribution without improving prediction)
FEATS = ['IP_UniqueBene', 'IP_AvgReimbursed', 'IP_AvgLOS', 'IP_AvgDiagCodes',
         'IP_UniqueAttPhysician', 'OP_AvgReimbursed', 'OP_AvgDiagCodes',
         'OP_UniqueAttPhysician', 'TotalUniqueBene', 'FemaleRatio',
         'AvgChronicCondCount', 'DeceasedRatio', 'ClaimsPerBene']

X = pf[FEATS].values
y = pf['Fraud'].values
folds = pf['fold'].values
sens = pf['IsExtremeAge'].values


def stratified_smote(Xa, ya, sa, seed=42):
    """Apply SMOTE separately within each sensitive group.
    Interpolating the sensitive attribute with the other features and rounding contaminates the group label
    (3.4% of synthetic samples fall between 0.3 and 0.7), so SMOTE is applied within groups."""
    Xo, yo, so = [], [], []
    for g in np.unique(sa):
        m = sa == g
        Xg, yg = Xa[m], ya[m]
        k = min(5, max(1, (yg == 1).sum() - 1))
        Xb, yb = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xg, yg)
        Xo.append(Xb); yo.append(yb); so.append(np.full(len(yb), g))
    return np.vstack(Xo), np.concatenate(yo), np.concatenate(so)


def run_oof(model_fn, seed=42):
    """Out-of-fold predictions"""
    oof = np.zeros(len(pf))
    for f in range(5):
        tr, va = folds != f, folds == f
        sc = StandardScaler()
        Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
        Xb, yb, _ = stratified_smote(Xtr, y[tr], sens[tr], seed)
        m = model_fn()
        m.fit(Xb, yb)
        oof[va] = m.predict(Xva)
    return oof


# ============ Table 1: model performance ============
print("=" * 70)
print("Table 1: Model Performance (5-fold CV)")
print("=" * 70)

MODELS = {
    'Logistic Regression': lambda: LogisticRegression(max_iter=500, random_state=42),
    'Random Forest': lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    'XGBoost': lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     eval_metric='logloss', random_state=42, n_jobs=-1),
}

rows = []
for name, mk in MODELS.items():
    f1s, aucs, prs, recs, precs = [], [], [], [], []
    for f in range(5):
        tr, va = folds != f, folds == f
        sc = StandardScaler()
        Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
        Xb, yb, _ = stratified_smote(Xtr, y[tr], sens[tr])
        m = mk(); m.fit(Xb, yb)
        p, pr = m.predict(Xva), m.predict_proba(Xva)[:, 1]
        f1s.append(f1_score(y[va], p)); aucs.append(roc_auc_score(y[va], pr))
        prs.append(average_precision_score(y[va], pr))
        recs.append(recall_score(y[va], p))
        precs.append(precision_score(y[va], p, zero_division=0))
    rows.append({'Model': name,
                 'F1': f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}",
                 'Precision': f"{np.mean(precs):.3f} ± {np.std(precs):.3f}",
                 'Recall': f"{np.mean(recs):.3f} ± {np.std(recs):.3f}",
                 'ROC-AUC': f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
                 'PR-AUC': f"{np.mean(prs):.3f} ± {np.std(prs):.3f}"})

t1 = pd.DataFrame(rows)
print(t1.to_string(index=False))
t1.to_csv(OUT / "Table1_model_performance.csv", index=False)


# ============ Table 2: odds ratios with and without panel-size adjustment ============
print("\n" + "=" * 70)
print("Table 2: Confounding Analysis (KEY RESULT)")
print("=" * 70)

oof = run_oof(lambda: LogisticRegression(max_iter=500, random_state=42))
fm = y == 1
det = oof[fm]
log_vol = np.log1p(pf.loc[fm, 'TotalUniqueBene'].values)

SENS_VARS = {
    'Extreme age-mix (Q1+Q4)': pf.loc[fm, 'IsExtremeAge'].values,
    'High-elderly-serving (top 25%)':
        (pf.loc[fm, 'AvgPatientAge'] >= pf['AvgPatientAge'].quantile(0.75)).astype(int).values,
    'High-minority-serving (top 25%)':
        (pf.loc[fm, 'MinorityRatio'] >= pf['MinorityRatio'].quantile(0.75)).astype(int).values,
}

rows2 = []
for name, sv in SENS_VARS.items():
    ms = sm.Logit(det, sm.add_constant(sv)).fit(disp=0)
    ma = sm.Logit(det, sm.add_constant(np.column_stack([sv, log_vol]))).fit(disp=0)
    rows2.append({'Sensitive attribute': name,
                  'Unadjusted OR': f"{np.exp(ms.params[1]):.3f}",
                  'Unadjusted p': f"{ms.pvalues[1]:.4f}",
                  'Volume-adjusted OR': f"{np.exp(ma.params[1]):.3f}",
                  'Volume-adjusted p': f"{ma.pvalues[1]:.4f}",
                  'Interpretation': 'Confounded' if ms.pvalues[1] < 0.05 and ma.pvalues[1] >= 0.05
                                    else ('Robust' if ma.pvalues[1] < 0.05 else 'Not significant')})

t2 = pd.DataFrame(rows2)
print(t2.to_string(index=False))
t2.to_csv(OUT / "Table2_confounding_analysis.csv", index=False)


# ============ Check 1: specificity of the control variable ============
print("\n" + "=" * 70)
print("Verification 1: Control Variable Specificity")
print("=" * 70)
ext = pf.loc[fm, 'IsExtremeAge'].values
for cname in ['TotalUniqueBene', 'AvgChronicCondCount', 'FemaleRatio', 'DeceasedRatio',
              'ClaimsPerBene', 'IP_UniqueBene', 'OP_AvgReimbursed', 'IP_AvgLOS']:
    cv = np.log1p(pf.loc[fm, cname].values) if cname == 'TotalUniqueBene' else pf.loc[fm, cname].values
    m = sm.Logit(det, sm.add_constant(np.column_stack([ext, cv]))).fit(disp=0)
    print(f"{cname:22s} control: age p={m.pvalues[1]:.4f} -> "
          f"{'ELIMINATED' if m.pvalues[1] >= 0.05 else 'retained'}")


# ============ Check 2: multiple seeds ============
print("\n" + "=" * 70)
print("Verification 2: Multi-seed Reproducibility")
print("=" * 70)
for seed in [42, 1, 7, 99, 2024]:
    oof_s = run_oof(lambda: LogisticRegression(max_iter=500, random_state=seed), seed)
    det_s = oof_s[fm]
    ms = sm.Logit(det_s, sm.add_constant(ext)).fit(disp=0)
    ma = sm.Logit(det_s, sm.add_constant(np.column_stack([ext, log_vol]))).fit(disp=0)
    verdict = 'CONFOUNDED' if ms.pvalues[1] < 0.05 and ma.pvalues[1] >= 0.05 else 'inconsistent'
    print(f"seed={seed:5d}: unadjusted p={ms.pvalues[1]:.4f}, "
          f"adjusted p={ma.pvalues[1]:.4f} -> {verdict}")


# ============ Check 3: evidence on the confounding structure ============
print("\n" + "=" * 70)
print("Verification 3: Evidence of Confounding Structure")
print("=" * 70)
ext_all = pf['IsExtremeAge'].values
vol_all = np.log1p(pf['TotalUniqueBene'].values)
t, p = stats.ttest_ind(vol_all[ext_all == 0], vol_all[ext_all == 1])
print(f"All providers, log(volume): middle group={vol_all[ext_all==0].mean():.3f}, "
      f"extreme group={vol_all[ext_all==1].mean():.3f}, p={p:.2e}")
print(f"Median panel size: middle group={pf[ext_all==0]['TotalUniqueBene'].median():.0f}, "
      f"extreme group={pf[ext_all==1]['TotalUniqueBene'].median():.0f}")

m_age = sm.Logit(det, sm.add_constant(ext)).fit(disp=0)
m_vol = sm.Logit(det, sm.add_constant(log_vol)).fit(disp=0)
m_both = sm.Logit(det, sm.add_constant(np.column_stack([ext, log_vol]))).fit(disp=0)
print(f"\nPseudo R²: age only={m_age.prsquared:.4f}, volume only={m_vol.prsquared:.4f}, "
      f"both={m_both.prsquared:.4f}")

pf['vol_q'] = pd.qcut(pf['TotalUniqueBene'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
print("\nFraud rate by panel-size quartile:")
print(pf.groupby('vol_q')['Fraud'].agg(['count', 'sum', 'mean']).round(4))


# ============ Fairlearn equalized-odds mitigation (secondary result) ============
print("\n" + "=" * 70)
print("Supplementary: Fairlearn Mitigation (secondary result)")
print("=" * 70)
oof_mit = np.zeros(len(pf))
for f in range(5):
    tr, va = folds != f, folds == f
    sc = StandardScaler()
    Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
    Xb, yb, sb = stratified_smote(Xtr, y[tr], sens[tr])
    base = LogisticRegression(max_iter=500, random_state=42)
    # Base learner: logistic regression, the model audited in the main analysis. The mitigation is reported for this learner only (see README).
    mit = ExponentiatedGradient(estimator=base, constraints=EqualizedOdds(),
                                eps=0.02, max_iter=30)
    mit.fit(Xb, yb, sensitive_features=sb)
    oof_mit[va] = mit.predict(Xva)

for lab, pred in [("Before mitigation", oof), ("After mitigation", oof_mit)]:
    eod = equalized_odds_difference(y, pred, sensitive_features=sens)
    dpd = demographic_parity_difference(y, pred, sensitive_features=sens)
    print(f"[{lab}] F1={f1_score(y,pred):.3f}, EOD={eod:.4f}, DPD={dpd:.4f}, "
          f"mid recall={recall_score(y[sens==0],pred[sens==0]):.3f}, "
          f"extreme recall={recall_score(y[sens==1],pred[sens==1]):.3f}")

print("\nDone. Table1 and Table2 CSV files saved.")
