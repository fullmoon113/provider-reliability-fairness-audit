"""
[02] 전체 실험 실행 -> Table 1, Table 2, 검증 결과 출력
실행: python3 02_experiments.py
입력: provider_features_FINAL.csv
출력: Table1_model_performance.csv, Table2_confounding_analysis.csv

필요 패키지: pandas numpy scikit-learn xgboost imbalanced-learn fairlearn statsmodels
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

pf = pd.read_csv(DATA / "provider_features_FINAL.csv")

# 다중공선성 제거 후 최종 13개 피처
# (원래 18개였으나 IP_ClaimCount/IP_TotalReimbursed/OP_ClaimCount 등이
#  상관계수 0.93~0.99로 중복되어 제거함. SHAP 해석 안정성 확보 목적)
FEATS = ['IP_UniqueBene', 'IP_AvgReimbursed', 'IP_AvgLOS', 'IP_AvgDiagCodes',
         'IP_UniqueAttPhysician', 'OP_AvgReimbursed', 'OP_AvgDiagCodes',
         'OP_UniqueAttPhysician', 'TotalUniqueBene', 'FemaleRatio',
         'AvgChronicCondCount', 'DeceasedRatio', 'ClaimsPerBene']

X = pf[FEATS].values
y = pf['Fraud'].values
folds = pf['fold'].values
sens = pf['IsExtremeAge'].values


def stratified_smote(Xa, ya, sa, seed=42):
    """민감그룹별로 SMOTE를 따로 적용.
    민감변수를 다른 피처와 함께 보간 후 반올림하면 그룹 라벨이 오염되므로
    (합성샘플의 3.4%가 0.3~0.7 사이 애매한 값), 그룹을 나눠서 적용함."""
    Xo, yo, so = [], [], []
    for g in np.unique(sa):
        m = sa == g
        Xg, yg = Xa[m], ya[m]
        k = min(5, max(1, (yg == 1).sum() - 1))
        Xb, yb = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xg, yg)
        Xo.append(Xb); yo.append(yb); so.append(np.full(len(yb), g))
    return np.vstack(Xo), np.concatenate(yo), np.concatenate(so)


def run_oof(model_fn, seed=42):
    """out-of-fold 예측 생성"""
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


# ============ Table 1: 모델 성능 비교 ============
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


# ============ Table 2: 교란변수 분석 (핵심 결과) ============
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


# ============ 검증 1: 통제변수 특이성 ============
print("\n" + "=" * 70)
print("Verification 1: Control Variable Specificity")
print("=" * 70)
ext = pf.loc[fm, 'IsExtremeAge'].values
for cname in ['TotalUniqueBene', 'AvgChronicCondCount', 'FemaleRatio', 'DeceasedRatio',
              'ClaimsPerBene', 'IP_UniqueBene', 'OP_AvgReimbursed', 'IP_AvgLOS']:
    cv = np.log1p(pf.loc[fm, cname].values) if cname == 'TotalUniqueBene' else pf.loc[fm, cname].values
    m = sm.Logit(det, sm.add_constant(np.column_stack([ext, cv]))).fit(disp=0)
    print(f"{cname:22s} 통제: age p={m.pvalues[1]:.4f} -> "
          f"{'ELIMINATED' if m.pvalues[1] >= 0.05 else 'retained'}")


# ============ 검증 2: 다중시드 재현성 ============
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


# ============ 검증 3: 교란 구조 근거 ============
print("\n" + "=" * 70)
print("Verification 3: Evidence of Confounding Structure")
print("=" * 70)
ext_all = pf['IsExtremeAge'].values
vol_all = np.log1p(pf['TotalUniqueBene'].values)
t, p = stats.ttest_ind(vol_all[ext_all == 0], vol_all[ext_all == 1])
print(f"전체 Provider log(volume): 중간군={vol_all[ext_all==0].mean():.3f}, "
      f"극단군={vol_all[ext_all==1].mean():.3f}, p={p:.2e}")
print(f"청구량 중앙값: 중간군={pf[ext_all==0]['TotalUniqueBene'].median():.0f}, "
      f"극단군={pf[ext_all==1]['TotalUniqueBene'].median():.0f}")

m_age = sm.Logit(det, sm.add_constant(ext)).fit(disp=0)
m_vol = sm.Logit(det, sm.add_constant(log_vol)).fit(disp=0)
m_both = sm.Logit(det, sm.add_constant(np.column_stack([ext, log_vol]))).fit(disp=0)
print(f"\nPseudo R²: age only={m_age.prsquared:.4f}, volume only={m_vol.prsquared:.4f}, "
      f"both={m_both.prsquared:.4f}")

pf['vol_q'] = pd.qcut(pf['TotalUniqueBene'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
print("\n청구량 분위별 사기율:")
print(pf.groupby('vol_q')['Fraud'].agg(['count', 'sum', 'mean']).round(4))


# ============ 참고: Fairlearn 완화 실험 (부수 결과) ============
print("\n" + "=" * 70)
print("Supplementary: Fairlearn Mitigation (secondary result)")
print("=" * 70)
# ExponentiatedGradient.predict() draws from a randomised classifier, so one draw is one sample from a
# distribution. The fit is deterministic given the data; we draw R predictions per fold (fixed seeds) and report
# the mean and SD over draws.
R = 30
oof_mit = np.zeros((R, len(pf)))
for f in range(5):
    tr, va = folds != f, folds == f
    sc = StandardScaler()
    Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
    Xb, yb, sb = stratified_smote(Xtr, y[tr], sens[tr])
    base = LogisticRegression(max_iter=500, random_state=42)
    # 주의: RandomForest를 base로 쓰면 시드에 따라 결과 불안정(3/3 시드 중 1개 악화)
    #      LogisticRegression이 안정적임 (4/4 시드 일관 개선)
    mit = ExponentiatedGradient(estimator=base, constraints=EqualizedOdds(),
                                eps=0.02, max_iter=30)
    mit.fit(Xb, yb, sensitive_features=sb)
    for r in range(R):
        oof_mit[r, va] = mit.predict(Xva, random_state=r)

def _metrics(pred):
    return (f1_score(y, pred), equalized_odds_difference(y, pred, sensitive_features=sens),
            demographic_parity_difference(y, pred, sensitive_features=sens),
            recall_score(y[sens == 0], pred[sens == 0]), recall_score(y[sens == 1], pred[sens == 1]))

names = ["F1", "EOD", "DPD", "mid recall", "extreme recall"]
b0 = _metrics(oof)
print("[Before mitigation] " + ", ".join(f"{n}={v:.4f}" for n, v in zip(names, b0)))
M = np.array([_metrics(oof_mit[r]) for r in range(R)])
print(f"[After mitigation, mean (SD) over {R} prediction draws] " +
      ", ".join(f"{n}={m:.4f} ({sd:.4f})" for n, m, sd in zip(names, M.mean(0), M.std(0, ddof=1))))

print(f"EOD after < EOD before in {int((M[:, 1] < b0[1]).sum())} of {R} draws")

def _eod(pred, yy, ss):
    g = []
    for lab in (1, 0):  # TPR gap, FPR gap
        m = yy == lab
        g.append(abs(pred[m & (ss == 0)].mean() - pred[m & (ss == 1)].mean()))
    return max(g)

rng = np.random.default_rng(20261014)
red = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    before = _eod(oof[i], y[i], sens[i])
    after = np.mean([_eod(oof_mit[r][i], y[i], sens[i]) for r in range(0, R, 3)])
    red.append(before - after)
lo, hi = np.percentile(red, [2.5, 97.5])
print(f"Bootstrap (1000) reduction in EOD, mean over prediction draws: {np.mean(red):.4f} 95% CI [{lo:.4f}, {hi:.4f}]")

print("\n완료. Table1, Table2 CSV 저장됨.")
