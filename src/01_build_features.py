"""
[01] 원본 Kaggle CSV -> Provider 단위 피처 테이블 생성
실행: python3 01_build_features.py
출력: provider_features_FINAL.csv

RAW_DIR 경로만 본인 환경에 맞게 수정하면 됨.
검증 완료: 5410 providers, 506 fraud(9.35%), 결측 0건
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

RAW_DIR = DATA / "raw"   # put the four Kaggle "Train_*" CSV files here (see data/DATA.md)

bene = pd.read_csv(RAW_DIR / "Train_Beneficiarydata-1542865627584.csv")
inpat = pd.read_csv(RAW_DIR / "Train_Inpatientdata-1542865627584.csv")
outpat = pd.read_csv(RAW_DIR / "Train_Outpatientdata-1542865627584.csv")
label = pd.read_csv(RAW_DIR / "Train-1542865627584.csv")

# ---------- Beneficiary 파생변수 ----------
# 기준일: 청구 데이터 마지막 시점(2009-12-31)
bene['DOB'] = pd.to_datetime(bene['DOB'])
bene['DOD'] = pd.to_datetime(bene['DOD'], errors='coerce')
bene['Age'] = ((pd.Timestamp('2009-12-31') - bene['DOB']).dt.days / 365.25)
bene['IsDeceased'] = bene['DOD'].notna().astype(int)

# 만성질환 컬럼은 1=있음, 2=없음 인코딩
chronic = [c for c in bene.columns if c.startswith('ChronicCond_')]
bene['ChronicCondCount'] = (bene[chronic] == 1).sum(axis=1)

# Race 코드 (CMS 공식): 1=White, 2=Black, 3=Other, 4=Asian, 5=Hispanic, 6=North American Native
# 본 데이터에는 1,2,3,5만 존재함
bene['IsMinority'] = (bene['Race'] != 1).astype(int)

# ---------- 청구 데이터 파생변수 ----------
inpat['AdmissionDt'] = pd.to_datetime(inpat['AdmissionDt'])
inpat['DischargeDt'] = pd.to_datetime(inpat['DischargeDt'])
inpat['LOS'] = (inpat['DischargeDt'] - inpat['AdmissionDt']).dt.days
inpat['DiagCount'] = inpat[[c for c in inpat.columns
                             if c.startswith('ClmDiagnosisCode_')]].notna().sum(axis=1)
outpat['DiagCount'] = outpat[[c for c in outpat.columns
                               if c.startswith('ClmDiagnosisCode_')]].notna().sum(axis=1)

# ---------- Provider 단위 집계 ----------
ip_agg = inpat.groupby('Provider').agg(
    IP_ClaimCount=('ClaimID', 'count'),
    IP_AvgReimbursed=('InscClaimAmtReimbursed', 'mean'),
    IP_AvgLOS=('LOS', 'mean'),
    IP_AvgDiagCodes=('DiagCount', 'mean'),
    IP_UniqueBene=('BeneID', 'nunique'),
    IP_UniqueAttPhysician=('AttendingPhysician', 'nunique'),
).reset_index()

op_agg = outpat.groupby('Provider').agg(
    OP_AvgReimbursed=('InscClaimAmtReimbursed', 'mean'),
    OP_AvgDiagCodes=('DiagCount', 'mean'),
    OP_UniqueAttPhysician=('AttendingPhysician', 'nunique'),
    OP_ClaimCount=('ClaimID', 'count'),
).reset_index()

# 환자 인구통계를 Provider 단위로 집계 (프록시 변수)
all_claims = pd.concat([inpat[['BeneID', 'Provider']],
                        outpat[['BeneID', 'Provider']]]).drop_duplicates()
all_claims = all_claims.merge(
    bene[['BeneID', 'Age', 'IsMinority', 'Gender', 'ChronicCondCount', 'IsDeceased']],
    on='BeneID', how='left')

demo = all_claims.groupby('Provider').agg(
    TotalUniqueBene=('BeneID', 'nunique'),
    AvgPatientAge=('Age', 'mean'),
    MinorityRatio=('IsMinority', 'mean'),
    FemaleRatio=('Gender', lambda x: (x == 2).mean()),
    AvgChronicCondCount=('ChronicCondCount', 'mean'),
    DeceasedRatio=('IsDeceased', 'mean'),
).reset_index()

# ---------- 병합 ----------
pf = (label.merge(ip_agg, on='Provider', how='left')
           .merge(op_agg, on='Provider', how='left')
           .merge(demo, on='Provider', how='left'))

# 입원/외래 서비스를 제공하지 않은 Provider는 0으로 처리
for c in pf.columns:
    if c.startswith('IP_') or c.startswith('OP_'):
        pf[c] = pf[c].fillna(0)

pf['ClaimsPerBene'] = (pf['IP_ClaimCount'] + pf['OP_ClaimCount']) / pf['TotalUniqueBene']
pf['Fraud'] = (pf['PotentialFraud'] == 'Yes').astype(int)

# ---------- 공정성 분석용 그룹 정의 ----------
pf['age_quartile'] = pd.qcut(pf['AvgPatientAge'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
pf['IsExtremeAge'] = pf['age_quartile'].isin(['Q1', 'Q4']).astype(int)

# ---------- Stratified 5-fold ----------
skf = StratifiedKFold(5, shuffle=True, random_state=42)
pf['fold'] = -1
for i, (tr, va) in enumerate(skf.split(pf, pf['Fraud'])):
    pf.loc[va, 'fold'] = i

# ---------- 검증 ----------
assert pf.shape[0] == 5410, f"Provider 수 불일치: {pf.shape[0]}"
assert pf['Fraud'].sum() == 506, f"사기 건수 불일치: {pf['Fraud'].sum()}"
assert pf.isna().sum().sum() == 0, "결측치 발생"
print(f"검증 통과: {pf.shape[0]} providers, {pf['Fraud'].sum()} fraud "
      f"({pf['Fraud'].mean():.4f}), 결측 {pf.isna().sum().sum()}건")
print(f"fold별 사기율: {pf.groupby('fold')['Fraud'].mean().round(4).tolist()}")

pf.to_csv(DATA / "provider_features_FINAL.csv", index=False)
print("저장 완료: provider_features_FINAL.csv")
