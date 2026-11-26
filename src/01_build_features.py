"""
[01] Build the provider-level feature table from the raw Kaggle CSV files
Run from the repository root: python src/01_build_features.py
Output: data/provider_features.csv

Put the four raw Kaggle files in data/raw/ (see data/DATA.md).
Checks: 5,410 providers, 506 fraud (9.35%), no missing values
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

# ---------- Beneficiary-level variables ----------
# Reference date: end of the claims window (2009-12-31)
bene['DOB'] = pd.to_datetime(bene['DOB'])
bene['DOD'] = pd.to_datetime(bene['DOD'], errors='coerce')
bene['Age'] = ((pd.Timestamp('2009-12-31') - bene['DOB']).dt.days / 365.25)
bene['IsDeceased'] = bene['DOD'].notna().astype(int)

# Chronic-condition columns are coded 1 = present, 2 = absent
chronic = [c for c in bene.columns if c.startswith('ChronicCond_')]
bene['ChronicCondCount'] = (bene[chronic] == 1).sum(axis=1)

# Race codes (CMS): 1 = White, 2 = Black, 3 = Other, 4 = Asian, 5 = Hispanic, 6 = North American Native
# Only codes 1, 2, 3 and 5 occur in this file
bene['IsMinority'] = (bene['Race'] != 1).astype(int)

# ---------- Claim-level variables ----------
inpat['AdmissionDt'] = pd.to_datetime(inpat['AdmissionDt'])
inpat['DischargeDt'] = pd.to_datetime(inpat['DischargeDt'])
inpat['LOS'] = (inpat['DischargeDt'] - inpat['AdmissionDt']).dt.days
inpat['DiagCount'] = inpat[[c for c in inpat.columns
                             if c.startswith('ClmDiagnosisCode_')]].notna().sum(axis=1)
outpat['DiagCount'] = outpat[[c for c in outpat.columns
                               if c.startswith('ClmDiagnosisCode_')]].notna().sum(axis=1)

# ---------- Aggregation to the provider level ----------
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

# Patient demographics aggregated to the provider (sensitive attributes)
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

# ---------- Merge ----------
pf = (label.merge(ip_agg, on='Provider', how='left')
           .merge(op_agg, on='Provider', how='left')
           .merge(demo, on='Provider', how='left'))

# Providers with no inpatient or outpatient activity get zero on that domain
for c in pf.columns:
    if c.startswith('IP_') or c.startswith('OP_'):
        pf[c] = pf[c].fillna(0)

pf['ClaimsPerBene'] = (pf['IP_ClaimCount'] + pf['OP_ClaimCount']) / pf['TotalUniqueBene']
pf['Fraud'] = (pf['PotentialFraud'] == 'Yes').astype(int)

# ---------- Partitions used in the fairness audit ----------
pf['age_quartile'] = pd.qcut(pf['AvgPatientAge'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
pf['IsExtremeAge'] = pf['age_quartile'].isin(['Q1', 'Q4']).astype(int)

# ---------- Stratified 5-fold ----------
skf = StratifiedKFold(5, shuffle=True, random_state=42)
pf['fold'] = -1
for i, (tr, va) in enumerate(skf.split(pf, pf['Fraud'])):
    pf.loc[va, 'fold'] = i

# ---------- Checks ----------
assert pf.shape[0] == 5410, f"provider count mismatch: {pf.shape[0]}"
assert pf['Fraud'].sum() == 506, f"fraud count mismatch: {pf['Fraud'].sum()}"
assert pf.isna().sum().sum() == 0, "missing values present"
print(f"checks passed: {pf.shape[0]} providers, {pf['Fraud'].sum()} fraud "
      f"({pf['Fraud'].mean():.4f}), missing {pf.isna().sum().sum()}")
print(f"fraud rate by fold: {pf.groupby('fold')['Fraud'].mean().round(4).tolist()}")

pf.to_csv(DATA / "provider_features.csv", index=False)
print("saved: provider_features.csv")
