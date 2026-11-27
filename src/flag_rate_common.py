"""Shared definitions of the Medicare flag-rate audit (paper Sections III-G and IV-G).
Inputs: the 2021-2022 provider panel (data/cms_2021_2022_merged.csv.gz), the 2022 "by Provider" file of CMS (columns listed in
EXTRACT_COLS; build the extract with src/make_cms2022_extract.py), and the OIG LEIE file (UPDATED.csv)."""
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; DATA = ROOT / 'data'; OUT = ROOT / 'output'
FILL = 5.5                                    # suppressed category counts (1-10) are set to the midpoint, as in Table 8
CUT = pd.Timestamp('2023-01-01')              # excluded on or after this date = positive label
EXTRACT_COLS = ['Rndrng_NPI', 'Rndrng_Prvdr_Type', 'Rndrng_Prvdr_Ent_Cd', 'Rndrng_Prvdr_Mdcr_Prtcptg_Ind', 'Tot_HCPCS_Cds', 'Tot_Benes',
                'Tot_Srvcs', 'Tot_Sbmtd_Chrg', 'Tot_Mdcr_Alowd_Amt', 'Tot_Mdcr_Pymt_Amt', 'Drug_Sprsn_Ind', 'Drug_Tot_Srvcs',
                'Drug_Mdcr_Pymt_Amt', 'Med_Sprsn_Ind', 'Med_Tot_Srvcs', 'Med_Mdcr_Pymt_Amt', 'Bene_Avg_Age', 'Bene_Age_LT_65_Cnt',
                'Bene_Age_65_74_Cnt', 'Bene_Age_75_84_Cnt', 'Bene_Age_GT_84_Cnt', 'Bene_Feml_Cnt', 'Bene_Male_Cnt', 'Bene_Race_Wht_Cnt',
                'Bene_Race_Black_Cnt', 'Bene_Race_API_Cnt', 'Bene_Race_Hspnc_Cnt', 'Bene_Race_NatInd_Cnt', 'Bene_Race_Othr_Cnt',
                'Bene_Dual_Cnt', 'Bene_Ndual_Cnt']
ATTRS = ['female', 'age 65-74', 'age 75-84', 'mean age', 'non-White', 'dual-eligible']
SPECIALTIES = 8                                # the eight largest provider types of the cohort
SW_MEAN_AGE = 12.52                            # fallback within-provider SD of age (CMS paired-year estimate)

def load_cohort(extract_path):
    """2022 provider table restricted to the NPIs of the 2021-2022 panel, with the audited attributes."""
    panel = pd.read_csv(DATA / 'cms_2021_2022_merged.csv.gz')
    d = pd.read_csv(extract_path, low_memory=False)
    d = d[d.Rndrng_NPI.isin(panel.Rndrng_NPI)].copy()
    p21 = panel.set_index('Rndrng_NPI')
    d['n21'] = d.Rndrng_NPI.map(p21.Tot_Benes_21); d['age21'] = d.Rndrng_NPI.map(p21.Bene_Avg_Age_21)
    d = d.rename(columns={'Rndrng_NPI': 'NPI', 'Rndrng_Prvdr_Type': 'spec', 'Tot_Benes': 'n', 'Bene_Avg_Age': 'age'})
    f = lambda c: d[c].fillna(FILL)
    d['female'] = f('Bene_Feml_Cnt') / (f('Bene_Feml_Cnt') + f('Bene_Male_Cnt'))
    d['age 65-74'] = f('Bene_Age_65_74_Cnt') / d.n; d['age 75-84'] = f('Bene_Age_75_84_Cnt') / d.n
    race = ['Bene_Race_Wht_Cnt', 'Bene_Race_Black_Cnt', 'Bene_Race_API_Cnt', 'Bene_Race_Hspnc_Cnt', 'Bene_Race_NatInd_Cnt', 'Bene_Race_Othr_Cnt']
    rt = d[race].fillna(FILL).sum(axis=1).replace(0, np.nan)
    d['non-White'] = 1 - f('Bene_Race_Wht_Cnt') / rt
    d['dual-eligible'] = f('Bene_Dual_Cnt') / (f('Bene_Dual_Cnt') + f('Bene_Ndual_Cnt'))
    d['mean age'] = d.age
    d['logn'] = np.log(d.n)
    return d.reset_index(drop=True)

def leie_labels(d, leie_path):
    le = pd.read_csv(leie_path, dtype=str)
    le['NPI'] = pd.to_numeric(le['NPI'], errors='coerce').fillna(0).astype(np.int64)
    le = le[le.NPI > 0].copy(); le['dt'] = pd.to_datetime(le.EXCLDATE, format='%Y%m%d', errors='coerce')
    return d.NPI.isin(set(le.loc[le.dt >= CUT, 'NPI'])).astype(int).values

def sigma_w_mean_age(g):
    """Within-provider SD of age from the 2021 -> 2022 change, as in Table S2: slope of squared change on (1/n21 + 1/n22), providers with >= 50 beneficiaries."""
    dA = (g.age - g.age21).values; inv = (1 / g.n21 + 1 / g.n).values; keep = np.minimum(g.n21, g.n).values >= 50
    return float(np.sqrt(max(np.polyfit(inv[keep], dA[keep] ** 2, 1)[0], 1.0)))

def top_specialties(d, k=SPECIALTIES):
    return d.spec.value_counts().head(k).index.tolist()
