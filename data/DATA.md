# Data

| File | What it is | Source |
|---|---|---|
| `provider_features.csv` | Provider-level feature table (5,410 providers, 506 labelled fraudulent), 5-fold assignment in column `fold`, derived by `src/01_build_features.py` | Kaggle "Healthcare Provider Fraud Detection Analysis" (rohitrox, 2019) |
| `cms_2018_2019_merged.csv.gz` | Provider-level panel: NPI, beneficiaries and mean patient age in 2018 and 2019, provider type (inner join on NPI) | CMS Medicare Physician & Other Practitioners - by Provider, 2018 and 2019 releases |
| `cms_2021_2022_merged.csv.gz` | Same, 2021 and 2022 | CMS Medicare Physician & Other Practitioners - by Provider, 2021 and 2022 releases |

Use the "by Provider" release (file name ends in `_Prov.csv`). The "by Provider and Service" release is at
procedure-code level and has no beneficiary demographics.

The Kaggle file's beneficiary-to-provider linkage is close to random (observed/expected variance ratio of mean patient
age = 1.012; for CMS it is far larger), so it should be treated as synthetic or semi-synthetic (see paper, Section III-A).
CMS files are public use files released by the Centers for Medicare & Medicaid Services (U.S. government data, no data use agreement). The two merged year-pair panels here contain only provider-level aggregates derived from those files and are redistributed under that release; the original CMS files are not included.

## Raw Kaggle files (not redistributed here)

To rebuild `provider_features.csv`, download the dataset "Healthcare Provider Fraud Detection Analysis" from Kaggle and place these
four files in `data/raw/` (that folder is git-ignored):

- `Train_Beneficiarydata-1542865627584.csv`
- `Train_Inpatientdata-1542865627584.csv`
- `Train_Outpatientdata-1542865627584.csv`
- `Train-1542865627584.csv`

then run `python src/01_build_features.py`; it checks the result (5,410 providers, 506 fraud, no missing values).

## LEIE file (optional)

`src/leie_cms_check.py` compares the audit with providers later excluded by the OIG. Download `UPDATED.csv` from the OIG "LEIE Downloadable Databases" page
(public, about 15 MB) and pass its path as the argument; the file is not redistributed here. The point estimates of this check can differ slightly between
platforms and library versions (see Table S12 of the supplement).
