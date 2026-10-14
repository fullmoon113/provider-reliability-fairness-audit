# Code and data: Unreliable Aggregate Sensitive Attributes Can Manufacture Fairness Gaps

Reproducibility package for

> M. Byeon and J. Yang, "Unreliable Aggregate Sensitive Attributes Can Manufacture Fairness Gaps: Bringing Provider
> Profiling Reliability into Algorithmic Fairness Auditing," submitted to *IEEE Access*, 2026.

The paper shows that, when a sensitive attribute is an aggregate of a few patients per provider, apparent fairness gaps in
fraud-detection audits can come from sampling noise in that attribute (its reliability depends on panel size), and
proposes a staged audit (Stages 0-5) to tell this apart from real disparity.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Python 3.11; versions used: numpy 2.4, pandas 3.0, scipy 1.17, statsmodels 0.15, scikit-learn 1.8, imbalanced-learn 0.14,
xgboost 3.2, fairlearn 0.15, matplotlib 3.10. Run every script **from the repository root**, e.g. `python src/refit_sweep.py`.
Results go to `output/`. All runs use fixed seeds (42 unless stated).

## Quick start

The derived feature table `data/provider_features_FINAL.csv` is included, so every analysis runs without any download:

```bash
python src/02_experiments.py        # Tables 1 and 3, specificity, seeds, Fairlearn (about 2 minutes)
python src/reliability_decomposition.py
python src/make_null_sim_csv.py && python src/simfig.py   # Figs. 4-5
```
To rebuild the feature table from the raw Kaggle files, put the four `Train_*` CSV files in `data/raw/` and run
`python src/01_build_features.py` (see `data/DATA.md`).

## What each script reproduces

| Script | Reproduces |
|---|---|
| `src/01_build_features.py` | Kaggle raw CSVs -> `data/provider_features_FINAL.csv` (feature table, 5-fold split, partition labels) |
| `src/02_experiments.py` | Main tables: model performance (Table 1), unadjusted and panel-size-adjusted odds ratios (Table 3), specificity test, multi-seed runs, Fairlearn |
| `src/reliability_decomposition.py` | Stage 0: variance decomposition (20.75 = 20.50 + 0.247), per-provider reliability, Table 2 |
| `src/prespecified_partitions.py` | Partition-independent checks: continuous age, each age tail separately, pooled Q1+Q4 (class-weighted model) |
| `src/cms_coupling_check.py` | Sampling-induced vs substantive coupling of mean patient age and panel size in CMS 2021-2022 |
| `src/leie_cms_check.py` | LEIE-labelled check on the CMS 2021-22 panel (supplementary): needs the public LEIE file `UPDATED.csv` from oig.hhs.gov, passed as the first argument; optional 2nd/3rd arguments are the exclusion-date cut and the flagged share. Output used in the paper: `results/leie_cms_check_output.txt` |
| `src/noisy_attribute_comparison.py` | Comparison with noise-tolerant fairness methods on the null simulation (conditional-independence check, noise-corrected gap, equalized-odds post-processing); `SD_TRUE=1.77` / `3.56` and `SIM_REPS` select regime and replicates (Table 9) |
| `src/table_s1_or.py` | Odds ratios across five random seeds (Table S1) |
| `src/table4_firth.py`, `src/firth.py`, `src/firth_validate.py` | Seven alternative covariates; Firth penalized-likelihood fits; Holm correction (Table 4). `firth.py` is our own implementation of Firth's logistic regression |
| `src/base_stability.py` | Unadjusted OR across resampling strategies and seeds |
| `src/pipeline_sensitivity.py` | Sensitivity of the shrunken-attribute result to feature set, resampling and sigma^2_true |
| `src/refit_sweep.py` | Shrunken attribute with the model refitted on the same partition (Table 8 logic: the partition used to resample is the one the disparity is evaluated on) |
| `src/fpr_class_weighting.py` | Class weighting; false-positive-rate side of equalized odds; shrunken rows (Tables 7, 8) |
| `src/pooled_smote_no_resampling.py` | Pooled SMOTE and no-resampling comparators |
| `src/fpr_by_strategy.py` | Flag rates among fraud / non-fraud providers by group and strategy |
| `src/null_sim.py`, `src/sim5k.py` | Null simulation (true age effect is zero by construction; 5,000 replicates, seed 20260910) and the CMS-calibrated regimes. `SIM_REPS=50 python src/sim5k.py` gives a quick smoke test |
| `src/make_figures_1_2.py` | Fig. 1 (causal graph) and Fig. 2 (detection rate and panel size by age quartile) |
| `src/make_figure_specificity.py` | p-values by control variable (supplementary figure; not in the main text) |
| `src/make_null_sim_csv.py`, `src/simfig.py` | Fig. 4 (null-simulation distribution, 5,000 replicates) and Fig. 5 (significant-disparity rate against the true between-provider SD; the plotted rates are those printed by `sim5k.py`) |
| `src/make_figure_cms.py` | Fig. 3 (CMS replication, reads `data/cms_*.csv.gz`) |

Scripts `refit_sweep.py`, `fpr_class_weighting.py`, `pooled_smote_no_resampling.py`, `fpr_by_strategy.py` and
`make_figures_1_2.py` (and `prespecified_partitions.py`) load one another by file name; keep them together in `src/`.

## Data

See [`data/DATA.md`](data/DATA.md). Everything is public and needs no data use agreement. The Kaggle provider file behaves
like synthetic data (random beneficiary-provider linkage), which the paper discloses; the real-data evidence comes from the
CMS Medicare "by Provider" public use files.

## Randomised mitigation (Fairlearn)

`ExponentiatedGradient.predict()` returns a draw from a randomised classifier, so one call is one sample. The script therefore
fits once per fold and draws 30 predictions per fold with fixed seeds (`random_state=r`), and prints the mean and SD over draws and a
bootstrap interval for the reduction in the equalized-odds difference. With Fairlearn 0.15 and scikit-learn 1.8 it prints
F1 0.4496 (SD 0.0040), equalized-odds difference 0.0282 (0.0053), mid-range recall 0.822, extreme-age recall 0.837, and the
difference falls in 30 of 30 draws (these are the values in the paper). Single draws under Fairlearn 0.12-0.14 give F1 0.442-0.454
and 0.026-0.031. Fairlearn 0.9 and 0.11 do not run with scikit-learn 1.8 (`force_all_finite` removed). Everything else in
`02_experiments.py` matches the paper exactly.

## Numerical sensitivity

The within-provider age SD is set to 12.7275 and the exact value matters: rounding it to 12.73 changes how a few providers are
classified into the extreme-age partition and so changes some counts. Use the scripts as given; do not retype constants.

## License

Code: MIT (see `LICENSE`). Data files are public-use releases of their original sources and keep their original terms.

## Citation

Please cite the paper (see journal page once published) and this repository.
