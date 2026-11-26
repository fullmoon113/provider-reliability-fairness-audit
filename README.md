# Code and data: Unreliable Sensitive Attributes Manufacture Fairness Gaps

Reproducibility package for

> M. Byeon and J. Yang, "Unreliable Aggregate Sensitive Attributes Manufacture Fairness Gaps: A Reliability Check for
> Provider-Level Audits," submitted to *IEEE Access*, 2026.

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
python src/make_null_sim_csv.py && python src/simfig.py   # Figs. 5-6
```
To rebuild the feature table from the raw Kaggle files, put the four `Train_*` CSV files in `data/raw/` and run
`python src/01_build_features.py` (see `data/DATA.md`).

## Stage 0 in five lines

```python
import sys; sys.path.insert(0, 'src')
from stage0 import stage0_reliability
res = stage0_reliability(units_df, unit='n_patients', attr='mean_age', partition='outer', sigma_w=12.7)
print(res['summary'])       # tau^2, Q test, median R, expected misclassification, decision
```

## What each script reproduces

| Script | Reproduces |
|---|---|
| `src/01_build_features.py` | Kaggle raw CSVs -> `data/provider_features_FINAL.csv` (feature table, 5-fold split, partition labels) |
| `src/02_experiments.py` | Main tables: model performance (Table 1), unadjusted and panel-size-adjusted odds ratios (Table 3), specificity test, multi-seed runs, Fairlearn |
| `src/reliability_decomposition.py` | Stage 0: variance decomposition (20.75 = 20.50 + 0.247 by moments; REML tau^2 = 0.812), per-provider reliability, Table 2 |
| `src/prespecified_partitions.py` | Partition-independent checks: continuous age, each age tail separately, pooled Q1+Q4 (class-weighted model) |
| `src/cms_coupling_check.py` | Sampling-induced vs substantive coupling of mean patient age and panel size in CMS 2021-2022 |
| `src/noisy_attribute_comparison.py` | Comparison with noise-tolerant fairness methods on the null simulation (conditional-independence check, noise-corrected gap, equalized-odds post-processing); `SD_TRUE=1.77` / `3.56` and `SIM_REPS` select regime and replicates (Table 10) |
| `src/table_s1_or.py` | Odds ratios across five random seeds (Table S1) |
| `src/table4_firth.py`, `src/firth.py`, `src/firth_validate.py` | Seven alternative covariates; Firth penalized-likelihood fits; Holm correction (Table 4). `firth.py` is our own implementation of Firth's logistic regression |
| `src/base_stability.py` | Unadjusted OR across resampling strategies and seeds |
| `src/pipeline_sensitivity.py` | Sensitivity of the shrunken-attribute result to feature set, resampling and sigma^2_true |
| `src/refit_sweep.py` | Shrunken attribute with the model refitted on the same partition (Table 9 logic: the partition used to resample is the one the disparity is evaluated on) |
| `src/fpr_class_weighting.py` | Class weighting; false-positive-rate side of equalized odds; shrunken rows (Tables 8, 9) |
| `src/pooled_smote_no_resampling.py` | Pooled SMOTE and no-resampling comparators |
| `src/fpr_by_strategy.py` | Flag rates among fraud / non-fraud providers by group and strategy |
| `src/null_sim.py`, `src/sim5k.py` | Null simulation (true age effect is zero by construction; 5,000 replicates, seed 20260910) and the CMS-calibrated regimes. `SIM_REPS=50 python src/sim5k.py` gives a quick smoke test |
| `src/make_figures_1_2.py` | Fig. 1 (causal graph) and Fig. 2 (detection rate and panel size by age quartile) |
| `src/make_figure_specificity.py` | Fig. 3 (p-values by control variable) |
| `src/make_null_sim_csv.py`, `src/simfig.py` | Fig. 5 (null-simulation distribution, 5,000 replicates) and Fig. 6 (false-positive rate against the true between-provider SD; the plotted rates are those printed by `sim5k.py`) |
| `src/make_figure_cms.py` | Fig. 4 (CMS replication, reads `data/cms_*.csv.gz`) |
| `src/stage0.py` | Importable Stage 0 check: `stage0_reliability(df, unit, attr, partition)` returns REML tau^2, Cochran Q, per-unit reliability, expected misclassification M and the decision |
| `src/supplement_runs.py`, `src/sim_panelB.py`, `src/make_supplement.py` | Supplement Tables S1 to S17: re-runs for S1, S4, S5, S8, S10, S11; slope sweep for S7 Panel B; `make_supplement.py OUTDIR` assembles `supplement.tex` from the saved outputs |
| `src/audit_common.py`, `src/simparams.py` | Shared helpers; `STRATEGY=class_weight` (default, primary) or `group_smote` selects the training strategy and the matching simulation coefficients |
| `src/primary_cw.py`, `src/table4_cw.py`, `src/cw_text_numbers.py` | Primary results under class weighting: Tables 1, 3, 4, 6, flagging among negatives, fairness and mitigation, shrinkage, partitions and Holm, thresholds, functional forms |
| `src/variance_component_inference.py` | REML tau^2, profile interval, Cochran Q and boundary LRT (Stage 0) |
| `src/rstar_table.py`, `src/misclassification_by_reliability.py` | m(R) and the required reliability R*(eps) (Table 1 of the paper) |
| `src/prop1_analytic.py` | Analytic prediction of Proposition 1 for the outer, top-quartile and median partitions |
| `src/ablation_size_features.py` | Training without panel-size features (13/10/8 features) for the three classifiers |
| `src/cms_stage0_sweep.py` | Stage 0 for 14 aggregate attributes in CMS files, by panel-size band and within specialty (`--zero` sets suppressed counts to 0) |
| `src/cms_exposure.py` | Probability that an audit of m fraud-labelled providers reports a spurious disparity, by specialty |
| `src/planted_effect_power.py`, `src/eiv_remedy.py` | Type I error and power with a planted true effect; errors-in-variables comparison |
| `src/make_fig2_cw.py`, `src/make_fig_null.py`, `src/make_fig_exposure.py` | Figures 3, 5 and the exposure figure |

Scripts `refit_sweep.py`, `fpr_class_weighting.py`, `pooled_smote_no_resampling.py`, `fpr_by_strategy.py` and
`make_figures_1_2.py` (and `prespecified_partitions.py`) load one another by file name; keep them together in `src/`.

## Data

See [`data/DATA.md`](data/DATA.md). Everything is public and needs no data use agreement. The Kaggle provider file behaves
like synthetic data (random beneficiary-provider linkage), which the paper discloses; the real-data evidence comes from the
CMS Medicare "by Provider" public use files.

## Known reproducibility caveat

The exponentiated-gradient (equalized-odds) mitigation at the end of `02_experiments.py` depends on the installed Fairlearn
version. The paper reports F1 0.462 and an equalized-odds difference of 0.0263; Fairlearn 0.15 gives F1 0.456 and 0.0275
(0.12 and 0.13 give F1 0.442 and 0.0244 and 0.0237). The qualitative result is the same in every version: the difference falls to about
0.02-0.03 and F1 falls by 0.06-0.08, with the loss on mid-range providers. Everything else in `02_experiments.py` matches the paper exactly.

## Numerical sensitivity

The within-provider age SD is set to 12.7275 and the exact value matters: rounding it to 12.73 changes how a few providers are
classified into the extreme-age partition and so changes some counts. Use the scripts as given; do not retype constants.

## License

Code: MIT (see `LICENSE`). Data files are public-use releases of their original sources and keep their original terms.

## Citation

Please cite the paper (see journal page once published) and this repository.
