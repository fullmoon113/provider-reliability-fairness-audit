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
pip install -r requirements-lock.txt      # exact versions of the v1.3 outputs (Python 3.13); requirements.txt gives minimum versions
```
Run every script **from the repository root**. Results go to `output/`. All runs use fixed seeds (42 unless stated).

## Quick start

The derived feature table `data/provider_features.csv` is included, so the primary analysis runs without any download:

```bash
python src/primary_cw.py      # primary specification (class weighting): performance, odds ratios, false-positive audit, shrinkage
python src/table4_cw.py       # specificity of the attenuation (seven alternative covariates)
python src/table12.py         # Stage 0 and Algorithm 1 applied end to end to the primary dataset
```
To regenerate everything that needs only the files in this repository:

```bash
python run_all.py             # all steps, full simulation sizes (several hours on 2 cores)
python run_all.py --quick     # same steps with small simulation sizes, as a smoke test
python run_all.py --list      # which step needs which input
```
The Medicare flag-rate audit and the LEIE check need two public files that are not redistributed (see `data/DATA.md`):
`python run_all.py --cms2022 cms2022_extract.csv.gz --leie UPDATED.csv`. The group-wise SMOTE specification of the earlier
analysis is kept as a sensitivity analysis (`src/02_experiments.py`). To rebuild the feature table from the raw Kaggle files,
put the four `Train_*` CSV files in `data/raw/` and run `python src/01_build_features.py`.

## Stage 0 and Algorithm 1 on your own table

```python
import sys; sys.path.insert(0, 'src')
from stage0 import stage0_reliability
res = stage0_reliability(df, unit='n_patients', attr='mean_age', partition='outer', sigma_w=12.7,
                         eps=0.10, weights=None, detection='detected')   # detection: 0/1 column, NaN-free for the audited rows
print(res['summary'])   # tau^2, Q test, median R, M, eta_G, eta_D, predicted spurious-disparity rate, branch, audit estimate, MDE, decision
```
Without a `detection` column the function returns the Stage 0a quantities only. `src/table12.py` prints the full table
for the primary dataset.

## What each script reproduces

| Script | Reproduces |
|---|---|
| `src/01_build_features.py` | Kaggle raw CSVs -> `data/provider_features.csv` (feature table, 5-fold split, partition labels) |
| `src/02_experiments.py` | Main tables: model performance, unadjusted and panel-size-adjusted odds ratios, specificity test, multi-seed runs, Fairlearn |
| `src/reliability_decomposition.py` | Stage 0: variance decomposition (20.75 = 20.50 + 0.247 by moments; REML tau^2 = 0.812), per-provider reliability |
| `src/prespecified_partitions.py` | Partition-independent checks: continuous age, each age tail separately, pooled Q1+Q4 (class-weighted model) |
| `src/cms_coupling_check.py` | Sampling-induced vs substantive coupling of mean patient age and panel size in CMS 2021-2022 |
| `src/noisy_attribute_comparison.py` | Comparison with noise-tolerant fairness methods on the null simulation (conditional-independence check, noise-corrected gap, equalized-odds post-processing); `SD_TRUE=1.65` / `3.47` and `SIM_REPS` select regime and replicates (Table S13 of the supplement) |
| `src/table_s1_or.py` | Odds ratios across five random seeds (Table S1) |
| `src/table4_firth.py`, `src/firth.py`, `src/firth_validate.py` | Seven alternative covariates; Firth penalized-likelihood fits; Holm correction (Table S10). `firth.py` is our own implementation of Firth's logistic regression |
| `src/base_stability.py` | Unadjusted OR across resampling strategies and seeds |
| `src/pipeline_sensitivity.py` | Sensitivity of the shrunken-attribute result to feature set, resampling and sigma^2_true |
| `src/refit_sweep.py` | Shrunken attribute with the model refitted on the same partition (the partition used to resample is the one the disparity is evaluated on) |
| `src/fpr_class_weighting.py` | Class weighting; false-positive-rate side of equalized odds; shrunken rows |
| `src/pooled_smote_no_resampling.py` | Pooled SMOTE and no-resampling comparators |
| `src/fpr_by_strategy.py` | Flag rates among fraud / non-fraud providers by group and strategy |
| `src/null_sim.py`, `src/sim5k.py` | Null simulation (true age effect is zero by construction; 5,000 replicates, seed 20260910) and the CMS-calibrated regimes. `SIM_REPS=50 python src/sim5k.py` gives a quick smoke test |
| `src/make_figures_1_2.py` | Fig. 1 (causal graph, `output/fig1.png`) |
| `src/make_fig2_cw.py` | Fig. 2 (detection rate and panel size by age quartile, `output/fig2.png`) |
| `src/make_null_sim_csv.py`, `src/make_fig_null.py` | Fig. 3 (null-simulation distribution, 5,000 replicates; `output/fig3.png`) |
| `src/make_figure_cms.py` | Fig. 4 (CMS replication, `output/fig4.png`; reads `data/cms_*.csv.gz`) |
| `src/make_fig_exposure.py` | Fig. 5 (exposure by specialty, `output/fig5.png`; reads `output/cms_exposure.csv`) |
| `src/stage0.py` | Importable Stage 0 check: `stage0_reliability(df, unit, attr, partition)` returns REML tau^2, Cochran Q, per-unit reliability, expected misclassification M and the decision |
| `src/supplement_runs.py`, `src/sim_panelB.py`, `src/make_supplement.py` | Supplement Tables S1 to S21: re-runs for S1, S4, S5, S8, S10, S11; slope sweep for S7 Panel B; `make_supplement.py OUTDIR` assembles `supplement.tex` from the saved outputs (S18 reads `output/cms_exposure.csv`) |
| `src/audit_common.py`, `src/simparams.py` | Shared helpers; `STRATEGY=class_weight` (default, primary) or `group_smote` selects the training strategy and the matching simulation coefficients |
| `src/primary_cw.py`, `src/table4_cw.py`, `src/cw_text_numbers.py` | Primary results under class weighting: Tables 2, 4, 5, 10, flagging among negatives, fairness and mitigation, shrinkage, partitions and Holm, thresholds, functional forms |
| `src/variance_component_inference.py` | REML tau^2, profile interval, Cochran Q and boundary LRT (Stage 0) |
| `src/rstar_table.py`, `src/misclassification_by_reliability.py` | m(R) and the required reliability R*(eps) |
| `src/prop1_analytic.py` | Analytic prediction of Proposition 1 for the outer, top-quartile and median partitions |
| `src/ablation_size_features.py` | Training without panel-size features (13/10/8 features) for the three classifiers |
| `src/cms_stage0_sweep.py` | Stage 0 for 14 aggregate attributes in CMS files, by panel-size band and within specialty (`--zero` sets suppressed counts to 0) |
| `src/cms_exposure.py` | Probability that an audit of m fraud-labeled providers reports a spurious disparity, by specialty, for a grid of size-detection slopes (Fig. 5, Table S18) |
| `src/leie_cms_check.py` | External check on CMS providers later excluded in the OIG LEIE list, including the size slope of the flag among excluded providers (Table S12; needs the LEIE file, see `data/DATA.md`) |
| `src/cms_multi_attribute.py` | Reliability of other aggregate attributes in CMS files |
| `src/corollary1.py` | Corollary 1: factors eta_G and eta_D and the predicted spurious-disparity rate against simulated cells (`output/corollary1_table.csv`) |
| `src/table12.py`, `src/table12_cms.py` | Output of Stage 0 and Algorithm 1 for the primary dataset and for CMS flag-rate audits |
| `src/table4b.py`, `src/mitigation_burden.py` | Detection and false-positive audit by partition; recall, false-positive rate, flag rate and precision before and after mitigation |
| `src/prop2_check.py`, `src/prop2_observed.py` | Proposition 2: partitions on posterior draws, posterior class probabilities and posterior means, in simulation and in the primary dataset |
| `src/flag_rate_common.py`, `src/flag_rate_predict.py`, `src/flag_rate_observe.py` | Medicare flag-rate audit: predictions are written first (`output/flag_rate_predictions.csv`, hash in `output/flag_rate_prereg.sha256`), observed associations afterwards |
| `src/make_cms2022_extract.py` | Extracts the columns used from the CMS 2022 "by Provider" file |
| `src/beneficiary_level_variance.py`, `src/permutation_null.py` | Beneficiary-level tests of the true between-provider variance; random re-linking of ages to providers (need `data/raw/`) |
| `run_all.py` | One command for every step above that its inputs allow |
| `src/planted_effect_power.py`, `src/eiv_remedy.py` | Type I error and power with a planted true effect; errors-in-variables and regression-calibration comparison (Table S13) |

Scripts `refit_sweep.py`, `fpr_class_weighting.py`, `pooled_smote_no_resampling.py`, `fpr_by_strategy.py` and
`make_figures_1_2.py` (and `prespecified_partitions.py`) load one another by file name; keep them together in `src/`.

## Data

See [`data/DATA.md`](data/DATA.md). Everything is public and needs no data use agreement. The Kaggle provider file behaves
like synthetic data (random beneficiary-provider linkage), which the paper discloses; the real-data evidence comes from the
CMS Medicare "by Provider" public use files.

## Known reproducibility caveats

* The exponentiated-gradient (equalized-odds) mitigation depends on the installed Fairlearn version. The mitigation uses a
  randomized classifier, so values are means over 30 prediction draws with fixed seeds (`src/mitigation_burden.py`,
  Fairlearn 0.15). Other versions give slightly different values; the qualitative result is the same: the equalized-odds
  difference falls sharply and recall falls in both groups.
* Base learner of the mitigation: logistic regression, the model audited in the main analysis. During development a random-forest
  base learner gave results that varied across seeds (it worsened the equalized-odds difference for one of three seeds), and
  it was not pursued; no result for it is reported.
* The LEIE check (`src/leie_cms_check.py`) fits a classifier on a random split. Its point estimates move with the seed, most of
  all the size slope among excluded providers (Table S12 reports the reference run and the range over 20 seeds).

## Numerical sensitivity

The within-provider age SD is set to 12.7275 and the exact value matters: rounding it to 12.73 changes how a few providers are
classified into the extreme-age partition and so changes some counts. Use the scripts as given; do not retype constants.

## License

Code: MIT (see `LICENSE`). Data files are public-use releases of their original sources and keep their original terms.

## Citation

Please cite the paper (see journal page once published) and this repository.
