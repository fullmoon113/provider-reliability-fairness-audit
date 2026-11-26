# Pre-specification of the Medicare flag-rate audit (written before any group-versus-flag association was computed)

Files: src/flag_rate_predict.py (this step), src/flag_rate_observe.py (run only after output/flag_rate_predictions.csv exists).

Data. CMS Medicare Physician and Other Practitioners by Provider, 2022 file, providers present in the 2021-2022 panel (1,105,159) with more than
ten beneficiaries; panel size n = Tot_Benes (2022). Labels: providers in the OIG LEIE excluded on or after 2023-01-01 (416).
Detectors (flag = the 10% highest out-of-fold scores within each specialty): T1 = histogram gradient boosting on indicators of the 20 most common provider
types and log panel size; T2 = T1 plus utilization and payment fields (no demographic counts, no risk score).
Audits: the eight largest specialties x six attributes (share female, share aged 65-74, share aged 75-84, mean age, share non-White, share dual-eligible)
x two detectors = 96. Audited units = providers not excluded; outcome = flagged. Partition: outer quartiles versus middle half of the specialty.
Stage 0 per audit: REML tau^2, sigma_w (shares: sqrt(p(1-p)) with p the specialty mean; mean age: paired-year estimate within the specialty), exact M.
An attribute "fails Stage 0" in an audit if M > 10%.
Predictions (output/flag_rate_predictions.csv): P1 = Proposition 1 with a constant mean (primary); P2 = Proposition 1 with the mean trend of the attribute in
log panel size. Both use the flag rate fitted as a quadratic logistic function of log panel size among audited units.
Criterion (fixed now): for an audit on an attribute that fails Stage 0, the prediction holds if (i) |log OR_unadjusted - log OR_P1| <= max(2 SE, 0.05),
(ii) |log OR| after adjustment for a 5-df natural spline of log panel size < 0.05 and (iii) |log OR| of the center split (above versus below the specialty median) < 0.05.
All audits are reported whether or not the criterion holds.
