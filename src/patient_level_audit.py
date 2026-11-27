"""Patient-level audit of the primary dataset (supplement Table S23).

The provider-level audit partitions providers on the mean age of their beneficiaries. Here the unit is the
(beneficiary, fraud-labeled provider) link, the attribute is the beneficiary's own age (no aggregation, so no
sampling error), and the outcome is whether the screening model detected the provider (reference class-weighted logistic
regression, out-of-fold, threshold 0.5). Standard errors are clustered on the provider.
Partitions: outer quartiles of beneficiary age versus the middle half; top quartile versus the rest; panel size added as a covariate.
Needs data/raw/ (Kaggle).   Usage: python src/patient_level_audit.py"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from audit_common import *
RAW = DATA / 'raw'
pf = load(); lab = pf.Fraud.values == 1
det = pd.Series(oof_lr(pf, outer(pf.AvgPatientAge.values), strategy='class_weight').astype(int), index=pf.Provider.values)
bene = pd.read_csv(RAW / 'Train_Beneficiarydata-1542865627584.csv', usecols=['BeneID', 'DOB'])
bene['Age'] = (pd.Timestamp('2009-12-31') - pd.to_datetime(bene['DOB'])).dt.days / 365.25
ip = pd.read_csv(RAW / 'Train_Inpatientdata-1542865627584.csv', usecols=['BeneID', 'Provider'])
op = pd.read_csv(RAW / 'Train_Outpatientdata-1542865627584.csv', usecols=['BeneID', 'Provider'])
pr = pd.concat([ip, op]).drop_duplicates().merge(bene[['BeneID', 'Age']], on='BeneID')
fraud_prov = set(pf.Provider[lab])
pr = pr[pr.Provider.isin(fraud_prov)].copy()
pr['D'] = det[pr.Provider.values].values
pr['logn'] = np.log1p(pf.set_index('Provider').TotalUniqueBene[pr.Provider.values].values.astype(float))
q = pr.Age.quantile([.25, .75]).values
pr['ext'] = ((pr.Age <= q[0]) | (pr.Age > q[1])).astype(float); pr['top'] = (pr.Age > q[1]).astype(float)
L = [f"links (beneficiary, fraud-labeled provider): {len(pr):,}; beneficiaries {pr.BeneID.nunique():,}; providers {pr.Provider.nunique()}; detection rate over links {pr.D.mean():.3f}"]
import statsmodels.api as sm
g = pd.factorize(pr.Provider)[0]
def fit(cols):
    X = sm.add_constant(pr[cols].values); r = sm.Logit(pr.D.values, X).fit(disp=0, cov_type='cluster', cov_kwds={'groups': g}); ci = np.exp(r.conf_int()[1])
    return f"{np.exp(r.params[1]):.3f} [{ci[0]:.3f}, {ci[1]:.3f}] p={r.pvalues[1]:.3f}"
for nm, c in (('outer quartiles of beneficiary age', 'ext'), ('top quartile of beneficiary age', 'top')):
    L.append(f"{nm}: unadjusted OR {fit([c])}; adjusted for log panel size {fit([c, 'logn'])}")
print('\n'.join(L)); open(OUT / 'patient_level_audit_output.txt', 'w').write('\n'.join(L) + '\n')
