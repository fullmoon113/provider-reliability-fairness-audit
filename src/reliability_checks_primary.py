"""Direct checks of the reliability model in the primary dataset (supplement Table S10).

(1) How many beneficiaries are linked to more than one provider (correlated sampling errors across providers).
(2) Random split-half reliability: each provider's unique beneficiaries are split at random into two halves; the correlation of the
    half means across providers, stepped up with the Spearman-Brown formula, is the reliability of the full-panel mean and is
    compared with the model-implied value R_i = tau^2 / (tau^2 + sigma_w^2 / n_i) (tau = 0.901, sigma_w = 12.7275).
(3) Regression of the squared deviation of the measured age from its mean on 1/n_i: slope = sigma_w^2, intercept = tau^2.
(4) Beneficiary-level bootstrap of the partition: beneficiaries are resampled with replacement (clusters across providers),
    provider mean ages and the outer-quartile partition are recomputed, and the detection outcomes of the reference model are held fixed.
Needs the raw Kaggle files in data/raw/ (see data/DATA.md).   Usage: python src/reliability_checks_primary.py"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from audit_common import *
import scipy.sparse as sp
RAW = DATA / 'raw'
pf = load(); n_i = pf.TotalUniqueBene.values.astype(float); lab = pf.Fraud.values == 1
bene = pd.read_csv(RAW / 'Train_Beneficiarydata-1542865627584.csv', usecols=['BeneID', 'DOB'])
bene['Age'] = (pd.Timestamp('2009-12-31') - pd.to_datetime(bene['DOB'])).dt.days / 365.25
ip = pd.read_csv(RAW / 'Train_Inpatientdata-1542865627584.csv', usecols=['BeneID', 'Provider'])
op = pd.read_csv(RAW / 'Train_Outpatientdata-1542865627584.csv', usecols=['BeneID', 'Provider'])
pr = pd.concat([ip, op]).drop_duplicates().merge(bene[['BeneID', 'Age']], on='BeneID')
pr = pr[pr.Provider.isin(pf.Provider)]
L = []
def P(s=''): print(s); L.append(s)
# (1)
k = pr.groupby('BeneID').Provider.nunique()
P(f"(1) beneficiaries {len(k):,}; linked to more than one provider: {(k > 1).mean() * 100:.1f}%; mean providers per beneficiary {k.mean():.2f}; "
  f"provider-beneficiary links {len(pr):,}")
prov_idx = pd.Series(np.arange(len(pf)), index=pf.Provider.values)
pi_ = prov_idx[pr.Provider.values].values
bidx = pd.factorize(pr.BeneID)[0]; age = pr.Age.values
A = np.asarray(pd.Series(age).groupby(pi_).mean().reindex(range(len(pf))))
TAU, SW = 0.901, 12.7275; MU = A.mean()
R_model = TAU ** 2 / (TAU ** 2 + SW ** 2 / n_i)
# (2) split-half, 200 random splits
rng = np.random.default_rng(20261126)
order = np.argsort(pi_, kind='stable'); pi_s, age_s = pi_[order], age[order]
starts = np.r_[0, np.cumsum(np.bincount(pi_s, minlength=len(pf)))]
bands = [(10, 19), (20, 49), (50, 99), (100, 10 ** 9)]
res = {b: [] for b in bands}
for _ in range(200):
    h1 = np.full(len(pf), np.nan); h2 = np.full(len(pf), np.nan)
    for i in range(len(pf)):
        a = age_s[starts[i]:starts[i + 1]]
        if len(a) < 2: continue
        p = rng.permutation(len(a)); m = len(a) // 2
        h1[i], h2[i] = a[p[:m]].mean(), a[p[m:2 * m]].mean()
    for b in bands:
        ok = (n_i >= b[0]) & (n_i <= b[1]) & ~np.isnan(h1)
        r = np.corrcoef(h1[ok], h2[ok])[0, 1]; res[b].append(2 * r / (1 + r))
P("\n(2) Random split-half reliability of the full-panel mean age (Spearman-Brown, mean over 200 random splits)")
P(f"{'panel size':>12s} {'providers':>10s} {'split-half':>11s} {'model R_i (mean)':>17s}")
for b in bands:
    ok = (n_i >= b[0]) & (n_i <= b[1])
    P(f"{str(b[0]) + ('+' if b[1] > 1e8 else '-' + str(b[1])):>12s} {ok.sum():10d} {np.mean(res[b]):11.3f} {R_model[ok].mean():17.3f}")
# (3) variance regression
import statsmodels.api as sm
d2 = (A - MU) ** 2; X = sm.add_constant(1 / n_i)
f = sm.OLS(d2, X).fit(cov_type='HC3'); ci = f.conf_int()
P(f"\n(3) (A_i - mean)^2 on 1/n_i: sigma_w^2 (slope) {f.params[1]:.1f} [{ci[1][0]:.1f}, {ci[1][1]:.1f}], sigma_w {np.sqrt(f.params[1]):.2f} "
  f"[{np.sqrt(max(ci[1][0], 0)):.2f}, {np.sqrt(ci[1][1]):.2f}]; tau^2 (intercept) {f.params[0]:.3f} [{ci[0][0]:.3f}, {ci[0][1]:.3f}], tau {np.sqrt(max(f.params[0], 0)):.2f}")
# (4) beneficiary bootstrap of the partition
ext_ref = outer(A)
det = oof_lr(pf, ext_ref, strategy='class_weight').astype(bool)[lab]
logn = np.log1p(n_i)[lab]
u0, a0 = audit(det, ext_ref[lab], logn)
nb = bidx.max() + 1
M = sp.csr_matrix((np.ones(len(pr)), (pi_, bidx)), shape=(len(pf), nb))
ors = []; aors = []; sig = 0; sig_adj = 0; B = 500
for b in range(B):
    w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
    num = M @ (w * np.asarray(pd.Series(age).groupby(bidx).first().reindex(range(nb))))
    den = M @ w; ok = den > 0
    Ab = np.where(ok, num / np.maximum(den, 1), MU)
    g = outer(Ab)[lab]
    u, a = audit(det, g, logn); ors.append(u[0]); aors.append(a[0]); sig += u[1] < .05; sig_adj += a[1] < .05
ors, aors = np.array(ors), np.array(aors)
P(f"\n(4) Beneficiary bootstrap (B={B}) of the outer-quartile partition, reference detection fixed: reference OR {u0[0]:.3f} (adjusted {a0[0]:.3f}); "
  f"bootstrap unadjusted OR median {np.median(ors):.3f}, 2.5-97.5 percentile [{np.percentile(ors, 2.5):.3f}, {np.percentile(ors, 97.5):.3f}], significant in {sig / B * 100:.1f}%; "
  f"adjusted OR median {np.median(aors):.3f} [{np.percentile(aors, 2.5):.3f}, {np.percentile(aors, 97.5):.3f}], significant in {sig_adj / B * 100:.1f}%")
open(OUT / 'reliability_checks_primary_output.txt', 'w').write('\n'.join(L) + '\n')
