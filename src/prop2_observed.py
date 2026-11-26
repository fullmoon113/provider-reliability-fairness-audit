"""Proposition 2 applied to the primary dataset: unadjusted odds ratios of detection (class-weighted logistic regression,
extreme age-mix definition replaced by the partitions below) for partitions on (i) the observed mean age, (ii) the posterior mean,
(iii) one posterior draw (median and 2.5-97.5% range over 1,000 draws) and (iv) the posterior probability of the true outer
quartiles (per SD, without and with log panel size).  tau^2 = 0.812 (REML), sigma_w = 12.7275.
Usage: python src/prop2_observed.py -> output/prop2_observed_output.txt"""
import numpy as np
from scipy.stats import norm
from audit_common import load, oof_lr, logit_or, outer, OUT
pf = load(); y = pf.Fraud.values; fm = y == 1
n = pf.TotalUniqueBene.values.astype(float); A = pf.AvgPatientAge.values; ext = pf.IsExtremeAge.values
det = oof_lr(pf, ext, strategy='class_weight')[fm]; lg = np.log1p(n[fm])
t2, sw2 = 0.812, 12.7275 ** 2; v = sw2 / n
mu = A.mean(); R = t2 / (t2 + v); Ah = mu + R * (A - mu); sd = np.sqrt(R * v)
rng = np.random.default_rng(20261127); out = []
def P(s): out.append(s); print(s)
o = logit_or(det, [outer(A)[fm]]); P(f'observed attribute (outer quartiles): OR {o[0]:.3f} [{o[2]:.3f}, {o[3]:.3f}] p={o[1]:.4f}')
o = logit_or(det, [outer(Ah)[fm]]); P(f'posterior mean partition: OR {o[0]:.3f} [{o[2]:.3f}, {o[3]:.3f}] p={o[1]:.4f}')
ors = []; ps = []
for _ in range(1000):
    g = outer(Ah + sd * rng.standard_normal(len(n)))[fm]
    r = logit_or(det, [g]); ors.append(r[0]); ps.append(r[1])
P(f'posterior draw partition (1,000 draws): median OR {np.median(ors):.3f}, 2.5-97.5% range [{np.quantile(ors,.025):.3f}, {np.quantile(ors,.975):.3f}]; share with p<0.05 {np.mean(np.array(ps)<.05)*100:.1f}%')
lo, hi = mu + norm.ppf(.25) * np.sqrt(t2), mu + norm.ppf(.75) * np.sqrt(t2)
X = norm.cdf((lo - Ah) / sd) + 1 - norm.cdf((hi - Ah) / sd); Z = (X[fm] - X[fm].mean()) / X[fm].std()
a = logit_or(det, [Z]); b = logit_or(det, [Z, lg])
P(f'posterior probability of the outer true quartiles (per SD): OR {a[0]:.3f} [{a[2]:.3f}, {a[3]:.3f}] p={a[1]:.4f}; with log panel size: OR {b[0]:.3f} [{b[2]:.3f}, {b[3]:.3f}] p={b[1]:.4f}')
(OUT / 'prop2_observed_output.txt').write_text('\n'.join(out) + '\n')
