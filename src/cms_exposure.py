"""
CMS exposure analysis: where does the manufactured disparity matter in real claims?

For each of the eight largest CMS specialties (and all providers pooled) the script takes the panel-size distribution
(2022, providers with >= 11 beneficiaries), the within-provider SD sigma_w and the true between-provider SD tau recovered
from the 2021 -> 2022 panel (no individual ages), and applies Proposition 1 to obtain the expected odds ratio of an
outer-quartile audit under a null (no age effect), for a grid of detection-on-size slopes gamma1.  The fraud-incidence slope
(1.398) and the mean detection rate (0.88) are the primary-dataset values: CMS carries no fraud labels.
It then converts the bias into the expected rate of significant disparities for audits of m fraud-labelled providers
(asymptotic Wald test on the log odds ratio, checked by Monte Carlo at m = 500 and 2,000).
Usage: python src/cms_exposure.py        (SIM_REPS=1000 for the Monte Carlo check)
"""
import os, sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from audit_common import *
from scipy.stats import norm
from scipy.optimize import brentq
logistic = lambda t: 1 / (1 + np.exp(-t))
d = pd.read_csv(DATA / 'cms_2021_2022_merged.csv.gz')
d = d[(d.Tot_Benes_21 >= 11) & (d.Tot_Benes_22 >= 11)].copy()
SPEC = ['Certified Registered Nurse Anesthetist (CRNA)', 'Anesthesiology', 'Emergency Medicine', 'Physician Assistant',
        'Family Practice', 'Physical Therapist in Private Practice', 'Internal Medicine', 'Nurse Practitioner']
lines = []
def emit(s=''): print(s, flush=True); lines.append(s)

def decompose(g):
    """sigma_w from the variance of the year-to-year change, tau^2 from the observed variance of the 2022 value."""
    dA = (g.Bene_Avg_Age_22 - g.Bene_Avg_Age_21).values
    inv = (1 / g.Tot_Benes_21 + 1 / g.Tot_Benes_22).values
    keep = np.minimum(g.Tot_Benes_21, g.Tot_Benes_22).values >= 50      # large panels: sampling noise is the smaller part
    # (dA)^2 = drift + sigma_w^2 (1/n1 + 1/n2): the slope is sigma_w^2, the intercept absorbs genuine year-to-year drift
    sw2 = max(np.polyfit(inv[keep], dA[keep] ** 2, 1)[0], 1.0)
    n22 = g.Tot_Benes_22.values.astype(float)
    V = g.Bene_Avg_Age_22.var(ddof=1)
    samp = sw2 * np.mean(1 / n22)
    return np.sqrt(sw2), max(V - samp, 1e-6), V, samp / V

def prop1(n, tau, sw, mu, g1, part='outer', g0=None, F1=1.398, target_det=0.858):
    z = (np.log1p(n) - np.log1p(n).mean()) / np.log1p(n).std(ddof=1)
    s = np.sqrt(tau ** 2 + sw ** 2 / n)
    # panel-size weights of fraud-labelled units: incidence logistic(F0 + F1 z) with mean 9.35%
    F0 = brentq(lambda a: logistic(a + F1 * z).mean() - 0.0935, -15, 5)
    w = logistic(F0 + F1 * z); w = w / w.sum()
    G0 = brentq(lambda a: (w * logistic(a + g1 * z)).sum() - target_det, -15, 15)
    h = logistic(G0 + g1 * z)
    f = lambda q, p: np.mean(norm.cdf((q - mu) / s)) - p
    q25, q75 = brentq(f, mu - 60, mu + 60, args=(.25,)), brentq(f, mu - 60, mu + 60, args=(.75,))
    pi = norm.cdf((q25 - mu) / s) + 1 - norm.cdf((q75 - mu) / s)
    pb = (w * pi).sum(); p1 = (w * h * pi).sum() / pb; p0 = (w * h * (1 - pi)).sum() / (1 - pb)
    lor = np.log((p1 / (1 - p1)) / (p0 / (1 - p0)))
    return lor, pb, p1, p0

def sig_rate(lor, pb, p1, p0, m):
    se = np.sqrt(1 / (m * pb * p1) + 1 / (m * pb * (1 - p1)) + 1 / (m * (1 - pb) * p0) + 1 / (m * (1 - pb) * (1 - p0)))
    t = abs(lor) / se
    return norm.cdf(-1.96 + t) + norm.cdf(-1.96 - t)

def mc_rate(n, tau, sw, mu, g1, m, reps, rng, F1=1.398, target_det=0.858):
    z = (np.log1p(n) - np.log1p(n).mean()) / np.log1p(n).std(ddof=1)
    F0 = brentq(lambda a: logistic(a + F1 * z).mean() - 0.0935, -15, 5)
    w = logistic(F0 + F1 * z); wn = w / w.sum()
    G0 = brentq(lambda a: (wn * logistic(a + g1 * z)).sum() - target_det, -15, 15)
    # an audit population of N providers whose expected number of fraud-labelled units is m
    N = int(round(m / 0.0935)); hits = 0; tot = 0; ors = []
    for _ in range(reps):
        idx = rng.integers(0, len(n), N)
        nn, zz = n[idx], z[idx]
        A = mu + rng.normal(0, tau, N) + rng.normal(0, sw / np.sqrt(nn))
        q = np.quantile(A, [.25, .75]); G = ((A <= q[0]) | (A > q[1])).astype(float)
        fr = rng.random(N) < logistic(F0 + F1 * zz)
        D = (rng.random(N) < logistic(G0 + g1 * zz))[fr].astype(float)
        if fr.sum() < 40 or D.sum() in (0, len(D)): continue
        o = logit_or(D, [G[fr]]); tot += 1; hits += o[1] < .05; ors.append(o[0])
    return hits / tot * 100, np.median(ors)

GRID = (0.25, 0.5, 0.901, 1.213)
MS = (500, 2000, 10000, 100000)
emit("Expected null odds ratio of an outer-quartile audit (Proposition 1) and expected significant-disparity rate")
emit(f"{'group':40} {'n':>8} {'med n':>6} {'sigma_w':>7} {'tau':>5} {'samp.share':>10} {'med R':>6} | " + ' '.join(f"OR@{g}" for g in GRID))
rows = []
groups = [('All providers (pooled)', d)] + [(s, d[d.Rndrng_Prvdr_Type_22 == s]) for s in SPEC]
for name, g in groups:
    sw, tau2, V, share = decompose(g); tau = np.sqrt(tau2)
    n = g.Tot_Benes_22.values.astype(float); mu = g.Bene_Avg_Age_22.mean()
    R = tau2 / (tau2 + sw ** 2 / n)
    res = {g1: prop1(n, tau, sw, mu, g1) for g1 in GRID}
    emit(f"{name:40} {len(g):8d} {np.median(n):6.0f} {sw:7.2f} {tau:5.2f} {share*100:9.1f}% {np.median(R):6.3f} | " +
         ' '.join(f"{np.exp(res[g1][0]):6.3f}" for g1 in GRID))
    for g1 in GRID:
        for m in MS:
            rows.append(dict(group=name, gamma1=g1, m=m, n_providers=len(g), tau=tau, sigma_w=sw, median_R=np.median(R),
                             expected_OR=np.exp(res[g1][0]), sig_rate=sig_rate(*res[g1], m) * 100))
t = pd.DataFrame(rows); t.to_csv(OUT / 'cms_exposure.csv', index=False)
emit("\nExpected significant-disparity rate (%) of the unadjusted audit under the null, nominal 5%, by number of fraud-labelled audited providers m")
for g1 in GRID:
    emit(f"\n-- detection slope gamma1 = {g1} per SD of log panel size")
    p = t[t.gamma1 == g1].pivot(index='group', columns='m', values='sig_rate').round(1)
    p = p.reindex([x[0] for x in groups]); emit(p.to_string())
reps = int(os.environ.get('SIM_REPS', 400)); rng = np.random.default_rng(20261126)
emit(f"\nMonte Carlo check of the asymptotic rate ({reps} reps; gamma1 = 1.213)")
for name, g in [groups[0], groups[7]]:
    sw, tau2, V, share = decompose(g); n = g.Tot_Benes_22.values.astype(float); mu = g.Bene_Avg_Age_22.mean()
    for m in (500, 2000):
        r, o = mc_rate(n, np.sqrt(tau2), sw, mu, 1.213, m, reps, rng)
        a = t[(t.group == name) & (t.gamma1 == 1.213) & (t.m == m)].sig_rate.iloc[0]
        emit(f"  {name:36} m={m:5d}: Monte Carlo {r:5.1f}% (median OR {o:.3f})   analytic {a:5.1f}%")
open(OUT / 'cms_exposure_output.txt', 'w').write('\n'.join(lines) + '\n')
