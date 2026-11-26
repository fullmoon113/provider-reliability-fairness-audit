"""
Type I error and power of the audit estimators when the attribute is an unreliable aggregate (a guard against
"fairness-washing").  A true detection penalty with odds ratio OR_true is planted on the TRUE group G* (outer quartiles of the
true provider mean age A*).  Panel sizes are the empirical ones of the primary dataset (or resampled to a larger audit).
Grid: OR_true in {1.0, 0.8, 0.7, 0.5}; tau in {0.9, 1.65, 3.47, 4.85}; corr(A*, log n) in {0, 0.24} (0.24 = coupling observed in CMS);
audit size: primary (5,410 providers, ~506 fraud-labelled) and large (40,000 providers, ~3,700 fraud-labelled).
Estimators (detection among fraud-labelled providers, odds ratio of the group indicator):
  naive      outer quartiles of the measured age
  adjusted   naive + log panel size
  shrunken   outer quartiles of the empirical-Bayes shrunken age + log panel size
  RC         regression-calibration probability of the true outer group + log panel size (per-SD scale)
  procedure  the staged rule of Section III-E: Stage 0 expected misclassification <= 10% -> naive audit; else if Cochran's Q rejects
             tau = 0 -> shrunken + size-adjusted audit; else decline to partition (counted as 'declined', not as a rejection)
  oracle     outer quartiles of the true age (unavailable in practice)
Reports: share of replicates with p < 0.05 (type I error at OR_true = 1, power otherwise), median odds ratio, share declined.
Usage: SIM_REPS=500 python src/planted_effect_power.py
"""
import os, sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from audit_common import *
from scipy.stats import norm, chi2
from multiprocessing import Pool
pf = load(); n0 = pf.TotalUniqueBene.values.astype(float)
SW = 12.7275; MU = pf.AvgPatientAge.mean(); import sys; sys.path.insert(0, str(HERE)); from simparams import B_DET, B_FRAUD
G0, G1 = B_DET; F0, F1 = B_FRAUD
logistic = lambda t: 1 / (1 + np.exp(-t))
REPS = int(os.environ.get('SIM_REPS', 500))
EPS = 0.10

# misclassification of the outer-vs-middle split as a function of reliability (MC, standard normal)
_rg = np.linspace(0.0, 1.0, 41); _r = np.random.default_rng(1); _t = _r.standard_normal(1_000_000); _e = _r.standard_normal(1_000_000)
def _mis(R):
    o = np.sqrt(R) * _t + np.sqrt(1 - R) * _e
    qt, qo = np.quantile(_t, [.25, .75]), np.quantile(o, [.25, .75])
    return (((_t < qt[0]) | (_t > qt[1])) != ((o < qo[0]) | (o > qo[1]))).mean()
_mv = np.array([_mis(R) for R in _rg])
mis = lambda R: np.interp(R, _rg, _mv)

def one(args):
    tau, orr, rho, big, seed = args
    rng = np.random.default_rng(seed)
    n = n0 if not big else rng.choice(n0, 40000, replace=True)
    N = len(n); logn = np.log1p(n); z = (logn - logn.mean()) / logn.std(ddof=1)
    v = SW ** 2 / n
    out = {k: [] for k in ('naive', 'adjusted', 'shrunken', 'RC', 'procedure', 'oracle')}; declined = 0; done = 0
    for _ in range(REPS):
        Astar = MU + tau * (rho * z + np.sqrt(1 - rho ** 2) * rng.standard_normal(N))
        A = Astar + rng.normal(0, np.sqrt(v))
        Gs = outer(Astar); Go = outer(A)
        fr = rng.random(N) < logistic(F0 + F1 * z)
        D = (rng.random(N) < logistic(G0 + G1 * z + np.log(orr) * Gs))
        if fr.sum() < 40 or D[fr].sum() in (0, fr.sum()): continue
        d = D[fr].astype(float); lg = logn[fr]
        w = 1 / v; Q = (w * (A - (w * A).sum() / w.sum()) ** 2).sum()
        s2 = max((Q - (N - 1)) / (w.sum() - (w ** 2).sum() / w.sum()), 1e-3); Rel = s2 / (s2 + v)   # DerSimonian-Laird (precision-weighted)
        wts = 1 / (s2 + v); mu_hat = (wts * A).sum() / wts.sum(); Ahat = Rel * A + (1 - Rel) * mu_hat
        Gsh = outer(Ahat)
        sd_post = np.sqrt(Rel * v); lo, hi = mu_hat + norm.ppf(.25) * np.sqrt(s2), mu_hat + norm.ppf(.75) * np.sqrt(s2)
        P = norm.cdf((lo - Ahat) / sd_post) + 1 - norm.cdf((hi - Ahat) / sd_post)
        Pz = (P[fr] - P[fr].mean()) / max(P[fr].std(), 1e-12)
        if Go[fr].sum() < 5 or Go[fr].sum() > fr.sum() - 5: continue
        done += 1
        res = {'naive': logit_or(d, [Go[fr]]), 'adjusted': logit_or(d, [Go[fr], lg]),
               'shrunken': logit_or(d, [Gsh[fr], lg]), 'RC': logit_or(d, [Pz, lg]), 'oracle': logit_or(d, [Gs[fr]])}
        # staged procedure
        q_sig = chi2.sf(Q, N - 1) < 0.05
        if mis(Rel).mean() <= EPS: res['procedure'] = res['naive']
        elif q_sig: res['procedure'] = res['shrunken']
        else: res['procedure'] = (np.nan, np.nan); declined += 1
        for k, r in res.items(): out[k].append((r[0], r[1]))
    row = dict(tau=tau, OR_true=orr, rho=rho, audit='large' if big else 'primary', reps=done, declined=declined / max(done, 1) * 100)
    for k, v_ in out.items():
        a = np.array(v_, float); ok = ~np.isnan(a[:, 1]); a = a[ok]
        row[k + '_rej'] = (a[:, 1] < .05).mean() * 100 if len(a) else np.nan
        row[k + '_or'] = np.median(a[:, 0]) if len(a) else np.nan
    return row

if __name__ == '__main__':
    grid = [(t, o, r, b, 1000 + i) for i, (t, o, r, b) in enumerate(
        [(t, o, r, b) for b in (0, 1) for r in (0.0, 0.24) for t in (0.9, 1.65, 3.47, 4.85) for o in (1.0, 0.8, 0.7, 0.5)])]
    with Pool(2) as p: rows = p.map(one, grid, chunksize=1)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'planted_effect_power.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    cols = ['tau', 'OR_true', 'rho', 'audit', 'reps', 'declined'] + [f'{k}_rej' for k in ('naive', 'adjusted', 'shrunken', 'RC', 'procedure', 'oracle')]
    txt = t[cols].round(1).to_string(index=False)
    txt2 = t[['tau', 'OR_true', 'rho', 'audit'] + [f'{k}_or' for k in ('naive', 'adjusted', 'shrunken', 'RC', 'procedure', 'oracle')]].round(3).to_string(index=False)
    s = "Rejection rate (%): type I error at OR_true = 1.0, power otherwise\n" + txt + "\n\nMedian odds ratio\n" + txt2 + "\n"
    open(OUT / 'planted_effect_power_output.txt', 'w').write(s); print(s)
