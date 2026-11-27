"""
Type I error and power of the audit estimators when the attribute is an unreliable aggregate (Table 9, Table S6 Panel D).
A true detection penalty with odds ratio OR_true is planted on the TRUE group G* (outer quartiles of the true provider mean age
A*).  Panel sizes are the empirical ones of the primary dataset (or resampled to a larger audit).
Grid: OR_true in {1.0, 0.8, 0.7, 0.5}; tau in {0.9, 1.65, 3.47, 4.85}; corr(A*, log n) in {0, 0.24} (0.24 = coupling observed in CMS);
audit size: primary (5,410 providers, ~506 fraud-labeled) and large (40,000 providers, ~3,700 fraud-labeled); plus two cells that
reach the other branches of Algorithm 1: tau = 0 (no true between-provider variance; the groups are random) and a high-reliability
audit (tau = 4.85, 20,000 providers resampled from those with at least 101 beneficiaries).
Estimators (detection among fraud-labeled providers, odds ratio of the group indicator):
  naive      outer quartiles of the measured age
  adjusted   naive + log panel size
  shrunken   outer quartiles of the empirical-Bayes shrunken age + log panel size
  RC         regression-calibration probability of the true outer group + log panel size (per-SD scale)
  RCnosize   the same probability without the size term (Proposition 2(b))
  postdraw   outer quartiles of one posterior draw of the true age (Proposition 2(a)), no size term
  procedure  Algorithm 1 (eps = 10%, alpha_s = 7.5%, delta = 0.70): if Cochran's Q does not reject tau = 0 -> decline to partition;
             else if M <= eps and r_hat <= alpha_s -> partition: unadjusted estimate; else size-adjusted estimate on the observed
             attribute, and decline to attribute if its minimum detectable odds ratio is above delta
  oracle     outer quartiles of the true age (unavailable in practice)
Reports: share of replicates with p < 0.05 (type I error at OR_true = 1, power otherwise; the procedure among replicates in which it does
not decline), the share of all replicates in which the procedure attributes a significant gap, and the branch shares.
Usage: SIM_REPS=1000 python src/planted_effect_power.py
"""
import os, sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from audit_common import *
from scipy.stats import norm, chi2
from multiprocessing import Pool
from stage0 import expected_misclassification
pf = load(); n0 = pf.TotalUniqueBene.values.astype(float)
SW = 12.7275; MU = pf.AvgPatientAge.mean(); import sys; sys.path.insert(0, str(HERE)); from simparams import B_DET, B_FRAUD
G0, G1 = B_DET; F0, F1 = B_FRAUD
logistic = lambda t: 1 / (1 + np.exp(-t))
REPS = int(os.environ.get('SIM_REPS', 1000))
EPS, ALPHA_S, DELTA = 0.10, 0.075, 0.70
TAUG = np.concatenate([[0.05], np.geomspace(0.2, 9.0, 24)])

def one(args):
    tau, orr, rho, kind, seed = args
    rng = np.random.default_rng(seed)
    n = n0 if kind == 'primary' else (rng.choice(n0, 40000, replace=True) if kind == 'large' else rng.choice(n0[n0 >= 101], 20000, replace=True))
    N = len(n); logn = np.log1p(n); z = (logn - logn.mean()) / logn.std(ddof=1)
    v = SW ** 2 / n; pfr = logistic(F0 + F1 * z)
    # M(tau) and attenuation lambda(tau) by simulation on this n vector (weights = fraud-labeling probability); interpolated per replicate
    Mg, Lg = [], []
    for t in TAUG:
        M_, _, l_ = expected_misclassification(n, MU, t ** 2, SW, 'outer', pfr, B=4, seed=int(t * 1000) + 7); Mg.append(M_); Lg.append(l_)
    Mfun = lambda t: np.interp(t, TAUG, Mg); Lfun = lambda t: np.interp(t, TAUG, Lg)
    keys = ('naive', 'adjusted', 'shrunken', 'RC', 'RCnosize', 'postdraw', 'procedure', 'oracle')
    out = {k: [] for k in keys}; br = dict(partition=0, adjust=0, decline_partition=0, decline_attribute=0); attrib = 0; done = 0
    for _ in range(REPS):
        if tau > 0: Astar = MU + tau * (rho * z + np.sqrt(1 - rho ** 2) * rng.standard_normal(N))
        else: Astar = MU + 1e-6 * rng.standard_normal(N)
        A = Astar + rng.normal(0, np.sqrt(v))
        Gs = outer(Astar); Go = outer(A)
        fr = rng.random(N) < pfr
        D = (rng.random(N) < logistic(G0 + G1 * z + np.log(orr) * Gs))
        if fr.sum() < 40 or D[fr].sum() in (0, fr.sum()): continue
        d = D[fr].astype(float); lg = logn[fr]
        w = 1 / v; Q = (w * (A - (w * A).sum() / w.sum()) ** 2).sum()
        s2 = max((Q - (N - 1)) / (w.sum() - (w ** 2).sum() / w.sum()), 1e-3); Rel = s2 / (s2 + v)   # DerSimonian-Laird
        wts = 1 / (s2 + v); mu_hat = (wts * A).sum() / wts.sum(); Ahat = Rel * A + (1 - Rel) * mu_hat
        Gsh = outer(Ahat)
        sd_post = np.sqrt(Rel * v); lo, hi = mu_hat + norm.ppf(.25) * np.sqrt(s2), mu_hat + norm.ppf(.75) * np.sqrt(s2)
        P = norm.cdf((lo - Ahat) / sd_post) + 1 - norm.cdf((hi - Ahat) / sd_post)
        Pz = (P[fr] - P[fr].mean()) / max(P[fr].std(), 1e-12)
        Gpd = outer(Ahat + sd_post * rng.standard_normal(N))
        if Go[fr].sum() < 5 or Go[fr].sum() > fr.sum() - 5: continue
        done += 1
        res = {'naive': logit_or(d, [Go[fr]]), 'adjusted': logit_or(d, [Go[fr], lg]),
               'shrunken': logit_or(d, [outer(Ahat)[fr], lg]), 'RC': logit_or(d, [Pz, lg]), 'RCnosize': logit_or(d, [Pz]),
               'postdraw': logit_or(d, [Gpd[fr]]), 'oracle': logit_or(d, [Gs[fr]])}
        # Algorithm 1
        sep = chi2.sf(Q, N - 1) < 0.05 and s2 > 1.5e-3
        if not sep:
            res['procedure'] = (np.nan, np.nan); br['decline_partition'] += 1
        else:
            t_hat = np.sqrt(s2); Mh = Mfun(t_hat); lam = max(Lfun(t_hat), 1e-6)
            s_ = np.sqrt(s2 + v); q = np.quantile(A, [.25, .75]); pi = norm.cdf((q[0] - mu_hat) / s_) + 1 - norm.cdf((q[1] - mu_hat) / s_)
            try:
                X = sm.add_constant(np.column_stack([lg, lg ** 2])); f = sm.Logit(d, X).fit(disp=0, maxiter=100)
                h = logistic(sm.add_constant(np.column_stack([logn, logn ** 2])) @ f.params)
            except Exception:
                h = np.full(N, d.mean())
            wa = fr / fr.sum(); pb = (wa * pi).sum(); hb = (wa * h).sum()
            phi = (wa * (h - hb) * (pi - pb)).sum() / np.sqrt(hb * (1 - hb) * pb * (1 - pb)); t_ = np.sqrt(fr.sum()) * abs(phi)
            r_hat = norm.cdf(t_ - 1.96) + norm.cdf(-t_ - 1.96)
            if Mh <= EPS and r_hat <= ALPHA_S:
                res['procedure'] = res['naive']; br['partition'] += 1
            else:
                se = np.log(res['adjusted'][2:4] if False else 1) or 0
                ra = logit_or(d, [Go[fr], lg])
                X2 = sm.add_constant(np.column_stack([Go[fr], lg]), has_constant='add')
                try: se = sm.Logit(d, X2).fit(disp=0, maxiter=100).bse[1]
                except Exception: se = np.nan
                mde = np.exp(-(1.96 + 0.8416) * se / lam)
                if not np.isfinite(mde) or mde < DELTA:
                    res['procedure'] = (np.nan, np.nan); br['decline_attribute'] += 1
                else:
                    res['procedure'] = res['adjusted']; br['adjust'] += 1
        if not np.isnan(res['procedure'][1]) and res['procedure'][1] < .05: attrib += 1
        for k, r in res.items(): out[k].append((r[0], r[1]))
    row = dict(tau=tau, OR_true=orr, rho=rho, audit=kind, reps=done, attrib_rate=attrib / max(done, 1) * 100,
               **{f'br_{k}': v_ / max(done, 1) * 100 for k, v_ in br.items()})
    for k, v_ in out.items():
        a = np.array(v_, float); ok = ~np.isnan(a[:, 1]); a = a[ok]
        row[k + '_rej'] = (a[:, 1] < .05).mean() * 100 if len(a) else np.nan
        row[k + '_or'] = np.median(a[:, 0]) if len(a) else np.nan
    return row

if __name__ == '__main__':
    cells = [(t, o, r, b) for b in ('primary', 'large') for r in (0.0, 0.24) for t in (0.9, 1.65, 3.47, 4.85) for o in (1.0, 0.8, 0.7, 0.5)]
    cells += [(0.0, 1.0, 0.0, 'primary')] + [(4.85, o, 0.0, 'highrel') for o in (1.0, 0.8, 0.7, 0.5)]
    if os.environ.get('CELLS'): cells = cells[:int(os.environ['CELLS'])]
    grid = [(*c, 2000 + i) for i, c in enumerate(cells)]
    with Pool(2) as p: rows = p.map(one, grid, chunksize=1)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'planted_effect_power.csv', index=False)
    pd.set_option('display.width', 280); pd.set_option('display.max_columns', 40)
    cols = ['tau', 'OR_true', 'rho', 'audit', 'reps', 'br_partition', 'br_adjust', 'br_decline_partition', 'br_decline_attribute', 'attrib_rate'] + [f'{k}_rej' for k in ('naive', 'adjusted', 'shrunken', 'RC', 'RCnosize', 'postdraw', 'procedure', 'oracle')]
    txt = t[cols].round(1).to_string(index=False)
    txt2 = t[['tau', 'OR_true', 'rho', 'audit'] + [f'{k}_or' for k in ('naive', 'adjusted', 'shrunken', 'RC', 'RCnosize', 'postdraw', 'procedure', 'oracle')]].round(3).to_string(index=False)
    s = "Rejection rate (%): type I error at OR_true = 1.0, power otherwise; branch shares (%)\n" + txt + "\n\nMedian odds ratio\n" + txt2 + "\n"
    open(OUT / 'planted_effect_power_output.txt', 'w').write(s); print(s)
