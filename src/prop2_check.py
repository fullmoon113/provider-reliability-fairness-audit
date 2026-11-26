"""Proposition 2 check (paper Section IV-J, Table S13 Panel A): which profiling-style estimates carry a size artifact under the null.
Same null simulation as Table 6 (true age effect zero; tau = 0.90).  Partitions (outer quartiles) of: the observed age (naive); the posterior
mean (Eq. 4); ONE draw from the posterior of the true age (Prop. 2(a)); and the posterior probability of the true outer group (Prop. 2(b),
entered as a standardized covariate, with and without log panel size).  Each is evaluated with tau^2 known and with tau^2 estimated
(DerSimonian-Laird).  Reports the median odds ratio per SD-scale or group and the rejection rate at 5%.
Usage: SIM_REPS=2000 python src/prop2_check.py"""
import os, sys, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from scipy.stats import norm
from audit_common import logit_or, outer, OUT
spec = importlib.util.spec_from_file_location("ns", str(HERE / "null_sim.py")); ns = importlib.util.module_from_spec(spec); sys.modules["ns"] = ns; spec.loader.exec_module(ns)
REPS = int(os.environ.get('SIM_REPS', 2000)); TAU = 0.90; rng = np.random.default_rng(20261127)
n, logn, z = ns.n, ns.logn, ns.z; v = ns.SW ** 2 / n; N = len(n)
pfr = ns.logistic(ns.B_FRAUD[0] + ns.B_FRAUD[1] * z); pdet = ns.logistic(ns.B_DET[0] + ns.B_DET[1] * z)
res = {}
def add(k, r): res.setdefault(k, []).append(r)
for _ in range(REPS):
    Astar = ns.MU_AGE + rng.normal(0, TAU, N); A = Astar + rng.normal(0, np.sqrt(v))
    fr = rng.random(N) < pfr; D = (rng.random(N) < pdet)
    if fr.sum() < 40 or D[fr].sum() in (0, fr.sum()): continue
    d = D[fr].astype(float); lg = logn[fr]
    w = 1 / v; Q = (w * (A - (w * A).sum() / w.sum()) ** 2).sum()
    s2h = max((Q - (N - 1)) / (w.sum() - (w ** 2).sum() / w.sum()), 1e-3)
    for lab, s2 in (('known', TAU ** 2), ('estimated', s2h)):
        Rl = s2 / (s2 + v); wm = 1 / (s2 + v); mu = (wm * A).sum() / wm.sum(); Ah = mu + Rl * (A - mu); sd = np.sqrt(Rl * v)
        lo, hi = mu + norm.ppf(.25) * np.sqrt(s2), mu + norm.ppf(.75) * np.sqrt(s2)
        P = norm.cdf((lo - Ah) / sd) + 1 - norm.cdf((hi - Ah) / sd); Pz = (P[fr] - P[fr].mean()) / max(P[fr].std(), 1e-12)
        Gd = outer(Ah + sd * rng.standard_normal(N))
        add(f'posterior mean partition ({lab} tau)', logit_or(d, [outer(Ah)[fr]])[:2])
        add(f'posterior draw partition ({lab} tau)', logit_or(d, [Gd[fr]])[:2])
        add(f'posterior probability, no size term ({lab} tau)', logit_or(d, [Pz])[:2])
        add(f'posterior probability + size ({lab} tau)', logit_or(d, [Pz, lg])[:2])
    add('observed partition (naive)', logit_or(d, [outer(A)[fr]])[:2])
rows = []
for k, v_ in res.items():
    a = np.array(v_, float); a = a[~np.isnan(a[:, 1])]
    rows.append(dict(estimate=k, median_OR=np.median(a[:, 0]), lo=np.quantile(a[:, 0], .025), hi=np.quantile(a[:, 0], .975), rejection=(a[:, 1] < .05).mean() * 100, reps=len(a)))
t = pd.DataFrame(rows); t.to_csv(OUT / 'prop2_check.csv', index=False); s = t.round(3).to_string(index=False)
print(s); open(OUT / 'prop2_check_output.txt', 'w').write(s + '\n')
