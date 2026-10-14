from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

"""
Errors-in-variables remedy for the disparity regression (reviewer item D-2).

Question: once the partition attribute is known to be an unreliable aggregate, does a standard measurement-error
correction (regression calibration) remove the manufactured disparity, and does it keep a real one?

Data-generating process = the paper's null simulation (src/null_sim.py).  True provider mean age A*_i = mu + a_i,
a_i ~ N(0, sd_true^2); observed A_i = A*_i + e_i, e_i ~ N(0, SW^2 / n_i).  Fraud incidence and detection depend on
log panel size only.  Scenario 'effect' adds a real detection penalty (odds ratio 0.70) for providers in the outer
quartiles of the TRUE age.

Estimators of the disparity odds ratio among fraud-positive providers (outcome: detected):
  naive       logit D ~ 1[A in outer quartiles of the observed age]
  adj. size   same, plus log panel size
  RC          regression calibration (OR per 1 SD of the calibrated probability, not comparable in scale to the others): replace the indicator by its posterior probability given (A_i, n_i),
              P(A*_i in the outer quartiles | A_i, n_i), using the empirical-Bayes posterior N(A^_i, R_i v_i)
              with sigma^2_true estimated from the data (Eq. eq:decomp of the paper)
  RC + size   RC plus log panel size
  oracle      logit D ~ 1[A* in outer quartiles of the true age]  (not available in practice)
Reported: median OR, share of replicates with p < 0.05, and, in the effect scenario, bias against the oracle.
Usage:  SIM_REPS=1000 python src/eiv_remedy.py
"""
import os, importlib.util, sys
import numpy as np, pandas as pd, statsmodels.api as sm, warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("ns", str(HERE / "null_sim.py"))
ns = importlib.util.module_from_spec(spec); sys.modules["ns"] = ns; spec.loader.exec_module(ns)

R = int(os.environ.get('SIM_REPS', 1000)); SEED = 20261015
n, logn, z = ns.n, ns.logn, ns.z
v_i = ns.SW ** 2 / n
p_f = ns.logistic(ns.B_FRAUD[0] + ns.B_FRAUD[1] * z)
LOG_OR_TRUE = np.log(0.70)

def outer(x):
    q = np.quantile(x, [.25, .75]); return ((x <= q[0]) | (x > q[1])).astype(float)

def logit_or(d, cols):
    X = sm.add_constant(np.column_stack(cols), has_constant='add')
    try:
        r = sm.Logit(d, X).fit(disp=0, maxiter=200)
        return np.exp(r.params[1]), r.pvalues[1]
    except Exception:
        return np.nan, np.nan

def run(sd_true, effect, rng):
    rows = []
    p_d0 = ns.B_DET[0] + ns.B_DET[1] * z
    for _ in range(R):
        Astar = ns.MU_AGE + rng.normal(0, sd_true, len(n))
        A = Astar + rng.normal(0, np.sqrt(v_i))
        Gs, Gobs = outer(Astar), outer(A)
        p_d = ns.logistic(p_d0 + (LOG_OR_TRUE * Gs if effect else 0.0))
        fraud = rng.random(len(n)) < p_f
        det = (rng.random(len(n)) < p_d)
        if fraud.sum() < 40 or det[fraud].sum() in (0, fraud.sum()): continue
        # moment estimate of sigma^2_true and reliability (Eq. decomp / rel)
        s2 = max(A.var(ddof=1) - v_i.mean(), 1e-3)
        mu = np.average(A, weights=1 / (s2 + v_i))
        Rel = s2 / (s2 + v_i)
        Ahat = Rel * A + (1 - Rel) * mu
        sd_post = np.sqrt(Rel * v_i)
        lo, hi = mu + norm.ppf(.25) * np.sqrt(s2), mu + norm.ppf(.75) * np.sqrt(s2)  # outer-quartile cuts of the TRUE age
        P = norm.cdf((lo - Ahat) / sd_post) + 1 - norm.cdf((hi - Ahat) / sd_post)
        f = fraud; d = det[f].astype(float); lg = logn[f]
        if Gobs[f].sum() < 5 or Gobs[f].sum() > f.sum() - 5: continue
        res = {}
        res['naive'] = logit_or(d, [Gobs[f]])
        res['adj. size'] = logit_or(d, [Gobs[f], lg])
        Pz = (P[f] - P[f].mean()) / max(P[f].std(), 1e-12)   # per-SD scale: the raw coefficient explodes when P barely varies
        res['RC'] = logit_or(d, [Pz])
        res['RC + size'] = logit_or(d, [Pz, lg])
        res['oracle'] = logit_or(d, [Gs[f]])
        rows.append({f'{k}_or': v[0] for k, v in res.items()} | {f'{k}_p': v[1] for k, v in res.items()})
    return pd.DataFrame(rows)

lines = []
def emit(s=''):
    print(s); lines.append(s)

rng = np.random.default_rng(SEED)
methods = ['naive', 'adj. size', 'RC', 'RC + size', 'oracle']
for sd_true in (0.50, 1.77):
    for effect in (False, True):
        o = run(sd_true, effect, rng)
        emit(f"\n=== true between-provider SD {sd_true}; true age effect: {'OR 0.70 in outer true quartiles' if effect else 'none (null)'}; {len(o)} replicates ===")
        emit(f"{'estimator':<10} {'median OR':>10} {'[2.5%, 97.5%]':>18} {'p<0.05 (%)':>11}")
        for m in methods:
            x = o[f'{m}_or'].dropna()
            emit(f"{m:<10} {x.median():>10.3f} {'[%.3f, %.3f]' % (x.quantile(.025), x.quantile(.975)):>18} {(o[f'{m}_p'] < .05).mean() * 100:>11.1f}")
open(OUT / 'eiv_remedy_output.txt', 'w').write('\n'.join(lines) + '\n')
