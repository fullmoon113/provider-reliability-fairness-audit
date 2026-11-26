"""Corollary 1 (paper Section III-E): factorization of the spurious association and the rate at which an audit reports it.

For an audit of the units with L = 1 (weights w(n) over the panel-size distribution), with
  pi(n) = Pr(G = 1 | n)  from Proposition 1 (partition boundaries = quantiles of the measured values of ALL units),
  h(n)  = Pr(D = 1 | n, L = 1),
the script returns eta_G, eta_D, rho, phi = Corr(D, G) = rho * eta_D * eta_G, the null gap Delta, and the approximate
probability that a two-sided level-alpha test of D vs G rejects in an audit of m units,
  Phi(sqrt(m)|phi| - z) + Phi(-sqrt(m)|phi| - z).

    python src/corollary1.py        # writes output/corollary1_table.csv and output/corollary1_output.txt (Table S19)
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
logistic = lambda t: 1 / (1 + np.exp(-t))

def boundaries(n, tau, sw, mu, probs=(.25, .75)):
    """Quantiles of the measured attribute over ALL units: A | n ~ N(mu, tau^2 + sw^2/n)."""
    s = np.sqrt(tau ** 2 + sw ** 2 / n)
    return [brentq(lambda q, p=p: np.mean(norm.cdf((q - mu) / s)) - p, mu - 80 * s.max(), mu + 80 * s.max()) for p in probs], s

def pi_of_n(n, tau, sw, mu, kind='outer'):
    (q25, q75), s = boundaries(n, tau, sw, mu)
    if kind == 'outer':
        return norm.cdf((q25 - mu) / s) + 1 - norm.cdf((q75 - mu) / s)
    return 1 - norm.cdf((q75 - mu) / s)          # top quartile

def corollary1(w, h, pi, m, alpha=0.05):
    """w: weights of audited units over the panel-size distribution (sum to 1); h, pi: h(n), pi(n) at the same points."""
    w = np.asarray(w, float) / np.sum(w)
    hb, pb = (w * h).sum(), (w * pi).sum()
    vh, vp = (w * (h - hb) ** 2).sum(), (w * (pi - pb) ** 2).sum()
    cov = (w * (h - hb) * (pi - pb)).sum()
    eta_D = np.sqrt(vh / (hb * (1 - hb))); eta_G = np.sqrt(vp / (pb * (1 - pb)))
    rho = cov / np.sqrt(vh * vp) if vh * vp > 0 else 0.0
    phi = rho * eta_D * eta_G                                  # = cov / sqrt(hb(1-hb) pb(1-pb))
    delta = cov / (pb * (1 - pb))                              # Prop. 1(i) null gap
    z = norm.ppf(1 - alpha / 2); t = np.sqrt(m) * abs(phi)
    rate = norm.cdf(t - z) + norm.cdf(-t - z)
    return dict(eta_G=eta_G, eta_D=eta_D, rho=rho, phi=phi, delta=delta, rate=rate * 100, pbar=pb, hbar=hb)

def null_sim_setting(n, tau, sw, mu, F, G, kind='outer', m=None):
    """Setting of null_sim.py: fraud ~ logistic(F0 + F1 z), detection ~ logistic(G0 + G1 z), z = standardized log(1 + n)."""
    logn = np.log1p(n); z = (logn - logn.mean()) / logn.std(ddof=1)
    pf = logistic(F[0] + F[1] * z); h = logistic(G[0] + G[1] * z)
    pi = pi_of_n(n, tau, sw, mu, kind)
    return corollary1(pf, h, pi, pf.sum() if m is None else m)

if __name__ == '__main__':
    from audit_common import load, DATA, OUT
    from simparams import B_FRAUD, B_DET
    SW = 12.7275; pf_ = load(); n = pf_.TotalUniqueBene.values.astype(float); mu = pf_.AvgPatientAge.mean()
    rows = []
    def add(panel, label, tau, g1mult, nvec, sim, sim_se_reps, **kw):
        r = null_sim_setting(nvec, tau, SW, mu, B_FRAUD, (B_DET[0], B_DET[1] * g1mult))
        rows.append(dict(panel=panel, setting=label, tau=tau, slope_mult=g1mult, eta_G=r['eta_G'], eta_D=r['eta_D'], rho=r['rho'], phi=r['phi'],
                         predicted=r['rate'], simulated=sim, mc_se=np.sqrt(sim / 100 * (1 - sim / 100) / sim_se_reps) * 100 if sim_se_reps else np.nan))
    A = json.load(open(OUT / 'sim5k.json')); B = {}
    for f in ('sim_panelB_0.5_0.75_1.0.json', 'sim_panelB_1.25_1.5.json'): B.update(json.load(open(OUT / f)))
    C = json.load(open(OUT / 'sim5k_C.json'))
    for sd in (0.0, 0.5, 1.0, 2.0, 4.0, 6.0):
        r = A[f'A|{sd}']; add('A', f'tau = {sd}', sd, 1.0, n, r['fpr_u'], r['reps'])
    for mult in (0.5, 0.75, 1.0, 1.25, 1.5):
        r = B[str(mult)]; add('B', f'slope x {mult}', 0.9, mult, n, r['fpr_u'], r['reps'])
    for k, r in C.items():
        thr = {'>=11': 11, '>=50': 50}.get(k.split(', ')[-1], 1)
        nv = n[n >= thr]; add('C', k[2:], r['sd'], 1.0, nv, r['fpr_u'], r['reps'])
    t = pd.DataFrame(rows); t.to_csv(OUT / 'corollary1_table.csv', index=False)
    pd.set_option('display.width', 220)
    s = t.round(3).to_string(index=False)
    s += f"\n\nmax |predicted - simulated| = {np.abs(t.predicted - t.simulated).max():.2f} points; mean {np.abs(t.predicted - t.simulated).mean():.2f}"
    print(s); open(OUT / 'corollary1_output.txt', 'w').write(s + '\n')
