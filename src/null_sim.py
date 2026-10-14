from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

"""
Null simulation: true age effect is exactly zero by construction.
Reproduces the paper's Table 6 / Table A7 and adds CMS-calibrated scenarios.
"""
import numpy as np, pandas as pd, statsmodels.api as sm, warnings
warnings.filterwarnings("ignore")

d = pd.read_csv(str(DATA/'provider_features_FINAL.csv'))
n = d.TotalUniqueBene.values.astype(float)
logn = np.log1p(n)
z = (logn - logn.mean()) / logn.std(ddof=1)

SW = 12.7275          # within-provider patient age SD
B_FRAUD = (-2.975, 1.398)
B_DET   = ( 1.261, 0.901)
MU_AGE  = d.AvgPatientAge.mean()

def logistic(t): return 1.0 / (1.0 + np.exp(-t))

def fit(y, X):
    X = sm.add_constant(X, has_constant='add')
    try:
        r = sm.Logit(y, X).fit(disp=0, maxiter=200)
        return np.exp(r.params[1]), r.pvalues[1]
    except Exception:
        return np.nan, np.nan

def run(sd_true, reps, seed, det_mult=1.0):
    rng = np.random.default_rng(seed)
    out = []
    p_f = logistic(B_FRAUD[0] + B_FRAUD[1] * z)
    p_d = logistic(B_DET[0]  + B_DET[1] * det_mult * z)
    for _ in range(reps):
        # measured provider mean age: true composition + sampling error, both
        # independent of claim volume. The true age effect is zero everywhere.
        A = MU_AGE + rng.normal(0, sd_true, len(n)) + rng.normal(0, SW / np.sqrt(n))
        q = np.quantile(A, [.25, .5, .75])
        extreme = ((A <= q[0]) | (A > q[2])).astype(float)
        fraud = rng.random(len(n)) < p_f
        if fraud.sum() < 40: continue
        det = (rng.random(len(n)) < p_d)[fraud].astype(float)
        if det.sum() in (0, fraud.sum()): continue
        e = extreme[fraud]
        if e.sum() < 5 or e.sum() > fraud.sum() - 5: continue
        or_u, p_u = fit(det, e[:, None])
        or_a, p_a = fit(det, np.column_stack([e, logn[fraud]]))
        out.append((or_u, p_u, or_a, p_a))
    o = pd.DataFrame(out, columns=['or_u', 'p_u', 'or_a', 'p_a']).dropna()
    return dict(reps=len(o),
                or_u=o.or_u.median(), lo=o.or_u.quantile(.025), hi=o.or_u.quantile(.975),
                fpr_u=(o.p_u < .05).mean() * 100,
                or_a=o.or_a.median(), fpr_a=(o.p_a < .05).mean() * 100)

if __name__ == '__main__':
    print("=== 재현 검증: 논문 Table 6 (sd_true=0.50, 500회) ===")
    print("   논문값: 미보정 OR 0.437 [0.248, 0.747], 유의 78.2% | 보정 OR 0.980, 유의 4.0%")
    r = run(0.50, 500, 42)
    print(f"   재현본: 미보정 OR {r['or_u']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}], "
          f"유의 {r['fpr_u']:.1f}% | 보정 OR {r['or_a']:.3f}, 유의 {r['fpr_a']:.1f}%  (n={r['reps']})")

    print("\n=== Table A7 Panel A 재현 (250회) ===")
    for sd in [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]:
        r = run(sd, 250, 7)
        print(f"   sd_true={sd:4.1f}  미보정 위양성 {r['fpr_u']:5.1f}%   보정 {r['fpr_a']:4.1f}%")

# ---------------------------------------------------------------- CMS scenarios
def run_with_n(nvec, sd_true, reps, seed):
    """Same generating process, but the provider claim-volume distribution is nvec."""
    global n, logn, z
    n_o, logn_o, z_o = n, logn, z
    n = nvec.astype(float); logn = np.log1p(n)
    z = (logn - logn.mean()) / logn.std(ddof=1)
    try:
        return run(sd_true, reps, seed)
    finally:
        n, logn, z = n_o, logn_o, z_o
