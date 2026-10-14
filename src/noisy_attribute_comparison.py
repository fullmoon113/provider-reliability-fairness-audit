from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

"""
Empirical comparison with the noisy-protected-attribute literature (Awasthi et al. 2020; Lamy et al. 2019 / Celis et al. 2021
style noise correction).  Data-generating process = the paper's null simulation (true age effect exactly zero; fraud
incidence and detection depend on log panel size only), with the TRUE group G* (outer age quartiles of true mean age A*)
known, so the methods can be scored against the truth.

(1) Assumption check.  Awasthi et al.'s guarantee needs the noisy group G^ to be conditionally independent of the
    prediction given the label and the true group.  We test this with a logistic model D ~ G* + G^ among fraud-positive
    providers and report how often G^ is significant.
(2) Noise-corrected disparity.  Dividing the measured recall gap by (1 - g0 - g1), with the AVERAGE (oracle) flip rates,
    is the standard correction.  We report the corrected gap and its rejection rate against a true gap of zero.
(3) Equalized-odds style post-processing on the noisy groups (lower the higher-recall group's detections at random until the
    recall rates match).  We report the measured-group, true-group and panel-size-tercile recall before and after.
"""
import importlib.util, sys, os
import numpy as np, pandas as pd, statsmodels.api as sm, warnings
warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("ns", str(HERE / "null_sim.py"))
ns = importlib.util.module_from_spec(spec); sys.modules["ns"] = ns; spec.loader.exec_module(ns)

R = int(os.environ.get('SIM_REPS', 2000)); SEED = 20261014; SD_TRUE = float(os.environ.get('SD_TRUE', 0.50))   # 0.50 = primary dataset; 1.77 and 3.56 = CMS-calibrated
rng = np.random.default_rng(SEED)
n, logn = ns.n, ns.logn
p_f = ns.logistic(ns.B_FRAUD[0] + ns.B_FRAUD[1] * ns.z)
p_d = ns.logistic(ns.B_DET[0] + ns.B_DET[1] * ns.z)
terc = np.digitize(n, np.quantile(n, [1/3, 2/3]))          # panel-size terciles (0 small, 1 mid, 2 large)

def outer(x):
    q = np.quantile(x, [.25, .75]); return ((x <= q[0]) | (x > q[1])).astype(float)

rows = []
for _ in range(R):
    Astar = ns.MU_AGE + rng.normal(0, SD_TRUE, len(n))
    A = Astar + rng.normal(0, ns.SW / np.sqrt(n))
    Gs, Gh = outer(Astar), outer(A)
    fraud = rng.random(len(n)) < p_f
    det = (rng.random(len(n)) < p_d)
    if fraud.sum() < 40 or det[fraud].sum() in (0, fraud.sum()): continue
    f = fraud; d = det[f].astype(float); gh = Gh[f]; gs = Gs[f]; t = terc[f]
    if gh.sum() < 5 or gh.sum() > f.sum() - 5: continue
    g0 = ((Gh == 1) & (Gs == 0)).sum() / max((Gs == 0).sum(), 1)      # P(G^=1 | G*=0)
    g1 = ((Gh == 0) & (Gs == 1)).sum() / max((Gs == 1).sum(), 1)      # P(G^=0 | G*=1)
    # (1) conditional-independence check: D ~ G* + G^
    try:
        r = sm.Logit(d, sm.add_constant(np.column_stack([gs, gh]))).fit(disp=0, maxiter=100)
        p_ci = r.pvalues[2]
    except Exception:
        p_ci = np.nan
    # (2) measured and corrected recall gaps (mid minus extreme)
    rec = lambda m: d[m].mean()
    gap_m = rec(gh == 0) - rec(gh == 1)
    gap_true = rec(gs == 0) - rec(gs == 1)
    se = np.sqrt(rec(gh == 0) * (1 - rec(gh == 0)) / max((gh == 0).sum(), 1) + rec(gh == 1) * (1 - rec(gh == 1)) / max((gh == 1).sum(), 1))
    k = 1 - g0 - g1
    z_m = gap_m / se
    # (3) post-processing on noisy groups: lower the higher-recall group's detections to the lower recall
    d_pp = d.copy()
    hi, lo = (0, 1) if rec(gh == 0) > rec(gh == 1) else (1, 0)
    keep = rec(gh == lo) / max(rec(gh == hi), 1e-9)
    idx = np.where((gh == hi) & (d == 1))[0]
    drop = rng.random(len(idx)) > keep
    d_pp[idx[drop]] = 0
    rec_pp = lambda m: d_pp[m].mean()
    rows.append(dict(
        g0=g0, g1=g1, p_ci=p_ci,
        gap_meas=gap_m, gap_corr=gap_m / k, gap_true=gap_true, sig_meas=abs(z_m) > 1.96,
        pp_gap_meas=rec_pp(gh == 0) - rec_pp(gh == 1), pp_gap_true=rec_pp(gs == 0) - rec_pp(gs == 1),
        rec_all=d.mean(), rec_all_pp=d_pp.mean(),
        **{f'rec_t{i}': (d[t == i].mean() if (t == i).any() else np.nan) for i in range(3)},
        **{f'rec_t{i}_pp': (d_pp[t == i].mean() if (t == i).any() else np.nan) for i in range(3)}))

o = pd.DataFrame(rows)
print(f"replicates used: {len(o)}   (true age effect = 0 by construction; true between-provider SD = {SD_TRUE})\n")
print(f"mean flip rates of the extreme-age label: P(G^=1|G*=0) = {o.g0.mean():.3f},  P(G^=0|G*=1) = {o.g1.mean():.3f}")
print(f"(1) conditional independence of G^ and detection given G*: rejected at 5% in {(o.p_ci < .05).mean()*100:.1f}% of replicates")
print(f"(2) recall gap, mid minus extreme (median [2.5%, 97.5%]):")
print(f"      true groups G*       {o.gap_true.median():+.3f} [{o.gap_true.quantile(.025):+.3f}, {o.gap_true.quantile(.975):+.3f}]")
print(f"      measured groups G^   {o.gap_meas.median():+.3f} [{o.gap_meas.quantile(.025):+.3f}, {o.gap_meas.quantile(.975):+.3f}]   significant in {o.sig_meas.mean()*100:.1f}%")
print(f"      noise-corrected      {o.gap_corr.median():+.3f} [{o.gap_corr.quantile(.025):+.3f}, {o.gap_corr.quantile(.975):+.3f}]   (same z, so same rejection rate)")
print(f"(3) equalized-odds post-processing on G^:")
print(f"      measured-group gap   {o.gap_meas.median():+.3f} -> {o.pp_gap_meas.median():+.3f}")
print(f"      true-group gap       {o.gap_true.median():+.3f} -> {o.pp_gap_true.median():+.3f}")
print(f"      overall recall       {o.rec_all.mean():.3f} -> {o.rec_all_pp.mean():.3f}")
for i, nm in enumerate(['small', 'middle', 'large']):
    print(f"      recall, {nm:6s} providers (tercile {i}): {o[f'rec_t{i}'].mean():.3f} -> {o[f'rec_t{i}_pp'].mean():.3f}")
print(f"      recall gap large minus small: {(o.rec_t2 - o.rec_t0).mean():+.3f} -> {(o.rec_t2_pp - o.rec_t0_pp).mean():+.3f}")
