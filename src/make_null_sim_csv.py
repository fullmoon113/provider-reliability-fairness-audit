from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

"""
Per-replicate results of the null simulation (true age effect zero, true between-provider SD 0.50, 5,000 replicates, seed 20260910)
-> data/null_sim_5000reps.csv, the input of src/simfig.py (Fig. 4).  Same generating process and random-number sequence as
null_sim.run(); columns: n = fraud-positive providers in the replicate, uOR/up = unadjusted odds ratio and p,
aOR/ap = panel-size-adjusted odds ratio and p.
"""
import importlib.util, sys
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location("ns", str(HERE / "null_sim.py"))
ns = importlib.util.module_from_spec(spec); sys.modules["ns"] = ns; spec.loader.exec_module(ns)

SD_TRUE, REPS, SEED = 0.50, 5000, 20260910   # same replicates as the paper's 5,000-replicate null simulation (sim5k.py, Panel A, SD 0.50)
rng = np.random.default_rng(SEED)
n, logn = ns.n, ns.logn
p_f = ns.logistic(ns.B_FRAUD[0] + ns.B_FRAUD[1] * ns.z)
p_d = ns.logistic(ns.B_DET[0] + ns.B_DET[1] * ns.z)
rows = []
for _ in range(REPS):
    A = ns.MU_AGE + rng.normal(0, SD_TRUE, len(n)) + rng.normal(0, ns.SW / np.sqrt(n))
    q = np.quantile(A, [.25, .5, .75])
    extreme = ((A <= q[0]) | (A > q[2])).astype(float)
    fraud = rng.random(len(n)) < p_f
    if fraud.sum() < 40: continue
    det = (rng.random(len(n)) < p_d)[fraud].astype(float)
    if det.sum() in (0, fraud.sum()): continue
    e = extreme[fraud]
    if e.sum() < 5 or e.sum() > fraud.sum() - 5: continue
    or_u, p_u = ns.fit(det, e[:, None])
    or_a, p_a = ns.fit(det, np.column_stack([e, logn[fraud]]))
    rows.append((int(fraud.sum()), or_u, p_u, or_a, p_a))
o = pd.DataFrame(rows, columns=['n', 'uOR', 'up', 'aOR', 'ap']).dropna()
o.to_csv(DATA / 'null_sim_5000reps.csv', index=False)
print(f"{len(o)} replicates; unadjusted median OR {o.uOR.median():.3f}, significant {100*(o.up<.05).mean():.1f}%; "
      f"adjusted median OR {o.aOR.median():.3f}, significant {100*(o.ap<.05).mean():.1f}%")
