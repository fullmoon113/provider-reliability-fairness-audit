from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

import numpy as np, pandas as pd, importlib.util, sys, json
spec = importlib.util.spec_from_file_location("ns", str(HERE/"null_sim.py"))
ns = importlib.util.module_from_spec(spec); sys.modules["ns"] = ns
spec.loader.exec_module(ns)

import os
R = int(os.environ.get('SIM_REPS', 5000))   # 5000 for the paper; set SIM_REPS=50 for a quick smoke test
SEED = 20260910
d = pd.read_csv(str(DATA/'provider_features_FINAL.csv'))
nfull = d.TotalUniqueBene.values.astype(float)
out = {}

def se(p, k):
    return 100 * np.sqrt((p / 100) * (1 - p / 100) / k)

print("=== Panel A: true between-provider SD (5,000 reps) ===", flush=True)
for sd in [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]:
    r = ns.run(sd, R, SEED)
    out[f'A|{sd}'] = dict(r, se_u=se(r['fpr_u'], r['reps']))
    print(f"  sd={sd:4.1f}  unadj {r['fpr_u']:5.2f}% (SE {se(r['fpr_u'],r['reps']):.2f})"
          f"  adj {r['fpr_a']:4.2f}%  OR {r['or_u']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]", flush=True)

print("\n=== Panel C: CMS-calibrated regimes (5,000 reps) ===", flush=True)
scen = [("Primary dataset (baseline)", 0.50, 1),
        ("CMS, most homogeneous specialty", 1.77, 1),
        ("CMS, specialty median", 3.56, 1),
        ("CMS, all providers pooled", 4.93, 1),
        ("CMS specialty median, >=11", 3.56, 11),
        ("CMS specialty median, >=50", 3.56, 50),
        ("CMS most homogeneous, >=11", 1.77, 11),
        ("CMS most homogeneous, >=50", 1.77, 50)]
for lbl, sd, thr in scen:
    sub = nfull[nfull >= thr]
    r = ns.run_with_n(sub, sd, R, SEED) if thr > 1 else ns.run(sd, R, SEED)
    out[f'C|{lbl}'] = dict(r, se_u=se(r['fpr_u'], r['reps']), sd=sd,
                           nprov=len(sub), medvol=float(np.median(sub)))
    print(f"  {lbl:36s} sd={sd:4.2f} med={np.median(sub):5.0f}"
          f"  unadj {r['fpr_u']:5.2f}% (SE {se(r['fpr_u'],r['reps']):.2f})"
          f"  adj {r['fpr_a']:4.2f}%", flush=True)

json.dump(out, open(OUT/'sim5k.json', 'w'))
print("\nDONE", flush=True)
