from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = ROOT / 'output'
"""Supplement S7 Panel B: strength of the panel-size-to-detection relationship (multiples of the fitted slope) at tau = 0.90."""
import sys, json, os, importlib.util
spec = importlib.util.spec_from_file_location("ns", str(HERE / "null_sim.py")); ns = importlib.util.module_from_spec(spec); sys.modules["ns"] = ns; spec.loader.exec_module(ns)
R = int(os.environ.get('SIM_REPS', 5000)); out = {}
for m in [float(x) for x in sys.argv[1:]] or [0.5, 0.75, 1.0, 1.25, 1.5]:
    r = ns.run(0.90, R, 20260910, det_mult=m); out[str(m)] = r
    print(m, {k: round(v, 3) for k, v in r.items()}, flush=True)
json.dump(out, open(OUT / f"sim_panelB_{'_'.join(sys.argv[1:]) or 'all'}.json", 'w'))
