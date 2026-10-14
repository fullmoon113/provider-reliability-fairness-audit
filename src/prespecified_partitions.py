from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

"""
Partition-independent checks of the age disparity (reviewer point: the Q1+Q4 partition was defined after inspecting the
quartile pattern). Detection is taken from a class-weighted logistic model, so no resampling is stratified on any
partition. Evaluated among the fraud-positive providers: (a) continuous mean patient age (per SD), (b) each tail
separately versus the middle half, (c) the pooled Q1+Q4 partition, each unadjusted and adjusted for log(1 + panel size).
"""
import warnings; warnings.filterwarnings("ignore")
exec(open(HERE/'fpr_class_weighting.py').read().split("oof0 = oof_smote(ext0)")[0])

oofc = oof_cw()
z = (a - a.mean()) / a.std(ddof=1)
q = np.quantile(a, [.25, .5, .75])
low, high = (a <= q[0]), (a > q[2])
mid = ~(low | high)

def run(label, col, mask):
    u = fit(oofc, mask, col); ad = fit(oofc, mask, col, True)
    print(f"{label:34s} n={mask.sum():4d}  unadj {fmt(u)}  | adj {fmt(ad)}")

print("Detection among fraud-positive providers (class-weighted model)\n")
run("continuous age, per SD", z, fm)
m = fm & (low | mid); run("lowest-age quartile vs middle half", low.astype(float), m)
m = fm & (high | mid); run("highest-age quartile vs middle half", high.astype(float), m)
run("pooled Q1+Q4 vs middle half", ext0, fm)
