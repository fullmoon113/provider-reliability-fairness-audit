from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

"""
Share of units placed in the wrong group by an outer-versus-middle split (outer quartiles vs middle half) of a
normally distributed attribute measured with classical error, as a function of reliability R (Section III-E, Stage 0).
True value t ~ N(0,1); measured value o = sqrt(R) t + sqrt(1-R) e; groups are defined by the 25th/75th percentiles
of t and of o respectively.  Usage:  python src/misclassification_by_reliability.py
"""
import numpy as np
rng = np.random.default_rng(1)
N = 2_000_000
lines = ['reliability  misclassified_share']
for R in (0.037, 0.3, 0.5, 0.7, 0.8, 0.9):
    t = rng.standard_normal(N)
    o = np.sqrt(R) * t + np.sqrt(1 - R) * rng.standard_normal(N)
    qt, qo = np.quantile(t, [.25, .75]), np.quantile(o, [.25, .75])
    gt = (t < qt[0]) | (t > qt[1]); go = (o < qo[0]) | (o > qo[1])
    lines.append(f'{R:<12} {(gt != go).mean():.3f}')
print('\n'.join(lines)); (OUT / 'misclassification_by_reliability.txt').write_text('\n'.join(lines) + '\n')
