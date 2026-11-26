"""Columns (d) and (e) of the Stage 0 and Algorithm 1 table: Stage 0 and Algorithm 1 applied to flag-rate audits of one CMS specialty.
Usage: python src/table12_cms.py CMS2022_EXTRACT.csv.gz [specialty]"""
import sys, io; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from flag_rate_common import *
from stage0 import stage0_reliability
d = load_cohort(sys.argv[1]); fl = pd.read_csv(OUT / 'flag_rate_flags.csv.gz'); assert (fl.NPI.values == d.NPI.values).all()
spec = sys.argv[2] if len(sys.argv) > 2 else 'Internal Medicine'
d['y'] = fl.y.values; d['det'] = fl.flag_T1.values; d['aud'] = (d.y == 0).astype(float)
g = d[d.spec == spec].reset_index(drop=True)
buf = io.StringIO()
for at in ('non-White', 'female', 'age 65-74', 'mean age'):
    sw = None
    if at == 'mean age': sw = sigma_w_mean_age(g)
    r = stage0_reliability(g, 'n', at, 'outer', sigma_w=sw, eps=0.10, weights='aud', detection='det')
    s = f'== {spec}: {at}, outer vs middle, flag rate (T1)\n' + r['summary'] + '\n'
    for k in ('tau2', 'p_Q', 'median_R', 'M', 'eta_G', 'eta_D', 'r_hat', 'branch', 'predicted_OR', 'MDE', 'power_at_delta', 'decision'):
        s += f'  {k}: {r.get(k)}\n'
    for k in ('unadjusted', 'adjusted', 'center', 'shrunken_adjusted'):
        if k in r: v = r[k]; s += f'  {k}: OR {v["OR"]:.3f} [{v["lo"]:.3f}, {v["hi"]:.3f}]\n'
    print(s); buf.write(s)
open(OUT / 'table12_cms_output.txt', 'w').write(buf.getvalue())
