#!/usr/bin/env python3
"""Regenerate the outputs of the paper with one command.

    python run_all.py                 # steps that need only the files in this repository
    python run_all.py --quick         # the same with small simulation sizes (smoke test, minutes)
    python run_all.py --cms2022 CMS2022_EXTRACT.csv.gz --leie UPDATED.csv    # adds the Medicare flag-rate audit and the LEIE check
    python run_all.py --list          # show the steps and what each needs
    python run_all.py --skip a.py,b.py   # leave out named steps (for example the slowest ones)

Steps whose inputs are not supplied are skipped and reported at the end. Each step writes into output/
(see README.md for the script-to-output map). Run from the repository root.
"""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
args = sys.argv[1:]
def opt(name):
    return args[args.index(name) + 1] if name in args else None
QUICK, LIST = '--quick' in args, '--list' in args
CMS22, LEIE = opt('--cms2022'), opt('--leie')
SKIP = set((opt('--skip') or '').split(','))
RAW = ROOT / 'data' / 'raw' / 'Train-1542865627584.csv'
reps = {'SIM_REPS': '50'} if QUICK else {}

# (script, arguments, environment, requirement or None)
CORE = [
    ('primary_cw.py', [], {}, None),
    ('table4_cw.py', [], {}, None),
    ('table4b.py', [], {}, None),
    ('cw_text_numbers.py', [], {}, None),
    ('reliability_decomposition.py', [], {}, None),
    ('variance_component_inference.py', [], {}, None),
    ('rstar_table.py', [], {}, None),
    ('misclassification_by_reliability.py', [], {}, None),
    ('prop1_analytic.py', [], {}, None),
    ('prespecified_partitions.py', [], {}, None),
    ('supplement_runs.py', [], {}, None),
    ('ablation_size_features.py', [], {}, None),
    ('mitigation_burden.py', [], {}, None),
    ('table12.py', [], {}, None),
    ('prop2_observed.py', [], {}, None),
    ('sim5k.py', [], reps, None),
    ('make_null_sim_csv.py', [], {}, None),
    ('prop2_check.py', [], reps, None),
    ('eiv_remedy.py', [], reps, None),
    ('planted_effect_power.py', [], reps, None),
    ('noisy_attribute_comparison.py', [], reps, None),
    ('cms_coupling_check.py', [], {}, None),
    ('cms_exposure.py', [], reps, None),
    ('corollary1.py', [], {}, None),
    ('prop1_nonwhite.py', [], {}, None),
    ('robustness_extras.py', [], {}, None),
    ('sensitivity_groupwise_smote.py', [], {}, None),
    ('make_figures_1_2.py', [], {}, None),
    ('make_fig2_cw.py', [], {}, None),
    ('make_fig_null.py', [], {}, None),
    ('make_figure_cms.py', [], {}, None),
    ('make_fig_exposure.py', [], {}, None),
]
OPTIONAL = [
    ('reliability_checks_primary.py', [], {}, ('data/raw (Kaggle Train_*.csv)', RAW.exists())),
    ('permutation_null.py', ['50' if QUICK else '5000'], {}, ('data/raw (Kaggle Train_*.csv)', RAW.exists())),
    ('patient_level_audit.py', [], {}, ('data/raw (Kaggle Train_*.csv)', RAW.exists())),
    ('flag_rate_predict.py', [CMS22 or '', LEIE or ''], {}, ('--cms2022 and --leie', bool(CMS22 and LEIE))),
    ('flag_rate_observe.py', [CMS22 or '', LEIE or ''], {}, ('--cms2022 and --leie', bool(CMS22 and LEIE))),
    ('table12_cms.py', [CMS22 or ''], {}, ('--cms2022', bool(CMS22))),
    ('mitigation_burden.py', [], {}, None),
    ('leie_cms_check.py', [LEIE or ''], {}, ('--leie', bool(LEIE))),
    ('cms_stage0_sweep.py', [CMS22 or ''], {}, ('--cms2022', bool(CMS22))),
]
SLOW = [] if QUICK else [('mitigation_burden.py', ['RF'], {}, None)]      # random-forest base of the mitigation (several minutes)
steps = CORE + SLOW + [s for s in OPTIONAL if s[0] != 'mitigation_burden.py']
if LIST:
    for s, a, e, need in steps:
        print(f'{s:42s} needs: {need[0] if need else "repository files only"}')
    sys.exit(0)

(ROOT / 'output').mkdir(exist_ok=True)
report = []
for script, a, e, need in steps:
    if script in SKIP:
        report.append((script, 'skipped', 'by --skip')); continue
    if need and not need[1]:
        report.append((script, 'skipped', f'needs {need[0]}')); continue
    t0 = time.time(); print(f'--- {script} {" ".join(a)}', flush=True)
    r = subprocess.run([sys.executable, str(ROOT / 'src' / script)] + a, cwd=ROOT, env={**os.environ, **e})
    report.append((script, 'ok' if r.returncode == 0 else f'FAILED ({r.returncode})', f'{time.time() - t0:.0f} s'))
print('\n=== summary')
for s, st, note in report: print(f'{s:42s} {st:14s} {note}')
print('\nSupplement: python src/make_supplement.py OUTDIR, then pdflatex supplement.tex twice in OUTDIR.')
sys.exit(1 if any(r[1].startswith('FAILED') for r in report) else 0)
