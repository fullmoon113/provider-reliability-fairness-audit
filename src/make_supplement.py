"""Builds supplement.tex (Tables S1-S21) from saved outputs and recorded run numbers.
Usage: python make_supplement.py OUTDIR   (then run pdflatex twice in OUTDIR)"""
import sys, re
from pathlib import Path
import pandas as pd, numpy as np
HERE = Path(__file__).resolve().parent; OUT = HERE.parent / 'output'
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
out = []
def w(s=''): out.append(s)
def tab(num, title, note, header, rows, spec=None, small='\\small', after=''):
    w(f'\\Needspace{{{min(len(rows)+7,45)}\\baselineskip}}'); w(f'\\subsection*{{Table S{num}. {title}}}')
    if note: w(note + '\n')
    spec = spec or ('l' + 'r' * (len(header) - 1))
    w('\\begin{center}' + small); w(f'\\begin{{tabular}}{{{spec}}}\\toprule')
    w(' & '.join(header) + r' \\ \midrule')
    for r in rows:
        if r == '--': w('\\midrule'); continue
        if isinstance(r, str): w(f'\\multicolumn{{{len(header)}}}{{l}}{{\\emph{{{r}}}}} \\\\'); continue
        w(' & '.join(str(c) for c in r) + r' \\')
    w('\\bottomrule\\end{tabular}\\end{center}')
    if after: w(after + '\n')
def pan(t): return t
w(r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}\usepackage{booktabs,amsmath,amssymb,graphicx,array}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}
\usepackage{hyperref}\usepackage{needspace}
\title{Supplementary Material\\[2pt]\large Unreliable Aggregate Sensitive Attributes Manufacture Fairness Gaps: A Reliability Check for Provider-Level Audits}
\author{}\date{}
\begin{document}\maketitle
\noindent Unless stated otherwise, the detection model is logistic regression with class weighting (the primary setting of the main text), the sensitive attribute is the extreme age-mix partition (outer quartiles versus the middle half of providers by mean beneficiary age), the unadjusted result is the odds ratio of detection for extreme versus middle age-mix among fraud-positive providers, and the adjusted result adds log(1+panel size). Odds ratios are given with 95\% confidence intervals where available. Simulation settings follow Table~7 of the main text: true between-provider standard deviation of mean age $\tau=0.9$ years unless stated otherwise, within-provider SD 12.73, 5,000 replications for the null simulations.
''')
# S1
tab(1, 'Detection strategy and random seed',
 'Extreme age-mix partition, logistic regression. SMOTE is applied within groups (group-wise) unless stated; the seed affects only the resampling.',
 ['Strategy','Seed','Unadj.\\ OR [95\\% CI]','$p$','Adj.\\ OR [95\\% CI]','$p$','Panel size $p$'],
 [['Class weighting (primary)','42','0.477 [0.282, 0.805]','0.0055','1.018 [0.566, 1.832]','0.952','$1.2\\times10^{-11}$'],'--',
  ['Group-wise SMOTE','42','0.462 [0.264, 0.809]','0.0070','0.823 [0.447, 1.516]','0.533','$6.7\\times10^{-7}$'],
  ['Group-wise SMOTE','1','0.466 [0.269, 0.809]','0.0066','0.835 [0.458, 1.521]','0.555','$3.5\\times10^{-7}$'],
  ['Group-wise SMOTE','7','0.439 [0.252, 0.765]','0.0037','0.776 [0.424, 1.419]','0.410','$7.7\\times10^{-7}$'],
  ['Group-wise SMOTE','99','0.519 [0.299, 0.901]','0.0199','0.811 [0.447, 1.472]','0.490','$1.0\\times10^{-4}$'],
  ['Group-wise SMOTE','2024','0.466 [0.269, 0.809]','0.0066','0.807 [0.444, 1.468]','0.483','$1.6\\times10^{-6}$'],'--',
  ['Pooled SMOTE','42','0.456 [0.271, 0.766]','0.0030','0.993 [0.553, 1.783]','0.982','$2.6\\times10^{-12}$'],
  ['No resampling or weighting','--','0.291 [0.181, 0.469]','$<$0.0001','0.941 [0.529, 1.672]','0.835','$9.7\\times10^{-25}$']],
 after='The unadjusted association is present under every strategy and seed and disappears after adjustment for panel size under every one of them; panel size is strongly associated with detection in every row (last column).')
# S2
tab(2, 'Age effect under restriction to larger providers',
 'Class weighting, no regression adjustment; the sample is restricted to providers above each panel-size threshold and the unadjusted test repeated.',
 ['Minimum panel size (beneficiaries)','Providers','Unadj.\\ OR','$p$'],
 [['$>0$ (all)','506','0.477','0.0055'],['$>10$','488','0.468','0.0075'],['$>20$','458','0.632','0.167'],['$>30$','432','0.612','0.179'],['$>50$','385','0.689','0.413'],['$>100$','281','0.559','0.326']],
 spec='lrrr', after='The unadjusted odds ratio rises from 0.477 to 0.689 as providers up to 50 beneficiaries are excluded and then falls to 0.559 above 100, so the pattern is not monotonic; the interval widens as the sample shrinks and the test loses significance above 20. We do not read the table as a dose-response. The size-adjusted analysis of Table 4 is the primary one.')
# S3
tab(3, 'Temporal stability of measured mean patient age by panel size (primary dataset)',
 'Claims split at 30 June 2009. Mean age is computed over unique beneficiaries in each half, matching the feature definition. This table does not depend on the detection model.',
 ['Minimum unique beneficiaries per half','Providers','Between-half correlation'],
 [['$\\ge1$','5,008','0.101'],['$\\ge5$','3,782','0.238'],['$\\ge10$','2,974','0.297'],['$\\ge20$','2,028','0.348'],['$\\ge30$','1,489','0.407'],['$\\ge50$','940','0.485'],['$\\ge100$','445','0.594']],
 spec='lrr', after='The reliability of a provider\'s measured age composition rises with the number of beneficiaries over which it is computed, as the aggregation mechanism of Section III predicts.')
# S4
mb = {}
for ln in (OUT / 'mitigation_burden_output.txt').read_text().split('\n')[1:]:
    q = ln.split()
    if len(q) == 4: mb[q[0]] = [float(x) for x in q[1:]]
def f3(k): return f'{mb[k][0]:.3f}', f'{mb[k][1]:.3f}'
tab(4, 'Fairness metrics and investigation burden before and after equalized-odds mitigation',
 'Exponentiated-gradient reduction ($\\epsilon=0.02$, 30 iterations), applied to the class-weighted logistic regression; the mitigated classifier is randomized, so the table reports means over 30 prediction draws. Gaps are between the middle and extreme age-mix groups.',
 ['Metric','Before','After'],
 [['Equalized-odds difference', *f3('eod')], ['Demographic parity difference', *f3('dpd')], ['F1', *f3('f1')],
  '--', ['Recall, middle age-mix', *f3('recall_mid')], ['Recall, extreme age-mix', *f3('recall_ext')],
  ['False-positive rate, middle age-mix', *f3('fpr_mid')], ['False-positive rate, extreme age-mix', *f3('fpr_ext')],
  ['Flag rate, middle age-mix', *f3('flag_mid')], ['Flag rate, extreme age-mix', *f3('flag_ext')], ['Flag rate, all providers', *f3('flag_all')],
  ['Precision, middle age-mix', *f3('ppv_mid')], ['Precision, extreme age-mix', *f3('ppv_ext')]],
 spec='lrr', after=f'Bootstrap (3,000 resamples of the 5,410 providers): recall gap (middle minus extreme) 0.100, 95\\% CI [0.028, 0.181]. After mitigation the equalized-odds difference is 0.031 (SD {mb["eod"][2]:.3f} over draws; reduced in 30 of 30 draws). Mitigation closes the gap by lowering recall in both groups, more in the middle group ({mb["recall_mid"][0]:.3f} to {mb["recall_mid"][1]:.3f}) than in the extreme group ({mb["recall_ext"][0]:.3f} to {mb["recall_ext"][1]:.3f}), and it raises the false-positive rate and the flag rate in both groups, so more providers without a fraud label are investigated (script mitigation\\_burden.py).')
# S5
tab(5, 'Robustness to outcome specification',
 'The binary detection indicator is replaced by continuous alternatives (ordinary least squares on the extreme age-mix indicator, fraud-positive providers).',
 ['Outcome','Unadj.\\ coefficient ($p$)','Adj.\\ coefficient ($p$)'],
 [['Predicted fraud probability','$-0.134$ ($2.5\\times10^{-8}$)','$-0.016$ (0.454)'],['Percentile rank of predicted probability','$-0.177$ ($1.1\\times10^{-9}$)','$-0.011$ (0.629)']],
 spec='lrr', after='The pattern does not depend on thresholding the model output.')
# S6
tab(6, 'Data integrity verification',
 'Checks on the primary dataset (Kaggle; four training files).',
 ['Check','Result'],
 [['Duplicate rows','0'],['Referential integrity (claims to beneficiaries and providers)','100\\%'],['Date of death missing','98.97\\%'],['Beneficiary age (years)','26.1 to 101.0, median 74.3'],['Length of stay (days)','mean 5.67, maximum 35'],['Gender codes','\\{1, 2\\}'],['Race codes (count)','1: 117,057; 2: 13,538; 3: 5,059; 5: 2,902'],['Largest absolute correlation with a training feature','0.130 (female ratio)'],['Fold prevalence across cross-validation folds','9.33\\% to 9.43\\%'],['Beneficiaries under 65','15.65\\% (of whom 14.8\\% carry the renal indicator)']],
 spec='ll')
# S7
w('\\Needspace{12\\baselineskip}'); w('\\subsection*{Table S7. Sensitivity of the null simulation}')
w('All panels use 5,000 replications unless stated; unadjusted = share of replications with a significant extreme age-mix disparity ($p<0.05$) before adjustment, adjusted = after adding log(1+panel size). Nominal level is 5\\%.\n')
def sub(title, header, rows, spec=None, size='\\small'):
    w(f'\\Needspace{{{min(len(rows)+5,48)}\\baselineskip}}'); w(f'\\textbf{{{title}}}'); w('\\begin{center}' + size); w(f'\\begin{{tabular}}{{{spec or "l"+"r"*(len(header)-1)}}}\\toprule'); w(' & '.join(header)+r' \\ \midrule')
    for r in rows: w(' & '.join(map(str,r))+r' \\')
    w('\\bottomrule\\end{tabular}\\end{center}')
sub('Panel A. True between-provider SD of mean age ($\\tau$)',['$\\tau$ (years)','Unadjusted \\%','Adjusted \\%'],
 [['0','96.9','4.62'],['0.5','94.9','4.46'],['0.9 (main)','87.1','4.30'],['1.0','83.9','4.40'],['2.0','43.7','4.92'],['4.0','12.4','4.96'],['6.0','7.4','5.30']])
sub('Panel B. Strength of the panel-size effect on detection ($\\tau=0.9$; multiples of the main slope 1.213)',['Slope multiple','Unadjusted \\%','Adjusted \\%','Median unadj.\\ OR'],
 [['0.50','47.0','4.66','0.633'],['0.75','72.7','4.52','0.517'],['1.00 (main)','87.1','4.32','0.431'],['1.25','93.9','4.44','0.369'],['1.50','96.9','4.72','0.324']])
w('Panels A and B were run with separate random seeds, so the main setting ($\\tau=0.9$, slope multiple 1.00) appears with 4.30\\% adjusted rejection in Panel A and 4.32\\% in Panel B, which differ by Monte Carlo error.\n')
sub('Panel C. Reliability regimes calibrated to CMS data',['Regime','$\\tau$','Median unadj.\\ OR [interval]','Unadj.\\ \\%','Adj.\\ \\%'],
 [['Homogeneous specialty','1.65','0.572 [0.343, 0.976]','57.1','4.78'],['Specialty median','3.47','0.777 [0.470, 1.315]','16.6','4.90'],['Pooled specialties','4.85','0.853 [0.519, 1.439]','9.1','4.70'],
  ['Specialty median, panels $\\ge$11 (n = 3,825)','3.47','--','7.0','5.28'],['Specialty median, panels $\\ge$50 (n = 1,713)','3.47','--','4.4','4.40'],['Homogeneous, panels $\\ge$11','1.65','--','23.6','5.24'],['Homogeneous, panels $\\ge$50','1.65','--','6.3','4.82']])
w('The median panel size is 25 beneficiaries in the full simulated population and 43 and 104 in the restricted populations above. Under the specialty-median regime the unadjusted rate approaches the nominal level once small panels are excluded.\n')
pe = pd.read_csv(OUT / 'planted_effect_power.csv')
def fm(x): return '--' if pd.isna(x) else f'{x:.1f}'
cols = [('naive_rej','Unadj.'),('adjusted_rej','Size-adj.'),('shrunken_rej','Shrunken'),('RC_rej','Reg.\\ calib.'),('RCnosize_rej','Calib., no size'),('postdraw_rej','Post.\\ draw'),('procedure_rej','Procedure'),('oracle_rej','Oracle')]
nrep = int(pe.reps.iloc[0])
w(f'\\Needspace{{10\\baselineskip}}\\textbf{{Panel D. Planted effect: rejection rate (\\%) of the age-mix partition at the 5\\% level, {nrep:,} replications per cell (type I error at OR = 1)}}\n')
w('Audits: primary (5,410 providers, about 506 fraud-labeled), large (40,000 providers, about 3,700 labeled) and high-reliability (20,000 providers drawn from panels of at least 101 beneficiaries, $\\tau=4.85$). $\\rho$ is the correlation between true provider mean age and standardized log panel size (0.24 is the coupling observed in CMS). Monte Carlo standard error of a rejection rate near 5\\% is 0.5 percentage points. Columns: Unadj., outer-quartile partition of the observed age with no control; Size-adj., the same with log panel size added (the audit estimate of Algorithm 1); Shrunken, outer quartiles of the empirical Bayes estimate with log panel size as covariate; Reg.\\ calib., the posterior probability of the true outer group, with log panel size added; Calib., no size, the same probability without the size term; Post.\\ draw, outer quartiles of one posterior draw of the true age; Procedure, Algorithm 1 as a whole (shown only where it attributes); Oracle, the true age. ``--\'\' marks cells where Algorithm 1 declined in every replicate.\n')
for kind, ttl in (('primary','Primary audit'),('large','Large audit'),('highrel','High-reliability audit')):
    q = pe[pe.audit == kind]
    if kind == 'primary': q = q[q.tau > 0].copy()
    rows = []
    for _, r in q.sort_values(['rho','tau','OR_true'], ascending=[True,True,False]).iterrows():
        rows.append([f'{r.rho:.2f}', f'{r.tau:.2f}', f'{r.OR_true:.1f}'] + [fm(r[c]) for c, _ in cols])
    sub(ttl, ['$\\rho$','$\\tau$','OR']+[h for _, h in cols], rows, spec='rrr' + 'r'*len(cols), size='\\footnotesize')
br = pe[(pe.attrib_rate > 0)]
w('Algorithm 1 branch shares (\\%): ' + '; '.join(f'{r.audit} audit, $\\tau={r.tau:.2f}$, OR {r.OR_true:.1f}, $\\rho={r.rho:.2f}$: attributes {r.attrib_rate:.1f}' for _, r in br.iterrows()) + '. In all other cells Algorithm 1 declined to attribute in 100\\% of replicates, except the $\\tau=0$ cell, where it declined to partition in 94.4\\%.\n')
w('At $\\tau=0.9$ no estimator on the observed attribute has power in the primary audit: the size-adjusted rejection rate is 4.9\\% to 6.4\\% at true odds ratios 0.8 to 0.5, against 8.3\\% to 71.6\\% for the oracle. The regression calibration with a size term has the highest power among the estimators on the observed attribute wherever any has power, and its null rejection stays between 3.6\\% and 5.5\\%. Without the size term the calibration is conservative in the primary audit (0.2\\% to 2.9\\% null rejection) and approaches the nominal level in the large audit (up to 5.7\\% at $\\tau=4.85$, $\\rho=0.24$), as Proposition 2(b) predicts. The posterior-draw partition is the one estimator whose null rejection exceeds the nominal level beyond Monte Carlo error (7.4\\% in the large audit at $\\tau=4.85$, $\\rho=0.24$), and it has less power than the size-adjusted estimate throughout. The shrunken estimate has lower power than the size-adjusted estimate in 21 of the 24 planted-effect cells of the primary audit, which is why the audit estimate of Algorithm 1 is the size-adjusted one.\n')
# S8
tab(8, 'Sensitivity to the functional form of panel size',
 'Adjusted odds ratio for extreme age-mix, class weighting.',
 ['Panel size term','Adj.\\ OR [95\\% CI]','$p$'],
 [['None (unadjusted)','0.477 [0.282, 0.805]','0.0055'],['$\\log(1+n)$ (primary)','1.018 [0.566, 1.832]','0.952'],['$n$ (raw)','0.919 [0.528, 1.602]','0.767'],['$\\sqrt n$','1.003 [0.568, 1.772]','0.991'],['Quadratic in $\\log(1+n)$','1.030 [0.576, 1.839]','0.921'],['Natural cubic spline, 4 df','1.022 [0.572, 1.827]','0.942'],['Decile indicators (9)','0.984 [0.543, 1.783]','0.957'],['$1/n$','0.614 [0.353, 1.068]','0.084']],
 spec='lrr', after='Stratified comparison without regression: the stratum-weighted detection gap is 0.002 across panel-size quartiles and 0.020 across deciles, against 0.100 unstratified. The Mantel--Haenszel common odds ratio is 0.949 across quartiles and 0.984 across deciles, against 0.477 unstratified. Only the $1/n$ form leaves a residual association, and it remains non-significant.')
# S9
w('\\Needspace{14\\baselineskip}'); w('\\subsection*{Table S9. Replication of the aggregation mechanism in CMS data}')
w('Medicare Physician and Other Practitioners public use file. Two independent year pairs: 2018--2019 and 2021--2022 (1,105,159 providers present in both 2021 and 2022). Providers serving fewer than eleven beneficiaries are suppressed by CMS.\n')
sub('Panel A. Stability of provider mean beneficiary age by panel size',['Panel size band','SD of year change 2018--19','SD 2021--22','Correlation 2018--19','Correlation 2021--22'],
 [['$\\le25$','4.11','3.99','0.895','0.884'],['26--50','3.01','2.89','0.905','0.900'],['51--100','2.33','2.19','0.915','0.904'],['101--200','1.76','1.64','0.925','0.906'],['201--500','1.26','1.19','0.942','0.920'],['$>500$','0.80','0.83','0.961','0.938']])
sub('Panel B. Within specialty: year-to-year correlation in the lowest and highest panel-size quartile',['Specialty','2018--19 low','2018--19 high','2021--22 low','2021--22 high'],
 [['CRNA','0.606','0.815','0.537','0.769'],['Anesthesiology','0.679','0.911','0.649','0.878'],['Emergency medicine','0.759','0.919','0.728','0.887'],['Physician assistant','0.788','0.933','0.778','0.904'],['Family practice','0.836','0.947','0.812','0.926'],['Physical therapist','0.808','0.884','0.817','0.861'],['Internal medicine','0.846','0.950','0.837','0.940'],['Nurse practitioner','0.852','0.949','0.856','0.934']])
w('The ordering holds in all eight groups and in both year pairs (sixteen of sixteen comparisons).\n')
sub('Panel C. Implied within-provider SD, sampling share and simulation $\\tau$',['Group','Implied $\\sigma_w$ 18--19 / 21--22','Sampling share \\% 18--19 / 21--22','$\\tau$ used'],
 [['All providers','12.92 / 12.51','6.3 / 7.6','--'],['CRNA','13.23 / 12.29','29.2 / 36.8','1.65'],['Anesthesiology','13.78 / 12.69','20.0 / 24.4','2.11'],['Emergency medicine','17.00 / 16.16','12.2 / 15.5','3.07'],['Physician assistant','15.15 / 14.58','11.6 / 14.3','4.13'],['Family practice','13.80 / 13.28','9.7 / 12.5','3.67'],['Physical therapist','10.01 / 9.52','15.3 / 15.4','3.26'],['Internal medicine','13.96 / 13.06','8.3 / 9.7','3.73'],['Nurse practitioner','16.65 / 15.49','7.9 / 8.5','6.22'],['Pooled (simulation)','--','--','4.85'],['Specialty median (simulation)','--','--','3.47'],['Primary dataset','12.73','98.8','0.90']])
w('The sampling share is 5.3\\% by the variance-ratio route of Section III-A and 6.3\\% to 7.6\\% by the paired-year route; the unweighted variance ratio is 11.8 in the 2022 file (1,230,293 providers with more than ten beneficiaries, $\\sigma_w=12.52$ years; \\texttt{cms\\_stage0\\_sweep.py}), and 8.5\\% is its reciprocal.\n')
# S10
tab(10, 'Maximum likelihood and Firth penalized likelihood',
 'Firth\'s penalized likelihood removes the first-order small-sample bias of maximum likelihood and remains finite under separation. Fraud-positive subgroups number 129, 56 and 103 providers. $p$-values in parentheses; for Firth they are penalized likelihood-ratio tests.',
 ['Partition','ML unadj.\\ (p)','Firth unadj.\\ (p)','ML adj.\\ (p)','Firth adj.\\ (p)'],
 ['Panel A. Sensitive-attribute partitions',['Extreme age-mix','0.477 (0.0055)','0.475 (0.0061)','1.018 (0.952)','1.010 (0.974)'],['High-elderly (top quartile)','0.444 (0.0166)','0.437 (0.0186)','0.807 (0.568)','0.793 (0.535)'],['High-minority (top quartile)','1.187 (0.601)','1.158 (0.647)','1.464 (0.285)','1.424 (0.304)']],
 spec='lrrrr')
w('Each row gives the $p$-value for the extreme age-mix effect when the named characteristic replaces panel size as the control variable (heavy right-skewed characteristics enter on the log scale, as panel size does). Holm adjustment over the eight rows.' + '\n')
sub('Panel B. Specificity: adjusted extreme age-mix effect when each alternative characteristic is used as the control',
 ['Control variable','$p$ (ML)','Holm (ML)','$p$ (Firth)','Holm (Firth)'],
 [['Panel size (log)','0.952','0.952','0.974','0.974'],['Mean length of stay','0.0004','0.0031','0.0004','0.0030'],['Chronic conditions','0.0023','0.0163','0.0022','0.0152'],['Claims per beneficiary (log)','0.0043','0.0207','0.0041','0.0189'],['Inpatient beneficiaries (log)','0.0025','0.0163','0.0026','0.0154'],['Outpatient reimbursement (log)','0.0115','0.0288','0.0109','0.0264'],['Deceased ratio','0.0041','0.0207','0.0038','0.0189'],['Female ratio','0.0096','0.0288','0.0088','0.0264']],
 spec='lrrrr')
w('The age-mix effect stays significant after Holm correction when any of the seven alternatives is the control, and disappears only when panel size is. Joint model with all seven alternatives: extreme age-mix OR 0.475 [0.234, 0.965], $p=0.040$; adding panel size, 0.716 [0.250, 2.045], $p=0.532$. The adjusted age-mix estimate has standard error 0.300 on the log scale; the smallest odds ratio detectable at 80\\% power is 0.432. An equivalence test (TOST) against odds ratios of 0.8 and 1.25 gives $p=0.247$, so equivalence is not established.' + '\n')
# S11
tab(11, 'Sensitivity of the adjusted age effect to label-selection strength',
 'The analysis conditions on a positive fraud label. If the probability of receiving a label scales with panel size, conditioning on it can induce association among other causes. Each row reweights fraud-positive providers by an assumed selection probability proportional to $\\exp(-\\gamma z)$, with $z$ the standardized log panel size. Robust (sandwich) $p$-values.',
 ['$\\gamma$','Adj.\\ OR (extreme age-mix)','$p$'],
 [['0','1.018','0.953'],['0.25','1.036','0.909'],['0.50','1.064','0.852'],['1.0','1.139','0.754'],['1.5','1.199','0.744'],['2.0','1.192','0.815']],
 spec='lrr', after='The adjusted extreme age-mix association is non-significant at every selection strength.')
# S12
import json as _json, re as _re2
_t = (OUT / 'leie_cms_check_output.txt').read_text()
def _blk(tag):
    i = _t.index(tag); j = _t.find('Stage 1-2', i + 5); return _t[i:] if j < 0 else _t[i:j]
def _or(blk, part, kind):
    m = _re2.search(part + r'\s+' + kind + r'\s+OR ([\d.]+) \[([\d.]+), ([\d.]+)\]', blk); return f'{float(m.group(1)):.2f} [{float(m.group(2)):.2f}, {float(m.group(3)):.2f}]'
bx, bi = _blk('Stage 1-2 [age excluded'), _blk('Stage 1-2 [age included')
rec = float(_re2.search(r'recall overall ([\d.]+)', bx).group(1)) * 100
sl = _re2.search(r'slope of the top-10% rule among excluded providers: (-?[\d.]+) per SD of log panel size \[(-?[\d.]+), (-?[\d.]+)\], p=([\d.]+)', _t)
pred = float(_re2.search(r'Prop\. 1 prediction[^:]*: OR ([\d.]+)', _t).group(1))
sd = [_json.loads(l) for l in (OUT / 'leie_seeds_output.txt').read_text().splitlines() if l.strip()]
def rg(k, f=lambda v: f'{v:.2f}'):
    v = np.array([x[k] for x in sd]); return f'{f(v.min())} to {f(v.max())} (median {f(np.median(v))})'
LEIE_SLOPE_TEX = f'$-{abs(float(sl.group(1))):.2f}$' if float(sl.group(1)) < 0 else f'${float(sl.group(1)):.2f}$'
tab(12, 'External check against the OIG exclusion list (LEIE)',
 'Providers in the 2021--2022 CMS file whose NPI appears in the OIG List of Excluded Individuals and Entities with an exclusion date from 1 January 2023: 416 of 1,105,159. A gradient-boosting classifier flags the top 10\\% of providers by out-of-fold score; odds ratios compare the share flagged among excluded providers. Reference run (seed 42), then the range over seeds 1 to 20 of the model fit.',
 ['Quantity','Reference run','Seeds 1--20: range (median)'],
 ['Age excluded from the model',
  ['Excluded providers flagged (recall), \\%', f'{rec:.1f}', rg('recall', lambda v: f'{100*v:.1f}')],
  ['Extreme age-mix, flagged: OR [95\\% CI], unadjusted', _or(bx, 'outer quartiles', 'unadjusted'), rg('or_unadj')],
  ['Extreme age-mix, adjusted for log panel size', _or(bx, 'outer quartiles', r'adj\. panel size'), rg('or_adj')],
  ['Outer age deciles versus the rest, unadjusted', _or(bx, 'outer deciles', 'unadjusted'), '--'],
  ['Outer age deciles, adjusted', _or(bx, 'outer deciles', r'adj\. panel size'), '--'],
  ['Size slope of the flag among excluded providers, per SD of log panel size', f'{LEIE_SLOPE_TEX} [${float(sl.group(2)):.2f}$, ${float(sl.group(3)):.2f}$], $p={float(sl.group(4)):.3f}$'.replace('$-', '$-'), rg('slope', lambda v: ('$-$' if v < 0 else '') + f'{abs(v):.2f}')],
  ['Proposition 1: predicted OR for the extreme age-mix partition', f'{pred:.2f}', '--'],
  '--', 'Age included as a model input',
  ['Extreme age-mix, flagged: OR [95\\% CI], unadjusted', _or(bi, 'outer quartiles', 'unadjusted'), '--'],
  ['Adjusted for log panel size', _or(bi, 'outer quartiles', r'adj\. panel size'), '--'],
  ['Outer age deciles, unadjusted', _or(bi, 'outer deciles', 'unadjusted'), '--']],
 spec='p{6.4cm}p{3.6cm}p{4.4cm}', after='Where age is not an input to the detection model, the reference run shows no significant disparity by age-mix; where it is an input, one appears. The size slope among excluded providers is small and depends on the random seed of the model fit: it is nominally significant in the reference run and between $-0.18$ and $+0.10$ across seeds 1 to 20, so no size dependence of the flag against real labels is established by this check. The label itself depends on unit size, because large billers are investigated more often, so the check cannot separate detection bias from label bias. All values come from the software environment pinned in [65]; across the 20 seeds the conclusions do not change. Script: \\texttt{leie\\_cms\\_check.py} (\\texttt{LEIE\\_SEED}, \\texttt{LEIE\\_SEED\\_ONLY}).')
# S13
import re as _re
w('\\Needspace{14\\baselineskip}'); w('\\subsection*{Table S13. Errors-in-variables remedies and methods for noisy sensitive attributes, in simulation}')
w('2,000 replications per setting. Median odds ratio and rejection rate (\\%) for the extreme age-mix partition. Null: no true effect. Planted: true odds ratio 0.70 in the outer quartiles of true age. RC = regression calibration: the standardized posterior probability $X_i=\\Pr(\\theta_i\\in S\\mid A_i,n_i)$ that the provider\'s true mean age lies in the outer quartiles of the true age distribution, entered per SD (Proposition 2(b)), with $\\tau^2$ estimated by the precision-weighted (DerSimonian--Laird) moment estimator; "moment $\\tau^2$" uses instead the unweighted estimator (variance of the observed ages minus the mean sampling variance); oracle = true provider mean age.\n')
blocks = _re.split(r'\n=== ', '\n' + (OUT / 'eiv_remedy_output.txt').read_text())[1:]
rows = []
for b in blocks:
    head = b.split('\n')[0]; tau = _re.search(r'SD ([\d.]+)', head).group(1); truth = 'Null' if 'null' in head else 'Planted 0.70'
    first = True
    for ln in b.split('\n')[3:]:
        m = _re.match(r'(.+?)\s{2,}([\d.]+)\s+\[([\d., ]+)\]\s+([\d.]+)$', ln)
        if m:
            lab = {'naive':'Unadjusted (naive)','adj. size':'Size-adjusted','RC':'RC','RC + size':'RC + size','RC, moment tau2':'RC, moment $\\tau^2$','RC + size, moment tau2':'RC + size, moment $\\tau^2$','oracle':'Oracle'}[m.group(1).strip()]
            rows.append([tau if first else '', truth if first else '', lab, f'{m.group(2)} [{m.group(3)}]', m.group(4)]); first = False
sub('Panel A. Remedies for the unadjusted partition', ['$\\tau$','Truth','Method','Median OR [interval]','Rejection \\%'], rows, spec='llllr')
w('The unweighted moment estimator of $\\tau^2$ is unstable when 96\\% of the variance is noise (interquartile range 0.27 to 1.32 for a true $\\tau^2$ of 0.81 at $\\tau=0.90$); with it, RC alone rejects in 15.3\\% of null replicates, whereas with the precision-weighted estimator it rejects in 0.4\\%, as Proposition 2(b) implies, and has little power at this reliability.\n')
rows = []
for tau, f in (('0.9','noisy_sd0.9.txt'),('1.65','noisy_sd1.65.txt'),('3.47','noisy_sd3.47.txt')):
    t = (OUT / f).read_text()
    g = lambda pat, k=1: _re.search(pat, t).group(k)
    flip = float(g(r'P\(G\^=1\|G\*=0\) = ([\d.]+),  P'))
    rows.append([tau, f'{flip:.3f} / {g(r"label-level error rate ([\d.]+)")}', g(r'rejected at 5% in ([\d.]+)%'), g(r'measured groups G\^\s+([+\-][\d.]+ \[[^\]]+\])') + ' / ' + g(r'significant in ([\d.]+)%'),
                 g(r'true gap, mean ([+\-][\d.]+)'), g(r'mean divisor 1 - g0 - g1 = ([\d.]+) \(all providers\), ([\d.]+)') + ' / ' + g(r'mean divisor 1 - g0 - g1 = ([\d.]+) \(all providers\), ([\d.]+)', 2),
                 g(r'correction with all-provider rates: ([+\-][\d.]+)'), g(r'with fraud-labeled rates: ([+\-][\d.]+)'), g(r'corrected ([+\-][\d.]+)\n'),
                 g(r'overall recall\s+([\d.]+) -> ([\d.]+)') + ' $\\to$ ' + g(r'overall recall\s+([\d.]+) -> ([\d.]+)', 2)])
names = ['Flip rate (all) / label-level error (labeled)','Conditional independence rejected, \\%','Measured recall gap / significant, \\%','True recall gap (mean)','Divisor $1-g_0-g_1$ (all / labeled)','Corrected gap, all-provider rates','Corrected gap, labeled rates','Corrected gap, size-independent noise','Overall recall before $\\to$ after post-processing']
sub('Panel B. Methods designed for noisy sensitive attributes (null truth)', ['Quantity'] + [f'$\\tau={r[0]}$' for r in rows], [[nm] + [r[k+1] for r in rows] for k, nm in enumerate(names) if k != 2] + [['Measured recall gap, median [2.5\\%, 97.5\\%]'] + [r[3].split(' / ')[0] for r in rows], ['Measured gap significant, \\%'] + [r[3].split(' / ')[1] for r in rows]], spec='p{5.6cm}ccc', size='\\footnotesize')
w('Flip rate = mean misclassification rate of the extreme-age label over all providers; label-level error = misclassification rate among fraud-labeled providers, over which recall is computed. Cond.\\ indep.\\ rejected = rejection rate of the test of conditional independence of the noisy group and detection given the true group, which is the assumption behind the guarantees of Awasthi et al.\\ (2020) and Lamy et al.\\ (2019). Both constructions are simplified. The correction divides the measured recall gap by one minus the sum of the two group flip rates, the mutual-contamination correction of Lamy et al.\\ (2019), with the rates supplied from the simulation (an oracle advantage). The post-processing is not the threshold-based equalized-odds post-processing of Hardt et al.\\ (2016) but a simplified equal-opportunity step that removes detections in the higher-recall group at random until the recall rates match. Size-independent noise: the same flip rates applied independently of panel size, as a control. The true gap is the mean recall gap between the true groups.\n')
pc = pd.read_csv(OUT / 'prop2_check.csv')
sub('Panel C. Partitions on profiling estimates under the null ($\\tau=0.90$, 2,000 replications)', ['Partition', 'Median OR', '2.5\\%', '97.5\\%', 'Rejection \\%'],
    [[str(r.iloc[0])] + [f'{r.iloc[k]:.3f}' for k in (1,2,3)] + [f'{r.iloc[4]:.1f}'] for _, r in pc.iterrows()], spec='lrrrr')
w('Posterior mean partitions invert the groups (Proposition 2(c)); posterior draws and posterior probabilities carry no size artifact (parts (a) and (b)). Reference audit (primary dataset, class-weighted logistic regression, $\\tau^2=0.812$): ' + ' '.join((OUT / 'prop2_observed_output.txt').read_text().split('\n')[1:4]).replace('%', '\\%') + '\n')

# S14
SD = OUT / 'cms_stage0'
order = ['mean age','age <65','age 65-74','age 75-84','age >84','female','non-White (any)','Black','Asian/Pacific Is.','Hispanic','Native American','Other race','dual-eligible']
def agg(fn):
    d = pd.read_csv(SD / fn); d = d[d.year == 'year 2']
    return d.groupby(['attribute', 'band']).apply(lambda x: pd.Series({'R': np.average(x.median_R, weights=x.providers), 'g': np.average(x.share_R_ge_0_7, weights=x.providers), 'm': np.average(x.mean_m, weights=x.providers)}), include_groups=False).reset_index()
sw = {f: pd.read_csv(SD / fn) for f, fn in (('5.5', 'cms_stage0_sweep.csv'), ('0', 'cms_stage0_sweep_zero.csv'))}
bands = ['11-20', '21-50', '51-100', '101+']
w('\\Needspace{22\\baselineskip}'); w('\\subsection*{Table S14. Reliability of multiple aggregate attributes in CMS claims (Stage 0 sweep)}')
w('Within-specialty analysis in the eight largest specialties, 2022 file, providers with more than ten beneficiaries, mean over specialties weighted by provider count. Bands are numbers of beneficiaries. Age-band shares are the band count divided by the number of beneficiaries; the share female is the female count divided by the sum of the female and male counts; race shares are the category count divided by the sum of the six race counts; non-White is one minus the White share. The sampling variance of a share is $\\bar p(1-\\bar p)/n_i$ with $\\bar p$ the specialty mean, and that of mean age is $\\sigma_w^2/n_i$ with $\\sigma_w=12.52$ (paired-year estimate). $R$ = median reliability; share = share of providers with $R_i\\ge0.7$; $m$ = expected share placed in the wrong group by an outer-versus-middle quartile split. Supp.\\ \\% = share of providers whose count for the category is blank in the file (suppressed, 1--10 beneficiaries, or zero); suppressed counts are set to 5.5 (midpoint) or 0. Mean age, share female and share dual-eligible are never blank; for non-White the White count is the blank count.\n')
for fill, lab, fn in (('5.5', 'midpoint fill (5.5)', 'cms_stage0_by_specialty_band.csv'), ('0', 'zero fill', 'cms_stage0_by_specialty_band_zero.csv')):
    G = agg(fn); sp = sw[fill]; sp = sp[sp.year == 'year 2'].set_index('attribute').suppressed_share
    rows = []
    for a in order:
        x = G[G.attribute == a].set_index('band')
        if x.empty: continue
        rows.append([a, f'{sp[a]*100:.1f}'] + [f'{x.loc[b,"R"]:.2f}' for b in bands] + [f'{x.loc[b,"g"]:.2f}' for b in bands] + [f'{x.loc[b,"m"]:.2f}' for b in bands])
    w('\\Needspace{18\\baselineskip}'); w(f'\\textbf{{Suppressed counts: {lab}}}'); w('\\begin{center}\\scriptsize\\setlength{\\tabcolsep}{2pt}')
    w('\\begin{tabular}{lr cccc cccc cccc}\\toprule'); w(r' & & \multicolumn{4}{c}{Median $R$} & \multicolumn{4}{c}{Share $R_i\ge0.7$} & \multicolumn{4}{c}{$m$} \\')
    w(r'Attribute & Supp.\ \% & 11--20 & 21--50 & 51--100 & 101+ & 11--20 & 21--50 & 51--100 & 101+ & 11--20 & 21--50 & 51--100 & 101+ \\ \midrule')
    for r in rows: w(' & '.join(r) + r' \\')
    w('\\bottomrule\\end{tabular}\\end{center}')
for fn, ttl in (('cms_stage0_testretest_within.csv', 'Panel A. Year-to-year (2021 to 2022) correlation of the provider value within specialty, by the smaller of the two years\' panel sizes'), ('cms_stage0_testretest.csv', 'Panel B. The same correlation pooled across specialties')):
    tr = pd.read_csv(SD / fn)
    sub(ttl, ['Attribute','All','11--20','21--50','51--100','101+'], [[r.attribute] + [f'{r[c]:.3f}' for c in ['all','n 11-20','n 21-50','n 51-100','n 101+']] for _, r in tr.iterrows()])
w('Panel A centers each year\'s value at its specialty mean (providers with the same specialty in both years), so differences between specialties do not count as stability; it rises with panel size for every attribute. It is not a second estimate of $R_i$: beneficiaries who stay with a provider from one year to the next enter both years, so their sampling errors are correlated and the correlation overstates reliability where panels are stable (share female: 0.51 against a computed $R$ of 0.27 in the smallest band). For the non-White share the opposite holds: the correlation (0.38 and 0.26 in the two smallest bands) is far below the computed reliability (0.87 and 0.94), because the White count is blank for most providers with fewer than 100 beneficiaries and the filled values drive the computed variance. Pooled across specialties (differences between specialties treated as true variance), median reliability is 0.84 to 0.99 across attributes and bands; this is the relevant figure only when an audit compares providers across specialties.\n')
# S15
tab(15, 'Sensitivity to the detection threshold',
 'Class weighting. The detection threshold applies to the predicted fraud probability; the primary analysis uses 0.5.',
 ['Threshold','Detection rate','Unadj.\\ OR','$p$','Adj.\\ OR','$p$'],
 [['0.3','0.951','0.348','0.011','0.629','0.296'],['0.4','0.921','0.428','0.012','0.737','0.401'],['0.5 (primary)','0.858','0.477','0.0055','1.018','0.952'],['0.6','0.787','0.440','0.0004','1.031','0.909'],['0.7','0.694','0.424','0.0001','1.055','0.833'],['0.8','0.595','0.379','$<$0.0001','1.139','0.621'],['0.9','0.468','0.292','$<$0.0001','0.910','0.738']],
 spec='lrrrrr', after='The unadjusted association is present at every threshold. The adjusted estimate is non-significant at every threshold and lies between 0.63 and 1.14.')
# S16
rows = []
for ln in (OUT / 'ablation_size_features_output.txt').read_text().split('\n')[1:]:
    m = re.match(r'(\w+)\s+(\w+)\s+(.+?)\s+([\d.]+)\s+(-?[\d.]+)\s+([\d.]+ \[[\d.,]+\])\s+([\d.]+)\s+([\d.]+ \[[\d.,]+\])\s+([\d.]+)', ln)
    if m:
        g = m.groups(); rows.append([{'LR':'Logit'}.get(g[0], g[0]), 'class weighting' if g[1]=='class_weight' else 'group-wise SMOTE', g[2].replace('(','(').strip(), g[3], g[5].replace(',',', '), g[6], g[7].replace(',',', '), g[8]])
tab(16, 'Ablation: removing panel-size features from the detection model',
 'Extreme age-mix partition. "No direct size (10)" removes the three direct size features (unique inpatient beneficiaries, total unique beneficiaries, claims per beneficiary); "no size-related (8)" also removes the two attending-physician counts. Logit = logistic regression, RF = random forest, XGB = gradient boosting. Detection rate in percent.',
 ['Model','Training','Features','Detect \\%','Unadj.\\ OR [CI]','$p$','Adj.\\ OR [CI]','$p$'], rows, spec='lllrlrlr', small='\\scriptsize',
 after='Removing the direct size features moves the unadjusted age-mix odds ratio of the logistic regression above one (not significant) and weakens it in the random forest; in the random forest and gradient boosting it stays significant, and the size-adjusted estimate is non-significant in every row.')
# S17
tab(17, 'Empirical Bayes shrinkage of provider mean age',
 'Provider mean age is replaced by its posterior mean under a normal model with REML $\\hat\\tau^2=0.812$ and known $\\sigma_w=12.73$.',
 ['Quantity','Value'],
 ['Group composition',['SD of provider mean age, raw $\\to$ shrunken','4.55 $\\to$ 0.376'],['Providers that keep their extreme age-mix group','62.3\\%'],['Median panel size of the extreme group, raw $\\to$ shrunken','13 $\\to$ 45'],['Share of the extreme group in the top panel-size quartile','9.9\\% $\\to$ 37.7\\%'],['Fraud prevalence of the extreme group','4.8\\% $\\to$ 12.9\\%'],'--','Extreme age-mix association',['Raw attribute (class weighting)','unadj.\\ 0.477, adj.\\ 1.018'],['Shrunken attribute, detection model fixed','unadj.\\ 2.262 [1.362, 3.758] ($p=0.0016$); adj.\\ 1.127 [0.628, 2.023] ($p=0.688$)'],['Shrunken attribute, model refit on shrunken age','unadj.\\ 1.447 [0.828, 2.530] ($p=0.194$); adj.\\ 0.807 [0.429, 1.518] ($p=0.507$)'],['Shrunken attribute, pooled SMOTE','unadj.\\ 2.348 [1.417, 3.889] ($p=0.0009$); adj.\\ 1.141 [0.636, 2.046] ($p=0.658$)']],
 spec='lp{7.4cm}', after='Sweep over the assumed $\\tau^2$ with the detection model fixed (unadjusted odds ratio): 0.247 gives 2.685 ($p=0.0002$); 0.812 gives 2.262; 0.948 (upper profile limit) gives 2.092 ($p=0.0043$); 1.97 gives 1.686 ($p=0.0415$). Shrinkage inverts the group composition: the extreme group becomes the larger, higher-prevalence providers, so the sign of the unadjusted association flips, and no adjusted estimate differs from one.')
# S18
ex = pd.read_csv(OUT / 'cms_exposure.csv'); ex['g'] = ex.gamma1.round(3)
nm = {'All providers (pooled)':'All providers (pooled)','Certified Registered Nurse Anesthetist (CRNA)':'CRNA','Physical Therapist in Private Practice':'Physical therapist'}
grid = [0.25, 0.5, 0.901, 1.213]
w('\\Needspace{16\\baselineskip}'); w('\\subsection*{Table S18. Exposure to a spurious age disparity by size--detection slope}')
w('Percentage of audits reporting a significant spurious age disparity when the true age effect is zero, by specialty, for audits of $m$ fraud-labeled providers (asymptotic Wald test). Panel sizes, $\\sigma_w$ and $\\tau$ come from CMS 2021--2022; the size--label slope (1.398) and the mean detection rate (0.88) come from the primary dataset, because CMS carries no fraud labels. The columns vary the size--detection slope in the model of the audited detector, in units of log odds per SD of log panel size (1.213 is the primary-dataset value). The two-sided rate depends on the magnitude of the slope; its sign sets the direction of the spurious gap (Proposition 1(iii)). The size slope of the flag among excluded providers in the LEIE check (Table S12) is $-0.25$ in the reference run and between $-0.18$ and $+0.10$ across 20 other seeds; the column for 0.25 is the upper end of the magnitudes seen.\n')
sub('Panel A. Audits of 10,000 fraud-labeled providers',['Group','$\\tau$']+[f'slope {g}' for g in grid],
    [[nm.get(r.group,r.group), f'{r.tau:.2f}'] + [f'{ex[(ex.group==r.group)&(ex.g==g)&(ex.m==10000)].sig_rate.iloc[0]:.1f}' for g in grid] for _, r in ex[(ex.g==0.25)&(ex.m==10000)].iterrows()])
rows=[]
for grp in ['All providers (pooled)','Certified Registered Nurse Anesthetist (CRNA)']:
    for m in [500,2000,10000,100000]:
        rows.append([nm.get(grp,grp), f'{m:,}'] + [f'{ex[(ex.group==grp)&(ex.g==g)&(ex.m==m)].sig_rate.iloc[0]:.1f}' for g in grid])
sub('Panel B. Audit size, for all providers and for the most homogeneous specialty',['Group','$m$']+[f'slope {g}' for g in grid], rows)
sub('Panel C. Audits of 500 fraud-labeled providers, all groups',['Group','$\\tau$']+[f'slope {g}' for g in grid],
    [[nm.get(r.group,r.group), f'{r.tau:.2f}'] + [f'{ex[(ex.group==r.group)&(ex.g==g)&(ex.m==500)].sig_rate.iloc[0]:.1f}' for g in grid] for _, r in ex[(ex.g==0.25)&(ex.m==500)].iterrows()])
w('The rate exceeds the nominal 5\\% only in large audits, and for small slopes only in homogeneous specialties.\n')

# S19
ct = pd.read_csv(OUT / 'corollary1_table.csv')
w('\\Needspace{20\\baselineskip}'); w('\\subsection*{Table S19. Corollary 1: predicted and simulated rate of a spurious disparity}')
w('Outer-versus-middle partition, true age effect zero, panel sizes and labels of the primary dataset unless stated. $\\eta_G$, $\\eta_D$ and $\\rho$ are the quantities of Corollary 1; $\\phi=\\rho\\,\\eta_D\\,\\eta_G$; predicted = $100\\{\\Phi(\\sqrt m|\\phi|-1.96)+\\Phi(-\\sqrt m|\\phi|-1.96)\\}$ with $m=506$ labeled providers (CMS rows: the audit size of the simulated population); simulated = share of 5,000 replicates with a significant unadjusted disparity; MC SE = Monte Carlo standard error.\n')
for pn, ttl in (('A','Panel A. True between-provider SD of mean age ($\\tau$)'),('B','Panel B. Strength of the panel-size effect on detection ($\\tau=0.9$)'),('C','Panel C. Primary dataset and CMS-calibrated reliability regimes')):
    x = ct[ct.panel == pn]
    sub(ttl, ['Setting','$\\tau$','$\\eta_G$','$\\eta_D$','$\\rho$','$\\phi$','Predicted \\%','Simulated \\%','MC SE'],
        [[r.setting.replace('>=','$\\ge$').replace('tau = ','$\\tau=$'), f'{r.tau:.2f}', f'{r.eta_G:.3f}', f'{r.eta_D:.3f}', f'{r.rho:.3f}', f'{r.phi:.3f}', f'{r.predicted:.1f}', f'{r.simulated:.1f}', f'{r.mc_se:.2f}'] for _, r in x.iterrows()], spec='lrrrrrrrr', size='\\footnotesize')
dev = (ct.predicted - ct.simulated).abs()
w(f'The prediction is within {dev.max():.1f} points of the simulated rate in every row (mean absolute difference {dev.mean():.1f}), above the Monte Carlo error in several rows. The prediction exceeds the simulation by 1 to 2 points where the rate is near 100\\% (Panels A and B); the Wald test on a $2\\times2$ table with a small cell is slightly conservative, and the number of labeled providers varies across replicates, whereas the formula uses the fixed expected number. Script: \\texttt{{corollary1.py}}.\n')
# S20
vc = (OUT / 'variance_component_inference_output.txt').read_text().split('\n')
def g(pat): return [l for l in vc if l.startswith(pat)][0]
import re as _re
def nums(l): return [float(x) for x in _re.findall(r'-?\d+\.\d+|-?\d+', l)]
tab(20, 'Between-provider variance of mean patient age: estimators and sensitivity',
 'Primary dataset, 5,410 providers; $\\sigma_w=12.73$ years from beneficiary-level data.',
 ['Estimator or check','Result'],
 [['REML, $\\hat\\tau^2$ ($\\hat\\tau$)', '0.812 (0.901); profile 95\\% interval [0.689, 0.948]; boundary-adjusted [0.708, 0.925]'],
  ['ML, $\\hat\\tau^2$', '0.811; profile 95\\% interval [0.689, 0.947]'],
  ['DerSimonian--Laird, $\\hat\\tau^2$', '0.802'],
  ['Likelihood-ratio test of $\\tau^2=0$ (REML), boundary mixture', '465.8, $p<0.001$'],
  ["Cochran's $Q$", '7,206.5 on 5,409 df, $p<0.001$; $Q/(k-1)=1.332$'],
  ['Median $R$ at $\\tau^2=0.812$ / 0.948', '0.111 / 0.128'],
  ['Share of providers with $R<0.7$ at $\\tau^2=0.812$ / 0.948', '97.9\\% / 97.3\\%'],
  ['Providers with $R\\ge0.7$ at $\\tau^2=0.812$ / 0.948', '115 / 144'],
  ["$Q$ with $\\sigma_w$ scaled by 0.98 / 1.02", '7,503.7 / 6,926.7 (both $p<0.001$)']],
 spec='p{6.4cm}p{8.8cm}', after='The three estimators agree to within 0.01, and the conclusion that a small true component exists does not depend on $\\sigma_w$ within $\\pm2\\%$. Script: \\texttt{variance\\_component\\_inference.py}.')
# S21
tab(21, 'Provider-level features of the detection model',
 'Seventeen provider-level aggregates are derived from the claims; thirteen are model features. Size-related: direct size features (three) and attending-physician counts (two) are removed in the ablation of Table S16.',
 ['Feature (code name)','Domain','Role'],
 [['Unique inpatient beneficiaries (IP\\_UniqueBene)','inpatient','direct size'],['Mean inpatient reimbursement (IP\\_AvgReimbursed)','inpatient','feature'],['Mean length of stay (IP\\_AvgLOS)','inpatient','feature'],['Mean distinct diagnosis codes, inpatient (IP\\_AvgDiagCodes)','inpatient','feature'],['Unique attending physicians, inpatient (IP\\_UniqueAttPhysician)','inpatient','size-related'],
  ['Mean outpatient reimbursement (OP\\_AvgReimbursed)','outpatient','feature'],['Mean diagnosis-code count, outpatient (OP\\_AvgDiagCodes)','outpatient','feature'],['Unique attending physicians, outpatient (OP\\_UniqueAttPhysician)','outpatient','size-related'],
  ['Total unique beneficiaries (TotalUniqueBene)','composition','direct size'],['Proportion female (FemaleRatio)','composition','feature'],['Mean chronic-condition count (AvgChronicCondCount)','composition','feature'],['Proportion deceased (DeceasedRatio)','composition','feature'],['Claims per beneficiary (ClaimsPerBene)','intensity','direct size'],
  '--','Excluded from the feature set',['Inpatient claim count (IP\\_ClaimCount)','inpatient','near-duplicate of IP\\_UniqueBene, $r=0.998$'],['Outpatient claim count (OP\\_ClaimCount)','outpatient','near-duplicate of TotalUniqueBene, $r=0.928$'],['Mean patient age (AvgPatientAge)','composition','sensitive attribute'],['Proportion non-White (MinorityRatio)','composition','sensitive attribute']],
 spec='p{7.6cm}lp{3.6cm}', small='\\footnotesize', after='Maximum pairwise correlation among the thirteen features 0.853; maximum variance inflation factor 5.14. Table 5 of the main text uses inpatient beneficiaries and claims per beneficiary, both size-related features, as alternative controls.')

w('\\end{document}')
D.mkdir(parents=True, exist_ok=True)
(D / 'supplement.tex').write_text('\n'.join(out))
print('written', D / 'supplement.tex')
