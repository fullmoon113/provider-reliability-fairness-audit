"""Builds supplement.tex (Tables S1-S17) from saved outputs and recorded run numbers.
Usage: python make_supplement.py OUTDIR   (then run pdflatex twice in OUTDIR)"""
import sys, re
from pathlib import Path
import pandas as pd
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
 spec='lrrr', after='The unadjusted odds ratio rises from 0.477 to 0.689 as providers up to 50 beneficiaries are excluded and then falls to 0.559 above 100, so the pattern is not monotonic; the interval widens as the sample shrinks and the test loses significance above 20. We do not read the table as a dose-response. The adjusted analysis of Table S8 is the primary one.')
# S3
tab(3, 'Temporal stability of measured mean patient age by panel size (primary dataset)',
 'Claims split at 30 June 2009. Mean age is computed over unique beneficiaries in each half, matching the feature definition. This table does not depend on the detection model.',
 ['Minimum unique beneficiaries per half','Providers','Between-half correlation'],
 [['$\\ge1$','5,008','0.101'],['$\\ge5$','3,782','0.238'],['$\\ge10$','2,974','0.297'],['$\\ge20$','2,028','0.348'],['$\\ge30$','1,489','0.407'],['$\\ge50$','940','0.485'],['$\\ge100$','445','0.594']],
 spec='lrr', after='The reliability of a provider\'s measured age composition rises with the number of beneficiaries over which it is computed, as the aggregation mechanism of Section III predicts.')
# S4
tab(4, 'Fairness metrics before and after equalized-odds mitigation',
 'Exponentiated-gradient reduction ($\\epsilon=0.02$, 30 iterations), applied to the class-weighted logistic regression. Gaps are between the middle and extreme age-mix groups.',
 ['Metric','Before','After'],
 [['Equalized-odds difference','0.100','0.031 (SD 0.015; reduced in 30 of 30 runs)'],['Demographic parity difference','0.153','--'],['F1','0.532','0.388'],['Recall, middle age-mix','0.883','0.781'],['Recall, extreme age-mix','0.783','0.763']],
 spec='lrl', after='Bootstrap (3,000 resamples of the 5,410 providers): recall gap (middle minus extreme) 0.100, 95\\% CI [0.028, 0.181]. Mitigation closes the gap by lowering recall in the middle group rather than raising it in the extreme group (F1 falls from 0.53 to 0.39).')
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
def sub(title, header, rows, spec=None):
    w(f'\\Needspace{{{min(len(rows)+5,48)}\\baselineskip}}'); w(f'\\textbf{{{title}}}'); w('\\begin{center}\\small'); w(f'\\begin{{tabular}}{{{spec or "l"+"r"*(len(header)-1)}}}\\toprule'); w(' & '.join(header)+r' \\ \midrule')
    for r in rows: w(' & '.join(map(str,r))+r' \\')
    w('\\bottomrule\\end{tabular}\\end{center}')
sub('Panel A. True between-provider SD of mean age ($\\tau$)',['$\\tau$ (years)','Unadjusted \\%','Adjusted \\%'],
 [['0','96.9','4.62'],['0.5','94.9','4.46'],['0.9 (main)','87.1','4.30'],['1.0','83.9','4.40'],['2.0','43.7','4.92'],['4.0','12.4','4.96'],['6.0','7.4','5.30']])
sub('Panel B. Strength of the panel-size effect on detection ($\\tau=0.9$; multiples of the main slope 1.213)',['Slope multiple','Unadjusted \\%','Adjusted \\%','Median unadj.\\ OR'],
 [['0.50','47.0','4.66','0.633'],['0.75','72.7','4.52','0.517'],['1.00 (main)','87.1','4.32','0.431'],['1.25','93.9','4.44','0.369'],['1.50','96.9','4.72','0.324']])
sub('Panel C. Reliability regimes calibrated to CMS data',['Regime','$\\tau$','Median unadj.\\ OR [interval]','Unadj.\\ \\%','Adj.\\ \\%'],
 [['Homogeneous specialty','1.65','0.572 [0.343, 0.976]','57.1','4.78'],['Specialty median','3.47','0.777 [0.470, 1.315]','16.6','4.90'],['Pooled specialties','4.85','0.853 [0.519, 1.439]','9.1','4.70'],
  ['Specialty median, panels $\\ge$11 (n = 3,825)','3.47','--','7.0','5.28'],['Specialty median, panels $\\ge$50 (n = 1,713)','3.47','--','4.4','4.40'],['Homogeneous, panels $\\ge$11','1.65','--','23.6','5.24'],['Homogeneous, panels $\\ge$50','1.65','--','6.3','4.82']])
w('The median panel size is 25 beneficiaries in the full simulated population and 43 and 104 in the restricted populations above. Under the specialty-median regime the unadjusted rate approaches the nominal level once small panels are excluded.\n')
pe = pd.read_csv(OUT / 'planted_effect_power.csv'); pe = pe[pe.audit == 'primary']
rows = []
for _, r in pe.sort_values(['rho', 'tau', 'OR_true'], ascending=[True, True, False]).iterrows():
    rows.append([f'{r.rho:.1f}', f'{r.tau:.1f}', f'{r.OR_true:.1f}', f'{r.naive_rej:.1f}', f'{r.adjusted_rej:.1f}', f'{r.shrunken_rej:.1f}', f'{r.RC_rej:.1f}', f'{r.procedure_rej:.1f}', f'{r.oracle_rej:.1f}'])
sub('Panel D. Planted effect: rejection rate (\\%), 300 replications (type I error at OR = 1)',
 ['$\\rho$','$\\tau$','True OR','Unadjusted','Size-adjusted','Shrunken','Reg.\\ calib.','Procedure','Oracle'],rows,spec='rrrrrrrrr')
w('$\\rho$ is the correlation between true provider mean age and standardized log panel size in the simulated population. The planted-effect table of the main text reports $\\rho=0$.\n')
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
w('The sampling share is 5.3\\% by the variance-ratio route of Section III-A (ratio 18.96 in the 2022 file) and 6.3\\% to 7.6\\% by the paired-year route.\n')
# S10
tab(10, 'Maximum likelihood and Firth penalized likelihood',
 'Firth\'s penalized likelihood removes the first-order small-sample bias of maximum likelihood and remains finite under separation. Fraud-positive subgroups number 129, 56 and 103 providers. $p$-values in parentheses; for Firth they are penalized likelihood-ratio tests.',
 ['Partition','ML unadj.\\ (p)','Firth unadj.\\ (p)','ML adj.\\ (p)','Firth adj.\\ (p)'],
 ['Panel A. Sensitive-attribute partitions',['Extreme age-mix','0.477 (0.0055)','0.475 (0.0061)','1.018 (0.952)','1.010 (0.974)'],['High-elderly (top quartile)','0.444 (0.0166)','0.437 (0.0186)','0.807 (0.568)','0.793 (0.535)'],['High-minority (top quartile)','1.187 (0.601)','1.158 (0.647)','1.464 (0.285)','1.424 (0.304)']],
 spec='lrrrr')
tab('10 (cont.)', 'Specificity: adjusted extreme age-mix effect when each alternative characteristic is used as the control',
 'Each row gives the $p$-value for the extreme age-mix effect when the named characteristic replaces panel size as the control variable (heavy right-skewed characteristics enter on the log scale, as panel size does). Holm adjustment over the eight rows.',
 ['Control variable','$p$ (ML)','Holm (ML)','$p$ (Firth)','Holm (Firth)'],
 [['Panel size (log)','0.952','0.952','0.974','0.974'],['Mean length of stay','0.0004','0.0031','0.0004','0.0030'],['Chronic conditions','0.0023','0.0163','0.0022','0.0152'],['Claims per beneficiary (log)','0.0043','0.0207','0.0041','0.0189'],['Inpatient beneficiaries (log)','0.0025','0.0163','0.0026','0.0154'],['Outpatient reimbursement (log)','0.0115','0.0288','0.0109','0.0264'],['Deceased ratio','0.0041','0.0207','0.0038','0.0189'],['Female ratio','0.0096','0.0288','0.0088','0.0264']],
 spec='lrrrr', after='The age-mix effect stays significant after Holm correction when any of the seven alternatives is the control, and disappears only when panel size is. Joint model with all seven alternatives: extreme age-mix OR 0.475 [0.234, 0.965], $p=0.040$; adding panel size, 0.716 [0.250, 2.045], $p=0.532$. The adjusted age-mix estimate has standard error 0.300 on the log scale; the smallest odds ratio detectable at 80\\% power is 0.432. An equivalence test (TOST) against odds ratios of 0.5 and 2 gives $p=0.247$, so equivalence is not established.')
# S11
tab(11, 'Sensitivity of the adjusted age effect to label-selection strength',
 'The analysis conditions on a positive fraud label. If the probability of receiving a label scales with panel size, conditioning on it can induce association among other causes. Each row reweights fraud-positive providers by an assumed selection probability proportional to $\\exp(-\\gamma z)$, with $z$ the standardized log panel size. Robust (sandwich) $p$-values.',
 ['$\\gamma$','Adj.\\ OR (extreme age-mix)','$p$'],
 [['0','1.018','0.953'],['0.25','1.036','0.909'],['0.50','1.064','0.852'],['1.0','1.139','0.754'],['1.5','1.199','0.744'],['2.0','1.192','0.815']],
 spec='lrr', after='The adjusted extreme age-mix association is non-significant at every selection strength. E-values (square-root approximation of the odds ratio as a risk ratio; direct odds-ratio value in parentheses): 1.10 (1.15) for the extreme age-mix estimate, whose interval contains one so the E-value for the interval is 1; and 1.47 (1.78) for the top-quartile estimate of 0.807 (Table S10).')
# S12
tab(12, 'External validation against the OIG exclusion list (LEIE)',
 'Providers in the 2021--2022 CMS file whose NPI appears in the OIG List of Excluded Individuals and Entities with an exclusion date after the claims window: 416 of 1,105,159. A gradient-boosting classifier flags the top 10\\% of providers by score; odds ratios compare the share flagged among excluded providers.',
 ['Quantity','Value'],
 [['Excluded providers recovered by the top-10\\% flag','24.8\\%'],['Extreme age-mix: OR for being flagged','1.26 [0.78, 2.05]'],['Adjusted for panel size','1.18 [0.72, 1.93]'],['Outer age deciles versus the rest','0.76 [0.47, 1.23]'],['Same model with mean age as an input: OR','4.03 [2.26, 7.18]'],['Same, adjusted','3.75 [2.10, 6.71]']],
 spec='lr', after='Where age is not an input to the detection model, no significant disparity by age-mix appears; where it is an input, one appears. The label itself depends on unit size, because large billers are investigated more often, so this check cannot separate detection bias from label bias.')
# S13
w('\\Needspace{14\\baselineskip}'); w('\\subsection*{Table S13. Errors-in-variables remedies in simulation}')
w('1,000 replications. Median odds ratio and rejection rate (\\%) for the extreme age-mix partition. Null: no true effect. Planted: true odds ratio 0.70. RC = regression calibration with the Stage 0 reliability; oracle = true provider mean age.\n')
sub('Panel A. Remedies for the unadjusted partition',['$\\tau$','Truth','Method','Median OR [interval]','Rejection \\%'],
 [['0.9','Null','Unadjusted (naive)','0.440 [0.266, 0.785]','83.6'],['','','Size-adjusted','1.039 [0.560, 2.127]','6.6'],['','','RC','1.036 [0.745, 12.345]','14.9'],['','','RC + size','1.025 [0.703, 5.835]','4.4'],['','','Oracle','1.020 [0.614, 1.682]','4.6'],
  ['0.9','Planted 0.70','Unadjusted (naive)','0.425','91.9'],['','','Size-adjusted','0.968','5.6'],['','','RC','0.979 [0.724, 11.150]','20.3'],['','','RC + size','0.965','5.4'],['','','Oracle','0.721 [0.458, 1.158]','26.2'],
  ['1.65','Null','Unadjusted (naive)','0.583','53.4'],['','','Size-adjusted','0.998','5.6'],['','','RC','1.019','3.5'],['','','RC + size','1.012','5.1'],['','','Oracle','1.003','4.8'],
  ['1.65','Planted 0.70','Unadjusted (naive)','0.544','69.5'],['','','Size-adjusted','0.918','7.3'],['','','RC','0.967','4.7'],['','','RC + size','0.922','8.0'],['','','Oracle','0.738','23.5']],spec='llllr')
sub('Panel B. Methods designed for noisy sensitive attributes (null truth)',['$\\tau$','Flip rate','Indep.\\ rejected \\%','Measured gap / sig.\\ \\%','Corrected gap','Recall before/after'],
 [['0.9','0.467','87.1','+0.111 [0.033, 0.190] / 81.9','+1.700','0.859 $\\to$ 0.777'],['1.65','0.417','60.6','+0.072 / 53.0','+0.427','0.859 $\\to$ 0.810'],['3.47','0.316','24.0','+0.029 / 15.3','+0.080','0.859 $\\to$ 0.838']],spec='lrrlrl')
w('Flip rate = mean misclassification rate of the extreme-age label; Indep.\\ rejected = rejection rate of the conditional-independence test of the noisy-attribute method; recall before/after = recall with and without equalized-odds post-processing on the noisy groups.\n')
w('The true gap in these simulations is $-0.014$ at $\\tau=0.9$; post-processing removes the measured gap (0.111 to 0) at a cost in recall.\n')
# S14
def parse_m5():
    t = (OUT / 'm5_summary.txt').read_text().split('\n'); blocks = {}; cur = None
    for ln in t:
        m = re.match(r'\[fill (\S+)\] (\S+)', ln)
        if m: cur = (m.group(1), m.group(2)); blocks[cur] = {}; continue
        if cur and ln.strip() and not ln.startswith('band') and not ln.startswith('attribute'):
            parts = re.split(r'\s{2,}', ln.strip())
            try: blocks[cur][parts[0]] = [float(x) for x in parts[1:5]]
            except ValueError: pass
    return blocks
B = parse_m5()
order = ['mean age','age <65','age 65-74','age 75-84','age >84','female','non-White (any)','Black','Asian/Pacific Is.','Hispanic','Native American','Other race','dual-eligible']
supp = {'age <65':0.465,'age 65-74':0.174,'age 75-84':0.282,'age >84':0.474,'Black':0.692,'Asian/Pacific Is.':0.817,'Hispanic':0.753,'Native American':0.666,'Other race':0.808}
w('\\Needspace{22\\baselineskip}'); w('\\subsection*{Table S14. Reliability of multiple aggregate attributes in CMS claims (Stage 0 sweep)}')
w('Within-specialty analysis in the eight largest specialties, 2022 file, provider-weighted. Suppressed category counts (1--10 beneficiaries) are set to 5.5 (midpoint) or 0. Bands are numbers of beneficiaries. $R$ = median reliability; share = share of providers with $R_i\\ge0.7$; $m$ = expected share placed in the wrong group by an outer-versus-middle quartile split. Mean age, share female and any-non-White are not suppressed.\n')
for fill, lab in (('5.5','midpoint fill (5.5)'),('0','zero fill')):
    rows = []
    for a in order:
        r = B[(fill,'R')].get(a); g = B[(fill,'ge07')].get(a); m = B[(fill,'m')].get(a)
        if r is None: continue
        rows.append([a, f'{supp[a]*100:.1f}' if a in supp else '0'] + [f'{x:.2f}' for x in r] + [f'{x:.2f}' for x in g] + [f'{x:.2f}' for x in m])
    w('\\Needspace{18\\baselineskip}'); w(f'\\textbf{{Suppressed counts: {lab}}}'); w('\\begin{center}\\scriptsize\\setlength{\\tabcolsep}{3pt}')
    w('\\begin{tabular}{lr cccc cccc cccc}\\toprule'); w(r' & & \multicolumn{4}{c}{Median $R$} & \multicolumn{4}{c}{Share $R_i\ge0.7$} & \multicolumn{4}{c}{$m$} \\')
    w(r'Attribute & Supp.\ \% & 11--20 & 21--50 & 51--100 & 101+ & 11--20 & 21--50 & 51--100 & 101+ & 11--20 & 21--50 & 51--100 & 101+ \\ \midrule')
    for r in rows: w(' & '.join(r) + r' \\')
    w('\\bottomrule\\end{tabular}\\end{center}')
tr = pd.read_csv(OUT / 'm5' / 'cms_stage0_testretest.csv')
rows = [[r.attribute] + [f'{r[c]:.3f}' for c in ['n 11-20','n 21-50','n 51-100','n 101+']] for _, r in tr.iterrows()]
sub('Year-to-year (2021 to 2022) correlation of the provider value, by the smaller of the two years\' panel sizes',['Attribute','11--20','21--50','51--100','101+'], rows)
w('Pooled across specialties (differences between specialties treated as true variance), median reliability is 0.84 to 0.99 across attributes and bands; this is the relevant figure only when an audit compares providers across specialties.\n')
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
        g = m.groups(); rows.append([g[0], 'class weighting' if g[1]=='class_weight' else 'group-wise SMOTE', g[2].replace('(','(').strip(), g[3], g[5].replace(',',', '), g[6], g[7].replace(',',', '), g[8]])
tab(16, 'Ablation: removing panel-size features from the detection model',
 'Extreme age-mix partition. "No direct size (10)" removes the three direct size features (unique inpatient beneficiaries, total unique beneficiaries, claims per beneficiary); "no size-related (8)" also removes the two attending-physician counts. LR = logistic regression, RF = random forest, XGB = gradient boosting. Detection rate in percent.',
 ['Model','Training','Features','Detect \\%','Unadj.\\ OR [CI]','$p$','Adj.\\ OR [CI]','$p$'], rows, spec='lllrlrlr', small='\\scriptsize',
 after='Removing the direct size features moves the unadjusted age-mix odds ratio of the logistic regression above one (not significant) and weakens it in the random forest; in the random forest and gradient boosting it stays significant, and the size-adjusted estimate is non-significant in every row.')
# S17
tab(17, 'Empirical Bayes shrinkage of provider mean age',
 'Provider mean age is replaced by its posterior mean under a normal model with REML $\\hat\\tau^2=0.812$ and known $\\sigma_w=12.73$.',
 ['Quantity','Value'],
 ['Group composition',['SD of provider mean age, raw $\\to$ shrunken','4.55 $\\to$ 0.376'],['Providers that keep their extreme age-mix group','62.3\\%'],['Median panel size of the extreme group, raw $\\to$ shrunken','13 $\\to$ 45'],['Share of the extreme group in the top panel-size quartile','9.9\\% $\\to$ 37.7\\%'],['Fraud prevalence of the extreme group','4.8\\% $\\to$ 12.9\\%'],'--','Extreme age-mix association',['Raw attribute (class weighting)','unadj.\\ 0.477, adj.\\ 1.018'],['Shrunken attribute, detection model fixed','unadj.\\ 2.262 [1.362, 3.758] ($p=0.0016$); adj.\\ 1.127 [0.628, 2.023] ($p=0.688$)'],['Shrunken attribute, model refit on shrunken age','unadj.\\ 1.447 [0.828, 2.530] ($p=0.194$); adj.\\ 0.807 [0.429, 1.518] ($p=0.507$)'],['Shrunken attribute, pooled SMOTE','unadj.\\ 2.348 [1.417, 3.889] ($p=0.0009$); adj.\\ 1.141 [0.636, 2.046] ($p=0.658$)']],
 spec='lp{7.4cm}', after='Sweep over the assumed $\\tau^2$ with the detection model fixed (unadjusted odds ratio): 0.247 gives 2.685 ($p=0.0002$); 0.812 gives 2.262; 0.948 (upper profile limit) gives 2.092 ($p=0.0043$); 1.97 gives 1.686 ($p=0.0415$). Shrinkage inverts the group composition: the extreme group becomes the larger, higher-prevalence providers, so the sign of the unadjusted association flips, and no adjusted estimate differs from one.')
w('\\end{document}')
D.mkdir(parents=True, exist_ok=True)
(D / 'supplement.tex').write_text('\n'.join(out))
print('written', D / 'supplement.tex')
