"""
Fig 4 (null simulation distribution) + Fig 5 (false-positive rate curve)

재생성 스크립트. 원 작업에서는 인라인 실행했으나 다른 그림들과 스타일을 맞춰 복원.

입력:
  null_sim_5000reps.csv   — 널 시뮬레이션 500회 결과 (중간산출물/)
                            컬럼: n, uOR, up, aOR, ap
  (Fig 5의 민감도 수치는 아래 SENS 딕셔너리에 하드코딩. 재계산하려면
   Table A7 산출 코드를 다시 돌려야 함)

출력:
  Fig4_null_simulation.png
  Fig5_false_positive.png

※ "claim volume" -> "panel size" 명명 변경 시 수정할 곳:
   - Fig 4: LBL_ADJ  (x축 라벨)
   - Fig 5: LBL_X, LEG_ADJ, LEG_UNADJ
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 다른 그림들과 동일한 스타일 ────────────────────────
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 17,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13,
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans'
})

RED = '#c44e52'
BLUE = '#4c72b0'
DARK = '#1f3864'

# ── 명명 변경 시 여기만 고치면 됨 ──────────────────────
LBL_ADJ = 'Panel-size-adjusted odds ratio'
LBL_X = 'True between-provider SD of mean patient age (years)'
LEG_UNADJ = 'Conventional (unadjusted) audit'
LEG_ADJ = 'Panel-size-adjusted audit'

CSV = DATA / 'null_sim_5000reps.csv'      # created by src/make_null_sim_csv.py
OBS_UNADJ, OBS_ADJ = 0.462, 0.823

# ══════════════════════════════════════════════════════
# Fig 4 — 널 시뮬레이션 분포
# ══════════════════════════════════════════════════════
r = pd.read_csv(CSV)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

a = ax[0]
a.hist(r.uOR, bins=32, color=RED, edgecolor='black', linewidth=.8, alpha=.85,
       label='Null simulation (true age effect = 0)')
a.axvline(OBS_UNADJ, color=DARK, lw=3, ls='-', label='Observed (OR = %.3f)' % OBS_UNADJ)
a.axvline(1.0, color='gray', lw=2, ls=':', label='No effect (OR = 1)')
a.set_xlabel('Unadjusted odds ratio', fontweight='bold')
a.set_ylabel('Frequency (5,000 replications)', fontweight='bold')
a.set_title('(a) Unadjusted: a null mechanism\nreproduces the observed disparity',
            fontweight='bold')
a.legend(fontsize=11, loc='upper right')
a.grid(axis='y', alpha=.3)

b = ax[1]
b.hist(r.aOR, bins=32, color=BLUE, edgecolor='black', linewidth=.8, alpha=.85,
       label='Null simulation, adjusted')
b.axvline(OBS_ADJ, color=DARK, lw=3, ls='-', label='Observed (OR = %.3f)' % OBS_ADJ)
b.axvline(1.0, color='gray', lw=2, ls=':', label='No effect (OR = 1)')
b.set_xlabel(LBL_ADJ, fontweight='bold')
b.set_ylabel('Frequency (5,000 replications)', fontweight='bold')
b.set_title('(b) Adjusted: distribution recenters on 1.0', fontweight='bold')
b.legend(fontsize=11, loc='upper right')
b.grid(axis='y', alpha=.3)

plt.tight_layout()
plt.savefig(str(OUT / 'Fig4_null_simulation.png'), bbox_inches='tight')
plt.close()
print('Fig4 저장. 관측값 백분위 %.0f%%' % (100 * (r.uOR <= OBS_UNADJ).mean()))

# ══════════════════════════════════════════════════════
# Fig 5 — 잡음대신호 비율별 위양성률
# ══════════════════════════════════════════════════════
# Table S7 Panel A (5,000 replicates; produced by src/sim5k.py, SD 0.50 row = 76.38%)
SENS = {
    'sd':     [0.0, 0.5, 1.0, 2.0, 4.0, 6.0],
    'unadj':  [81.50, 76.38, 58.88, 26.00, 8.70, 6.10],
    'adj':    [4.64, 4.72, 4.60, 4.96, 4.80, 4.78],
}
EST_SD = 0.50   # 본 데이터에서 추정된 참 기관간 SD

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(SENS['sd'], SENS['unadj'], 'o-', color=RED, lw=2.5, ms=9, label=LEG_UNADJ)
ax.plot(SENS['sd'], SENS['adj'], 's-', color=BLUE, lw=2.5, ms=9, label=LEG_ADJ)
ax.axhline(5, color='gray', ls=':', lw=2, label='Nominal 5% level')
ax.axvline(EST_SD, color=DARK, ls='--', lw=2.5)
ax.text(EST_SD + 0.85, 66,
        'Estimated true\nbetween-provider\nSD = %.2f yr' % EST_SD,
        fontsize=12, color=DARK, fontweight='bold')
ax.set_xlabel(LBL_X, fontweight='bold')
ax.set_ylabel('False-positive rate (%)', fontweight='bold')
ax.set_title('Spurious fairness gaps arise when demographic\naggregates are noise-dominated',
             fontweight='bold', fontsize=16)
ax.legend(fontsize=12)
ax.grid(alpha=.3)
ax.set_ylim(0, 90)

plt.tight_layout()
plt.savefig(str(OUT / 'Fig5_false_positive.png'), bbox_inches='tight')
plt.close()
print('Fig5 저장.')
