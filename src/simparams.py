"""Coefficients of the generating process of the null simulation (Eq. 3 of the paper).
fraud:  logit Pr(L=1)    = B_FRAUD[0] + B_FRAUD[1] z
detect: logit Pr(D=1|L=1) = B_DET[0]   + B_DET[1]   z,   z = standardized log(1 + unique beneficiaries)
The detection coefficients are fitted to the out-of-fold detections of the primary logistic regression:
class weighting (primary, DET_STRATEGY=cw) or group-wise SMOTE (sensitivity, DET_STRATEGY=smote)."""
import os
STRATEGY = os.environ.get('DET_STRATEGY', 'cw')
B_FRAUD = (-2.975, 1.398)
B_DET = (0.854, 1.213) if STRATEGY == 'cw' else (1.261, 0.901)
