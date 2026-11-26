"""STEP 1 of the Medicare flag-rate audit (paper Section IV-G): trains the two screening models, flags providers, and writes the
PREDICTIONS of Proposition 1 / Corollary 1 for every attribute-by-specialty audit. It never forms the association between a
group indicator and the flags; run src/flag_rate_observe.py only after output/flag_rate_predictions.csv has been written (and, in the
repository, committed).

Detectors (labels: providers excluded by the OIG on or after 2023-01-01; 5-fold out-of-fold scores; flag = the 10% highest scores
WITHIN each specialty, so that every specialty has flagged and unflagged providers):
  T1  histogram gradient boosting, balanced class weights; inputs: indicators of the 20 most common provider types, log panel size
  T2  T1 inputs plus utilization and payment fields (services, distinct procedure codes, allowed and submitted amounts, payments per
      beneficiary, drug and medical shares, entity type, participation); no demographic counts and no risk score
Audit unit = provider not excluded; outcome = flagged (investigation burden).  For each of the eight largest specialties and each of
six attributes the script runs Stage 0 (REML tau^2, sigma_w, R_i, exact M) and predicts the outer-versus-middle odds ratio:
  P1  Proposition 1 with a constant mean (the measured quartiles of the specialty set the boundaries)
  P2  Proposition 1 with the mean trend of the attribute in log panel size (A | n ~ N(m(n), tau^2 + sigma_w^2 / n))
Pre-specified criterion (observe script): the prediction holds for an audit on an attribute that fails Stage 0 (M > 10%) if
  (i) |log OR_unadjusted - log OR_predicted| <= max(2 SE, 0.05),  (ii) |log OR_size-adjusted| < 0.05 and (iii) |log OR_center split| < 0.05.
Usage: python src/flag_rate_predict.py CMS2022_EXTRACT.csv.gz UPDATED.csv"""
import sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from flag_rate_common import *
from stage0 import _reml, expected_misclassification

def oof_flags(X, y, groups, seed=42, top=0.10):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed); p = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = HistGradientBoostingClassifier(random_state=seed, max_iter=100, class_weight='balanced').fit(X.iloc[tr], y[tr])
        p[va] = m.predict_proba(X.iloc[va])[:, 1]
    thr = pd.Series(p).groupby(groups).transform(lambda s: s.quantile(1 - top)).values
    return (p >= thr).astype(int), p

def t2_features(d):
    X = pd.DataFrame({'tot_hcpcs': d.Tot_HCPCS_Cds, 'srv_per_bene': d.Tot_Srvcs / d.n, 'allowed_per_bene': d.Tot_Mdcr_Alowd_Amt / d.n,
                      'pay_per_bene': d.Tot_Mdcr_Pymt_Amt / d.n, 'allowed_to_charge': d.Tot_Mdcr_Alowd_Amt / d.Tot_Sbmtd_Chrg.replace(0, np.nan),
                      'drug_share': d.Drug_Mdcr_Pymt_Amt / d.Tot_Mdcr_Pymt_Amt.replace(0, np.nan),
                      'med_share': d.Med_Mdcr_Pymt_Amt / d.Tot_Mdcr_Pymt_Amt.replace(0, np.nan),
                      'entity_org': (d.Rndrng_Prvdr_Ent_Cd == 'O').astype(float), 'participating': (d.Rndrng_Prvdr_Mdcr_Prtcptg_Ind == 'Y').astype(float)})
    return X.replace([np.inf, -np.inf], np.nan)

def predict_audit(n, a, sw, aud, flag, kind='outer', B=20):
    """Stage 0 and the Proposition 1 / Corollary 1 predictions for one attribute-by-specialty audit (no group-flag association)."""
    v = sw ** 2 / n; t2, _ = _reml(a, v)
    w = 1 / (t2 + v); mu = (w * a).sum() / w.sum(); R = t2 / (t2 + v) if t2 > 0 else np.zeros(len(a))
    wt = aud / aud.sum()
    M_aud, _, lam = expected_misclassification(n, mu, t2, sw, kind, aud, B=B, seed=11)
    q = np.quantile(a, [.25, .75]); ln = np.log(n)
    X = sm.add_constant(np.column_stack([ln, ln ** 2]))
    h = 1 / (1 + np.exp(-X @ sm.Logit(flag[aud > 0], X[aud > 0]).fit(disp=0, maxiter=100).params))
    def predict(pi):
        pb = (wt * pi).sum(); hb = (wt * h).sum(); cov = (wt * (h - hb) * (pi - pb)).sum()
        p1 = (wt * h * pi).sum() / pb; p0 = (wt * h * (1 - pi)).sum() / (1 - pb)
        eta_G = np.sqrt((wt * (pi - pb) ** 2).sum() / (pb * (1 - pb))); eta_D = np.sqrt((wt * (h - hb) ** 2).sum() / (hb * (1 - hb)))
        phi = cov / np.sqrt(hb * (1 - hb) * pb * (1 - pb)); m = aud.sum(); t = np.sqrt(m) * abs(phi)
        return dict(logOR=float(np.log((p1 / (1 - p1)) / (p0 / (1 - p0)))), eta_G=float(eta_G), eta_D=float(eta_D), phi=float(phi),
                    rate=float((norm.cdf(t - 1.96) + norm.cdf(-t - 1.96)) * 100))
    s = np.sqrt(t2 + v)
    P1 = predict(norm.cdf((q[0] - mu) / s) + 1 - norm.cdf((q[1] - mu) / s))
    # P2: mean trend of the attribute in log panel size (uses attribute and size only), tau^2 from the residuals
    Z = np.column_stack([np.ones(len(a)), ln, ln ** 2]); beta = np.linalg.lstsq(Z, a, rcond=None)[0]; mn = Z @ beta
    t2b, _ = _reml(a - mn, v); s2 = np.sqrt(t2b + v)
    P2 = predict(norm.cdf((q[0] - mn) / s2) + 1 - norm.cdf((q[1] - mn) / s2))
    return dict(tau2=t2, tau2_trend=t2b, sigma_w=sw, median_R=float(np.median(R)), share_R07=float((R >= .7).mean()), M=M_aud, lam=lam,
                eta_G=P1['eta_G'], eta_D=P1['eta_D'], phi=P1['phi'], pred_rate=P1['rate'], pred_logOR_P1=P1['logOR'], pred_logOR_P2=P2['logOR'],
                eta_G_trend=P2['eta_G'])

if __name__ == '__main__':
    t0 = time.time(); d = load_cohort(sys.argv[1]); y = leie_labels(d, sys.argv[2])
    print(f'{len(d):,} providers, {y.sum()} excluded on or after {CUT.date()}', flush=True)
    types = d.spec.value_counts().index[:20]
    X1 = pd.get_dummies(d.spec.where(d.spec.isin(types), 'other')).astype(float); X1['logn'] = d.logn.values
    X2 = pd.concat([X1, t2_features(d)], axis=1)
    f1, p1 = oof_flags(X1, y, d.spec.values); print(f'T1 done {time.time()-t0:.0f}s, recall {f1[y==1].mean():.3f}', flush=True)
    f2, p2 = oof_flags(X2, y, d.spec.values); print(f'T2 done {time.time()-t0:.0f}s, recall {f2[y==1].mean():.3f}', flush=True)
    pd.DataFrame(dict(NPI=d.NPI, spec=d.spec, n=d.n, y=y, flag_T1=f1, flag_T2=f2, score_T1=p1, score_T2=p2)).to_csv(OUT / 'flag_rate_flags.csv.gz', index=False)
    rows = []
    for s in top_specialties(d):
        m_ = (d.spec == s).values; g = d[m_]; aud = (y[m_] == 0).astype(float)
        for at in ATTRS:
            sw = sigma_w_mean_age(g) if at == 'mean age' else None
            a = g[at].values.astype(float)
            if sw is None:
                p = np.average(a, weights=g.n.values); sw = float(np.sqrt(p * (1 - p)))
            for det, fl in (('T1', f1), ('T2', f2)):
                r = predict_audit(g.n.values.astype(float), a, sw, aud, fl[m_])
                rows.append(dict(specialty=s, attribute=at, detector=det, m_audited=int(aud.sum()), flag_rate=float(fl[m_][aud > 0].mean()), **r))
        print(s, f'{time.time()-t0:.0f}s', flush=True)
    pr = pd.DataFrame(rows); pr['fails_stage0'] = pr.M > 0.10
    pr.to_csv(OUT / 'flag_rate_predictions.csv', index=False)
    print(pr.groupby(['attribute']).agg(M=('M', 'mean'), medR=('median_R', 'mean'), etaG=('eta_G', 'mean'), predOR1=('pred_logOR_P1', 'mean'), predOR2=('pred_logOR_P2', 'mean')).round(3))
