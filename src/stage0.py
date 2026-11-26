"""Stage 0 and Algorithm 1 of the paper (Section III-F), importable and runnable on any table of units.

    from stage0 import stage0_reliability
    res = stage0_reliability(df, unit='TotalUniqueBene', attr='AvgPatientAge', partition='outer', sigma_w=12.7275,
                             eps=0.10, weights='Fraud', detection='det')
    print(res['summary'])

df         one row per unit (e.g. provider)
unit       column with the number of individuals aggregated in each unit (n_i)
attr       column with the unit's measured mean (or share) of the attribute
partition  'outer' (outer quartiles vs middle half), 'top' (top quartile) or 'median'
sigma_w    within-unit SD of the individual attribute. Required for a mean; for a share in [0, 1] it defaults to
           sqrt(p(1 - p)) with p the weighted mean over the units, which is the specialty-level value when df is one specialty
eps        tolerated expected misclassification M of the audited units
weights    column of audit weights (0/1 for 'unit is audited'); default: all units
detection  column with the audited outcome D (0/1) for the audited units; needed for eta_D, the predicted spurious-disparity rate
           and the decision (without it Stage 0a only is reported)
alpha_s, delta, B   tolerated spurious-disparity rate (7.5%), smallest odds ratio that matters (0.70), simulation draws (400)

Returns a dict whose 'summary' is one column of the Stage 0 and Algorithm 1 table of the paper: tau^2 (REML) with Cochran's Q and boundary likelihood-ratio tests, the
reliability R_i, the expected misclassification M by simulation (boundaries set by the measured distribution), eta_G, eta_D, the
predicted spurious-disparity rate r_hat (Corollary 1), the branch, the audit estimate with its companions, the minimum
detectable effect and the decision."""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats, optimize
from scipy.stats import norm

def _reml(a, v):
    def ll(t2):
        w = 1 / (t2 + v); mu = (w * a).sum() / w.sum()
        return -0.5 * (np.log(t2 + v).sum() + (w * (a - mu) ** 2).sum() + np.log(w.sum()))
    hi = max(np.var(a), 1e-9) * 2
    t2 = optimize.minimize_scalar(lambda x: -ll(x), bounds=(0, hi), method='bounded', options={'xatol': 1e-10}).x
    return (0.0 if ll(0) >= ll(t2) else t2), ll

def _group(x, kind):
    if kind == 'outer':
        q = np.quantile(x, [.25, .75]); return ((x <= q[0]) | (x > q[1])).astype(int)
    if kind == 'top':
        return (x >= np.quantile(x, .75)).astype(int)
    return (x > np.quantile(x, .5)).astype(int)

def expected_misclassification(n, mu, tau2, sigma_w, kind='outer', weights=None, B=200, seed=2026):
    """M = sum_i w_i Pr(measured group != true group): theta_i ~ N(mu, tau2), A_i = theta_i + N(0, sigma_w^2 / n_i), the boundaries
    of the measured and of the true partition are the quantiles of A and of theta over ALL units, as the audit draws them.
    Also returns the mean attenuation lambda = Pr(G*=1 | G=1) - Pr(G*=1 | G=0) of an effect planted on the true group."""
    rng = np.random.default_rng(seed); k = len(n); w = np.ones(k) / k if weights is None else np.asarray(weights, float) / np.sum(weights)
    mis = np.zeros(k); lam = []
    for _ in range(B):
        th = mu + np.sqrt(tau2) * rng.standard_normal(k); A = th + sigma_w / np.sqrt(n) * rng.standard_normal(k)
        gt, gm = _group(th, kind), _group(A, kind); mis += gt != gm
        a1 = (w * gm).sum(); a0 = (w * (1 - gm)).sum()
        if a1 > 0 and a0 > 0: lam.append((w * gm * gt).sum() / a1 - (w * (1 - gm) * gt).sum() / a0)
    mis /= B
    return float((w * mis).sum()), mis, float(np.mean(lam))

def _size_terms(ln, m, flexible_from=10000):
    """Panel-size terms of the adjusted audit: log(1 + n) for audits of fewer than `flexible_from` units, a natural cubic spline
    with 5 degrees of freedom in log(1 + n) for larger audits, where a linear term leaves residual structure."""
    if m < flexible_from: return [ln]
    from patsy import dmatrix
    B = np.asarray(dmatrix('cr(x, df=5) - 1', {'x': ln}))
    return [B[:, j] for j in range(1, B.shape[1])] if B.shape[1] > 1 else [ln]

def _logit(d, cols):
    X = sm.add_constant(np.column_stack(cols), has_constant='add')
    try:
        r = sm.Logit(d, X).fit(disp=0, maxiter=200); ci = r.conf_int()[1]
        return dict(OR=float(np.exp(r.params[1])), lo=float(np.exp(ci[0])), hi=float(np.exp(ci[1])), p=float(r.pvalues[1]), se=float(r.bse[1]))
    except Exception:
        return dict(OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan, se=np.nan)

def stage0_reliability(df, unit, attr, partition='outer', sigma_w=None, eps=0.10, weights=None, detection=None,
                       alpha_s=0.075, delta=0.70, B=400, seed=2026):
    n = df[unit].to_numpy(float); a = df[attr].to_numpy(float); k = len(a)
    wt_raw = np.ones(k) if weights is None else df[weights].to_numpy(float)
    aud = wt_raw > 0; wt = wt_raw / wt_raw.sum()
    if sigma_w is None:
        if a.min() < 0 or a.max() > 1:
            raise ValueError('sigma_w is required for a continuous attribute; it is derived only for shares in [0, 1]')
        p = np.average(a, weights=n); sigma_w = float(np.sqrt(p * (1 - p)))
    v = sigma_w ** 2 / n
    t2, ll = _reml(a, v)
    w = 1 / v; mu0 = (w * a).sum() / w.sum(); Q = (w * (a - mu0) ** 2).sum()
    p_Q = stats.chi2.sf(Q, k - 1); lr = 2 * (ll(t2) - ll(0)); p_LR = 0.5 * stats.chi2.sf(lr, 1) if lr > 0 else 1.0
    wm = 1 / (t2 + v); mu = (wm * a).sum() / wm.sum()
    R = t2 / (t2 + v) if t2 > 0 else np.zeros(k)
    out = dict(tau2=t2, Q=Q, p_Q=p_Q, p_LR=p_LR, R=R, median_R=float(np.median(R)), mu=mu, sigma_w=sigma_w, n_audited=int(aud.sum()))
    separable = (p_Q < 0.05) and (p_LR < 0.05) and t2 > 0
    out['separable'] = separable
    if not separable:
        out.update(branch='decline to partition', decision='decline to partition; report the detection gradient in panel size', M=np.nan)
        out['summary'] = (f'tau^2 = {t2:.4g}; Q p = {p_Q:.3g}; LR p = {p_LR:.3g}; median R = {out["median_R"]:.3f} -> {out["decision"]}')
        return out
    M, mis, lam = expected_misclassification(n, mu, t2, sigma_w, partition, wt, B, seed)
    out.update(M=M, lam=lam)
    # boundaries set by the measured distribution (all units); pi(n) from the Stage 0 estimates
    s = np.sqrt(t2 + v)
    if partition == 'outer':
        q = np.quantile(a, [.25, .75]); pi = norm.cdf((q[0] - mu) / s) + 1 - norm.cdf((q[1] - mu) / s)
    elif partition == 'top':
        pi = 1 - norm.cdf((np.quantile(a, .75) - mu) / s)
    else:
        pi = 1 - norm.cdf((np.quantile(a, .5) - mu) / s)
    pb = (wt * pi).sum(); eta_G = float(np.sqrt((wt * (pi - pb) ** 2).sum() / (pb * (1 - pb))))
    out.update(eta_G=eta_G, pbar=float(pb))
    if detection is None:
        out['summary'] = (f'tau^2 = {t2:.4g}; Q p = {p_Q:.3g}; median R = {out["median_R"]:.3f}; M = {M*100:.1f}%; eta_G = {eta_G:.3f} '
                          f'(no detection outcome supplied: Stage 0a only)')
        out.update(branch=None, decision=None); return out
    D = df[detection].to_numpy(float); ln = np.log1p(n); m = int(aud.sum())
    X = np.column_stack([ln[aud], ln[aud] ** 2]) if np.ptp(ln[aud]) > 0 else ln[aud][:, None]
    try:
        f = sm.Logit(D[aud], sm.add_constant(X)).fit(disp=0, maxiter=200)
        h = 1 / (1 + np.exp(-sm.add_constant(np.column_stack([ln, ln ** 2]) if X.shape[1] == 2 else ln[:, None], has_constant='add') @ f.params))
    except Exception:
        h = np.full(k, D[aud].mean())
    wa = wt.copy(); hb = (wa * h).sum(); vh = (wa * (h - hb) ** 2).sum(); vp = (wa * (pi - pb) ** 2).sum()
    cov = (wa * (h - hb) * (pi - pb)).sum()
    eta_D = float(np.sqrt(vh / (hb * (1 - hb)))); rho = float(cov / np.sqrt(vh * vp)) if vh * vp > 0 else 0.0
    phi = rho * eta_D * eta_G; z = norm.ppf(.975); tt = np.sqrt(m) * abs(phi); r_hat = float(norm.cdf(tt - z) + norm.cdf(-tt - z))
    p1 = (wa * h * pi).sum() / pb; p0 = (wa * h * (1 - pi)).sum() / (1 - pb)
    pred_or = float((p1 / (1 - p1)) / (p0 / (1 - p0)))
    out.update(eta_D=eta_D, rho=rho, phi=phi, r_hat=r_hat, predicted_OR=pred_or)
    # observed audit: unadjusted, size-adjusted, center split, shrunken (adjusted)
    G = _group(a, partition); Gc = _group(a, 'median'); Rr = R; shr = a.mean() + Rr * (a - a.mean()); Gs = _group(shr, partition)
    d_, l_ = D[aud], ln[aud]
    S_ = _size_terms(ln, m)
    S_a = [c[aud] for c in S_]
    unadj = _logit(d_, [G[aud]]); adj = _logit(d_, [G[aud]] + S_a); ctr = _logit(d_, [Gc[aud]]); shr_adj = _logit(d_, [Gs[aud]] + S_a)
    slope = _logit(d_, [(l_ - l_.mean()) / l_.std()])
    # minimum detectable effect of the size-adjusted audit: a planted log odds ratio beta on the true group appears on the measured
    # group attenuated by lam (Pr(G*=1|G=1) - Pr(G*=1|G=0)), so MDE_OR = exp(-2.80 SE / lam) (80% power, 5% two-sided)
    se = adj['se']; mde = float(np.exp(-(1.96 + 0.8416) * se / max(lam, 1e-9)))
    power_delta = float(norm.cdf(abs(np.log(delta)) * lam / se - 1.96) + norm.cdf(-abs(np.log(delta)) * lam / se - 1.96))
    out.update(unadjusted=unadj, adjusted=adj, center=ctr, shrunken_adjusted=shr_adj, size_slope=slope, MDE=mde, power_at_delta=power_delta)
    if M <= eps and r_hat <= alpha_s:
        branch = 'partition'; decision = 'attribute permitted: partition on the observed attribute'
    else:
        branch = 'adjust'
        decision = ('decline to attribute: the audit cannot detect a true odds ratio of %.2f with 80%% power; report the detection gradient in n_i' % delta
                    if mde < delta else 'attribute permitted: size-adjusted estimate reported with its companions')
    out.update(branch=branch, decision=decision)
    out['summary'] = (f'tau^2 = {t2:.4g}; Q p = {p_Q:.3g}; median R = {out["median_R"]:.3f}; M = {M*100:.1f}% (eps {eps*100:.0f}%); eta_G = {eta_G:.3f}; '
                      f'eta_D = {eta_D:.3f}; r_hat = {r_hat*100:.1f}% (alpha_s {alpha_s*100:.1f}%); branch = {branch}; '
                      f'audit OR (size-adjusted) = {adj["OR"]:.3f} [{adj["lo"]:.3f}, {adj["hi"]:.3f}]; unadjusted {unadj["OR"]:.3f}; center {ctr["OR"]:.3f}; '
                      f'Prop. 1 {pred_or:.3f}; shrunken(adj.) {shr_adj["OR"]:.3f}; power at OR {delta:.2f} = {power_delta*100:.1f}%, MDE = {mde:.3g} -> {decision}')
    return out

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_common import load
    from sklearn.preprocessing import StandardScaler
    d = load(); fm = d.Fraud.values == 1
    from audit_common import oof_lr, outer
    d['det'] = np.nan; d.loc[fm, 'det'] = oof_lr(d, outer(d.AvgPatientAge.values), strategy='class_weight')[fm]
    for part in ('outer', 'top'):
        r = stage0_reliability(d, 'TotalUniqueBene', 'AvgPatientAge', part, sigma_w=12.7275, eps=0.10, weights='Fraud', detection='det')
        print(part, r['summary'])
