"""
stats_utils.py — shared, from-scratch statistical primitives.
No internet access in the sandbox => scipy/statsmodels unavailable.
All p-values implemented via classical numerical-recipes algorithms and
validated against known textbook values where used.
"""
import math
import numpy as np

# ---------- regularized incomplete beta -> F and t p-values ----------
def betacf(a, b, x, maxit=300, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h

def betai(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    else:
        return 1.0 - bt * betacf(b, a, 1 - x) / b

def f_pvalue(f_stat, d1, d2):
    if f_stat <= 0: return 1.0
    x = d2 / (d2 + d1 * f_stat)
    return betai(d2 / 2.0, d1 / 2.0, x)

def t_pvalue(t_stat, df):
    x = df / (df + t_stat ** 2)
    return betai(df / 2.0, 0.5, x)

def r_pvalue(r, n):
    """two-sided p-value for Pearson r with sample size n."""
    if abs(r) >= 1: return 0.0
    t = r * math.sqrt((n - 2) / (1 - r ** 2))
    return t_pvalue(t, n - 2)

# ---------- regularized incomplete gamma -> chi-square p-value ----------
def gammainc_series(a, x):
    ap = a
    summ = 1.0 / a
    delta = summ
    for n in range(1, 500):
        ap += 1
        delta *= x / ap
        summ += delta
        if abs(delta) < abs(summ) * 3e-12:
            break
    return summ * math.exp(-x + a * math.log(x) - math.lgamma(a))

def gammainc_cf(a, x, maxit=300, eps=3e-12):
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, maxit + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30: d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h

def chi2_sf(x, df):
    """Upper-tail (survival) p-value of chi-square(df) at x = Q(df/2, x/2)."""
    a = df / 2.0
    xx = x / 2.0
    if xx < 0 or a <= 0:
        return 1.0
    if xx == 0:
        return 1.0
    if xx < a + 1.0:
        return 1.0 - gammainc_series(a, xx)
    else:
        return gammainc_cf(a, xx)

# ---------- PCA via SVD ----------
def pca(X, n_components=None):
    """X: (n_samples, n_features), standardized internally (z-score)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    n = X.shape[0]
    explained_var = (S ** 2) / (n - 1)
    explained_ratio = explained_var / explained_var.sum()
    scores = U * S
    if n_components is not None:
        scores = scores[:, :n_components]
        Vt = Vt[:n_components, :]
        explained_ratio = explained_ratio[:n_components]
    return scores, Vt, explained_ratio, mu, sd

# ---------- VIF ----------
def vif(X):
    """X: (n,p) design matrix of predictors (no intercept). Returns VIF per column."""
    n, p = X.shape
    vifs = []
    for j in range(p):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(n), others])
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vifs.append(1.0 / (1.0 - r2) if r2 < 0.9999 else np.inf)
    return vifs

# ---------- OLS via lstsq, returns RSS and df ----------
def ols_rss(X, y):
    """X includes intercept column. Returns residual sum of squares, n_params."""
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float((resid ** 2).sum()), X.shape[1], coef

# ---------- validation block (run once at import time if __main__) ----------
if __name__ == "__main__":
    print("F-dist checks (expect ~0.05):")
    for d1, d2, fcrit in [(1, 30, 4.17), (2, 30, 3.32), (5, 30, 2.53), (3, 100, 2.70)]:
        print(f"  F({d1},{d2})={fcrit} -> p={f_pvalue(fcrit, d1, d2):.4f}")
    print("Chi2 checks (expect ~0.05):")
    for df, xcrit in [(1, 3.841), (2, 5.991), (5, 11.07), (10, 18.31)]:
        print(f"  chi2(df={df})={xcrit} -> p={chi2_sf(xcrit, df):.4f}")

# ---------- generic mixed (split-plot) two-way ANOVA ----------
def split_plot_anova(long_df, subject_col, between_col, within_col, value_col):
    """
    Cultivar (between-subjects) x Time (within-subjects) mixed ANOVA.
    Returns dict with F/df/p/eta2 for between, within, and interaction effects.
    """
    df = long_df.dropna(subset=[value_col]).copy()
    GM = df[value_col].mean()
    subj_mean = df.groupby(subject_col)[value_col].mean()
    between_of_subj = df.drop_duplicates(subject_col).set_index(subject_col)[between_col]
    between_levels = sorted(df[between_col].unique())
    within_levels = sorted(df[within_col].unique())
    n_j = {b: (between_of_subj == b).sum() for b in between_levels}
    K = len(within_levels)
    N = len(subj_mean)

    cult_mean = df.groupby(between_col)[value_col].mean()
    time_mean = df.groupby(within_col)[value_col].mean()
    cellmean = df.groupby([between_col, within_col])[value_col].mean()

    SS_total = ((df[value_col] - GM) ** 2).sum()
    SS_between = sum(K * n_j[b] * (cult_mean[b] - GM) ** 2 for b in between_levels)
    df_between = len(between_levels) - 1

    tmp = subj_mean.to_frame('m').join(between_of_subj)
    tmp['cm'] = tmp[between_col].map(cult_mean)
    SS_suberr = (K * (tmp['m'] - tmp['cm']) ** 2).sum()
    df_suberr = N - len(between_levels)

    SS_within = sum(N * (time_mean[t] - GM) ** 2 for t in within_levels)
    df_within = K - 1

    SS_int = 0.0
    for b in between_levels:
        for t in within_levels:
            if (b, t) in cellmean.index:
                SS_int += n_j[b] * (cellmean[(b, t)] - cult_mean[b] - time_mean[t] + GM) ** 2
    df_int = (len(between_levels) - 1) * (K - 1)

    SS_errB = SS_total - SS_between - SS_suberr - SS_within - SS_int
    df_errB = (N - len(between_levels)) * (K - 1)

    F_b = (SS_between / df_between) / (SS_suberr / df_suberr) if df_suberr > 0 else np.nan
    p_b = f_pvalue(F_b, df_between, df_suberr) if df_suberr > 0 else np.nan
    F_w = (SS_within / df_within) / (SS_errB / df_errB) if df_errB > 0 else np.nan
    p_w = f_pvalue(F_w, df_within, df_errB) if df_errB > 0 else np.nan
    F_i = (SS_int / df_int) / (SS_errB / df_errB) if df_errB > 0 else np.nan
    p_i = f_pvalue(F_i, df_int, df_errB) if df_errB > 0 else np.nan

    return {
        'between': dict(F=F_b, df1=df_between, df2=df_suberr, p=p_b, eta2=SS_between / SS_total),
        'within': dict(F=F_w, df1=df_within, df2=df_errB, p=p_w, eta2=SS_within / SS_total),
        'interaction': dict(F=F_i, df1=df_int, df2=df_errB, p=p_i, eta2=SS_int / SS_total),
        'n_subjects': N, 'n_per_group': n_j,
    }

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"
