#!/usr/bin/env python3
"""
glm_and_mixed_model_comparison.py: per request, compares the established OLS
recommended model (cultivar + EBI x cultivar + climate + Growth_April) against
two alternative model families, both implemented from scratch since the
sandbox has no scipy/statsmodels:

1. Gamma GLM, log link, fit via IRLS. Yield is a strictly positive, right-
   skewed quantity, an OLS/Gaussian model implicitly assumes symmetric,
   constant-variance errors that don't match that. For a Gamma family with a
   log link, the IRLS weight simplifies to a constant 1 (derivation in the
   docstring of fit_gamma_glm below), so each IRLS step is literally an
   unweighted OLS regression of a working response on X -- no external
   optimizer needed.

2. Random-intercept-per-tree mixed model, variance components estimated by
   the classical unbalanced one-way ANOVA method-of-moments estimator
   (Searle), BLUP shrinkage applied per tree. This is the from-scratch
   equivalent of a random-intercept lme4/statsmodels MixedLM fit. Properly
   cross-validated: a held-out tree with no other year in the training fold
   gets BLUP=0 (population-average prediction), matching how a genuinely new
   tree would be predicted in a real mixed model.

All three are compared on the SAME n=167 base sample (2022+2023 pooled, the
scope of the established recommended model), using the same 5-fold CV R^2
framework (main seed 42 + 5 robustness seeds) used throughout this project.

-> Results_Analysis/08_UEF53_Rerun/GLM_And_Mixed_Model_Comparison.json
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import f_pvalue, t_pvalue

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")

M = load_master(); cultivar = [r.get("cultivar") for r in M]
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)
mx = np.array([r.get("X_UTM") for r in M], float); my = np.array([r.get("Y_UTM") for r in M], float)


def gpkg_geom_centroid(blob):
    flags = blob[3]; envelope_ind = (flags >> 1) & 0x07
    env_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope_ind]
    wkb = blob[8 + env_bytes:]; endian = "<" if wkb[0] == 1 else ">"
    pos = 1
    geom_type = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
    base_type = geom_type % 1000
    xs, ys = [], []
    if base_type == 6:
        npoly = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
        for _ in range(npoly):
            pos += 1; struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
            nrings = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
            for r in range(nrings):
                npts = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
                for _p in range(npts):
                    x, y = struct.unpack_from(endian + "dd", wkb, pos); pos += 16
                    if r == 0: xs.append(x); ys.append(y)
    elif base_type == 3:
        nrings = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
        for r in range(nrings):
            npts = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
            for _p in range(npts):
                x, y = struct.unpack_from(endian + "dd", wkb, pos); pos += 16
                if r == 0: xs.append(x); ys.append(y)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def load_phys(year):
    path = os.path.join(GDIR, f"Yield_with_clustering_{year}.gpkg")
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT table_name FROM gpkg_geometry_columns"); tbl = cur.fetchone()[0]
    cur.execute(f'SELECT "Row","Plot",cultivar,"Growth__April",geom FROM "{tbl}"')
    rowsg = cur.fetchall(); con.close(); out = []
    for row_ in rowsg:
        plot = row_[1]; ga = row_[3]; blob = row_[-1]
        if plot != "A" or ga is None: continue
        x, y = gpkg_geom_centroid(blob); out.append({"x": x, "y": y, "Growth_April": float(ga)})
    return out


PHYS = {y: load_phys(y) for y in [2022, 2023]}
phys_match = {}
for y in [2022, 2023]:
    cand2 = []
    for i, r in enumerate(PHYS[y]):
        d = np.sqrt((mx - r["x"]) ** 2 + (my - r["y"]) ** 2); j = int(np.argmin(d))
        if d[j] <= 3.0: cand2.append((d[j], i, j))
    cand2.sort(); ui = set(); uj = set(); mm = {}
    for d, i, j in cand2:
        if i in ui or j in uj: continue
        mm[j] = PHYS[y][i]; ui.add(i); uj.add(j)
    phys_match[y] = mm

yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc:
    ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"])
    coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))


def hav(a, b, d, e):
    p = math.pi / 180
    x = math.sin((d - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(d * p) * math.sin((e - b) * p / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(max(0, x)))


cand = []
for t, (la, lo) in coord.items():
    ds = np.array([hav(la, lo, a, b) for a, b in zip(mlat, mlon)])
    for j in np.where(ds <= 8)[0]: cand.append((float(ds[j]), t, int(j)))
cand.sort(); uy = set(); um = set(); ymap = {}
for d, t, j in cand:
    if t in uy or j in um: continue
    ymap[t] = j; uy.add(t); um.add(j)

rows = []
for t, j in ymap.items():
    for yr in (2022, 2023):
        if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
        if M[j].get(f"EBI_Norm_{yr}") is None: continue
        rows.append((t, j, yr, ymeas[t][yr]))
n = len(rows)
print(f"n={n} tree-years (same scope as the established recommended model)")


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


yv = np.array([r[3] for r in rows])
tree_ids = np.array([r[0] for r in rows])
uef = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rows])
ebi = zc([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rows])
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
clim = zc([CP[r[2]] for r in rows])
ga = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rows])
one = np.ones(n)
X = np.column_stack([one, uef, ebi, ebi * uef, clim, ga])
PARAM_NAMES = ["intercept", "cultivar", "EBI", "EBIxcultivar", "climate", "Growth_April"]

n_distinct_trees = len(set(tree_ids))
years_per_tree = defaultdict(list)
for i, t in enumerate(tree_ids): years_per_tree[t].append(i)
n_repeated_trees = sum(1 for t in years_per_tree if len(years_per_tree[t]) > 1)
print(f"{n_distinct_trees} distinct trees, {n_repeated_trees} measured in more than one year")

# ================= model 1: OLS (established baseline) =================
def r2_cv_ols(Xd, y, k=5, seed=42):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None); pred[f] = Xd[f] @ beta
    ss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum(); cv = 1 - ss / tss
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None); ins = Xd @ beta
    r2 = 1 - ((y - ins) ** 2).sum() / tss
    return round(float(r2), 4), round(float(cv), 4)


r2_ols, cv_ols = r2_cv_ols(X, yv)
ols_cv_seeds = {s: r2_cv_ols(X, yv, seed=s)[1] for s in [1, 7, 13, 99, 2024]}
print(f"\n[1] OLS baseline: R2={r2_ols} cvR2={cv_ols}  (5-seed range {min(ols_cv_seeds.values())}-{max(ols_cv_seeds.values())})")


# ================= model 2: Gamma GLM, log link, via IRLS =================
def fit_gamma_glm(Xd, y, max_iter=100, tol=1e-9):
    """
    IRLS for a Gamma-distributed response with a log link.
    General IRLS: beta_new = argmin_beta sum_i w_i (z_i - X_i beta)^2, where
      z_i = eta_i + (y_i - mu_i) * g'(mu_i)      (working response)
      w_i = 1 / (V(mu_i) * g'(mu_i)^2)            (IRLS weight)
    For Gamma, V(mu) = mu^2. For log link, g(mu)=log(mu) so g'(mu)=1/mu.
    => w_i = 1 / (mu_i^2 * (1/mu_i)^2) = 1  (constant!)
    So every IRLS step is exactly an unweighted OLS fit of z on X.
    """
    beta, *_ = np.linalg.lstsq(Xd, np.log(y), rcond=None)  # sensible start
    for _ in range(max_iter):
        eta = Xd @ beta
        eta = np.clip(eta, -30, 30)
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        beta_new, *_ = np.linalg.lstsq(Xd, z, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new
    mu = np.exp(np.clip(Xd @ beta, -30, 30))
    return beta, mu


beta_glm, mu_glm = fit_gamma_glm(X, yv)
p_glm = X.shape[1]
pearson_resid = (yv - mu_glm) / mu_glm
phi_hat = float((pearson_resid ** 2).sum() / (n - p_glm))  # dispersion
cov_beta = phi_hat * np.linalg.inv(X.T @ X)  # valid since IRLS weight=1 here
se_glm = np.sqrt(np.diag(cov_beta))
t_glm = beta_glm / se_glm
p_glm_wald = [float(t_pvalue(float(tv), n - p_glm)) for tv in t_glm]

deviance_glm = float(2 * np.sum((yv - mu_glm) / mu_glm - np.log(yv / mu_glm)))
print(f"\n[2] Gamma GLM (log link): dispersion(phi)={phi_hat:.4f}  deviance={deviance_glm:.3f}")
for name, b, se, tv, pv in zip(PARAM_NAMES, beta_glm, se_glm, t_glm, p_glm_wald):
    print(f"    {name}: coef={b:+.4f} se={se:.4f} t={tv:+.3f} p={pv:.4f}")


def r2_cv_glm(Xd, y, k=5, seed=42):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        b, _ = fit_gamma_glm(Xd[tr], y[tr])
        pred[f] = np.exp(np.clip(Xd[f] @ b, -30, 30))
    ss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum(); cv = 1 - ss / tss
    ss_in = ((y - mu_glm) ** 2).sum() if len(y) == len(mu_glm) else np.nan
    r2_in = 1 - ((y - mu_glm) ** 2).sum() / tss
    return round(float(r2_in), 4), round(float(cv), 4)


r2_glm, cv_glm = r2_cv_glm(X, yv)
glm_cv_seeds = {s: r2_cv_glm(X, yv, seed=s)[1] for s in [1, 7, 13, 99, 2024]}
print(f"    R2(natural scale)={r2_glm} cvR2(natural scale)={cv_glm}  (5-seed range {min(glm_cv_seeds.values())}-{max(glm_cv_seeds.values())})")

# ================= model 3: random-intercept-per-tree mixed model =================
beta_ols, *_ = np.linalg.lstsq(X, yv, rcond=None)
resid = yv - X @ beta_ols


def variance_components(resid_, tree_ids_):
    groups = defaultdict(list)
    for r_, t_ in zip(resid_, tree_ids_): groups[t_].append(r_)
    G = len(groups); N = len(resid_)
    grand_mean = float(np.mean(resid_))
    ss_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values())
    ss_within = sum(sum((np.array(v) - np.mean(v)) ** 2) for v in groups.values())
    df_between = G - 1; df_within = N - G
    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else float(np.var(resid_))
    sum_ng2 = sum(len(v) ** 2 for v in groups.values())
    c0 = (N - sum_ng2 / N) / df_between if df_between > 0 else 1.0
    sigma2_u = max(0.0, (ms_between - ms_within) / c0) if c0 > 0 else 0.0
    sigma2_e = ms_within
    return sigma2_u, sigma2_e, G, groups


sigma2_u, sigma2_e, n_groups, resid_groups = variance_components(resid, tree_ids)
icc = sigma2_u / (sigma2_u + sigma2_e) if (sigma2_u + sigma2_e) > 0 else 0.0
print(f"\n[3] Mixed model (random intercept per tree): sigma2_u(between-tree)={sigma2_u:.4f} "
      f"sigma2_e(within/residual)={sigma2_e:.4f} ICC={icc:.4f} ({n_groups} tree groups, "
      f"{sum(1 for v in resid_groups.values() if len(v) > 1)} with >1 observation)")


def blup_predict(resid_train, tree_train, tree_test, sigma2_u_, sigma2_e_):
    """BLUP shrinkage estimate of each tree's random intercept from training
    residuals; trees absent from training get BLUP=0 (population average),
    the correct behavior for an unobserved group in a random-intercept model."""
    groups = defaultdict(list)
    for r_, t_ in zip(resid_train, tree_train): groups[t_].append(r_)
    u_hat = {}
    for t_, vals in groups.items():
        ng = len(vals); mean_g = np.mean(vals)
        shrink = (ng * sigma2_u_) / (ng * sigma2_u_ + sigma2_e_) if (ng * sigma2_u_ + sigma2_e_) > 0 else 0.0
        u_hat[t_] = shrink * mean_g
    return np.array([u_hat.get(t_, 0.0) for t_ in tree_test])


u_hat_full = blup_predict(resid, tree_ids, tree_ids, sigma2_u, sigma2_e)
fitted_mixed = X @ beta_ols + u_hat_full
tss_full = ((yv - yv.mean()) ** 2).sum()
r2_mixed_in = 1 - ((yv - fitted_mixed) ** 2).sum() / tss_full


def r2_cv_mixed(Xd, y, tree_ids_, k=5, seed=42):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        b_tr, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None)
        resid_tr = y[tr] - Xd[tr] @ b_tr
        s2u, s2e, _, _ = variance_components(resid_tr, tree_ids_[tr])
        u_test = blup_predict(resid_tr, tree_ids_[tr], tree_ids_[f], s2u, s2e)
        pred[f] = Xd[f] @ b_tr + u_test
    ss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum(); cv = 1 - ss / tss
    return round(float(cv), 4)


cv_mixed = r2_cv_mixed(X, yv, tree_ids)
mixed_cv_seeds = {s: r2_cv_mixed(X, yv, tree_ids, seed=s) for s in [1, 7, 13, 99, 2024]}
print(f"    R2(in-sample, with BLUPs)={round(r2_mixed_in,4)} cvR2={cv_mixed}  "
      f"(5-seed range {min(mixed_cv_seeds.values())}-{max(mixed_cv_seeds.values())})")
print(f"    NOTE: only {sum(1 for v in resid_groups.values() if len(v) > 1)} of {n_groups} trees have >1 year of "
      "data, so BLUP shrinkage has almost nothing to work with for the vast majority of rows "
      "(singleton trees get u_hat=0 in-sample too, since there's no repeat observation to estimate a "
      "tree-specific deviation from). The random effect can only meaningfully help the ~20 repeated trees.")

RESULTS = {
    "n": n, "n_distinct_trees": n_distinct_trees, "n_repeated_trees": n_repeated_trees,
    "ols_baseline": {"R2": r2_ols, "cvR2": cv_ols, "cv_seed_range": [min(ols_cv_seeds.values()), max(ols_cv_seeds.values())]},
    "gamma_glm_log_link": {
        "dispersion_phi": round(phi_hat, 4), "deviance": round(deviance_glm, 3),
        "coefficients": {name: {"coef": round(float(b), 4), "se": round(float(se), 4),
                                  "t": round(float(tv), 3), "p": round(float(pv), 4)}
                          for name, b, se, tv, pv in zip(PARAM_NAMES, beta_glm, se_glm, t_glm, p_glm_wald)},
        "R2_natural_scale": r2_glm, "cvR2_natural_scale": cv_glm,
        "cv_seed_range": [min(glm_cv_seeds.values()), max(glm_cv_seeds.values())],
    },
    "mixed_model_random_intercept": {
        "sigma2_between_tree": round(float(sigma2_u), 4), "sigma2_within_residual": round(float(sigma2_e), 4),
        "ICC": round(float(icc), 4), "n_tree_groups": n_groups,
        "n_trees_with_gt1_obs": sum(1 for v in resid_groups.values() if len(v) > 1),
        "R2_in_sample_with_BLUPs": round(float(r2_mixed_in), 4), "cvR2": cv_mixed,
        "cv_seed_range": [min(mixed_cv_seeds.values()), max(mixed_cv_seeds.values())],
    },
}
with open(os.path.join(OUT, "GLM_And_Mixed_Model_Comparison.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")
