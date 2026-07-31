#!/usr/bin/env python3
"""
more_models_comparison.py: per follow-up request, fits several more model
families beyond the OLS/Gamma-GLM/mixed-model comparison in section 48, all
implemented from scratch (no scipy/statsmodels in this sandbox):

1. Log-normal model: OLS on log(yield), retransformed to the natural scale
   via Duan's smearing estimator (nonparametric, avoids the Jensen's-
   inequality bias of naively exponentiating a mean log-prediction).
2. Inverse Gaussian GLM, log link, via IRLS (V(mu)=mu^3, weight=1/mu; a
   standard alternative positive-continuous family to the Gamma GLM already
   tried in section 48).
3. Huber-loss robust regression (IRLS M-estimation), which downweights
   extreme points smoothly instead of deleting them -- a direct, principled
   alternative to the hard outlier removal used in sections 45-47. Run BOTH
   on the full n=167 sample (its natural use case: no deletion needed) and on
   the already-outlier-removed n=152 sample, for comparison.
4. Ridge regression over the full 75-column design (5 base recommended-model
   terms + all 70 engineered candidates from section 47's feature bank),
   letting L2 shrinkage handle the many correlated candidates instead of
   stepwise forward selection + FDR. Penalty strength chosen by GCV
   (generalized cross-validation, analytic, computed fresh on each fold's
   training data only, so there is no leakage into the CV estimate).

Per the standing instruction to work only with the outlier-removed data going
forward, models 1/2/4 use the n=152 both-tails-removed sample (section 46).
Model 3 (Huber) is deliberately run on the full n=167 sample too, since
avoiding the need to delete points is the whole point of a robust-regression
alternative; this deviation is intentional and reported as such, not an
oversight.

-> Results_Analysis/08_UEF53_Rerun/More_Models_Comparison.json
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict
import numpy as np
import openpyxl
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

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

no_out = json.load(open(os.path.join(OUT, "Four_Way_Comparison_No_Outliers_Both_Tails.json")))
excluded = set()
for entry in no_out["removed_trees_high"] + no_out["removed_trees_low"]:
    excluded.add((entry["tree_id"], entry["year"]))


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


def build_base(exclude_outliers):
    rows = []
    for t, j in ymap.items():
        for yr in (2022, 2023):
            if exclude_outliers and (t, yr) in excluded: continue
            if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
            if M[j].get(f"EBI_Norm_{yr}") is None: continue
            rows.append((t, j, yr, ymeas[t][yr]))
    n_ = len(rows)
    yv_ = np.array([r[3] for r in rows])
    uef_ = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rows])
    ebi_ = zc([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rows])
    CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
    clim_ = zc([CP[r[2]] for r in rows])
    ga_ = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rows])
    one_ = np.ones(n_)
    X_ = np.column_stack([one_, uef_, ebi_, ebi_ * uef_, clim_, ga_])
    return rows, X_, yv_, n_


rows152, X152, y152, n152 = build_base(True)
rows167, X167, y167, n167 = build_base(False)
print(f"n (outlier-removed, both tails) = {n152}")
print(f"n (full sample) = {n167}")


def r2_cv_generic(fit_fn, predict_fn, Xd, y, k=5, seed=42):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        model = fit_fn(Xd[tr], y[tr])
        pred[f] = predict_fn(model, Xd[f])
    ss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum()
    return round(float(1 - ss / tss), 4)


def robustness(fit_fn, predict_fn, Xd, y):
    seeds = [42, 1, 7, 13, 99, 2024]
    vals = {s: r2_cv_generic(fit_fn, predict_fn, Xd, y, seed=s) for s in seeds}
    return vals


SEEDS_ALL = [42, 1, 7, 13, 99, 2024]

# ================= reference: plain OLS on n=152 (from section 46/48) =================
def fit_ols(Xd, y):
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta


def pred_ols(beta, Xd):
    return Xd @ beta


ols_cv = robustness(fit_ols, pred_ols, X152, y152)
print(f"\n[0] OLS (reference, n=152): cvR2 seed42={ols_cv[42]}  range={min(ols_cv.values())}-{max(ols_cv.values())}")

# ================= 1. Log-normal + Duan smearing =================
def fit_lognormal(Xd, y):
    beta_log, *_ = np.linalg.lstsq(Xd, np.log(y), rcond=None)
    resid_log = np.log(y) - Xd @ beta_log
    smear = float(np.mean(np.exp(resid_log)))
    return (beta_log, smear)


def pred_lognormal(model, Xd):
    beta_log, smear = model
    return np.exp(Xd @ beta_log) * smear


lognormal_cv = robustness(fit_lognormal, pred_lognormal, X152, y152)
beta_ln, smear_ln = fit_lognormal(X152, y152)
in_sample_ln = pred_lognormal((beta_ln, smear_ln), X152)
r2_ln_in = 1 - ((y152 - in_sample_ln) ** 2).sum() / ((y152 - y152.mean()) ** 2).sum()
print(f"\n[1] Log-normal + Duan smearing (n=152): smear factor={smear_ln:.4f}  R2(in-sample)={r2_ln_in:.4f} "
      f"cvR2 seed42={lognormal_cv[42]}  range={min(lognormal_cv.values())}-{max(lognormal_cv.values())}")

# ================= 2. Inverse Gaussian GLM, log link, IRLS =================
def fit_invgauss_glm(Xd, y, max_iter=100, tol=1e-9):
    beta, *_ = np.linalg.lstsq(Xd, np.log(y), rcond=None)
    for _ in range(max_iter):
        eta = np.clip(Xd @ beta, -30, 30); mu = np.exp(eta)
        z = eta + (y - mu) / mu
        w = 1.0 / mu  # Inverse Gaussian: V(mu)=mu^3, log link weight = 1/(mu^3 * (1/mu)^2) = 1/mu
        W = np.diag(w)
        XtW = Xd.T @ W
        beta_new = np.linalg.solve(XtW @ Xd, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new
    return beta


def pred_glm(beta, Xd):
    return np.exp(np.clip(Xd @ beta, -30, 30))


invgauss_cv = robustness(fit_invgauss_glm, pred_glm, X152, y152)
beta_ig = fit_invgauss_glm(X152, y152)
mu_ig = pred_glm(beta_ig, X152)
r2_ig_in = 1 - ((y152 - mu_ig) ** 2).sum() / ((y152 - y152.mean()) ** 2).sum()
print(f"\n[2] Inverse Gaussian GLM, log link (n=152): R2(in-sample)={r2_ig_in:.4f} "
      f"cvR2 seed42={invgauss_cv[42]}  range={min(invgauss_cv.values())}-{max(invgauss_cv.values())}")

# ================= 3. Huber robust regression (IRLS M-estimation) =================
def fit_huber(Xd, y, k=1.345, max_iter=100, tol=1e-8):
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    for _ in range(max_iter):
        resid = y - Xd @ beta
        mad = np.median(np.abs(resid - np.median(resid))) / 0.6745
        if mad < 1e-8: mad = 1e-8
        u = resid / mad
        w = np.where(np.abs(u) <= k, 1.0, k / np.abs(u))
        W = np.diag(w)
        XtW = Xd.T @ W
        beta_new = np.linalg.solve(XtW @ Xd, XtW @ y)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new
    return beta


huber_cv_152 = robustness(fit_huber, pred_ols, X152, y152)
huber_cv_167 = robustness(fit_huber, pred_ols, X167, y167)
beta_huber_167 = fit_huber(X167, y167)
resid_huber_167 = y167 - X167 @ beta_huber_167
mad167 = np.median(np.abs(resid_huber_167 - np.median(resid_huber_167))) / 0.6745
final_weights_167 = np.where(np.abs(resid_huber_167 / mad167) <= 1.345, 1.0, 1.345 / np.abs(resid_huber_167 / mad167))
n_downweighted = int(np.sum(final_weights_167 < 0.999))
print(f"\n[3] Huber robust regression: n=152 cvR2 seed42={huber_cv_152[42]} range={min(huber_cv_152.values())}-{max(huber_cv_152.values())}")
print(f"    n=167 (no deletion) cvR2 seed42={huber_cv_167[42]} range={min(huber_cv_167.values())}-{max(huber_cv_167.values())}")
print(f"    on n=167, {n_downweighted} of 167 tree-years downweighted below full weight "
      f"(median downweighted-point weight={np.median(final_weights_167[final_weights_167 < 0.999]) if n_downweighted else 'n/a'})")

# ================= 4. Ridge over base + full 70-candidate engineered feature bank =================
def load_xlsx_by_id(path):
    wb = openpyxl.load_workbook(path, read_only=True); ws = wb.active
    rowsx = list(ws.iter_rows(values_only=True)); header = rowsx[0]; hidx = {h: i for i, h in enumerate(header)}
    return {r[hidx["Tree_ID"]]: {h: r[hidx[h]] for h in header} for r in rowsx[1:]}


canopy_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_CanopyStructure_v5.xlsx"))
bloomfrac_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_BloomFraction_v3.xlsx"))
v6_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_EBIPixelFeatures_v6.xlsx"))
canopy_for_master = [canopy_by_id.get(r.get("Tree_ID")) for r in M]
bloomfrac_for_master = [bloomfrac_by_id.get(r.get("Tree_ID")) for r in M]
v6_for_master = [v6_by_id.get(r.get("Tree_ID")) for r in M]

CANOPY_NAMES = ["CanopyArea", "CanopyCover", "ShadowFraction", "BloomFraction", "BrightFraction",
                "EBI_density", "BloomVolume", "BloomPixelVolume", "EBI_x_BloomFraction",
                "EBI_x_CanopyCover", "BrightFraction_x_CanopyArea"]
V6_NAMES = ["EBI_Norm_std", "EBI_Norm_var", "EBI_Norm_median", "EBI_Norm_P10", "EBI_Norm_P25",
            "EBI_Norm_P75", "EBI_Norm_P90", "EBI_Norm_IQR", "EBI_Norm_skew", "EBI_Norm_kurtosis",
            "EBI_Norm_CV", "EBI_OtsuFrac", "HighEBI_frac_055", "HighEBI_density_055",
            "HighEBI_frac_065", "HighEBI_density_065", "HighEBI_frac_075", "HighEBI_density_075"]


def build_all_features(j, yr):
    c = canopy_for_master[j]; b = bloomfrac_for_master[j]; v6 = v6_for_master[j]
    ebi = M[j].get(f"EBI_Norm_{yr}")
    if c is None or b is None or v6 is None or ebi is None: return None
    area = c.get(f"CanopyArea_m2_{yr}"); cover = c.get(f"CanopyCoverFraction_{yr}")
    shadow = c.get(f"ShadowFraction_{yr}"); pix = c.get(f"CanopyPixelCount_{yr}")
    bf = b.get(f"BloomFraction_{yr}"); brf = b.get(f"BrightFraction_{yr}")
    frac065 = v6.get(f"HighEBI_frac_065_{yr}"); ebi_std = v6.get(f"EBI_Norm_std_{yr}"); ebi_cv = v6.get(f"EBI_Norm_CV_{yr}")
    base_needed = [area, cover, shadow, pix, bf, brf, frac065, ebi_std, ebi_cv]
    v6_needed = [v6.get(f"{name}_{yr}") for name in V6_NAMES]
    if any(v is None for v in base_needed) or any(v is None for v in v6_needed): return None
    ebi = float(ebi); area = float(area); cover = float(cover); shadow = float(shadow); pix = float(pix)
    bf = float(bf); brf = float(brf); frac065 = float(frac065); ebi_std = float(ebi_std); ebi_cv = float(ebi_cv)
    out = {
        "CanopyArea": area, "CanopyCover": cover, "ShadowFraction": shadow, "BloomFraction": bf,
        "BrightFraction": brf, "EBI_density": ebi / area if area > 0 else np.nan, "BloomVolume": ebi * area,
        "BloomPixelVolume": bf * pix, "EBI_x_BloomFraction": ebi * bf, "EBI_x_CanopyCover": ebi * cover,
        "BrightFraction_x_CanopyArea": brf * area,
    }
    out.update({name: float(v6.get(f"{name}_{yr}")) for name in V6_NAMES})
    return out


rowsRidge = []
for t, j in ymap.items():
    for yr in (2022, 2023):
        if (t, yr) in excluded: continue
        if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
        if M[j].get(f"EBI_Norm_{yr}") is None: continue
        feats = build_all_features(j, yr)
        if feats is None: continue
        rowsRidge.append((t, j, yr, ymeas[t][yr], feats))
nR = len(rowsRidge)
print(f"\n[4] Ridge: n (with full 29-feature match, outlier-removed) = {nR}")

yR = np.array([r[3] for r in rowsRidge])
uefR = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rowsRidge])
ebiR = zc([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rowsRidge])
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
climR = zc([CP[r[2]] for r in rowsRidge])
gaR = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rowsRidge])
oneR = np.ones(nR)

feat_names = CANOPY_NAMES + V6_NAMES
feat_cols = [zc([r[4][name] for r in rowsRidge]) for name in feat_names]
XR_cols = [oneR, uefR, ebiR, ebiR * uefR, climR, gaR] + feat_cols
XR = np.column_stack(XR_cols)
col_names = ["intercept", "cultivar", "EBI", "EBIxcultivar", "climate", "Growth_April"] + feat_names
print(f"    Ridge design: {XR.shape[1]} columns (5 base terms + intercept + {len(feat_names)} engineered features)")


def ridge_fit_gcv(Xd, y, lambdas=None):
    if lambdas is None:
        lambdas = np.logspace(-2, 3, 40)
    n_, p_ = Xd.shape
    best_lam, best_gcv, best_beta = None, np.inf, None
    XtX = Xd.T @ Xd
    Xty = Xd.T @ y
    for lam in lambdas:
        P = np.eye(p_) * lam; P[0, 0] = 0.0  # don't penalize intercept
        A = XtX + P
        A_inv = np.linalg.pinv(A)
        beta = A_inv @ Xty
        H = Xd @ A_inv @ Xd.T
        tr_h = np.trace(H)
        rss = float(((y - Xd @ beta) ** 2).sum())
        denom = (1 - tr_h / n_) ** 2
        gcv = (rss / n_) / denom if denom > 1e-8 else np.inf
        if gcv < best_gcv:
            best_gcv, best_lam, best_beta = gcv, lam, beta
    return best_beta, best_lam, best_gcv


def fit_ridge(Xd, y):
    beta, lam, gcv = ridge_fit_gcv(Xd, y)
    return beta


ridge_cv = robustness(fit_ridge, pred_ols, XR, yR)
beta_ridge, lam_ridge, gcv_ridge = ridge_fit_gcv(XR, yR)
in_sample_ridge = XR @ beta_ridge
r2_ridge_in = 1 - ((yR - in_sample_ridge) ** 2).sum() / ((yR - yR.mean()) ** 2).sum()
top_ridge_coefs = sorted(zip(col_names, beta_ridge), key=lambda kv: -abs(kv[1]))[:10]
print(f"    chosen lambda (GCV, full n)={lam_ridge:.3f}  R2(in-sample)={r2_ridge_in:.4f} "
      f"cvR2 seed42={ridge_cv[42]}  range={min(ridge_cv.values())}-{max(ridge_cv.values())}")
print("    top 10 |coef| (standardized scale):")
for name, b in top_ridge_coefs:
    print(f"      {name}: {b:+.4f}")

# also fit a "base-terms-only" ridge as a fair apples-to-apples check against OLS
ridge_base_cv = robustness(fit_ridge, pred_ols, X152, y152)
print(f"\n    Ridge on JUST the 5 base recommended-model terms (n=152, no engineered features): "
      f"cvR2 seed42={ridge_base_cv[42]}  range={min(ridge_base_cv.values())}-{max(ridge_base_cv.values())}")

RESULTS = {
    "n152": n152, "n167": n167, "n_ridge_full_feature_match": nR,
    "ols_reference_n152": ols_cv,
    "lognormal_smearing_n152": {"smear_factor": round(smear_ln, 4), "R2_in_sample": round(float(r2_ln_in), 4), "cvR2": lognormal_cv},
    "invgauss_glm_loglink_n152": {"R2_in_sample": round(float(r2_ig_in), 4), "cvR2": invgauss_cv},
    "huber_robust_n152": huber_cv_152,
    "huber_robust_n167_no_deletion": {"cvR2": huber_cv_167, "n_downweighted_of_167": n_downweighted},
    "ridge_full_feature_bank": {
        "n_columns": XR.shape[1], "chosen_lambda_gcv": round(float(lam_ridge), 4),
        "R2_in_sample": round(float(r2_ridge_in), 4), "cvR2": ridge_cv,
        "top10_abs_coefficients": [[name, round(float(b), 4)] for name, b in top_ridge_coefs],
    },
    "ridge_base_terms_only_n152": ridge_base_cv,
}
with open(os.path.join(OUT, "More_Models_Comparison.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")
