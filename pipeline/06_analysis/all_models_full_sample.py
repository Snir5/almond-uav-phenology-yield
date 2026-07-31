#!/usr/bin/env python3
"""
all_models_full_sample.py: per follow-up request ("add the outliers back and
retry all the models"), re-runs every alternative model family from sections
48-52 on the FULL n=167 sample (actual-yield anomalies included), for direct
comparison against the outlier-removed (n=152) results already logged.

Gamma GLM and the random-intercept mixed model (section 48) were already run
on the full n=167 sample originally (that section predates the "outliers
removed" standing instruction) -- not repeated here. Huber robust regression
was already run on both n=152 and n=167 (section 49) -- not repeated here.

This script covers the remaining six: log-normal + Duan smearing, Inverse
Gaussian GLM (log link), Ridge over the full 75-column engineered bank,
Lasso, Elastic Net, k-NN regression, Random Forest, AND the AIC/MSE/MAE
add-one/remove-one test across all 70 candidates -- all on n=167.

Selectable via argv: `python3 all_models_full_sample.py <stage> [seeds]`
  stage in {glm, lasso, enet, nonlinear, aic}. Split into stages because the
  full run exceeds this sandbox's per-call execution budget.

-> Results_Analysis/08_UEF53_Rerun/All_Models_Full_Sample.json (merged across stages)
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

STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
SEEDS_ALL = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [42, 1, 7, 13, 99, 2024]

M = load_master(); cultivar = [r.get("cultivar") for r in M]
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)
mx = np.array([r.get("X_UTM") for r in M], float); my = np.array([r.get("Y_UTM") for r in M], float)


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
    canopy_feats = {
        "CanopyArea": area, "CanopyCover": cover, "ShadowFraction": shadow, "BloomFraction": bf,
        "BrightFraction": brf, "EBI_density": ebi / area if area > 0 else np.nan, "BloomVolume": ebi * area,
        "BloomPixelVolume": bf * pix, "EBI_x_BloomFraction": ebi * bf, "EBI_x_CanopyCover": ebi * cover,
        "BrightFraction_x_CanopyArea": brf * area,
    }
    v6_feats = {name: float(v6.get(f"{name}_{yr}")) for name in V6_NAMES}
    composite_feats = {
        "TBL": area * frac065, "ICBI": ebi * (1 - shadow), "TICBL": area * ebi * (1 - shadow),
        "UBI": ebi * (1 - ebi_cv), "CCAB": frac065 * cover,
    }
    out = {}
    out.update(canopy_feats); out.update(v6_feats); out.update(composite_feats)
    out["_raw_area"] = area; out["_raw_frac065"] = frac065; out["_raw_ebi_std"] = ebi_std; out["_raw_shadow"] = shadow
    return out


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

# NOTE: no outlier exclusion in this script -- full n=167 sample, per request.


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


rows = []
for t, j in ymap.items():
    for yr in (2022, 2023):
        if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
        if M[j].get(f"EBI_Norm_{yr}") is None: continue
        rows.append((t, j, yr, ymeas[t][yr]))
n = len(rows)

yv = np.array([r[3] for r in rows])
uef = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rows])
ebi = zc([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rows])
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
clim = zc([CP[r[2]] for r in rows])
ga = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rows])
one = np.ones(n)
X_ols = np.column_stack([one, uef, ebi, ebi * uef, clim, ga])
X_raw = np.column_stack([uef, ebi, clim, ga])
print(f"n={n} (FULL sample, outliers included)")


def r2_mse_mae(y_true, y_pred):
    resid = y_true - y_pred
    tss = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / tss), float(np.mean(resid ** 2)), float(np.mean(np.abs(resid)))


def cv_ols(Xd, y, seed, k=5):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None); pred[f] = Xd[f] @ beta
    return r2_mse_mae(y, pred)


def load_prior():
    path = os.path.join(OUT, "All_Models_Full_Sample.json")
    return json.load(open(path)) if os.path.exists(path) else {"n": n, "seeds": []}


def save_merged(new_data):
    prior = load_prior()
    for k_, v_ in new_data.items():
        if isinstance(v_, dict) and isinstance(prior.get(k_), dict):
            prior[k_] = {**prior[k_], **v_}
        else:
            prior[k_] = v_
    prior["n"] = n
    prior["seeds"] = sorted(set(prior.get("seeds", [])) | set(SEEDS_ALL))
    with open(os.path.join(OUT, "All_Models_Full_Sample.json"), "w") as f:
        json.dump(prior, f, indent=2, default=str)


ols_ref = {s: dict(zip(["cvR2", "cvMSE", "cvMAE"], cv_ols(X_ols, yv, s))) for s in SEEDS_ALL}
for s in ols_ref: ols_ref[s] = {k_: round(v_, 5) for k_, v_ in ols_ref[s].items()}
print("OLS reference (with interaction):", ols_ref)

# =====================================================================
if STAGE in ("glm", "all"):
    def fit_lognormal(Xd, y):
        beta_log, *_ = np.linalg.lstsq(Xd, np.log(y), rcond=None)
        resid_log = np.log(y) - Xd @ beta_log
        return beta_log, float(np.mean(np.exp(resid_log)))

    def pred_lognormal(model, Xd):
        beta_log, smear = model
        return np.exp(Xd @ beta_log) * smear

    def fit_invgauss_glm(Xd, y, max_iter=100, tol=1e-9):
        beta, *_ = np.linalg.lstsq(Xd, np.log(y), rcond=None)
        for _ in range(max_iter):
            eta = np.clip(Xd @ beta, -30, 30); mu = np.exp(eta)
            z = eta + (y - mu) / mu
            w = 1.0 / mu
            W = np.diag(w); XtW = Xd.T @ W
            beta_new = np.linalg.solve(XtW @ Xd, XtW @ z)
            if np.max(np.abs(beta_new - beta)) < tol: beta = beta_new; break
            beta = beta_new
        return beta

    def pred_glm(beta, Xd):
        return np.exp(np.clip(Xd @ beta, -30, 30))

    def cv_generic(fit_fn, pred_fn, Xd, y, seed, k=5):
        n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
        pred = np.full(n_, np.nan)
        for f in folds:
            tr = np.setdiff1d(np.arange(n_), f)
            model = fit_fn(Xd[tr], y[tr]); pred[f] = pred_fn(model, Xd[f])
        return r2_mse_mae(y, pred)

    lognormal_by_seed = {s: dict(zip(["cvR2", "cvMSE", "cvMAE"], cv_generic(fit_lognormal, pred_lognormal, X_ols, yv, s))) for s in SEEDS_ALL}
    invgauss_by_seed = {s: dict(zip(["cvR2", "cvMSE", "cvMAE"], cv_generic(fit_invgauss_glm, pred_glm, X_ols, yv, s))) for s in SEEDS_ALL}
    for d in (lognormal_by_seed, invgauss_by_seed):
        for s in d: d[s] = {k_: round(v_, 5) for k_, v_ in d[s].items()}
    print("Log-normal:", lognormal_by_seed)
    print("Inverse Gaussian GLM:", invgauss_by_seed)

    # ---- Ridge over full 75-column engineered bank, full sample ----
    rowsRidge = []
    for t, j in ymap.items():
        for yr in (2022, 2023):
            if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
            if M[j].get(f"EBI_Norm_{yr}") is None: continue
            feats = build_all_features(j, yr)
            if feats is None: continue
            rowsRidge.append((t, j, yr, ymeas[t][yr], feats))
    nR = len(rowsRidge)
    yR = np.array([r[3] for r in rowsRidge])
    uefR = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rowsRidge])
    ebiR = zc([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rowsRidge])
    climR = zc([CP[r[2]] for r in rowsRidge])
    gaR = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rowsRidge])
    oneR = np.ones(nR)
    feat_names = CANOPY_NAMES + V6_NAMES
    feat_cols = [zc([r[4][name] for r in rowsRidge]) for name in feat_names]
    XR = np.column_stack([oneR, uefR, ebiR, ebiR * uefR, climR, gaR] + feat_cols)
    print(f"Ridge design (full sample): n={nR}, {XR.shape[1]} columns")

    def ridge_fit_gcv(Xd, y, lambdas=None):
        if lambdas is None: lambdas = np.logspace(-2, 3, 40)
        n_, p_ = Xd.shape
        best_lam, best_gcv, best_beta = None, np.inf, None
        XtX = Xd.T @ Xd; Xty = Xd.T @ y
        for lam in lambdas:
            P = np.eye(p_) * lam; P[0, 0] = 0.0
            A_inv = np.linalg.pinv(XtX + P)
            beta = A_inv @ Xty
            H = Xd @ A_inv @ Xd.T
            rss = float(((y - Xd @ beta) ** 2).sum())
            denom = (1 - np.trace(H) / n_) ** 2
            gcv = (rss / n_) / denom if denom > 1e-8 else np.inf
            if gcv < best_gcv: best_gcv, best_lam, best_beta = gcv, lam, beta
        return best_beta, best_lam

    def fit_ridge(Xd, y):
        beta, lam = ridge_fit_gcv(Xd, y); return beta

    def pred_ols_beta(beta, Xd): return Xd @ beta

    ridge_by_seed = {s: dict(zip(["cvR2", "cvMSE", "cvMAE"], cv_generic(fit_ridge, pred_ols_beta, XR, yR, s))) for s in SEEDS_ALL}
    for s in ridge_by_seed: ridge_by_seed[s] = {k_: round(v_, 5) for k_, v_ in ridge_by_seed[s].items()}
    print("Ridge (full 75-col bank, full sample):", ridge_by_seed)

    save_merged({
        "ols_reference": ols_ref, "lognormal": lognormal_by_seed, "invgauss_glm": invgauss_by_seed,
        "ridge_full_bank": ridge_by_seed, "ridge_n": nR,
    })

# =====================================================================
if STAGE in ("lasso", "all"):
    rowsRidge = []
    for t, j in ymap.items():
        for yr in (2022, 2023):
            if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
            if M[j].get(f"EBI_Norm_{yr}") is None: continue
            feats = build_all_features(j, yr)
            if feats is None: continue
            rowsRidge.append((t, j, yr, ymeas[t][yr], feats))
    nR = len(rowsRidge)
    yR = np.array([r[3] for r in rowsRidge])
    uefR = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rowsRidge])
    ebiR = zc([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rowsRidge])
    climR = zc([CP[r[2]] for r in rowsRidge])
    gaR = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rowsRidge])
    feat_names = CANOPY_NAMES + V6_NAMES
    feat_cols = [zc([r[4][name] for r in rowsRidge]) for name in feat_names]
    col_names = ["cultivar", "EBI", "EBIxcultivar", "climate", "Growth_April"] + feat_names
    Xall = np.column_stack([uefR, ebiR, ebiR * uefR, climR, gaR] + feat_cols)

    def soft_threshold(x, t): return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

    def fit_glmnet_cd(Xd, y, lam, alpha, max_iter=300, tol=2e-6):
        n_, p_ = Xd.shape
        y_mean = y.mean(); yc_ = y - y_mean
        beta = np.zeros(p_); col_ss = (Xd ** 2).sum(axis=0) / n_
        for it in range(max_iter):
            beta_old = beta.copy(); r = yc_ - Xd @ beta
            for j_ in range(p_):
                r_j = r + Xd[:, j_] * beta[j_]
                rho = (Xd[:, j_] @ r_j) / n_
                denom = col_ss[j_] + lam * (1 - alpha)
                beta[j_] = soft_threshold(rho, lam * alpha) / denom if denom > 1e-12 else 0.0
                r = r_j - Xd[:, j_] * beta[j_]
            if np.max(np.abs(beta - beta_old)) < tol: break
        return y_mean, beta

    def predict_glmnet(model, Xd):
        intercept, beta = model; return intercept + Xd @ beta

    LAMBDA_GRID = np.logspace(-3, 1, 12)

    def inner_cv_select_lambda(Xd, y, alpha, k=3, seed=0):
        n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
        best_lam, best_mse = None, np.inf
        for lam in LAMBDA_GRID:
            errs = []
            for f in folds:
                tr = np.setdiff1d(np.arange(n_), f)
                model = fit_glmnet_cd(Xd[tr], y[tr], lam, alpha)
                pred = predict_glmnet(model, Xd[f]); errs.append(np.mean((y[f] - pred) ** 2))
            mse = np.mean(errs)
            if mse < best_mse: best_mse, best_lam = mse, lam
        return best_lam

    def nested_cv_glmnet(Xd, y, alpha, k=5, seed=42):
        n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
        pred_all = np.full(n_, np.nan)
        for f in folds:
            tr = np.setdiff1d(np.arange(n_), f)
            lam = inner_cv_select_lambda(Xd[tr], y[tr], alpha, seed=seed)
            model = fit_glmnet_cd(Xd[tr], y[tr], lam, alpha)
            pred_all[f] = predict_glmnet(model, Xd[f])
        return r2_mse_mae(y, pred_all)

    lasso_by_seed = {}
    enet_by_seed = {}
    for s in SEEDS_ALL:
        r2, mse, mae = nested_cv_glmnet(Xall, yR, 1.0, seed=s)
        lasso_by_seed[s] = {"cvR2": round(r2, 4), "cvMSE": round(mse, 5), "cvMAE": round(mae, 5)}
        r2, mse, mae = nested_cv_glmnet(Xall, yR, 0.5, seed=s)
        enet_by_seed[s] = {"cvR2": round(r2, 4), "cvMSE": round(mse, 5), "cvMAE": round(mae, 5)}
        print(f"seed {s}: lasso={lasso_by_seed[s]} enet={enet_by_seed[s]}")

    lam_full_lasso = inner_cv_select_lambda(Xall, yR, 1.0, seed=42)
    _, beta_lasso_full = fit_glmnet_cd(Xall, yR, lam_full_lasso, 1.0)
    nonzero_lasso = sorted([(col_names[i], round(float(beta_lasso_full[i]), 4)) for i in range(len(col_names)) if abs(beta_lasso_full[i]) > 1e-6], key=lambda kv: -abs(kv[1]))
    lam_full_enet = inner_cv_select_lambda(Xall, yR, 0.5, seed=42)
    _, beta_enet_full = fit_glmnet_cd(Xall, yR, lam_full_enet, 0.5)
    nonzero_enet = sorted([(col_names[i], round(float(beta_enet_full[i]), 4)) for i in range(len(col_names)) if abs(beta_enet_full[i]) > 1e-6], key=lambda kv: -abs(kv[1]))
    print("Lasso nonzero (full data):", nonzero_lasso)
    print("Elastic Net nonzero (full data):", nonzero_enet)

    save_merged({
        "lasso": lasso_by_seed, "elastic_net": enet_by_seed,
        "lasso_nonzero_full_data": nonzero_lasso, "elastic_net_nonzero_full_data": nonzero_enet,
    })

# =====================================================================
if STAGE in ("nonlinear", "all"):
    def knn_predict(Xtr, ytr, Xte, k):
        preds = np.zeros(len(Xte))
        for i in range(len(Xte)):
            d = np.sqrt(((Xtr - Xte[i]) ** 2).sum(axis=1))
            nn = np.argsort(d)[:k]; preds[i] = ytr[nn].mean()
        return preds

    def cv_knn(Xd, y, seed, k_neighbors, k_folds=5):
        n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k_folds)
        pred = np.full(n_, np.nan)
        for f in folds:
            tr = np.setdiff1d(np.arange(n_), f)
            pred[f] = knn_predict(Xd[tr], y[tr], Xd[f], k_neighbors)
        return r2_mse_mae(y, pred)

    def select_k_inner(Xd, y, seed):
        best_k, best_mse = None, np.inf
        for k_ in [3, 5, 7, 9, 11, 15, 20, 25]:
            _, mse, _ = cv_knn(Xd, y, seed, k_, k_folds=3)
            if mse < best_mse: best_mse, best_k = mse, k_
        return best_k

    def build_tree(X, y, depth, max_depth, min_leaf, rng, mtry):
        n_, p_ = X.shape
        if depth >= max_depth or n_ < 2 * min_leaf:
            return {"leaf": True, "value": float(y.mean())}
        feat_idx = rng.choice(p_, size=min(mtry, p_), replace=False)
        best_feat, best_thr, best_score = None, None, np.inf
        parent_rss = float(((y - y.mean()) ** 2).sum())
        for fj in feat_idx:
            vals = np.unique(X[:, fj])
            if len(vals) < 2: continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            for thr in thresholds:
                left = X[:, fj] <= thr; right = ~left
                nl, nr = left.sum(), right.sum()
                if nl < min_leaf or nr < min_leaf: continue
                rss = float(((y[left] - y[left].mean()) ** 2).sum() + ((y[right] - y[right].mean()) ** 2).sum())
                if rss < best_score: best_score, best_feat, best_thr = rss, fj, thr
        if best_feat is None or best_score >= parent_rss:
            return {"leaf": True, "value": float(y.mean())}
        left = X[:, best_feat] <= best_thr; right = ~left
        return {"leaf": False, "feat": int(best_feat), "thr": float(best_thr),
                "left": build_tree(X[left], y[left], depth + 1, max_depth, min_leaf, rng, mtry),
                "right": build_tree(X[right], y[right], depth + 1, max_depth, min_leaf, rng, mtry)}

    def predict_tree(tree, x):
        node = tree
        while not node["leaf"]:
            node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
        return node["value"]

    def fit_forest(X, y, n_trees, max_depth, min_leaf, mtry, seed):
        rng = np.random.default_rng(seed); trees = []; n_ = len(y)
        for _ in range(n_trees):
            boot_idx = rng.integers(0, n_, size=n_)
            trees.append(build_tree(X[boot_idx], y[boot_idx], 0, max_depth, min_leaf, rng, mtry))
        return trees

    def predict_forest(trees, X):
        return np.array([np.mean([predict_tree(t, X[i]) for t in trees]) for i in range(len(X))])

    def cv_forest(Xd, y, seed, n_trees=80, max_depth=3, min_leaf=10, mtry=3, k=5):
        n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
        pred = np.full(n_, np.nan)
        for f in folds:
            tr = np.setdiff1d(np.arange(n_), f)
            trees = fit_forest(Xd[tr], y[tr], n_trees, max_depth, min_leaf, mtry, seed)
            pred[f] = predict_forest(trees, Xd[f])
        return r2_mse_mae(y, pred)

    knn_by_seed = {}; rf_by_seed = {}
    for s in SEEDS_ALL:
        kbest = select_k_inner(X_raw, yv, s)
        r2, mse, mae = cv_knn(X_raw, yv, s, kbest)
        knn_by_seed[s] = {"k": kbest, "cvR2": round(r2, 4), "cvMSE": round(mse, 5), "cvMAE": round(mae, 5)}
        r2, mse, mae = cv_forest(X_raw, yv, s)
        rf_by_seed[s] = {"cvR2": round(r2, 4), "cvMSE": round(mse, 5), "cvMAE": round(mae, 5)}
        print(f"seed {s}: knn={knn_by_seed[s]} rf={rf_by_seed[s]}")

    save_merged({"knn": knn_by_seed, "random_forest": rf_by_seed})

# =====================================================================
if STAGE in ("aic", "all"):
    rowsAll = []
    for t, j in ymap.items():
        for yr in (2022, 2023):
            if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
            if M[j].get(f"EBI_Norm_{yr}") is None: continue
            feats = build_all_features(j, yr)
            if feats is None: continue
            rowsAll.append((t, j, yr, ymeas[t][yr], feats))
    nF = len(rowsAll)
    yF = np.array([r[3] for r in rowsAll])
    uefF = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rowsAll])
    ebiF = zc([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rowsAll])
    climF = zc([CP[r[2]] for r in rowsAll])
    gaF = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rowsAll])
    oneF = np.ones(nF)
    base_names = ["intercept", "cultivar", "EBI", "EBIxcultivar", "climate", "Growth_April"]
    base_cols_arr = [oneF, uefF, ebiF, ebiF * uefF, climF, gaF]

    raw_area = zc([r[4]["_raw_area"] for r in rowsAll])
    raw_frac065 = zc([r[4]["HighEBI_frac_065"] for r in rowsAll])
    raw_ebistd = zc([r[4]["EBI_Norm_std"] for r in rowsAll])
    raw_shadow_inv = zc([-(r[4]["ShadowFraction"]) for r in rowsAll])
    sbs = raw_area + raw_frac065 + raw_ebistd + raw_shadow_inv
    feat_arrays = {}
    for name in CANOPY_NAMES + V6_NAMES:
        feat_arrays[name] = zc([r[4][name] for r in rowsAll])
    for name in ["TBL", "ICBI", "TICBL", "UBI", "CCAB"]:
        feat_arrays[name] = zc([r[4][name] for r in rowsAll])
    feat_arrays["SBS"] = zc(sbs)
    candidates = {}
    for name, arr in feat_arrays.items():
        candidates[name] = arr; candidates[f"{name}_x_cultivar"] = arr * uefF

    def fit_ols_metrics(Xd, y):
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        resid = y - Xd @ beta; n_, p_ = Xd.shape
        rss = float((resid ** 2).sum()); k = p_ + 1
        logL = -n_ / 2 * np.log(2 * np.pi) - n_ / 2 * np.log(rss / n_) - n_ / 2
        aic = -2 * logL + 2 * k; bic = -2 * logL + k * np.log(n_)
        return {"AIC": round(float(aic), 3), "BIC": round(float(bic), 3),
                "MSE_in_sample": round(rss / n_, 5), "MAE_in_sample": round(float(np.mean(np.abs(resid))), 5)}

    def cv_mse_mae(Xd, y, seed=42, k=5):
        n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
        pred = np.full(n_, np.nan)
        for f in folds:
            tr = np.setdiff1d(np.arange(n_), f)
            beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None); pred[f] = Xd[f] @ beta
        resid = y - pred
        return round(float(np.mean(resid ** 2)), 5), round(float(np.mean(np.abs(resid))), 5)

    X_base_full = np.column_stack(base_cols_arr)
    base_metrics = fit_ols_metrics(X_base_full, yF)
    base_cv_mse, base_cv_mae = cv_mse_mae(X_base_full, yF)
    print(f"BASELINE (n={nF}): AIC={base_metrics['AIC']} MSE_in={base_metrics['MSE_in_sample']} "
          f"MAE_in={base_metrics['MAE_in_sample']} CV_MSE={base_cv_mse} CV_MAE={base_cv_mae}")

    add_results = {}
    for name, col in candidates.items():
        Xt = np.column_stack(base_cols_arr + [col])
        m = fit_ols_metrics(Xt, yF)
        cvm, cva = cv_mse_mae(Xt, yF)
        add_results[name] = {
            "dAIC": round(m["AIC"] - base_metrics["AIC"], 3),
            "dCV_MSE": round(cvm - base_cv_mse, 5), "dCV_MAE": round(cva - base_cv_mae, 5),
        }
    n_improve_aic = sum(1 for v in add_results.values() if v["dAIC"] < 0)
    n_improve_mse = sum(1 for v in add_results.values() if v["dCV_MSE"] < 0)
    by_aic = sorted(add_results.keys(), key=lambda n_: add_results[n_]["dAIC"])
    print(f"[ADD, n={nF}] improve AIC: {n_improve_aic}/70, improve CV-MSE: {n_improve_mse}/70. Top 5 by dAIC:")
    for n_ in by_aic[:5]:
        print(f"  {n_}: {add_results[n_]}")

    remove_results = {}
    for i, name in enumerate(base_names):
        if name == "intercept": continue
        keep_idx = [j2 for j2 in range(len(base_names)) if j2 != i]
        Xr = np.column_stack([base_cols_arr[j2] for j2 in keep_idx])
        m = fit_ols_metrics(Xr, yF)
        cvm, cva = cv_mse_mae(Xr, yF)
        remove_results[name] = {"dAIC": round(m["AIC"] - base_metrics["AIC"], 3),
                                 "dCV_MSE": round(cvm - base_cv_mse, 5), "dCV_MAE": round(cva - base_cv_mae, 5)}
    print(f"[REMOVE, n={nF}]:")
    for name in base_names[1:]:
        print(f"  remove {name}: {remove_results[name]}")

    save_merged({
        "aic_baseline_n167": {**base_metrics, "CV_MSE": base_cv_mse, "CV_MAE": base_cv_mae},
        "aic_add_candidates_n167": add_results, "aic_n_improve_AIC": n_improve_aic, "aic_n_improve_CVMSE": n_improve_mse,
        "aic_remove_terms_n167": remove_results, "aic_n_rows": nF,
    })

print("\nstage complete:", STAGE)
