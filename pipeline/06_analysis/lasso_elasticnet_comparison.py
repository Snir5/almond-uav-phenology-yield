#!/usr/bin/env python3
"""
lasso_elasticnet_comparison.py: per follow-up request ("have we tried lasso,
ridge, or other regularization"), adds Lasso (L1) and Elastic Net (L1+L2
blend) to the model-family comparison. Ridge (L2) was already tried in
section 49 (More_Models_Comparison.json); this fills in the two remaining
standard regularized-regression variants.

Both implemented from scratch via coordinate descent (soft-thresholding),
since the sandbox has no scikit-learn. All 75 predictor columns are used at
once (5 base recommended-model terms + all 70 engineered candidates from
section 47), letting the penalty itself decide what survives, instead of
stepwise forward selection. Unlike Ridge, Lasso/Elastic Net can zero
coefficients out entirely, so this doubles as an independent feature-
selection method to compare against sections 43/47's FDR-controlled search.

Penalty strength (lambda) selected by NESTED cross-validation: within each
outer fold's training data, a further 5-fold inner CV grid-search picks
lambda, then the final fit for that outer fold uses only that fold's
training data. This avoids any leakage of the outer test fold into lambda
selection. Same outlier-removed n=152 sample as section 47 (per the standing
instruction).

-> Results_Analysis/08_UEF53_Rerun/Lasso_ElasticNet_Comparison.json
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
SEEDS_ALL = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [42]

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

no_out = json.load(open(os.path.join(OUT, "Four_Way_Comparison_No_Outliers_Both_Tails.json")))
excluded = set()
for entry in no_out["removed_trees_high"] + no_out["removed_trees_low"]:
    excluded.add((entry["tree_id"], entry["year"]))

rowsAll = []
for t, j in ymap.items():
    for yr in (2022, 2023):
        if (t, yr) in excluded: continue
        if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
        if M[j].get(f"EBI_Norm_{yr}") is None: continue
        feats = build_all_features(j, yr)
        if feats is None: continue
        rowsAll.append((t, j, yr, ymeas[t][yr], feats))
nF = len(rowsAll)
print(f"n={nF} (outlier-removed, both tails, full feature match)")


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


yF = np.array([r[3] for r in rowsAll])
uefF = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rowsAll])
ebiF = zc([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rowsAll])
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
climF = zc([CP[r[2]] for r in rowsAll])
gaF = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rowsAll])

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

col_names = ["cultivar", "EBI", "EBIxcultivar", "climate", "Growth_April"]
cols = [uefF, ebiF, ebiF * uefF, climF, gaF]
for name, arr in feat_arrays.items():
    col_names.append(name); cols.append(arr)
    col_names.append(f"{name}_x_cultivar"); cols.append(arr * uefF)
Xall = np.column_stack(cols)  # NOTE: no intercept column here; handled via centering
print(f"design: {Xall.shape[1]} penalized predictor columns (5 base + 70 engineered candidates)")

# ================= coordinate descent: Lasso / Elastic Net (X columns already ~standardized) =================
def soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def fit_glmnet_cd(Xd, y, lam, alpha, max_iter=300, tol=2e-6):
    n_, p_ = Xd.shape
    y_mean = y.mean(); yc_ = y - y_mean
    beta = np.zeros(p_)
    col_ss = (Xd ** 2).sum(axis=0) / n_  # ~1 for z-scored columns
    for it in range(max_iter):
        beta_old = beta.copy()
        r = yc_ - Xd @ beta
        for j_ in range(p_):
            r_j = r + Xd[:, j_] * beta[j_]
            rho = (Xd[:, j_] @ r_j) / n_
            denom = col_ss[j_] + lam * (1 - alpha)
            beta[j_] = soft_threshold(rho, lam * alpha) / denom if denom > 1e-12 else 0.0
            r = r_j - Xd[:, j_] * beta[j_]
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    intercept = y_mean
    return intercept, beta


def predict_glmnet(model, Xd):
    intercept, beta = model
    return intercept + Xd @ beta


LAMBDA_GRID = np.logspace(-3, 1, 12)


def inner_cv_select_lambda(Xd, y, alpha, k=3, seed=0):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    best_lam, best_mse = None, np.inf
    for lam in LAMBDA_GRID:
        errs = []
        for f in folds:
            tr = np.setdiff1d(np.arange(n_), f)
            model = fit_glmnet_cd(Xd[tr], y[tr], lam, alpha)
            pred = predict_glmnet(model, Xd[f])
            errs.append(np.mean((y[f] - pred) ** 2))
        mse = np.mean(errs)
        if mse < best_mse:
            best_mse, best_lam = mse, lam
    return best_lam


def nested_cv_glmnet(Xd, y, alpha, k=5, seed=42):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred_all = np.full(n_, np.nan)
    lambdas_used = []
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        lam = inner_cv_select_lambda(Xd[tr], y[tr], alpha, seed=seed)
        lambdas_used.append(lam)
        model = fit_glmnet_cd(Xd[tr], y[tr], lam, alpha)
        pred_all[f] = predict_glmnet(model, Xd[f])
    resid = y - pred_all
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / tss
    mse = float(np.mean(resid ** 2)); mae = float(np.mean(np.abs(resid)))
    return round(float(r2), 4), round(mse, 5), round(mae, 5), lambdas_used


def run_family(alpha, label):
    print(f"\n=== {label} (alpha={alpha}) ===")
    results_by_seed = {}
    for s in SEEDS_ALL:
        r2, mse, mae, lambdas_used = nested_cv_glmnet(Xall, yF, alpha, seed=s)
        results_by_seed[s] = {"cvR2": r2, "cvMSE": mse, "cvMAE": mae, "lambdas": [round(float(l), 4) for l in lambdas_used]}
        print(f"  seed {s}: cvR2={r2} cvMSE={mse} cvMAE={mae} lambdas={results_by_seed[s]['lambdas']}")
    r2s = [v["cvR2"] for v in results_by_seed.values()]
    mses = [v["cvMSE"] for v in results_by_seed.values()]
    maes = [v["cvMAE"] for v in results_by_seed.values()]
    print(f"  cvR2 range {min(r2s)}-{max(r2s)}  cvMSE range {min(mses)}-{max(mses)}  cvMAE range {min(maes)}-{max(maes)}")

    # final fit on full data (lambda via CV on full data) to inspect sparsity pattern
    lam_full = inner_cv_select_lambda(Xall, yF, alpha, seed=42)
    intercept_full, beta_full = fit_glmnet_cd(Xall, yF, lam_full, alpha)
    nonzero = [(col_names[i], round(float(beta_full[i]), 4)) for i in range(len(col_names)) if abs(beta_full[i]) > 1e-6]
    nonzero_sorted = sorted(nonzero, key=lambda kv: -abs(kv[1]))
    print(f"  final full-data fit: lambda={lam_full:.4f}, {len(nonzero)} of {len(col_names)} coefficients nonzero:")
    for name, b in nonzero_sorted:
        print(f"    {name}: {b:+.4f}")
    return {
        "cv_by_seed": results_by_seed, "cvR2_range": [round(min(r2s), 4), round(max(r2s), 4)],
        "cvMSE_range": [round(min(mses), 5), round(max(mses), 5)], "cvMAE_range": [round(min(maes), 5), round(max(maes), 5)],
        "final_lambda_full_data": round(float(lam_full), 4), "n_nonzero": len(nonzero), "n_total": len(col_names),
        "nonzero_coefficients": nonzero_sorted,
    }


lasso_results = run_family(1.0, "LASSO (pure L1)")
enet_results = run_family(0.5, "ELASTIC NET (L1+L2, alpha=0.5)")

# ================= reference: OLS on just the 5 base terms, n=152 (already established) =================
X_base_only = np.column_stack([np.ones(nF), uefF, ebiF, ebiF * uefF, climF, gaF])


def r2_cv_ols(Xd, y, seed=42, k=5):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None); pred[f] = Xd[f] @ beta
    resid = y - pred; tss = ((y - y.mean()) ** 2).sum()
    return round(float(1 - (resid ** 2).sum() / tss), 4), round(float(np.mean(resid ** 2)), 5), round(float(np.mean(np.abs(resid))), 5)


ols_ref = {s: r2_cv_ols(X_base_only, yF, seed=s) for s in SEEDS_ALL}
print(f"\n=== OLS reference (5 base terms only, n={nF}) ===")
for s, v in ols_ref.items():
    print(f"  seed {s}: cvR2={v[0]} cvMSE={v[1]} cvMAE={v[2]}")

RESULTS = {
    "n": nF, "n_predictor_columns": Xall.shape[1],
    "ols_reference_base_terms_only": {str(s): {"cvR2": v[0], "cvMSE": v[1], "cvMAE": v[2]} for s, v in ols_ref.items()},
    "lasso": lasso_results, "elastic_net": enet_results,
}
with open(os.path.join(OUT, "Lasso_ElasticNet_Comparison.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")
