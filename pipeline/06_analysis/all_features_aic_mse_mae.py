#!/usr/bin/env python3
"""
all_features_aic_mse_mae.py: re-evaluates every candidate feature from
section 47 (add-one-at-a-time) AND every term already in the recommended
model (remove-one-at-a-time) using AIC, MSE, and MAE, not just R^2/CV-R^2.

AIC (Gaussian-OLS, standard form with the sigma^2 parameter counted):
  logL = -n/2*log(2*pi) - n/2*log(RSS/n) - n/2
  AIC  = -2*logL + 2*(p+1),  p = number of beta coefficients
Lower AIC = better (penalizes added parameters, unlike R^2 which never
decreases when a term is added).

MSE/MAE reported both in-sample (on the fitted model, optimistic) and
cross-validated (5-fold, main seed 42 + 5 robustness seeds, honest).

Per the standing instruction, uses the outlier-removed n=152 sample
(sections 45-46). Same 70-candidate universe as section 47 (11 canopy +
18 EBI pixel-distribution + 6 composite bloom indices, each plain and x
cultivar).

-> Results_Analysis/08_UEF53_Rerun/All_Features_AIC_MSE_MAE.json
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
CV_SEED = 42
SEEDS_ALL = [42, 1, 7, 13, 99, 2024]

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
oneF = np.ones(nF)
base_names = ["intercept", "cultivar", "EBI", "EBIxcultivar", "climate", "Growth_April"]
base_cols_arr = [oneF, uefF, ebiF, ebiF * uefF, climF, gaF]
X_base = np.column_stack(base_cols_arr)

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
    candidates[name] = arr
    candidates[f"{name}_x_cultivar"] = arr * uefF

# ================= metric helpers =================
def fit_ols_metrics(Xd, y):
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    n_, p_ = Xd.shape
    rss = float((resid ** 2).sum())
    mse_in = rss / n_
    mae_in = float(np.mean(np.abs(resid)))
    k = p_ + 1
    logL = -n_ / 2 * np.log(2 * np.pi) - n_ / 2 * np.log(rss / n_) - n_ / 2
    aic = -2 * logL + 2 * k
    bic = -2 * logL + k * np.log(n_)
    return {"beta": beta, "AIC": round(float(aic), 3), "BIC": round(float(bic), 3),
            "MSE_in_sample": round(float(mse_in), 5), "MAE_in_sample": round(float(mae_in), 5), "p": p_}


def cv_mse_mae(Xd, y, seed=CV_SEED, k=5):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None); pred[f] = Xd[f] @ beta
    resid = y - pred
    return round(float(np.mean(resid ** 2)), 5), round(float(np.mean(np.abs(resid))), 5)


def cv_robust(Xd, y):
    vals = {s: cv_mse_mae(Xd, y, seed=s) for s in SEEDS_ALL}
    mses = [v[0] for v in vals.values()]; maes = [v[1] for v in vals.values()]
    return {"MSE_seed42": vals[42][0], "MAE_seed42": vals[42][1],
            "MSE_range": [round(min(mses), 5), round(max(mses), 5)],
            "MAE_range": [round(min(maes), 5), round(max(maes), 5)]}


# ================= baseline =================
base_metrics = fit_ols_metrics(X_base, yF)
base_cv = cv_robust(X_base, yF)
print(f"\nBASELINE (n={nF}): AIC={base_metrics['AIC']} BIC={base_metrics['BIC']} "
      f"MSE_in={base_metrics['MSE_in_sample']} MAE_in={base_metrics['MAE_in_sample']} | "
      f"CV MSE={base_cv['MSE_seed42']} (range {base_cv['MSE_range']}) CV MAE={base_cv['MAE_seed42']} (range {base_cv['MAE_range']})")

# ================= ADD: each of the 70 candidates =================
add_results = {}
for name, col in candidates.items():
    Xt = np.column_stack(base_cols_arr + [col])
    m = fit_ols_metrics(Xt, yF)
    cvm = cv_robust(Xt, yF)
    add_results[name] = {
        "AIC": m["AIC"], "dAIC_vs_baseline": round(m["AIC"] - base_metrics["AIC"], 3),
        "BIC": m["BIC"], "dBIC_vs_baseline": round(m["BIC"] - base_metrics["BIC"], 3),
        "MSE_in_sample": m["MSE_in_sample"], "MAE_in_sample": m["MAE_in_sample"],
        "CV_MSE_seed42": cvm["MSE_seed42"], "CV_MAE_seed42": cvm["MAE_seed42"],
        "dCV_MSE": round(cvm["MSE_seed42"] - base_cv["MSE_seed42"], 5),
        "dCV_MAE": round(cvm["MAE_seed42"] - base_cv["MAE_seed42"], 5),
    }

by_aic = sorted(add_results.keys(), key=lambda n: add_results[n]["dAIC_vs_baseline"])
by_cvmse = sorted(add_results.keys(), key=lambda n: add_results[n]["dCV_MSE"])
by_cvmae = sorted(add_results.keys(), key=lambda n: add_results[n]["dCV_MAE"])
n_improve_aic = sum(1 for n in add_results if add_results[n]["dAIC_vs_baseline"] < 0)
n_improve_cvmse = sum(1 for n in add_results if add_results[n]["dCV_MSE"] < 0)
n_improve_cvmae = sum(1 for n in add_results if add_results[n]["dCV_MAE"] < 0)
print(f"\n[ADD] {len(add_results)} candidates tested. Improve AIC (dAIC<0): {n_improve_aic}. "
      f"Improve CV-MSE: {n_improve_cvmse}. Improve CV-MAE: {n_improve_cvmae}.")
print("Top 8 by dAIC (most negative = best):")
for n in by_aic[:8]:
    r = add_results[n]
    print(f"  {n}: dAIC={r['dAIC_vs_baseline']:+.3f} dBIC={r['dBIC_vs_baseline']:+.3f} "
          f"dCV_MSE={r['dCV_MSE']:+.5f} dCV_MAE={r['dCV_MAE']:+.5f}")
print("Top 8 by dCV_MSE:")
for n in by_cvmse[:8]:
    r = add_results[n]
    print(f"  {n}: dCV_MSE={r['dCV_MSE']:+.5f} dAIC={r['dAIC_vs_baseline']:+.3f}")
print("Top 8 by dCV_MAE:")
for n in by_cvmae[:8]:
    r = add_results[n]
    print(f"  {n}: dCV_MAE={r['dCV_MAE']:+.5f} dAIC={r['dAIC_vs_baseline']:+.3f}")

# ================= REMOVE: each of the 5 non-intercept baseline terms =================
remove_results = {}
for i, name in enumerate(base_names):
    if name == "intercept": continue
    keep_idx = [j for j in range(len(base_names)) if j != i]
    Xr = np.column_stack([base_cols_arr[j] for j in keep_idx])
    m = fit_ols_metrics(Xr, yF)
    cvm = cv_robust(Xr, yF)
    remove_results[name] = {
        "AIC": m["AIC"], "dAIC_vs_baseline": round(m["AIC"] - base_metrics["AIC"], 3),
        "BIC": m["BIC"], "dBIC_vs_baseline": round(m["BIC"] - base_metrics["BIC"], 3),
        "MSE_in_sample": m["MSE_in_sample"], "MAE_in_sample": m["MAE_in_sample"],
        "CV_MSE_seed42": cvm["MSE_seed42"], "CV_MAE_seed42": cvm["MAE_seed42"],
        "dCV_MSE": round(cvm["MSE_seed42"] - base_cv["MSE_seed42"], 5),
        "dCV_MAE": round(cvm["MAE_seed42"] - base_cv["MAE_seed42"], 5),
    }
print(f"\n[REMOVE] each existing baseline term removed one at a time (negative dAIC/dCV = "
      f"removing it IMPROVES the model, i.e. that term is not pulling its weight):")
for name in base_names[1:]:
    r = remove_results[name]
    print(f"  remove {name}: dAIC={r['dAIC_vs_baseline']:+.3f} dBIC={r['dBIC_vs_baseline']:+.3f} "
          f"dCV_MSE={r['dCV_MSE']:+.5f} dCV_MAE={r['dCV_MAE']:+.5f}")

RESULTS = {
    "n": nF,
    "baseline": {**{k: v for k, v in base_metrics.items() if k != "beta"}, **base_cv},
    "add_candidates": add_results, "n_candidates": len(add_results),
    "n_improve_AIC": n_improve_aic, "n_improve_CV_MSE": n_improve_cvmse, "n_improve_CV_MAE": n_improve_cvmae,
    "top10_by_dAIC": by_aic[:10], "top10_by_dCV_MSE": by_cvmse[:10], "top10_by_dCV_MAE": by_cvmae[:10],
    "remove_existing_terms": remove_results,
}
with open(os.path.join(OUT, "All_Features_AIC_MSE_MAE.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")
