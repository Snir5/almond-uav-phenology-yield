#!/usr/bin/env python3
"""
poly_regression_comparison.py: answers "did we try polynomial regression?"
Not tried in sections 45-56 (Gamma/Inverse-Gaussian GLM, mixed model,
log-normal, Huber, Ridge, Lasso, Elastic Net, k-NN, Random Forest, grid
search). This adds polynomial terms of the continuous predictors (EBI,
climate/Chill Portions, Growth_April) to the recommended OLS model and
checks whether curvature helps, on the full n=167 sample (per section 54).

Same base data pipeline as all_models_full_sample.py / random_forest_grid_search.py
(haversine yield match, GPKG Growth_April match, z-scored predictors).

-> Results_Analysis/08_UEF53_Rerun/Polynomial_Regression_Comparison.json
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict
import numpy as np
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

# NOTE: no outlier exclusion -- full n=167 sample, per section 54.


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
print(f"n={n} (FULL sample, outliers included, per section 54)")

ebi2 = ebi ** 2; clim2 = clim ** 2; ga2 = ga ** 2; ebi3 = ebi ** 3

X_base = np.column_stack([one, uef, ebi, ebi * uef, clim, ga])
base_names = ["intercept", "cultivar", "EBI", "EBIxcultivar", "climate", "Growth_April"]

variants = {
    "baseline_linear": X_base,
    "quad_EBI": np.column_stack([X_base, ebi2]),
    "quad_EBI_x_cultivar": np.column_stack([X_base, ebi2, ebi2 * uef]),
    "quad_climate": np.column_stack([X_base, clim2]),
    "quad_Growth_April": np.column_stack([X_base, ga2]),
    "quad_all_three": np.column_stack([X_base, ebi2, clim2, ga2]),
    "cubic_EBI": np.column_stack([X_base, ebi2, ebi3]),
    "full_quadratic_surface": np.column_stack([X_base, ebi2, clim2, ga2, ebi * clim, ebi * ga, clim * ga]),
}
variant_extra_names = {
    "baseline_linear": [],
    "quad_EBI": ["EBI^2"],
    "quad_EBI_x_cultivar": ["EBI^2", "EBI^2 x cultivar"],
    "quad_climate": ["climate^2"],
    "quad_Growth_April": ["Growth_April^2"],
    "quad_all_three": ["EBI^2", "climate^2", "Growth_April^2"],
    "cubic_EBI": ["EBI^2", "EBI^3"],
    "full_quadratic_surface": ["EBI^2", "climate^2", "Growth_April^2", "EBIxclimate", "EBIxGrowth_April", "climatexGrowth_April"],
}


def fit_ols_metrics(Xd, y):
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta; n_, p_ = Xd.shape
    rss = float((resid ** 2).sum()); k = p_ + 1
    logL = -n_ / 2 * np.log(2 * np.pi) - n_ / 2 * np.log(rss / n_) - n_ / 2
    aic = -2 * logL + 2 * k; bic = -2 * logL + k * np.log(n_)
    return {"AIC": round(float(aic), 3), "BIC": round(float(bic), 3),
            "MSE_in_sample": round(rss / n_, 5), "MAE_in_sample": round(float(np.mean(np.abs(resid))), 5),
            "R2_in_sample": round(float(1 - rss / ((y - y.mean()) ** 2).sum()), 4)}


def cv_ols(Xd, y, seed, k=5):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None); pred[f] = Xd[f] @ beta
    resid = y - pred
    tss = ((y - y.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / tss), float(np.mean(resid ** 2)), float(np.mean(np.abs(resid)))


SEEDS_ALL = [42, 1, 7, 13, 99, 2024]
results = {}
base_metrics = fit_ols_metrics(X_base, yv)
for name, Xd in variants.items():
    m = fit_ols_metrics(Xd, yv)
    cv_by_seed = {}
    for s in SEEDS_ALL:
        r2, mse, mae = cv_ols(Xd, yv, s)
        cv_by_seed[s] = {"cvR2": round(r2, 4), "cvMSE": round(mse, 5), "cvMAE": round(mae, 5)}
    r2s = [v["cvR2"] for v in cv_by_seed.values()]
    results[name] = {
        "extra_terms": variant_extra_names[name],
        "in_sample": m,
        "dAIC_vs_baseline": round(m["AIC"] - base_metrics["AIC"], 3),
        "dBIC_vs_baseline": round(m["BIC"] - base_metrics["BIC"], 3),
        "cv_by_seed": cv_by_seed,
        "cvR2_range": [round(min(r2s), 4), round(max(r2s), 4)],
        "cvR2_mean": round(float(np.mean(r2s)), 4),
    }
    print(f"{name}: AIC={m['AIC']} (d={results[name]['dAIC_vs_baseline']:+.3f}) "
          f"R2_in={m['R2_in_sample']} cvR2_range={results[name]['cvR2_range']} mean={results[name]['cvR2_mean']}")

with open(os.path.join(OUT, "Polynomial_Regression_Comparison.json"), "w") as f:
    json.dump({"n": n, "base_names": base_names, "results": results}, f, indent=2, default=str)
print("\nsaved json")
