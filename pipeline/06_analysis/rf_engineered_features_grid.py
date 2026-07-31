#!/usr/bin/env python3
"""
rf_engineered_features_grid.py: per follow-up ("did you check RF with other
features too?"), Random Forest / kNN in sections 52-53-56 only ever used the
4 base features (cultivar, EBI, climate, Growth_April). Ridge/Lasso/Elastic
Net (sections 49/51/53) were tested against the full ~34-column engineered
bank (canopy shape, EBI-pixel-distribution, composite indices). This gives
Random Forest that same fair shot: grid search over hyperparameters with the
full engineered bank as candidate split features, full n=167 sample.

Selectable via argv: `python3 rf_engineered_features_grid.py screen [mtry_csv]`
                      `python3 rf_engineered_features_grid.py final [seeds_csv]`
Split into stages/mtry-chunks because the full grid exceeds the sandbox's
per-call execution budget.

-> Results_Analysis/08_UEF53_Rerun/RF_Engineered_Features_Grid.json
"""
import os, sys, csv, math, json, sqlite3, struct, time
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
COMPOSITE_NAMES = ["TBL", "ICBI", "TICBL", "UBI", "CCAB"]


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

# NOTE: no outlier exclusion -- full n=167 sample, per section 54.


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


rows = []
for t, j in ymap.items():
    for yr in (2022, 2023):
        if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
        if M[j].get(f"EBI_Norm_{yr}") is None: continue
        feats = build_all_features(j, yr)
        if feats is None: continue
        rows.append((t, j, yr, ymeas[t][yr], feats))
n = len(rows)

yv = np.array([r[3] for r in rows])
uef = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rows])
ebi = zc([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rows])
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
clim = zc([CP[r[2]] for r in rows])
ga = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rows])
eng_names = CANOPY_NAMES + V6_NAMES + COMPOSITE_NAMES
feat_cols = [zc([r[4][name] for r in rows]) for name in eng_names]
col_names = ["cultivar", "EBI", "climate", "Growth_April"] + eng_names
X_eng = np.column_stack([uef, ebi, clim, ga] + feat_cols)
p = X_eng.shape[1]
print(f"n={n} (FULL sample, outliers included, per section 54), p={p} features (4 base + {len(eng_names)} engineered)")


def r2_mse_mae(y_true, y_pred):
    resid = y_true - y_pred
    tss = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / tss), float(np.mean(resid ** 2)), float(np.mean(np.abs(resid)))


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


def cv_forest(Xd, y, seed, n_trees=50, max_depth=3, min_leaf=10, mtry=6, k=3):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        trees = fit_forest(Xd[tr], y[tr], n_trees, max_depth, min_leaf, mtry, seed)
        pred[f] = predict_forest(trees, Xd[f])
    return r2_mse_mae(y, pred)


STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
json_path = os.path.join(OUT, "RF_Engineered_Features_Grid.json")


def load_prior():
    return json.load(open(json_path)) if os.path.exists(json_path) else {}


MAX_DEPTHS = [2, 3, 4, 5]
MIN_LEAVES = [5, 8, 10, 15]
MTRYS_FULL = [3, 6, 10, 15, 20]

if STAGE in ("screen", "all"):
    mtrys_this_call = [int(m) for m in sys.argv[2].split(",")] if len(sys.argv) > 2 else MTRYS_FULL
    min_leaves_this_call = [int(m) for m in sys.argv[3].split(",")] if len(sys.argv) > 3 else MIN_LEAVES
    t0 = time.time()
    grid_results = []
    for md in MAX_DEPTHS:
        for ml in min_leaves_this_call:
            for mt in mtrys_this_call:
                if 2 * ml > n * 0.8: continue
                r2, mse, mae = cv_forest(X_eng, yv, 42, n_trees=50, max_depth=md, min_leaf=ml, mtry=mt, k=3)
                grid_results.append({"max_depth": md, "min_leaf": ml, "mtry": mt, "screen_r2": round(r2, 4)})
    print(f"stage 1 (mtry={mtrys_this_call}): {len(grid_results)} combos screened in {time.time()-t0:.1f}s")
    prior = load_prior()
    all_results = prior.get("all_screened", []) + grid_results
    all_results.sort(key=lambda d: -d["screen_r2"])
    print("Top 10 so far:")
    for g in all_results[:10]:
        print(f"  max_depth={g['max_depth']} min_leaf={g['min_leaf']} mtry={g['mtry']}: screen_r2={g['screen_r2']}")
    prior.update({"n": n, "p": p, "eng_feature_names": eng_names,
                  "grid_dims": {"max_depth": MAX_DEPTHS, "min_leaf": MIN_LEAVES, "mtry": MTRYS_FULL},
                  "all_screened": all_results, "top10_screened": all_results[:10], "best_combo": all_results[0]})
    with open(json_path, "w") as f:
        json.dump(prior, f, indent=2, default=str)
    print("saved")

if STAGE in ("final", "all"):
    prior = load_prior()
    best = prior["best_combo"]
    print(f"\nusing best combo: {best}")
    SEEDS_ALL = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [42, 1, 7, 13, 99, 2024]
    final_by_seed = prior.get("final_cv_by_seed", {})
    for s in SEEDS_ALL:
        r2, mse, mae = cv_forest(X_eng, yv, s, n_trees=150, max_depth=best["max_depth"], min_leaf=best["min_leaf"], mtry=best["mtry"], k=5)
        final_by_seed[str(s)] = {"cvR2": round(r2, 4), "cvMSE": round(mse, 5), "cvMAE": round(mae, 5)}
        print(f"  seed {s}: {final_by_seed[str(s)]}")
    r2s = [v["cvR2"] for v in final_by_seed.values()]
    print(f"\nRF w/ engineered bank, n=167: cvR2 range so far {min(r2s)}-{max(r2s)}")
    print("(4-feature RF grid-tuned: 0.316-0.385; 4-feature RF untuned: 0.316-0.380; OLS: 0.448-0.488)")
    prior["final_cv_by_seed"] = final_by_seed
    prior["final_cvR2_range"] = [round(min(r2s), 4), round(max(r2s), 4)]
    prior["four_feature_rf_grid_tuned_cvR2_range"] = [0.316, 0.385]
    prior["ols_reference_cvR2_range"] = [0.448, 0.488]
    with open(json_path, "w") as f:
        json.dump(prior, f, indent=2, default=str)
    print("saved")
