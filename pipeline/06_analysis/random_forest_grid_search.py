#!/usr/bin/env python3
"""
random_forest_grid_search.py: per follow-up ("I don't believe Random Forest
performed that poorly, try a grid search"), the Random Forest in section 52
used one fixed, hand-picked hyperparameter set (80 trees, max_depth=3,
min_leaf=10, mtry=3) with no tuning. This grid-searches max_depth, min_leaf,
and mtry properly, on the FULL n=167 sample (per section 54, the sample this
project now uses throughout).

Two-stage procedure, disclosed as such:
1. Screening grid search: every (max_depth, min_leaf, mtry) combination
   scored by 3-fold CV (single seed, 50 trees per forest for speed), to find
   the best region of hyperparameter space cheaply.
2. Final evaluation: the single best combination from stage 1 is re-run with
   the project's standard 5-fold CV across all 6 seeds and a larger 150-tree
   forest, for a fair, directly comparable number to every other model in
   sections 48-54.

This is NOT full nested CV (that would re-run the whole grid inside every
outer fold, computationally out of reach in this sandbox), so the stage-1
selection carries a small optimism risk verified by making sure stage 2's
numbers are still computed as genuine 5-fold-held-out CV, not re-using
stage 1's folds.

-> Results_Analysis/08_UEF53_Rerun/Random_Forest_Grid_Search.json
"""
import os, sys, csv, math, json, sqlite3, struct, time
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


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


rows = []
for t, j in ymap.items():
    for yr in (2022, 2023):
        if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
        if M[j].get(f"EBI_Norm_{yr}") is None: continue
        rows.append((t, j, yr, ymeas[t][yr]))
n = len(rows)
print(f"n={n} (FULL sample, outliers included, per section 54)")

yv = np.array([r[3] for r in rows])
uef = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rows])
ebi = zc([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rows])
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
clim = zc([CP[r[2]] for r in rows])
ga = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rows])
X_raw = np.column_stack([uef, ebi, clim, ga])
one = np.ones(n)
X_ols = np.column_stack([one, uef, ebi, ebi * uef, clim, ga])


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


def cv_forest(Xd, y, seed, n_trees, max_depth, min_leaf, mtry, k=5):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        trees = fit_forest(Xd[tr], y[tr], n_trees, max_depth, min_leaf, mtry, seed)
        pred[f] = predict_forest(trees, Xd[f])
    return r2_mse_mae(y, pred)


STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"

# ================= stage 1: screening grid search (3-fold, 1 seed, 50 trees) =================
MAX_DEPTHS = [2, 3, 4, 5]
MIN_LEAVES = [5, 8, 10, 15]
MTRYS = [2, 3, 4]

json_path = os.path.join(OUT, "Random_Forest_Grid_Search.json")


def load_prior():
    return json.load(open(json_path)) if os.path.exists(json_path) else {}


if STAGE in ("screen", "all"):
    t0 = time.time()
    grid_results = []
    for md in MAX_DEPTHS:
        for ml in MIN_LEAVES:
            for mt in MTRYS:
                if 2 * ml > n * 0.8:
                    continue
                r2, mse, mae = cv_forest(X_raw, yv, 42, n_trees=50, max_depth=md, min_leaf=ml, mtry=mt, k=3)
                grid_results.append({"max_depth": md, "min_leaf": ml, "mtry": mt, "screen_r2": round(r2, 4)})
    print(f"stage 1: {len(grid_results)} combinations screened in {time.time()-t0:.1f}s")
    grid_results.sort(key=lambda d: -d["screen_r2"])
    print("Top 10 screened combos:")
    for g in grid_results[:10]:
        print(f"  max_depth={g['max_depth']} min_leaf={g['min_leaf']} mtry={g['mtry']}: screen_r2={g['screen_r2']}")
    prior = load_prior()
    prior.update({"n": n, "grid_screened": len(grid_results),
                  "grid_dims": {"max_depth": MAX_DEPTHS, "min_leaf": MIN_LEAVES, "mtry": MTRYS},
                  "top10_screened": grid_results[:10], "best_combo": grid_results[0]})
    with open(json_path, "w") as f:
        json.dump(prior, f, indent=2, default=str)
    print("saved stage 1 json")

# ================= stage 2: honest evaluation, 5-fold CV, 6 seeds, 150 trees =================
if STAGE in ("final", "all"):
    prior = load_prior()
    best = prior["best_combo"]
    print(f"\nusing best combo from stage 1: {best}")
    SEEDS_ALL = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [42, 1, 7, 13, 99, 2024]
    final_by_seed = prior.get("final_cv_by_seed", {})
    for s in SEEDS_ALL:
        r2, mse, mae = cv_forest(X_raw, yv, s, n_trees=150, max_depth=best["max_depth"], min_leaf=best["min_leaf"], mtry=best["mtry"], k=5)
        final_by_seed[str(s)] = {"cvR2": round(r2, 4), "cvMSE": round(mse, 5), "cvMAE": round(mae, 5)}
        print(f"  seed {s}: {final_by_seed[str(s)]}")
    r2s = [v["cvR2"] for v in final_by_seed.values()]
    print(f"\nTuned Random Forest, n=167: cvR2 range so far {min(r2s)}-{max(r2s)}")
    print(f"(for comparison, section 52/53's untuned RF, n=167: cvR2 range 0.316-0.380; OLS reference: 0.448-0.488)")
    prior["final_cv_by_seed"] = final_by_seed
    prior["final_cvR2_range"] = [round(min(r2s), 4), round(max(r2s), 4)]
    prior["untuned_rf_reference_cvR2_range"] = [0.316, 0.380]
    prior["ols_reference_cvR2_range"] = [0.448, 0.488]
    with open(json_path, "w") as f:
        json.dump(prior, f, indent=2, default=str)
    print("saved stage 2 json")
