#!/usr/bin/env python3
"""
nonlinear_models_comparison.py: per follow-up request ("is there a nonlinear
model worth trying"), everything tried so far (sections 48-51: OLS, Gamma
GLM, Inverse Gaussian GLM, mixed model, log-normal, Huber, Ridge, Lasso,
Elastic Net) is linear in the predictors, even the GLMs are a linear
predictor pushed through a link function. This adds two genuinely nonlinear,
from-scratch model families (no sklearn in this sandbox):

1. k-Nearest-Neighbors regression: fully nonparametric, makes no functional
   form assumption at all. k chosen by CV.
2. A small bagged regression-tree ensemble (Random Forest): CART-style
   recursive splitting (minimize weighted child RSS), bootstrap resampling,
   random feature subsampling per split, shallow depth and a minimum leaf
   size to control overfitting given n~150. Trees automatically capture
   nonlinearities AND interactions (e.g. a cultivar x EBI split) without
   needing an explicit interaction term.

Both are fit on the 4 RAW predictors (cultivar, EBI, climate, Growth_April,
no z-scoring needed for trees, EBI/climate/Growth_April z-scored for kNN
distance comparability) rather than the 75-column engineered bank, since the
point here is to test whether a flexible functional form finds structure the
linear model's specific functional form (main effects + one interaction) is
missing on the ESTABLISHED features, not to re-run another 75-candidate
search. Same outlier-removed n=152 sample, same 5-fold CV framework.

-> Results_Analysis/08_UEF53_Rerun/Nonlinear_Models_Comparison.json
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
SEEDS_ALL = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [42, 1, 7, 13, 99, 2024]

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

rows = []
for t, j in ymap.items():
    for yr in (2022, 2023):
        if (t, yr) in excluded: continue
        if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
        if M[j].get(f"EBI_Norm_{yr}") is None: continue
        rows.append((t, j, yr, ymeas[t][yr]))
n = len(rows)
print(f"n={n} (outlier-removed, both tails)")


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


yv = np.array([r[3] for r in rows])
uef = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rows])
ebi_raw = np.array([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rows], float)
ebi = zc(ebi_raw)
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
clim = zc([CP[r[2]] for r in rows])
ga = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rows])
one = np.ones(n)

X_ols = np.column_stack([one, uef, ebi, ebi * uef, clim, ga])       # linear reference (with interaction)
X_raw = np.column_stack([uef, ebi, clim, ga])                        # 4 raw features for kNN / RF, no manual interaction


def r2_mse_mae(y_true, y_pred):
    resid = y_true - y_pred
    tss = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / tss
    return float(r2), float(np.mean(resid ** 2)), float(np.mean(np.abs(resid)))


# ================= reference: OLS with interaction =================
def cv_ols(Xd, y, seed, k=5):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None); pred[f] = Xd[f] @ beta
    return r2_mse_mae(y, pred)


# ================= 1. k-NN regression =================
def knn_predict(Xtr, ytr, Xte, k):
    preds = np.zeros(len(Xte))
    for i in range(len(Xte)):
        d = np.sqrt(((Xtr - Xte[i]) ** 2).sum(axis=1))
        nn = np.argsort(d)[:k]
        preds[i] = ytr[nn].mean()
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


# ================= 2. small Random Forest (bagged regression trees) =================
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
            if rss < best_score:
                best_score, best_feat, best_thr = rss, fj, thr
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
    rng = np.random.default_rng(seed)
    trees = []
    n_ = len(y)
    for _ in range(n_trees):
        boot_idx = rng.integers(0, n_, size=n_)
        Xb, yb = X[boot_idx], y[boot_idx]
        trees.append(build_tree(Xb, yb, 0, max_depth, min_leaf, rng, mtry))
    return trees


def predict_forest(trees, X):
    preds = np.zeros(len(X))
    for i in range(len(X)):
        preds[i] = np.mean([predict_tree(t, X[i]) for t in trees])
    return preds


def cv_forest(Xd, y, seed, n_trees=80, max_depth=3, min_leaf=10, mtry=3, k=5):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        trees = fit_forest(Xd[tr], y[tr], n_trees, max_depth, min_leaf, mtry, seed)
        pred[f] = predict_forest(trees, Xd[f])
    return r2_mse_mae(y, pred)


# ================= run all =================
print("\n=== OLS reference (with EBIxcultivar interaction, n=152) ===")
ols_by_seed = {}
for s in SEEDS_ALL:
    r2, mse, mae = cv_ols(X_ols, yv, s)
    ols_by_seed[s] = {"cvR2": round(r2, 4), "cvMSE": round(mse, 5), "cvMAE": round(mae, 5)}
    print(f"  seed {s}: cvR2={ols_by_seed[s]['cvR2']} cvMSE={ols_by_seed[s]['cvMSE']} cvMAE={ols_by_seed[s]['cvMAE']}")

print("\n=== k-NN regression (4 raw features, k selected via inner CV) ===")
knn_by_seed = {}
for s in SEEDS_ALL:
    kbest = select_k_inner(X_raw, yv, s)
    r2, mse, mae = cv_knn(X_raw, yv, s, kbest)
    knn_by_seed[s] = {"k": kbest, "cvR2": round(r2, 4), "cvMSE": round(mse, 5), "cvMAE": round(mae, 5)}
    print(f"  seed {s}: k={kbest} cvR2={knn_by_seed[s]['cvR2']} cvMSE={knn_by_seed[s]['cvMSE']} cvMAE={knn_by_seed[s]['cvMAE']}")

print("\n=== Random Forest (bagged CART, 4 raw features, 80 trees, depth<=3, min_leaf=10, mtry=3) ===")
rf_by_seed = {}
for s in SEEDS_ALL:
    r2, mse, mae = cv_forest(X_raw, yv, s)
    rf_by_seed[s] = {"cvR2": round(r2, 4), "cvMSE": round(mse, 5), "cvMAE": round(mae, 5)}
    print(f"  seed {s}: cvR2={rf_by_seed[s]['cvR2']} cvMSE={rf_by_seed[s]['cvMSE']} cvMAE={rf_by_seed[s]['cvMAE']}")

# full-data forest for feature importance (permutation importance, cheap and honest)
trees_full = fit_forest(X_raw, yv, n_trees=150, max_depth=3, min_leaf=10, mtry=3, seed=42)
base_pred_full = predict_forest(trees_full, X_raw)
base_mse_full = float(np.mean((yv - base_pred_full) ** 2))
importances = {}
rng_imp = np.random.default_rng(0)
feat_labels = ["cultivar", "EBI", "climate", "Growth_April"]
for fi, name in enumerate(feat_labels):
    Xp = X_raw.copy()
    Xp[:, fi] = rng_imp.permutation(Xp[:, fi])
    pred_p = predict_forest(trees_full, Xp)
    mse_p = float(np.mean((yv - pred_p) ** 2))
    importances[name] = round(mse_p - base_mse_full, 5)
print(f"\nPermutation importance (increase in MSE when shuffled, full-data forest): {importances}")

RESULTS = {
    "n": n, "seeds": SEEDS_ALL,
    "ols_reference": ols_by_seed,
    "knn": knn_by_seed,
    "random_forest": rf_by_seed,
    "rf_permutation_importance_full_data": importances,
}
json_path = os.path.join(OUT, "Nonlinear_Models_Comparison.json")
if os.path.exists(json_path) and len(sys.argv) > 1:
    prior = json.load(open(json_path))
    for k_ in ["ols_reference", "knn", "random_forest"]:
        prior[k_] = {**prior.get(k_, {}), **RESULTS[k_]}
    prior["seeds"] = sorted(set(prior.get("seeds", [])) | set(SEEDS_ALL))
    prior["rf_permutation_importance_full_data"] = importances
    RESULTS = prior
with open(json_path, "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")
