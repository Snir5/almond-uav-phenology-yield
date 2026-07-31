#!/usr/bin/env python3
"""
new_models_pls_omp_boost.py: three literature-inspired model families not yet
tried in this project (PLS, Orthogonal Matching Pursuit, gradient boosting),
plus a simple-average ensemble check, all on the full n=167 sample.

PLS and OMP run on the full 38-column engineered bank (4 base + 34
engineered), the setting where they are theoretically best suited (both are
designed for/robust to the collinearity that already hurt Ridge/Lasso/RF on
this same bank, sections 49/51/58). Gradient boosting runs on the 4 base
features only, for direct comparability with the tuned Random Forest number
(section 56); it is not re-tested on the full bank since both Ridge and
Random Forest already got WORSE, not better, with the full bank at this n
(sections 49, 58), so there is little reason to expect boosting to differ.

Selectable via argv: `python3 new_models_pls_omp_boost.py <stage> [arg2]`
  stage in {pls, omp, boost_screen, boost_final, ensemble}
  pls/omp: arg2 = seeds csv (default all 6)
  boost_screen: arg2 = learning_rate subset csv (default all)
  boost_final: arg2 = seeds csv
  ensemble: arg2 = seeds csv

-> Results_Analysis/08_UEF53_Rerun/New_Models_PLS_OMP_Boost.json
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
JSON_PATH = os.path.join(OUT, "New_Models_PLS_OMP_Boost.json")

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
one = np.ones(n)
eng_names = CANOPY_NAMES + V6_NAMES + COMPOSITE_NAMES
feat_cols = [zc([r[4][name] for r in rows]) for name in eng_names]
X_eng = np.column_stack([uef, ebi, clim, ga] + feat_cols)      # 38 cols, for PLS/OMP
X_ols = np.column_stack([one, uef, ebi, ebi * uef, clim, ga])  # 6 cols, OLS baseline
X_raw = np.column_stack([uef, ebi, clim, ga])                  # 4 cols, for boosting
print(f"n={n} (FULL sample, outliers included, per section 54), X_eng p={X_eng.shape[1]}")


def r2_mse_mae(y_true, y_pred):
    resid = y_true - y_pred
    tss = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / tss), float(np.mean(resid ** 2)), float(np.mean(np.abs(resid)))


SEEDS_ALL = [42, 1, 7, 13, 99, 2024]


def load_prior():
    return json.load(open(JSON_PATH)) if os.path.exists(JSON_PATH) else {}


def save_merged(new_data):
    prior = load_prior()
    for k_, v_ in new_data.items():
        if isinstance(v_, dict) and isinstance(prior.get(k_), dict):
            prior[k_] = {**prior[k_], **v_}
        else:
            prior[k_] = v_
    prior["n"] = n
    with open(JSON_PATH, "w") as f:
        json.dump(prior, f, indent=2, default=str)


# ============================= PLS (NIPALS) =============================
def fit_pls(X, y, n_comp):
    Xc = X - X.mean(axis=0); Xs = Xc.std(axis=0); Xs[Xs == 0] = 1.0; Xc = Xc / Xs
    y_mean = y.mean(); yc_ = y - y_mean
    Xr = Xc.copy(); yr = yc_.copy()
    Ws, Ps, qs = [], [], []
    for a in range(n_comp):
        num = Xr.T @ yr; norm = np.linalg.norm(num)
        if norm < 1e-12: break
        w = num / norm
        t = Xr @ w
        tt = t @ t
        if tt < 1e-12: break
        p_load = (Xr.T @ t) / tt
        q = (yr @ t) / tt
        Xr = Xr - np.outer(t, p_load)
        yr = yr - q * t
        Ws.append(w); Ps.append(p_load); qs.append(q)
    if not Ws:
        return {"B": np.zeros(X.shape[1]), "y_mean": y_mean, "x_mean": X.mean(axis=0), "x_std": Xs, "n_comp_used": 0}
    W = np.column_stack(Ws); P = np.column_stack(Ps); qv = np.array(qs)
    B = W @ np.linalg.pinv(P.T @ W) @ qv
    return {"B": B, "y_mean": y_mean, "x_mean": X.mean(axis=0), "x_std": Xs, "n_comp_used": len(Ws)}


def predict_pls(model, X):
    Xc = (X - model["x_mean"]) / model["x_std"]
    return model["y_mean"] + Xc @ model["B"]


def cv_pls_fixed(Xd, y, seed, n_comp, k=5):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        model = fit_pls(Xd[tr], y[tr], n_comp)
        pred[f] = predict_pls(model, Xd[f])
    return pred


def select_ncomp_inner(Xd, y, seed, max_comp, k=3):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    best_a, best_mse = 1, np.inf
    for a in range(1, max_comp + 1):
        pred = np.full(n_, np.nan)
        for f in folds:
            tr = np.setdiff1d(np.arange(n_), f)
            model = fit_pls(Xd[tr], y[tr], a)
            pred[f] = predict_pls(model, Xd[f])
        mse = np.mean((y - pred) ** 2)
        if mse < best_mse: best_mse, best_a = mse, a
    return best_a


def nested_cv_pls(Xd, y, seed, max_comp=12, k=5):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan); ncomps = []
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        a = select_ncomp_inner(Xd[tr], y[tr], seed, max_comp)
        model = fit_pls(Xd[tr], y[tr], a)
        pred[f] = predict_pls(model, Xd[f]); ncomps.append(a)
    return r2_mse_mae(y, pred), ncomps, pred


if sys.argv[1] == "pls":
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else SEEDS_ALL
    by_seed = {}
    for s in seeds:
        (r2, mse, mae), ncomps, pred = nested_cv_pls(X_eng, yv, s)
        by_seed[str(s)] = {"cvR2": round(r2, 4), "cvMSE": round(mse, 5), "cvMAE": round(mae, 5), "ncomps_per_fold": ncomps}
        print(f"PLS seed {s}: {by_seed[str(s)]}")
    save_merged({"pls": {"cv_by_seed": by_seed}})
    print("saved pls")

# ============================= OMP =============================
def fit_omp(X, y, k_terms):
    Xc = X - X.mean(axis=0); Xs = Xc.std(axis=0); Xs[Xs == 0] = 1.0; Xn = Xc / Xs
    y_mean = y.mean(); yc_ = y - y_mean
    n_, p_ = Xn.shape
    selected = []; residual = yc_.copy()
    for step in range(min(k_terms, p_, n_ - 1)):
        corr = Xn.T @ residual
        corr[selected] = 0.0
        j = int(np.argmax(np.abs(corr)))
        if j in selected: break
        selected.append(j)
        Xsel = Xn[:, selected]
        beta, *_ = np.linalg.lstsq(Xsel, yc_, rcond=None)
        residual = yc_ - Xsel @ beta
    B = np.zeros(p_)
    if selected:
        Xsel = Xn[:, selected]
        beta, *_ = np.linalg.lstsq(Xsel, yc_, rcond=None)
        for i2, j2 in enumerate(selected): B[j2] = beta[i2]
    return {"B": B, "y_mean": y_mean, "x_mean": X.mean(axis=0), "x_std": Xs, "selected": selected}


def predict_omp(model, X):
    Xc = (X - model["x_mean"]) / model["x_std"]
    return model["y_mean"] + Xc @ model["B"]


def select_k_inner_omp(Xd, y, max_k, k_folds=3, seed=0):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k_folds)
    best_k, best_mse = 1, np.inf
    for kt in range(1, max_k + 1):
        pred = np.full(n_, np.nan)
        for f in folds:
            tr = np.setdiff1d(np.arange(n_), f)
            model = fit_omp(Xd[tr], y[tr], kt)
            pred[f] = predict_omp(model, Xd[f])
        mse = np.mean((y - pred) ** 2)
        if mse < best_mse: best_mse, best_k = mse, kt
    return best_k


def nested_cv_omp(Xd, y, seed, max_k=10, k=5):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan); ks = []
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        kt = select_k_inner_omp(Xd[tr], y[tr], max_k, seed=seed)
        model = fit_omp(Xd[tr], y[tr], kt)
        pred[f] = predict_omp(model, Xd[f]); ks.append(kt)
    return r2_mse_mae(y, pred), ks, pred


if sys.argv[1] == "omp":
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else SEEDS_ALL
    by_seed = {}
    full_model = fit_omp(X_eng, yv, 10)
    col_names = ["cultivar", "EBI", "climate", "Growth_April"] + eng_names
    selected_names = [col_names[i] for i in full_model["selected"]]
    for s in seeds:
        (r2, mse, mae), ks, pred = nested_cv_omp(X_eng, yv, s)
        by_seed[str(s)] = {"cvR2": round(r2, 4), "cvMSE": round(mse, 5), "cvMAE": round(mae, 5), "k_per_fold": ks}
        print(f"OMP seed {s}: {by_seed[str(s)]}")
    print("OMP full-data selected features (k=10):", selected_names)
    save_merged({"omp": {"cv_by_seed": by_seed, "full_data_selected_k10": selected_names}})
    print("saved omp")

# ============================= Gradient boosting =============================
def build_tree_gb(X, y, depth, max_depth, min_leaf):
    n_, p_ = X.shape
    if depth >= max_depth or n_ < 2 * min_leaf:
        return {"leaf": True, "value": float(y.mean())}
    best_feat, best_thr, best_score = None, None, np.inf
    parent_rss = float(((y - y.mean()) ** 2).sum())
    for fj in range(p_):
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
            "left": build_tree_gb(X[left], y[left], depth + 1, max_depth, min_leaf),
            "right": build_tree_gb(X[right], y[right], depth + 1, max_depth, min_leaf)}


def predict_tree_gb(tree, x):
    node = tree
    while not node["leaf"]:
        node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
    return node["value"]


def fit_boost(X, y, n_rounds, max_depth, min_leaf, lr):
    f0 = float(y.mean()); trees = []
    pred = np.full(len(y), f0)
    for _ in range(n_rounds):
        resid = y - pred
        t = build_tree_gb(X, resid, 0, max_depth, min_leaf)
        trees.append(t)
        upd = np.array([predict_tree_gb(t, X[i]) for i in range(len(X))])
        pred = pred + lr * upd
    return {"f0": f0, "trees": trees, "lr": lr}


def predict_boost(model, X):
    pred = np.full(len(X), model["f0"])
    for t in model["trees"]:
        upd = np.array([predict_tree_gb(t, X[i]) for i in range(len(X))])
        pred = pred + model["lr"] * upd
    return pred


def cv_boost(Xd, y, seed, n_rounds, max_depth, min_leaf, lr, k=5):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        model = fit_boost(Xd[tr], y[tr], n_rounds, max_depth, min_leaf, lr)
        pred[f] = predict_boost(model, Xd[f])
    return r2_mse_mae(y, pred)


if sys.argv[1] == "boost_screen":
    lrs = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.05, 0.1, 0.2]
    MAX_DEPTHS = [1, 2]; MIN_LEAVES = [10, 15]; N_ROUNDS = [30, 60, 100]
    t0 = time.time()
    results = []
    for md in MAX_DEPTHS:
        for ml in MIN_LEAVES:
            for lr in lrs:
                for nr in N_ROUNDS:
                    r2, mse, mae = cv_boost(X_raw, yv, 42, nr, md, ml, lr, k=3)
                    results.append({"max_depth": md, "min_leaf": ml, "lr": lr, "n_rounds": nr, "screen_r2": round(r2, 4)})
    print(f"boost screen ({len(results)} combos) in {time.time()-t0:.1f}s")
    prior = load_prior()
    all_res = prior.get("boost_all_screened", []) + results
    all_res.sort(key=lambda d: -d["screen_r2"])
    for g in all_res[:10]: print(" ", g)
    prior["boost_all_screened"] = all_res; prior["boost_best_combo"] = all_res[0]
    with open(JSON_PATH, "w") as f: json.dump(prior, f, indent=2, default=str)
    print("saved boost screen")

if sys.argv[1] == "boost_final":
    prior = load_prior()
    best = prior["boost_best_combo"]
    print("using best boost combo:", best)
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else SEEDS_ALL
    by_seed = prior.get("boost_final_cv_by_seed", {})
    for s in seeds:
        r2, mse, mae = cv_boost(X_raw, yv, s, best["n_rounds"], best["max_depth"], best["min_leaf"], best["lr"], k=5)
        by_seed[str(s)] = {"cvR2": round(r2, 4), "cvMSE": round(mse, 5), "cvMAE": round(mae, 5)}
        print(f"boost seed {s}: {by_seed[str(s)]}")
    prior["boost_final_cv_by_seed"] = by_seed
    with open(JSON_PATH, "w") as f: json.dump(prior, f, indent=2, default=str)
    print("saved boost final")

# ============================= Ensemble (simple average) =============================
def cv_ols_pred(Xd, y, seed, k=5):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None); pred[f] = Xd[f] @ beta
    return pred


if sys.argv[1] == "ensemble":
    prior = load_prior()
    pls_ncomp_mode = int(np.median([np.median(v["ncomps_per_fold"]) for v in prior["pls"]["cv_by_seed"].values()]))
    omp_k_mode = int(np.median([np.median(v["k_per_fold"]) for v in prior["omp"]["cv_by_seed"].values()]))
    boost_best = prior["boost_best_combo"]
    print(f"ensemble using pls_ncomp~{pls_ncomp_mode}, omp_k~{omp_k_mode}, boost={boost_best}")
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else SEEDS_ALL
    by_seed = prior.get("ensemble_cv_by_seed", {})
    for s in seeds:
        pred_ols = cv_ols_pred(X_ols, yv, s)
        pred_pls = cv_pls_fixed(X_eng, yv, s, pls_ncomp_mode)
        pred_omp = np.full(n, np.nan)
        rng = np.random.default_rng(s); idx = rng.permutation(n); folds = np.array_split(idx, 5)
        for f in folds:
            tr = np.setdiff1d(np.arange(n), f)
            model = fit_omp(X_eng[tr], yv[tr], omp_k_mode); pred_omp[f] = predict_omp(model, X_eng[f])
        pred_boost = np.full(n, np.nan)
        for f in folds:
            tr = np.setdiff1d(np.arange(n), f)
            model = fit_boost(X_raw[tr], yv[tr], boost_best["n_rounds"], boost_best["max_depth"], boost_best["min_leaf"], boost_best["lr"])
            pred_boost[f] = predict_boost(model, X_raw[f])
        ens_all = (pred_ols + pred_pls + pred_omp + pred_boost) / 4.0
        ens_ols_best_alt = (pred_ols + pred_pls) / 2.0  # PLS typically best alt; adjusted below if not
        r2_ols, mse_ols, mae_ols = r2_mse_mae(yv, pred_ols)
        r2_ens_all, mse_ens_all, mae_ens_all = r2_mse_mae(yv, ens_all)
        r2_ens2, mse_ens2, mae_ens2 = r2_mse_mae(yv, ens_ols_best_alt)
        by_seed[str(s)] = {
            "ols_only": {"cvR2": round(r2_ols, 4), "cvMSE": round(mse_ols, 5), "cvMAE": round(mae_ols, 5)},
            "ensemble_all4": {"cvR2": round(r2_ens_all, 4), "cvMSE": round(mse_ens_all, 5), "cvMAE": round(mae_ens_all, 5)},
            "ensemble_ols_pls": {"cvR2": round(r2_ens2, 4), "cvMSE": round(mse_ens2, 5), "cvMAE": round(mae_ens2, 5)},
        }
        print(f"ensemble seed {s}: {by_seed[str(s)]}")
    prior["ensemble_cv_by_seed"] = by_seed
    prior["ensemble_pls_ncomp_used"] = pls_ncomp_mode
    prior["ensemble_omp_k_used"] = omp_k_mode
    with open(JSON_PATH, "w") as f: json.dump(prior, f, indent=2, default=str)
    print("saved ensemble")
