#!/usr/bin/env python3
"""
predicted_yield_random_forest.py: per follow-up ("try RF on the last
section, see if it improves the results"), tests Random Forest against the
predicted_Yield appendix (section 63). Reuses the exact same data pipeline
and three feature sets as predicted_yield_feature_search.py:
  (a) base 4 features (cultivar, EBI, climate, Growth_April)
  (b) base 4 + the 34-column remote-sensing engineered bank (38 total)
  (c) base 4 + engineered bank + the 6 "circular" physiology fields (44 total)
so RF's ceiling can be compared directly against the OLS numbers already
logged: (a)/(b) = 84.4% CV R^2, (c) = 93.5% CV R^2.

n=1793 here (much larger than the 167 used for the measured-yield model),
so unlike sections 52/53/56/58, overfitting risk is low and RF has a
realistic chance to add value if genuine nonlinearity exists.

Selectable via argv: `python3 predicted_yield_random_forest.py <stage> [arg2]`
  stage in {screen_base, screen_eng, screen_phys, final}
  final: arg2 = which featureset's best combo to finalize {base, eng, phys}, arg3 = seeds csv

-> Results_Analysis/08_UEF53_Rerun/Predicted_Yield_Random_Forest.json
"""
import os, sys, json, time
import numpy as np
import openpyxl
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
JSON_PATH = os.path.join(OUT, "Predicted_Yield_Random_Forest.json")

M = load_master(); cultivar = [r.get("cultivar") for r in M]


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
ENG_NAMES = CANOPY_NAMES + V6_NAMES + COMPOSITE_NAMES
PHYS_NAMES = ["SWP_April", "SWP_May_June", "SWP_June", "Growth_May_June", "Growth_June", "CNC_June"]


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


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


rows = []
for j in range(len(M)):
    if cultivar[j] not in ("UEF", "53"): continue
    for yr in (2022, 2023):
        py = M[j].get(f"predicted_Yield_{yr}")
        if py is None: continue
        ga = M[j].get(f"Growth_April_{yr}")
        cp = M[j].get(f"Chill_Portions_{yr}")
        if ga is None or cp is None: continue
        feats = build_all_features(j, yr)
        if feats is None: continue
        phys_vals = {name: M[j].get(f"{name}_{yr}") for name in PHYS_NAMES}
        if any(v is None for v in phys_vals.values()): continue
        feats = dict(feats); feats.update(phys_vals)
        rows.append((j, yr, float(py), feats))
n = len(rows)
print(f"n={n} tree-years (UEF + cultivar 53, 2022-2023)")

yv = np.array([r[2] for r in rows])
uef = np.array([1.0 if cultivar[r[0]] == "UEF" else 0.0 for r in rows])
ebi = zc([M[r[0]].get(f"EBI_Norm_{r[1]}") for r in rows])
clim = zc([M[r[0]].get(f"Chill_Portions_{r[1]}") for r in rows])
ga = zc([M[r[0]].get(f"Growth_April_{r[1]}") for r in rows])
feat_cols = {name: zc([r[3][name] for r in rows]) for name in ENG_NAMES + PHYS_NAMES}

X_base = np.column_stack([uef, ebi, clim, ga])
X_eng = np.column_stack([uef, ebi, clim, ga] + [feat_cols[n_] for n_ in ENG_NAMES])
X_phys = np.column_stack([uef, ebi, clim, ga] + [feat_cols[n_] for n_ in ENG_NAMES] + [feat_cols[n_] for n_ in PHYS_NAMES])
FEATURESETS = {"base": X_base, "eng": X_eng, "phys": X_phys}
print(f"X_base p={X_base.shape[1]}  X_eng p={X_eng.shape[1]}  X_phys p={X_phys.shape[1]}")


def r2_mse_mae(y_true, y_pred):
    resid = y_true - y_pred
    tss = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / tss), float(np.mean(resid ** 2)), float(np.mean(np.abs(resid)))


QUANT_CAP = 24  # histogram-style candidate thresholds per node (LightGBM-style binning);
# n here (~1793, up to 4-fold-shrunk within CV) is far larger than the n=167 measured-yield
# scripts, so full unique-value threshold enumeration is O(n^2) per node and too slow;
# quantile binning keeps split quality effectively identical while making this tractable.


def candidate_thresholds(col):
    vals = np.unique(col)
    if len(vals) <= QUANT_CAP + 1:
        return (vals[:-1] + vals[1:]) / 2.0
    qs = np.linspace(0, 100, QUANT_CAP + 2)[1:-1]
    thr = np.unique(np.percentile(col, qs))
    return thr


def build_tree(X, y, depth, max_depth, min_leaf, rng, mtry):
    n_, p_ = X.shape
    if depth >= max_depth or n_ < 2 * min_leaf:
        return {"leaf": True, "value": float(y.mean())}
    feat_idx = rng.choice(p_, size=min(mtry, p_), replace=False)
    best_feat, best_thr, best_score = None, None, np.inf
    parent_rss = float(((y - y.mean()) ** 2).sum())
    for fj in feat_idx:
        col = X[:, fj]
        thresholds = candidate_thresholds(col)
        if len(thresholds) < 1: continue
        for thr in thresholds:
            left = col <= thr; right = ~left
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


STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
GRID = {
    "base": {"max_depth": [3, 4, 5, 6], "min_leaf": [10, 20, 30], "mtry": [2, 3, 4]},
    "eng": {"max_depth": [3, 4, 5, 6], "min_leaf": [10, 20, 30], "mtry": [4, 6, 10]},
    "phys": {"max_depth": [3, 4, 5, 6], "min_leaf": [10, 20, 30], "mtry": [4, 8, 14]},
}

if STAGE.startswith("screen_"):
    fs = STAGE.split("_", 1)[1]
    Xd = FEATURESETS[fs]
    g = GRID[fs]
    mds_this_call = [int(m) for m in sys.argv[2].split(",")] if len(sys.argv) > 2 else g["max_depth"]
    mls_this_call = [int(m) for m in sys.argv[3].split(",")] if len(sys.argv) > 3 else g["min_leaf"]
    mtrys_this_call = [int(m) for m in sys.argv[4].split(",")] if len(sys.argv) > 4 else g["mtry"]
    t0 = time.time()
    prior = load_prior()
    key = f"screen_{fs}_all"
    all_res = prior.get(key, [])
    seen = {(r["max_depth"], r["min_leaf"], r["mtry"]) for r in all_res}
    for md in mds_this_call:
        for ml in mls_this_call:
            for mt in mtrys_this_call:
                if (md, ml, mt) in seen: continue
                tt = time.time()
                r2, mse, mae = cv_forest(Xd, yv, 42, n_trees=15, max_depth=md, min_leaf=ml, mtry=mt, k=3)
                rec = {"max_depth": md, "min_leaf": ml, "mtry": mt, "screen_r2": round(r2, 4)}
                all_res.append(rec); seen.add((md, ml, mt))
                print(f"  {rec}  ({time.time()-tt:.1f}s)", flush=True)
                all_res.sort(key=lambda d: -d["screen_r2"])
                prior[key] = all_res
                prior[f"screen_{fs}_best"] = all_res[0]
                with open(JSON_PATH, "w") as f: json.dump(prior, f, indent=2, default=str)
    print(f"screen_{fs} chunk done in {time.time()-t0:.1f}s, {len(all_res)} combos total so far")
    for r in all_res[:5]: print(" best:", r)

if STAGE == "final":
    fs = sys.argv[2] if len(sys.argv) > 2 else "base"
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [42, 1, 7, 13, 99, 2024]
    n_trees_final = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    prior = load_prior()
    best = prior[f"screen_{fs}_best"]
    print(f"using best {fs} combo: {best}  n_trees={n_trees_final}")
    Xd = FEATURESETS[fs]
    key = f"final_{fs}_cv_by_seed"
    by_seed = prior.get(key, {})
    for s in seeds:
        tt = time.time()
        r2, mse, mae = cv_forest(Xd, yv, s, n_trees=n_trees_final, max_depth=best["max_depth"], min_leaf=best["min_leaf"], mtry=best["mtry"], k=5)
        by_seed[str(s)] = {"cvR2": round(r2, 4), "cvMSE": round(mse, 5), "cvMAE": round(mae, 5)}
        print(f"  seed {s}: {by_seed[str(s)]}  ({time.time()-tt:.1f}s)", flush=True)
        prior[key] = by_seed
        with open(JSON_PATH, "w") as f: json.dump(prior, f, indent=2, default=str)
    prior[key] = by_seed
    r2s = [v["cvR2"] for v in by_seed.values()]
    prior[f"final_{fs}_cvR2_range"] = [round(min(r2s), 4), round(max(r2s), 4)] if r2s else None
    prior[f"final_{fs}_cvR2_mean"] = round(float(np.mean(r2s)), 4) if r2s else None
    with open(JSON_PATH, "w") as f: json.dump(prior, f, indent=2, default=str)
    print(f"\nRF {fs}: cvR2 so far {prior[f'final_{fs}_cvR2_range']} mean {prior[f'final_{fs}_cvR2_mean']}")
    print("saved")
