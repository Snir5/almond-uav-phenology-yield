#!/usr/bin/env python3
"""
predicted_yield_rf_permutation_importance.py: replaces the split-based
("Gini-style") importance in predicted_yield_rf_importance.py with
permutation importance, per follow-up discussion. Split-based importance is
biased toward high-cardinality/continuous features (a feature can only be
split on as many times as it has distinct values worth splitting at,
regardless of how much real signal it carries); this was caught concretely
in this project when "climate" (Chill_Portions) turned out to be exactly
2-valued here (35 for 2022, 25 for 2023, since it is station-level) yet
still dominated split-based importance (61.9%) while cultivar (also
2-valued, one-hot) scored almost nothing (0.17%). The real reason climate
dominates is that its one available split separates predicted_Yield by a
much larger gap (2.43 kg between years) than cultivar's one available split
(0.21 kg between UEF and cultivar 53), not cardinality bias, but split-based
importance cannot distinguish "few splits, huge effect" from "few splits,
no effect" as cleanly as permutation importance can.

Permutation importance: for each of the 5 CV folds (already the honest,
held-out evaluation used throughout this project), after training the fold's
forest, shuffle one feature at a time WITHIN the test fold only, re-predict
with the SAME trained trees (no retraining), and measure how much CV R^2
drops. A feature that matters a lot will hurt a lot when shuffled; a feature
the model barely uses will barely move the score, regardless of how many
distinct values it has. Averaged over multiple seeds for stability.

Selectable via argv: `python3 predicted_yield_rf_permutation_importance.py <seeds_csv> <n_trees>`

-> Results_Analysis/08_UEF53_Rerun/Predicted_Yield_RF_Permutation_Importance.json
"""
import os, sys, json, time
import numpy as np
import openpyxl
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
JSON_PATH = os.path.join(OUT, "Predicted_Yield_RF_Permutation_Importance.json")

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
BASE_NAMES = ["cultivar(UEF)", "EBI", "climate", "Growth_April"]
ALL_38_NAMES = BASE_NAMES + ENG_NAMES


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
        rows.append((j, yr, float(py), feats))
n = len(rows)
print(f"n={n} tree-years (UEF + cultivar 53, 2022-2023)")

yv = np.array([r[2] for r in rows])
uef = np.array([1.0 if cultivar[r[0]] == "UEF" else 0.0 for r in rows])
ebi = zc([M[r[0]].get(f"EBI_Norm_{r[1]}") for r in rows])
clim = zc([M[r[0]].get(f"Chill_Portions_{r[1]}") for r in rows])
ga = zc([M[r[0]].get(f"Growth_April_{r[1]}") for r in rows])
feat_cols = {name: zc([r[3][name] for r in rows]) for name in ENG_NAMES}

X_eng = np.column_stack([uef, ebi, clim, ga] + [feat_cols[n_] for n_ in ENG_NAMES])
assert X_eng.shape[1] == len(ALL_38_NAMES)

QUANT_CAP = 24


def candidate_thresholds(col):
    vals = np.unique(col)
    if len(vals) <= QUANT_CAP + 1:
        return (vals[:-1] + vals[1:]) / 2.0
    qs = np.linspace(0, 100, QUANT_CAP + 2)[1:-1]
    return np.unique(np.percentile(col, qs))


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


def r2_of(y_true, y_pred):
    resid = y_true - y_pred
    tss = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / tss)


MAX_DEPTH, MIN_LEAF, MTRY = 6, 10, 20  # tuned combo from screen_eng_best (section 63 RF update)

seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [42]
n_trees = int(sys.argv[2]) if len(sys.argv) > 2 else 25
k = 5

prior = {}
if os.path.exists(JSON_PATH):
    prior = json.load(open(JSON_PATH))
per_seed = prior.get("per_seed_drop", {})  # seed(str) -> {feature: mean_R2_drop}

t0 = time.time()
for s in seeds:
    if str(s) in per_seed:
        continue
    rng = np.random.default_rng(s); idx = rng.permutation(n); folds = np.array_split(idx, k)
    baseline_pred = np.full(n, np.nan)
    drop_sum = {name: 0.0 for name in ALL_38_NAMES}
    perm_rng = np.random.default_rng(s + 10000)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        trees = fit_forest(X_eng[tr], yv[tr], n_trees, MAX_DEPTH, MIN_LEAF, MTRY, s)
        base_pred_fold = predict_forest(trees, X_eng[f])
        baseline_pred[f] = base_pred_fold
        base_r2_fold = r2_of(yv[f], base_pred_fold)
        for fj, name in enumerate(ALL_38_NAMES):
            X_perm = X_eng[f].copy()
            X_perm[:, fj] = perm_rng.permutation(X_perm[:, fj])
            perm_pred = predict_forest(trees, X_perm)
            perm_r2_fold = r2_of(yv[f], perm_pred)
            # weight each fold's drop by fold size so folds contribute proportionally
            drop_sum[name] += (base_r2_fold - perm_r2_fold) * len(f)
    overall_base_r2 = r2_of(yv, baseline_pred)
    drop_pct = {name: drop_sum[name] / n for name in ALL_38_NAMES}  # size-weighted mean R2 drop
    per_seed[str(s)] = {"overall_cvR2": overall_base_r2, "drop": drop_pct}
    print(f"seed {s} done, baseline CV R2={overall_base_r2:.4f} ({time.time()-t0:.1f}s elapsed)", flush=True)
    prior["per_seed_drop"] = per_seed
    prior["n_trees_per_seed"] = n_trees
    prior["max_depth"] = MAX_DEPTH; prior["min_leaf"] = MIN_LEAF; prior["mtry"] = MTRY
    prior["n"] = n
    with open(JSON_PATH, "w") as f_:
        json.dump(prior, f_, indent=2, default=str)

avg_drop = {}
avg_base_r2 = float(np.mean([d["overall_cvR2"] for d in per_seed.values()])) if per_seed else None
for name in ALL_38_NAMES:
    vals = [d["drop"].get(name, 0.0) for d in per_seed.values()]
    avg_drop[name] = float(np.mean(vals)) if vals else 0.0
ranked = sorted(avg_drop.items(), key=lambda kv: -kv[1])

prior["seeds_run"] = sorted(int(s) for s in per_seed.keys())
prior["avg_baseline_cvR2"] = avg_base_r2
prior["r2_drop_ranked"] = ranked
with open(JSON_PATH, "w") as f_:
    json.dump(prior, f_, indent=2, default=str)

print(f"\nbaseline CV R2 (avg over {len(per_seed)} seeds): {avg_base_r2:.4f}")
print("top 10 by permutation R2 drop:")
for name, d in ranked[:10]:
    print(f"  {name}: {d*100:+.3f} pts")
print("saved")
