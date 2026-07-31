#!/usr/bin/env python3
"""
predicted_yield_rf_importance.py: per follow-up request, ranks the 34
engineered remote-sensing candidates (plus the 4 base features, 38 total)
by their contribution to the tuned Random Forest model from section 63's
"Random Forest, + 34 engineered candidates" row (85.4% CV R^2 on
predicted_Yield). Reuses the exact data pipeline, feature set, and tuned
hyperparameters (max_depth=6, min_leaf=10, mtry=20) already established in
predicted_yield_random_forest.py.

Importance measure: split-based RSS reduction (the standard Random Forest
"Gini-style" importance for regression trees), accumulated per feature
across every split in every tree, then normalized to sum to 100%. This is
computed on the model refit on the FULL n=1793 sample (not per-CV-fold),
since the goal here is to describe what the fitted model actually uses,
not to re-estimate predictive accuracy (already reported honestly via CV
in section 63). Averaged over several seeds for stability, since a single
seed's bagged forest can have noisy per-feature importance.

Selectable via argv: `python3 predicted_yield_rf_importance.py <seeds_csv> <n_trees>`

-> Results_Analysis/08_UEF53_Rerun/Predicted_Yield_RF_Importance.json
"""
import os, sys, json, time
import numpy as np
import openpyxl
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
JSON_PATH = os.path.join(OUT, "Predicted_Yield_RF_Importance.json")

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

X_eng = np.column_stack([uef, ebi, clim, ga] + [feat_cols[n_] for n_ in ENG_NAMES])
assert X_eng.shape[1] == len(ALL_38_NAMES)

QUANT_CAP = 24


def candidate_thresholds(col):
    vals = np.unique(col)
    if len(vals) <= QUANT_CAP + 1:
        return (vals[:-1] + vals[1:]) / 2.0
    qs = np.linspace(0, 100, QUANT_CAP + 2)[1:-1]
    return np.unique(np.percentile(col, qs))


def build_tree_track_importance(X, y, depth, max_depth, min_leaf, rng, mtry, importance):
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
    # record this split's contribution: RSS reduction, weighted by the
    # fraction of the full sample reaching this node (standard weighting
    # so splits near the root, which affect more rows, count more).
    reduction = (parent_rss - best_score) * (n_ / len(yv))
    importance[best_feat] = importance.get(best_feat, 0.0) + reduction
    left = X[:, best_feat] <= best_thr; right = ~left
    return {"leaf": False, "feat": int(best_feat), "thr": float(best_thr),
            "left": build_tree_track_importance(X[left], y[left], depth + 1, max_depth, min_leaf, rng, mtry, importance),
            "right": build_tree_track_importance(X[right], y[right], depth + 1, max_depth, min_leaf, rng, mtry, importance)}


MAX_DEPTH, MIN_LEAF, MTRY = 6, 10, 20  # tuned combo from screen_eng_best (section 63 RF update)

seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [42, 1, 7]
n_trees = int(sys.argv[2]) if len(sys.argv) > 2 else 100

prior = {}
if os.path.exists(JSON_PATH):
    prior = json.load(open(JSON_PATH))
per_seed_pct = prior.get("per_seed_pct", {})  # seed(str) -> {feature: pct}

t0 = time.time()
for s in seeds:
    if str(s) in per_seed_pct:
        continue  # already computed in a prior call
    rng = np.random.default_rng(s)
    importance = {}
    for _ in range(n_trees):
        boot_idx = rng.integers(0, n, size=n)
        build_tree_track_importance(X_eng[boot_idx], yv[boot_idx], 0, MAX_DEPTH, MIN_LEAF, rng, MTRY, importance)
    total = sum(importance.values())
    pct = {ALL_38_NAMES[k]: (v / total * 100.0) for k, v in importance.items()}
    per_seed_pct[str(s)] = pct
    print(f"seed {s} done ({time.time()-t0:.1f}s elapsed)", flush=True)
    # save incrementally so a timeout mid-run doesn't lose completed seeds
    prior["per_seed_pct"] = per_seed_pct
    prior["n_trees_per_seed"] = n_trees
    prior["max_depth"] = MAX_DEPTH; prior["min_leaf"] = MIN_LEAF; prior["mtry"] = MTRY
    prior["n"] = n
    with open(JSON_PATH, "w") as f:
        json.dump(prior, f, indent=2, default=str)

# average pct across all seeds computed so far (features absent from a given
# seed's importance get 0 for that seed, i.e. it was never split on)
avg = {}
for name in ALL_38_NAMES:
    vals = [d.get(name, 0.0) for d in per_seed_pct.values()]
    avg[name] = float(np.mean(vals)) if vals else 0.0
total_avg = sum(avg.values())
avg = {k: (v / total_avg * 100.0 if total_avg else 0.0) for k, v in avg.items()}  # renormalize to 100%
ranked = sorted(avg.items(), key=lambda kv: -kv[1])

prior["seeds_run"] = sorted(int(s) for s in per_seed_pct.keys())
prior["importance_pct_ranked"] = ranked
with open(JSON_PATH, "w") as f:
    json.dump(prior, f, indent=2, default=str)

print(f"\ntop 10 (averaged over {len(per_seed_pct)} seeds):")
for name, pct in ranked[:10]:
    print(f"  {name}: {pct:.2f}%")
print("saved")
