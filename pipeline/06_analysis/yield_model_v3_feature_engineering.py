#!/usr/bin/env python3
"""
yield_model_v3_feature_engineering.py: engineers new candidate features from the same
GPKG physiology data used in sections 21-22, rather than just testing the raw fields:

  1. Season-trajectory features: SWP_change = SWP_June - SWP_April (does water status
     worsen or recover across the season) and Growth_change = Growth_June - Growth_April
     (does vegetative growth accelerate or decelerate), the DYNAMICS the raw snapshots
     do not capture on their own.
  2. A PCA composite "vigor score": the 7 raw physiology fields are correlated with each
     other (they are all facets of the same underlying tree condition), so their first two
     principal components are computed (via stats_utils.pca, SVD-based) as compact,
     decorrelated composite features, an alternative to testing 7 raw + 7 interaction
     terms one at a time.
  3. TG_clusters / SWP_clusters (categorical cluster IDs from the original,
     already-invalidated predicted_Yield model). TG_clusters turns out to be CONSTANT
     across all of Plot A within a year (2022 all cluster 2, 2023 all cluster 3), so it
     carries no tree-level information here and is dropped; SWP_clusters does vary
     (2022: 34 vs 1076 trees; 2023: 213 vs 897) and is tested as a binary dummy.
  4. DEM_Jul2024 (elevation, already in the master, never tested as a yield feature),
     a bonus, cheap addition (drainage/cold-air-pooling/soil-depth proxy).

Same rigor as section 22: forward selection scored on 5-fold cross-validated R^2 (never
in-sample fit), then a 5-independent-fold-split robustness check on whatever is selected,
since a single fold split can produce a "winner" that is fold-specific noise.
-> Results_Analysis/08_UEF53_Rerun/Yield_Model_v3_FeatureEngineering.json + figure.
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict, Counter
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import pca

BASE = find_thesis_root()
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
CV_SEED = 42
PHYS_FIELDS = ["SWP_April", "Growth_April", "SWP_MayJune", "Growth_MayJune", "CNC_June", "SWP_June", "Growth_June"]
PHYS_GPKG_COLS = ["SWP_April", "Growth__April", "SWP_May.June", "Growth_May.June", "CNC_June", "SWP_June", "Growth_June"]
MATCH_THRESHOLD_M = 3.0


def gpkg_geom_centroid(blob):
    flags = blob[3]
    envelope_ind = (flags >> 1) & 0x07
    env_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope_ind]
    wkb = blob[8 + env_bytes:]
    endian = "<" if wkb[0] == 1 else ">"
    pos = 1
    geom_type = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
    base_type = geom_type % 1000
    xs, ys = [], []
    if base_type == 6:
        npoly = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
        for _ in range(npoly):
            pos += 1
            struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
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
    if not xs: return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


M = load_master()
mx = np.array([r.get("X_UTM") for r in M], float); my = np.array([r.get("Y_UTM") for r in M], float)
cultivar = [r.get("cultivar") for r in M]


def load_phys(year):
    con = sqlite3.connect(os.path.join(GDIR, f"Yield_with_clustering_{year}.gpkg")); cur = con.cursor()
    cols_sql = ", ".join(f'"{c}"' for c in PHYS_GPKG_COLS)
    cur.execute(f'SELECT geom, Plot, cultivar, TG_clusters, SWP_clusters, {cols_sql} FROM "Yield_with_clustering_{year}" WHERE Plot="A"')
    matched = {}
    for row in cur.fetchall():
        geom, plot, cult, tg, swpc = row[0], row[1], row[2], row[3], row[4]
        c = gpkg_geom_centroid(geom)
        if c is None: continue
        d = np.sqrt((mx - c[0]) ** 2 + (my - c[1]) ** 2); j = int(np.argmin(d))
        if d[j] > MATCH_THRESHOLD_M or M[j].get("cultivar") != cult: continue
        vals = dict(zip(PHYS_FIELDS, row[5:]))
        rec = {f: (float(vals[f]) if vals[f] is not None else np.nan) for f in PHYS_FIELDS}
        rec["TG_clusters"] = tg; rec["SWP_clusters"] = swpc
        matched[j] = rec
    con.close()
    return matched


PHYS = {2022: load_phys(2022), 2023: load_phys(2023)}
print("TG_clusters distinct values per year:",
      {y: sorted(set(v["TG_clusters"] for v in PHYS[y].values())) for y in [2022, 2023]})
print("SWP_clusters distribution per year:",
      {y: dict(Counter(v["SWP_clusters"] for v in PHYS[y].values())) for y in [2022, 2023]})

yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc:
    ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"])
    coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)


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
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


def r2_cv(Xd, y, k=5, seed=None):
    n = len(y); local_rng = np.random.default_rng(seed if seed is not None else CV_SEED)
    idx = local_rng.permutation(n); folds = np.array_split(idx, k)
    pred = np.full(n, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None)
        pred[f] = Xd[f] @ beta
    ss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum()
    cv = 1 - ss / tss
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None); ins = Xd @ beta
    r2 = 1 - ((y - ins) ** 2).sum() / tss
    return round(float(r2), 3), round(float(cv), 3)


def forward_select(base_cols, candidate_cols, y, seed=None):
    selected = []
    X = np.column_stack(list(base_cols.values()))
    _, cur_cv = r2_cv(X, y, seed=seed)
    history = [{"step": 0, "added": None, "cvR2": cur_cv}]
    remaining = list(candidate_cols.keys())
    improved = True
    while improved and remaining:
        improved = False; best = None; best_cv = cur_cv
        for name in remaining:
            Xtry = np.column_stack(list(base_cols.values()) + [candidate_cols[n] for n in selected] + [candidate_cols[name]])
            _, cv = r2_cv(Xtry, y, seed=seed)
            if cv > best_cv + 1e-6:
                best_cv = cv; best = name
        if best is not None:
            selected.append(best); remaining.remove(best); cur_cv = best_cv
            history.append({"step": len(selected), "added": best, "cvR2": round(cur_cv, 3)})
            improved = True
    return selected, history


# ================= build the pooled 2022-2023 dataset with engineered features =================
rowsP = [(t, j, yr) for t, j in ymap.items() for yr in [2022, 2023]
         if ymeas[t].get(yr) is not None and M[j].get(f"EBI_Norm_{yr}") is not None and j in PHYS[yr]]
yP = np.array([ymeas[t][yr] for t, j, yr in rowsP]); nP = len(yP)
uefP = np.array([1.0 if cultivar[j] == "UEF" else 0.0 for t, j, yr in rowsP])
ebiP = zc([M[j].get(f"EBI_Norm_{yr}") for t, j, yr in rowsP])
cpP = zc([CP[yr] for t, j, yr in rowsP])
oneP = np.ones(nP)
base_cols_P = {"intercept": oneP, "cultivar": uefP, "EBI": ebiP, "EBIxcultivar": ebiP * uefP, "climate": cpP}

# raw physiology matrix (for PCA); stats_utils.pca standardizes internally, so pass raw values
raw_mat = np.column_stack([[PHYS[yr][j][f] for t, j, yr in rowsP] for f in PHYS_FIELDS])
scores, Vt, evr, pca_mu, pca_sd = pca(raw_mat, n_components=3)
print("PCA explained variance ratio (first 3 PCs):", evr)
print("PCA loadings (fields x PCs):")
for i, f in enumerate(PHYS_FIELDS):
    print(f"  {f}: PC1={Vt[0, i]:+.3f} PC2={Vt[1, i]:+.3f}")

engineered_cols = {}
# 1. trajectory / change features
swp_change = zc([PHYS[yr][j]["SWP_June"] - PHYS[yr][j]["SWP_April"] for t, j, yr in rowsP])
growth_change = zc([PHYS[yr][j]["Growth_June"] - PHYS[yr][j]["Growth_April"] for t, j, yr in rowsP])
engineered_cols["SWP_change_JuneMinusApril"] = swp_change
engineered_cols["SWP_change_x_cultivar"] = swp_change * uefP
engineered_cols["Growth_change_JuneMinusApril"] = growth_change
engineered_cols["Growth_change_x_cultivar"] = growth_change * uefP
# 2. PCA composite vigor scores
pc1 = zc(scores[:, 0]); pc2 = zc(scores[:, 1])
engineered_cols["Physiology_PC1"] = pc1
engineered_cols["Physiology_PC1_x_cultivar"] = pc1 * uefP
engineered_cols["Physiology_PC2"] = pc2
engineered_cols["Physiology_PC2_x_cultivar"] = pc2 * uefP
# 3. SWP_clusters (categorical, binary dummy; TG_clusters dropped, constant within Plot A)
swpc = np.array([1.0 if PHYS[yr][j]["SWP_clusters"] == 1 else 0.0 for t, j, yr in rowsP])
engineered_cols["SWP_cluster1_dummy"] = zc(swpc)
engineered_cols["SWP_cluster1_dummy_x_cultivar"] = zc(swpc) * uefP
# 4. DEM (already in master, bonus). Some trees lack a DEM value; mean-impute (-> 0 after
# z-scoring) rather than drop rows, since DEM is a secondary bonus candidate here, not
# central; noted explicitly, not hidden.
dem_raw = np.array([M[j].get("DEM_Jul2024") for t, j, yr in rowsP], float)
n_dem_missing = int(np.isnan(dem_raw).sum())
dem_raw[np.isnan(dem_raw)] = np.nanmean(dem_raw)
dem = zc(dem_raw)
engineered_cols["DEM"] = dem
engineered_cols["DEM_x_cultivar"] = dem * uefP
print(f"DEM: {n_dem_missing}/{nP} missing, mean-imputed")

selP, histP = forward_select(base_cols_P, engineered_cols, yP)
XfinalP = np.column_stack(list(base_cols_P.values()) + [engineered_cols[n] for n in selP])
r2_P, cv_P = r2_cv(XfinalP, yP); r2_P_base, cv_P_base = r2_cv(np.column_stack(list(base_cols_P.values())), yP)

robustness_seeds = [1, 7, 13, 99, 2024]
robustness = []
for sd in robustness_seeds:
    s, h = forward_select(base_cols_P, engineered_cols, yP, seed=sd)
    _, cv_base_sd = r2_cv(np.column_stack(list(base_cols_P.values())), yP, seed=sd)
    if s:
        _, cv_final_sd = r2_cv(np.column_stack(list(base_cols_P.values()) + [engineered_cols[n] for n in s]), yP, seed=sd)
    else:
        cv_final_sd = cv_base_sd
    robustness.append({"seed": sd, "selected": s, "baseline_cvR2": cv_base_sd, "final_cvR2": cv_final_sd})
pick_counts = Counter(f for r in robustness for f in r["selected"])

RESULTS = {
    "note": "Engineered features from the same GPKG physiology data (sections 21-22): season "
            "trajectory/change, PCA composite vigor scores, SWP_clusters category, plus DEM as "
            "a bonus. TG_clusters excluded, constant across all of Plot A within each year "
            "(2022 all cluster 2, 2023 all cluster 3), carries no tree-level information here.",
    "TG_clusters_diagnostic": {str(y): sorted(set(v["TG_clusters"] for v in PHYS[y].values())) for y in [2022, 2023]},
    "SWP_clusters_diagnostic": {str(y): dict(Counter(v["SWP_clusters"] for v in PHYS[y].values())) for y in [2022, 2023]},
    "PCA_explained_variance_ratio_first3": [round(float(x), 3) for x in evr],
    "PCA_loadings_PC1_PC2": {f: {"PC1": round(float(Vt[0, i]), 3), "PC2": round(float(Vt[1, i]), 3)} for i, f in enumerate(PHYS_FIELDS)},
    "pooled_2022_2023_with_climate": {
        "n": nP, "baseline_features": list(base_cols_P.keys())[1:],
        "baseline_cvR2": cv_P_base, "baseline_R2": r2_P_base,
        "candidates_tested": list(engineered_cols.keys()),
        "forward_selection_history": histP, "selected_features": selP,
        "final_cvR2": cv_P, "final_R2": r2_P, "improvement_cvR2": round(cv_P - cv_P_base, 3),
    },
    "robustness_across_5_independent_CV_fold_splits": {
        "per_seed": robustness,
        "how_often_each_feature_selected_of_5": dict(pick_counts),
    },
}
print("\nEngineered-feature forward selection:", json.dumps(RESULTS["pooled_2022_2023_with_climate"], indent=2, default=str))
print("\nRobustness:", json.dumps(RESULTS["robustness_across_5_independent_CV_fold_splits"], indent=2, default=str))

# combine with section 22's Growth_April to see if an engineered feature adds anything ON TOP
ga_raw = zc([PHYS[yr][j]["Growth_April"] for t, j, yr in rowsP])
base_plus_ga = dict(base_cols_P); base_plus_ga["Growth_April"] = ga_raw
selP2, histP2 = forward_select(base_plus_ga, engineered_cols, yP)
_, cv_ga_base = r2_cv(np.column_stack(list(base_plus_ga.values())), yP)
Xfinal2 = np.column_stack(list(base_plus_ga.values()) + [engineered_cols[n] for n in selP2])
r2_2, cv_2 = r2_cv(Xfinal2, yP)
RESULTS["on_top_of_Growth_April"] = {
    "n": nP, "baseline_cvR2_with_GrowthApril": cv_ga_base,
    "forward_selection_history": histP2, "selected_features": selP2,
    "final_cvR2": cv_2, "improvement_cvR2": round(cv_2 - cv_ga_base, 3),
    "note": "does any engineered feature add value ON TOP of cultivar+EBIxcultivar+climate+Growth_April (the section-22 recommended model)?",
}
print("\nOn top of Growth_April:", json.dumps(RESULTS["on_top_of_Growth_April"], indent=2, default=str))

json.dump(RESULTS, open(os.path.join(OUT, "Yield_Model_v3_FeatureEngineering.json"), "w"), indent=2, default=str)

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
a = ax[0]
labels = list(engineered_cols.keys())
names_seq = ["base"] + [h["added"] or "" for h in histP[1:]]
steps = [h["cvR2"] for h in histP]
a.plot(range(len(steps)), steps, "o-", color="#16A085")
a.set_xticks(range(len(steps))); a.set_xticklabels(names_seq, rotation=35, ha="right", fontsize=7.5)
a.set_ylabel("5-fold CV R²"); a.set_title("A. Forward selection, engineered features\n(pooled 2022-23 + climate)", fontsize=9.5, fontweight="bold")
a.grid(alpha=.3)

a = ax[1]
names_seq2 = ["base(+GrowthApril)"] + [h["added"] or "" for h in histP2[1:]]
steps2 = [h["cvR2"] for h in histP2]
a.plot(range(len(steps2)), steps2, "o-", color="#8E44AD")
a.set_xticks(range(len(steps2))); a.set_xticklabels(names_seq2, rotation=35, ha="right", fontsize=7.5)
a.set_ylabel("5-fold CV R²"); a.set_title("B. On top of Growth_April\n(section-22 model + engineered features)", fontsize=9.5, fontweight="bold")
a.grid(alpha=.3)

a = ax[2]
if pick_counts:
    ks = list(pick_counts.keys()); vs = [pick_counts[k] for k in ks]
    a.barh(ks, vs, color="#C0392B")
    a.set_xlim(0, 5); a.set_xlabel("times selected out of 5 independent fold splits")
else:
    a.text(0.5, 0.5, "no engineered feature was\nrobustly selected in any split", ha="center", va="center", fontsize=10)
    a.set_xticks([]); a.set_yticks([])
a.set_title("C. Robustness: how often each\nengineered feature is picked", fontsize=9.5, fontweight="bold")
fig.suptitle("Feature-engineered candidates from GPKG physiology: trajectory, PCA composites, clusters, DEM", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Yield_Model_v3_FeatureEngineering.png"), dpi=140, bbox_inches="tight")
print("\nsaved figure")
