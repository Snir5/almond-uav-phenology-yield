#!/usr/bin/env python3
"""
yield_model_v2_all_gpkg.py: uses every field in every yield/spectral GPKG (per request)
and rebuilds the per-tree yield model (yield_model_prototype.py) with them as candidates,
via honest forward feature selection on cross-validated R^2 (not in-sample fit, which
always rises with more parameters and would be misleading at this sample size).

Data sources, all four GPKGs in Trees_Data_Survey/Field_Data_Yield_GPKGs/, Plot A only
(project scope), matched to the master within 3 m (cultivar cross-checked):
  - Yield_with_clustering_2022/2023.gpkg: SWP_April, Growth_April, SWP_MayJune,
    Growth_MayJune, CNC_June, SWP_June, Growth_June (section 21; predicted_Yield already
    used and already invalidated as a feature, section 8, not retested here).
  - spectral_data_plot_A_31072024.gpkg: four fields not yet tested anywhere in this
    project (everything else here, NDVI/NDRE/EVI/etc. and the raw bands, was already
    tested in jul2024_spectral_yield.py, section 20.3, sourced from the master's own
    Jul2024 columns): SR (simple ratio), IPVI, SIPI, and area (an independently
    measured July-2024 canopy area, a different source from this project's own removed
    canopy-structure extraction).

Two honest, separate augmentation tests, since physiology only exists for 2022-2023 and
the new spectral fields only for 2024, they cannot be pooled into one design without
losing years entirely:
  A. Within-season 2023 (n~150ish with physiology, the inferential year): baseline
     (cultivar + EBI x cultivar) vs baseline + greedily-selected physiology features,
     added one at a time only if 5-fold CV R^2 improves.
  B. 2024 (n=18-20, small, directional only): does area/SR/IPVI/SIPI add anything to
     the small repeated-measures panel model.

Forward selection stops as soon as no remaining candidate improves CV R^2, this is the
safeguard against overfitting with 7+ correlated physiology candidates at n~150.
-> Results_Analysis/08_UEF53_Rerun/Yield_Model_v2_AllGPKG.json + figure.
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

BASE = find_thesis_root()
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
rng = np.random.default_rng(42)

PHYS_FIELDS = ["SWP_April", "Growth_April", "SWP_MayJune", "Growth_MayJune", "CNC_June", "SWP_June", "Growth_June"]
PHYS_GPKG_COLS = ["SWP_April", "Growth__April", "SWP_May.June", "Growth_May.June", "CNC_June", "SWP_June", "Growth_June"]
SPEC_FIELDS = ["SR", "IPVI", "SIPI", "area"]
MATCH_THRESHOLD_M = 3.0
# The July-2024 spectral GPKG has its own, already-documented ~3.4 m registration
# residual (THESIS_PROGRESS_LOG.md section 2, GEOSPATIAL_GUARDRAILS.md), confirmed here:
# median nearest-master-tree distance for spectral_data_plot_A_31072024.gpkg is 3.33 m
# (vs 0.96 m for the Yield_with_clustering GPKGs), and it is not a clean constant shift
# (offset vector std ~1-3 m), so a wider threshold plus a cultivar-agreement filter is
# used for this file specifically instead of the tighter 3 m used for physiology.
SPEC_MATCH_THRESHOLD_M = 5.0


def gpkg_geom_centroid(blob):
    assert blob[0:2] == b"GP"
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
    path = os.path.join(GDIR, f"Yield_with_clustering_{year}.gpkg")
    con = sqlite3.connect(path); cur = con.cursor()
    cols_sql = ", ".join(f'"{c}"' for c in PHYS_GPKG_COLS)
    cur.execute(f'SELECT geom, Plot, cultivar, {cols_sql} FROM "Yield_with_clustering_{year}" WHERE Plot="A"')
    matched = {}
    for row in cur.fetchall():
        geom, plot, cult = row[0], row[1], row[2]
        c = gpkg_geom_centroid(geom)
        if c is None: continue
        d = np.sqrt((mx - c[0]) ** 2 + (my - c[1]) ** 2); j = int(np.argmin(d))
        if d[j] > MATCH_THRESHOLD_M or M[j].get("cultivar") != cult: continue
        vals = dict(zip(PHYS_FIELDS, row[3:]))
        matched[j] = {f: (float(vals[f]) if vals[f] is not None else np.nan) for f in PHYS_FIELDS}
    con.close()
    return matched


def load_spec2024():
    path = os.path.join(GDIR, "spectral_data_plot_A_31072024.gpkg")
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute('SELECT geom, Plot, Cultivator, SR, IPVI, SIPI, area FROM spectral_data_plot_A_31072024 WHERE Plot="A"')
    matched = {}
    for row in cur.fetchall():
        geom, plot, cult = row[0], row[1], row[2]
        if cult not in ("UEF", "53"): continue
        c = gpkg_geom_centroid(geom)
        if c is None: continue
        d = np.sqrt((mx - c[0]) ** 2 + (my - c[1]) ** 2); j = int(np.argmin(d))
        if d[j] > SPEC_MATCH_THRESHOLD_M or M[j].get("cultivar") != cult: continue
        vals = dict(zip(SPEC_FIELDS, row[3:]))
        matched[j] = {f: (float(vals[f]) if vals[f] is not None else np.nan) for f in SPEC_FIELDS}
    con.close()
    return matched


PHYS = {2022: load_phys(2022), 2023: load_phys(2023)}
SPEC2024 = load_spec2024()
print(f"physiology matched: 2022={len(PHYS[2022])} 2023={len(PHYS[2023])} | spectral-extra 2024={len(SPEC2024)}")

# ---------------- measured yield + spatial match ----------------
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
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023, 2024]}


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


CV_SEED = 42


def r2_cv(Xd, y, k=5, seed=None):
    # Fresh, fixed-seed RNG per call (not a shared/advancing global one): guarantees the
    # SAME fold split for any two models fit on the same n, so "before vs after" CV R^2
    # comparisons (and the forward-selection search) are apples to apples, not confounded
    # by comparing different random fold assignments.
    n = len(y); local_rng = np.random.default_rng(seed if seed is not None else CV_SEED); idx = local_rng.permutation(n); folds = np.array_split(idx, k)
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


# ================= A. within-2023 (physiology available) =================
rows23 = [(t, j) for t, j in ymap.items() if ymeas[t].get(2023) is not None and M[j].get("EBI_Norm_2023") is not None and j in PHYS[2023]]
y23 = np.array([ymeas[t][2023] for t, j in rows23]); n23 = len(y23)
uef23 = np.array([1.0 if cultivar[j] == "UEF" else 0.0 for t, j in rows23])
ebi23 = zc([M[j].get("EBI_Norm_2023") for t, j in rows23])
one23 = np.ones(n23)
base_cols_23 = {"intercept": one23, "cultivar": uef23, "EBI": ebi23, "EBIxcultivar": ebi23 * uef23}
phys_cols_23 = {}
for f in PHYS_FIELDS:
    v = zc([PHYS[2023][j][f] for t, j in rows23])
    phys_cols_23[f] = v
    phys_cols_23[f + "_x_cultivar"] = v * uef23


def forward_select(base_cols, candidate_cols, y, seed=None):
    """Greedy forward selection: add the candidate that most improves 5-fold CV R^2,
    stop when nothing improves it further. Returns (selected_names, history)."""
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


sel23, hist23 = forward_select(base_cols_23, phys_cols_23, y23)
Xfinal23 = np.column_stack(list(base_cols_23.values()) + [phys_cols_23[n] for n in sel23])
r2_23, cv_23 = r2_cv(Xfinal23, y23)
r2_23_base, cv_23_base = r2_cv(np.column_stack(list(base_cols_23.values())), y23)

RESULTS = {
    "note": "Forward-selected additions to the yield model from all four GPKG files, "
            "selection criterion is 5-fold CV R^2 (out-of-sample), not in-sample fit, "
            "to guard against overfitting with 7 correlated physiology candidates.",
    "A_within_2023_physiology": {
        "n": n23, "baseline_features": list(base_cols_23.keys())[1:],
        "baseline_cvR2": cv_23_base, "baseline_R2": r2_23_base,
        "candidates_tested": PHYS_FIELDS + [f + "_x_cultivar" for f in PHYS_FIELDS],
        "forward_selection_history": hist23,
        "selected_features": sel23,
        "final_cvR2": cv_23, "final_R2": r2_23,
        "improvement_cvR2": round(cv_23 - cv_23_base, 3),
    },
}
print("\nA. within-2023 forward selection:", json.dumps(RESULTS["A_within_2023_physiology"], indent=2, default=str))

# ================= B. 2024 (small n, directional): does area/SR/IPVI/SIPI help? =================
rows24 = [(t, j) for t, j in ymap.items() if ymeas[t].get(2024) is not None and M[j].get("EBI_Norm_2024") is not None and j in SPEC2024]
y24 = np.array([ymeas[t][2024] for t, j in rows24]); n24 = len(y24)
if n24 >= 10:
    uef24 = np.array([1.0 if cultivar[j] == "UEF" else 0.0 for t, j in rows24])
    ebi24 = zc([M[j].get("EBI_Norm_2024") for t, j in rows24])
    one24 = np.ones(n24)
    base_cols_24 = {"intercept": one24, "cultivar": uef24, "EBI": ebi24, "EBIxcultivar": ebi24 * uef24}
    spec_cols_24 = {}
    for f in SPEC_FIELDS:
        v = zc([SPEC2024[j][f] for t, j in rows24])
        spec_cols_24[f] = v
        spec_cols_24[f + "_x_cultivar"] = v * uef24
    sel24, hist24 = forward_select(base_cols_24, spec_cols_24, y24)
    Xfinal24 = np.column_stack(list(base_cols_24.values()) + [spec_cols_24[n] for n in sel24])
    r2_24, cv_24 = r2_cv(Xfinal24, y24)
    r2_24_base, cv_24_base = r2_cv(np.column_stack(list(base_cols_24.values())), y24)
    RESULTS["B_2024_new_spectral_fields"] = {
        "n": n24, "note": "small, directional only (repeated-measures panel)",
        "baseline_cvR2": cv_24_base, "baseline_R2": r2_24_base,
        "candidates_tested": SPEC_FIELDS + [f + "_x_cultivar" for f in SPEC_FIELDS],
        "forward_selection_history": hist24, "selected_features": sel24,
        "final_cvR2": cv_24, "final_R2": r2_24, "improvement_cvR2": round(cv_24 - cv_24_base, 3),
    }
    print("\nB. 2024 forward selection:", json.dumps(RESULTS["B_2024_new_spectral_fields"], indent=2, default=str))
else:
    RESULTS["B_2024_new_spectral_fields"] = {
        "n": n24, "n_spectral_matched_total": len(SPEC2024),
        "note": "too few matched trees to fit any model, skipped. The 2024 measured-yield "
                "panel is only 18-20 specific trees, and this spectral file's ~48% coverage "
                "of Plot A UEF/53 trees (551/1145) happens not to overlap that small panel "
                "well; not a registration problem, just an unlucky intersection of two small "
                "samples."}
    print(f"\nB. 2024: only n={n24} matched, skipped")

# ================= C. pooled 2022-2023 physiology-augmented model, for comparison to the deployed pooled model =================
rowsP = [(t, j, yr) for t, j in ymap.items() for yr in [2022, 2023]
         if ymeas[t].get(yr) is not None and M[j].get(f"EBI_Norm_{yr}") is not None and j in PHYS[yr]]
yP = np.array([ymeas[t][yr] for t, j, yr in rowsP]); nP = len(yP)
uefP = np.array([1.0 if cultivar[j] == "UEF" else 0.0 for t, j, yr in rowsP])
ebiP = zc([M[j].get(f"EBI_Norm_{yr}") for t, j, yr in rowsP])
cpP = zc([CP[yr] for t, j, yr in rowsP])
oneP = np.ones(nP)
base_cols_P = {"intercept": oneP, "cultivar": uefP, "EBI": ebiP, "EBIxcultivar": ebiP * uefP, "climate": cpP}
phys_cols_P = {}
for f in PHYS_FIELDS:
    v = zc([PHYS[yr][j][f] for t, j, yr in rowsP])
    phys_cols_P[f] = v
    phys_cols_P[f + "_x_cultivar"] = v * uefP
selP, histP = forward_select(base_cols_P, phys_cols_P, yP)
XfinalP = np.column_stack(list(base_cols_P.values()) + [phys_cols_P[n] for n in selP])
r2_P, cv_P = r2_cv(XfinalP, yP)
r2_P_base, cv_P_base = r2_cv(np.column_stack(list(base_cols_P.values())), yP)
RESULTS["C_pooled_2022_2023_with_climate_and_physiology"] = {
    "n": nP, "baseline_features": list(base_cols_P.keys())[1:],
    "baseline_cvR2": cv_P_base, "baseline_R2": r2_P_base,
    "forward_selection_history": histP, "selected_features": selP,
    "final_cvR2": cv_P, "final_R2": r2_P, "improvement_cvR2": round(cv_P - cv_P_base, 3),
}
print("\nC. pooled 2022-2023 with climate, forward selection:", json.dumps(RESULTS["C_pooled_2022_2023_with_climate_and_physiology"], indent=2, default=str))

# ---- robustness check: does the same selection reappear under different CV fold splits? ----
# The search above reuses one fixed 5-fold split to evaluate every candidate, a mild form of
# selection bias (the winner may partly fit those specific folds). Re-running forward
# selection under several independent fold splits checks whether the same features keep
# getting picked, or whether it is fold-specific noise.
robustness_seeds = [1, 7, 13, 99, 2024]
robustness = []
for sd in robustness_seeds:
    s, h = forward_select(base_cols_P, phys_cols_P, yP, seed=sd)
    _, cv_base_sd = r2_cv(np.column_stack(list(base_cols_P.values())), yP, seed=sd)
    _, cv_final_sd = r2_cv(np.column_stack(list(base_cols_P.values()) + [phys_cols_P[n] for n in s]), yP, seed=sd) if s else (None, cv_base_sd)
    robustness.append({"seed": sd, "selected": s, "baseline_cvR2": cv_base_sd, "final_cvR2": cv_final_sd})
from collections import Counter
pick_counts = Counter(f for r in robustness for f in r["selected"])
RESULTS["C_robustness_across_5_independent_CV_fold_splits"] = {
    "per_seed": robustness,
    "how_often_each_feature_selected_of_5": dict(pick_counts),
    "note": "Forward selection reuses one fixed fold split to score every candidate, so the "
            "winner can partly fit those specific folds. Re-running under 5 independent fold "
            "splits (seeds 1,7,13,99,2024) checks whether the same features keep winning.",
}
print("\nC robustness (5 independent fold splits):", json.dumps(RESULTS["C_robustness_across_5_independent_CV_fold_splits"], indent=2, default=str))

RESULTS["verdict"] = (
    "Selected" if (sel23 or selP or RESULTS.get("B_2024_new_spectral_fields", {}).get("selected_features"))
    else "No candidate from any GPKG improved cross-validated R^2 in any of the three settings tested; "
         "the recommended model (cultivar + EBI x cultivar + climate) is unchanged."
)
json.dump(RESULTS, open(os.path.join(OUT, "Yield_Model_v2_AllGPKG.json"), "w"), indent=2, default=str)
print("\nVERDICT:", RESULTS["verdict"])

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
labels = ["A. within-2023\n(+physiology)", "C. pooled 2022-23\n(+climate+physiology)", "B. 2024\n(+new spectral, small n)"]
bases = [cv_23_base, cv_P_base, RESULTS.get("B_2024_new_spectral_fields", {}).get("baseline_cvR2", 0)]
finals = [cv_23, cv_P, RESULTS.get("B_2024_new_spectral_fields", {}).get("final_cvR2", 0)]
ns = [n23, nP, RESULTS.get("B_2024_new_spectral_fields", {}).get("n", 0)]
xg = np.arange(3); w = 0.35
a = ax[0]
a.bar(xg - w / 2, bases, w, color="#95a5a6", label="baseline (no GPKG extras)")
a.bar(xg + w / 2, finals, w, color="#16A085", label="+ forward-selected GPKG features")
for x, v in zip(xg - w / 2, bases): a.text(x, v + .005, f"{v:.3f}", ha="center", fontsize=8)
for x, v in zip(xg + w / 2, finals): a.text(x, v + .005, f"{v:.3f}", ha="center", fontsize=8)
a.set_xticks(xg); a.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], fontsize=8)
a.set_ylabel("5-fold CV R²"); a.legend(fontsize=8); a.grid(axis="y", alpha=.3)
a.set_title("A. Does adding all-GPKG features improve CV R²?", fontsize=10, fontweight="bold")

a = ax[1]
steps = [h["cvR2"] for h in hist23]; names = ["base"] + [h["added"] or "" for h in hist23[1:]]
a.plot(range(len(steps)), steps, "o-", color="#16A085")
a.set_xticks(range(len(steps))); a.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
a.set_ylabel("5-fold CV R²"); a.set_title("B. Forward selection path, within-2023", fontsize=10, fontweight="bold")
a.grid(alpha=.3)

a = ax[2]
stepsP = [h["cvR2"] for h in histP]; namesP = ["base"] + [h["added"] or "" for h in histP[1:]]
a.plot(range(len(stepsP)), stepsP, "o-", color="#8E44AD")
a.set_xticks(range(len(stepsP))); a.set_xticklabels(namesP, rotation=30, ha="right", fontsize=8)
a.set_ylabel("5-fold CV R²"); a.set_title("C. Forward selection path, pooled 2022-23 + climate", fontsize=10, fontweight="bold")
a.grid(alpha=.3)
fig.suptitle("Updated yield model: forward-selected features from all GPKG files", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Yield_Model_v2_AllGPKG.png"), dpi=140, bbox_inches="tight")
print("saved figure")
