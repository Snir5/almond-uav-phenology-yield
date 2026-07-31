#!/usr/bin/env python3
"""
predicted_yield_model_full_features.py: per request, re-runs section 27's
predicted_Yield circularity check, this time offering EVERY feature used anywhere
in this thesis (not just the 7 raw physiology fields), PLUS every pairwise
interaction among them (mirroring section 30's full pairwise sweep methodology),
to a single forward selection against predicted_Yield_2022/2023.

STILL EXPLICITLY NOT A REAL YIELD-ACCURACY RESULT. predicted_Yield is a modelled
value (the clustering-model output), not measured ground truth (project convention,
non-negotiable). The point of this script is the CONTRAST with section 30: when the
exact same kind of exhaustive feature/interaction search was run against MEASURED
yield (section 30), it found nothing, the ceiling (CV R^2=0.456) held. When the same
kind of search is run against predicted_Yield here, it should climb much higher,
because predicted_Yield is itself a model's output built partly from these same
physiology fields, so an exhaustive search mechanically reconstructs that formula
rather than discovering anything new about the orchard. This contrast is the
argument for why more measured-yield field seasons (not more feature engineering)
is what would actually improve the REAL model.

12 base features (identical set to section 30): cultivar, EBI, climate, DEM,
Growth_April, Growth_MayJune, Growth_June, SWP_April, SWP_MayJune, SWP_June,
CNC_June, NGRDI, plus their C(12,2)=66 pairwise interactions, offered together
(78 candidates total) to one forward selection against predicted_Yield, pooled
2022-2023 (n=167, matched to the master's own predicted_Yield_{year} columns).

-> Results_Analysis/08_UEF53_Rerun/Predicted_Yield_Model_FullFeatures.json + figure.
"""
import os, sys, csv, math, json, sqlite3, struct, itertools
from collections import defaultdict, Counter
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

BASE = find_thesis_root()
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
CV_SEED = 42
PHYS_FIELDS = ["SWP_April", "Growth_April", "SWP_MayJune", "Growth_MayJune", "CNC_June", "SWP_June", "Growth_June"]
PHYS_GPKG_COLS = ["SWP_April", "Growth__April", "SWP_May.June", "Growth_May.June", "CNC_June", "SWP_June", "Growth_June"]
MATCH_THRESHOLD_M = 3.0

M = load_master()
cultivar = [r.get("cultivar") for r in M]


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
    cur.execute("SELECT table_name FROM gpkg_geometry_columns")
    tbl = cur.fetchone()[0]
    cols = ",".join(f'"{c}"' for c in PHYS_GPKG_COLS)
    cur.execute(f'SELECT "Row","Plot",cultivar,{cols},geom FROM "{tbl}"')
    rows = cur.fetchall(); con.close()
    out = []
    for row_ in rows:
        plot = row_[1]; vals = row_[3:3 + len(PHYS_FIELDS)]; blob = row_[-1]
        if plot != "A" or any(v is None for v in vals): continue
        x, y = gpkg_geom_centroid(blob)
        d = {"x": x, "y": y}
        for f, v in zip(PHYS_FIELDS, vals): d[f] = float(v)
        out.append(d)
    return out


PHYS = {y: load_phys(y) for y in [2022, 2023]}
mx = np.array([r.get("X_UTM") for r in M], float); my = np.array([r.get("Y_UTM") for r in M], float)
phys_match = {}
for y in [2022, 2023]:
    cand = []
    for i, r in enumerate(PHYS[y]):
        d = np.sqrt((mx - r["x"]) ** 2 + (my - r["y"]) ** 2)
        j = int(np.argmin(d))
        if d[j] <= MATCH_THRESHOLD_M: cand.append((d[j], i, j))
    cand.sort(); ui = set(); uj = set(); m = {}
    for d, i, j in cand:
        if i in ui or j in uj: continue
        m[j] = PHYS[y][i]; ui.add(i); uj.add(j)
    phys_match[y] = m

rowsP = []
for y in [2022, 2023]:
    for j, phys in phys_match[y].items():
        if M[j].get(f"EBI_Norm_{y}") is None: continue
        pv = M[j].get(f"predicted_Yield_{y}")
        if pv is None: continue
        rowsP.append((j, y, phys, pv))
nP = len(rowsP)
print(f"pooled sample (matched to master's own predicted_Yield_{{year}}): n={nP}")

yP = np.array([pv for j, y, phys, pv in rowsP])
uefP = np.array([1.0 if cultivar[j] == "UEF" else 0.0 for j, y, phys, pv in rowsP])


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


ebiP = zc([M[j].get(f"EBI_Norm_{y}") for j, y, phys, pv in rowsP])
CP = {yr: float(np.nanmean([r.get(f"Chill_Portions_{yr}") for r in M if r.get(f"Chill_Portions_{yr}") is not None])) for yr in [2022, 2023]}
climP = zc([CP[y] for j, y, phys, pv in rowsP])
ngrdiP = zc([M[j].get(f"NGRDI_Norm_{y}") for j, y, phys, pv in rowsP])
dem_raw = np.array([M[j].get("DEM_Jul2024") for j, y, phys, pv in rowsP], float)
dem_raw[np.isnan(dem_raw)] = np.nanmean(dem_raw)
demP = zc(dem_raw)
oneP = np.ones(nP)

phys_cols = {}
for f in PHYS_FIELDS:
    phys_cols[f] = zc([phys[f] for j, y, phys, pv in rowsP])
gaP = phys_cols["Growth_April"]

CV_SEED_ = 42


def r2_cv(Xd, y, k=5, seed=None):
    n = len(y); rng = np.random.default_rng(seed if seed is not None else CV_SEED_)
    idx = rng.permutation(n); folds = np.array_split(idx, k)
    pred = np.full(n, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None)
        pred[f] = Xd[f] @ beta
    ss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum()
    cv = 1 - ss / tss
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None); ins = Xd @ beta
    r2 = 1 - ((y - ins) ** 2).sum() / tss
    return round(float(r2), 4), round(float(cv), 4)


# ---------------- baseline: cultivar + climate alone (the "just knows the season/cultivar" floor) ----------------
X_floor = np.column_stack([oneP, uefP, climP])
r2_floor, cv_floor = r2_cv(X_floor, yP)
print(f"floor (cultivar+climate only) -> predicted_Yield: R2={r2_floor} cvR2={cv_floor}")

# ---------------- every base feature (12, identical set to section 30) ----------------
features = {
    "cultivar": uefP, "EBI": ebiP, "climate": climP, "DEM": demP,
    "Growth_April": gaP, "Growth_MayJune": phys_cols["Growth_MayJune"], "Growth_June": phys_cols["Growth_June"],
    "SWP_April": phys_cols["SWP_April"], "SWP_MayJune": phys_cols["SWP_MayJune"], "SWP_June": phys_cols["SWP_June"],
    "CNC_June": phys_cols["CNC_June"], "NGRDI": ngrdiP,
}
feat_names = list(features.keys())
all_pairs = list(itertools.combinations(feat_names, 2))
candidates = dict(features)  # every main effect...
for a, b in all_pairs:
    candidates[f"{a}_x_{b}"] = features[a] * features[b]   # ...plus every pairwise interaction
print(f"\n{len(feat_names)} main effects + {len(all_pairs)} pairwise interactions = {len(candidates)} candidates offered")

base_cols0 = {"intercept": oneP}


def forward_select(base_cols_, candidate_cols, y, seed=None):
    selected = []; history = []
    Xd = np.column_stack(list(base_cols_.values()))
    cur_r2, cur_cv = r2_cv(Xd, y, seed=seed)
    history.append({"step": 0, "added": None, "cvR2": cur_cv})
    remaining = dict(candidate_cols)
    while remaining:
        best_name, best_cv, best_r2 = None, cur_cv, cur_r2
        for name, col in remaining.items():
            Xtry = np.column_stack(list(base_cols_.values()) + [col])
            r2t, cvt = r2_cv(Xtry, y, seed=seed)
            if cvt > best_cv + 1e-6:
                best_name, best_cv, best_r2 = name, cvt, r2t
        if best_name is None: break
        selected.append(best_name); base_cols_[best_name] = remaining.pop(best_name)
        cur_cv, cur_r2 = best_cv, best_r2
        history.append({"step": len(selected), "added": best_name, "cvR2": cur_cv})
    return selected, history, cur_r2, cur_cv


sel_main, hist_main, r2_main, cv_main = forward_select(dict(base_cols0), candidates, yP)
print(f"\nforward selection (main seed 42, {len(candidates)} candidates): selected={sel_main}")
print(f"final: R2={r2_main} cvR2={cv_main}")

robust = []
for seed in [1, 7, 13, 99, 2024]:
    sel_s, hist_s, r2_s, cv_s = forward_select(dict(base_cols0), dict(candidates), yP, seed=seed)
    robust.append({"seed": seed, "selected": sel_s, "final_cvR2": cv_s})
    print(f"  seed {seed}: final_cvR2={cv_s}  n_selected={len(sel_s)}")

# ---------------- for comparison: measured-yield model's real numbers (section 22/25/30) ----------------
MEASURED_YIELD_MODEL_CVR2 = 0.4563
MEASURED_YIELD_FULL_SEARCH_CVR2 = 0.4563  # section 30: exhaustive pairwise search found NOTHING beyond this

RESULTS = {
    "note": "TARGET IS MODELLED/PREDICTED YIELD (predicted_Yield_2022/2023), NOT measured yield. "
            "Extends section 27 by offering EVERY base feature used anywhere in this thesis (12, "
            "identical set to section 30) PLUS all 66 pairwise interactions (78 candidates total) to "
            "one exhaustive forward selection. The point is the CONTRAST with section 30: the exact "
            "same kind of exhaustive search against MEASURED yield found nothing beyond the existing "
            "model (CV R^2 stayed at 0.456); here it should climb much higher, because it mechanically "
            "reconstructs the clustering model's own generating formula, not a real predictive gain.",
    "n": nP,
    "measured_yield_for_comparison": {
        "recommended_model_cvR2": MEASURED_YIELD_MODEL_CVR2,
        "exhaustive_pairwise_search_cvR2": MEASURED_YIELD_FULL_SEARCH_CVR2,
        "note": "section 22/25 and section 30 respectively -- the real, honest, non-circular results; "
                "the exhaustive search of every feature and interaction found ZERO additional gain."
    },
    "floor_cultivar_climate_only": {"R2": r2_floor, "cvR2": cv_floor},
    "n_candidates_offered": len(candidates),
    "n_main_effects": len(feat_names), "n_pairwise_interactions": len(all_pairs),
    "forward_selection_main_seed42": {"selected_features": sel_main, "history": hist_main,
                                       "final_R2": r2_main, "final_cvR2": cv_main},
    "robustness_across_5_seeds": robust,
}
with open(os.path.join(OUT, "Predicted_Yield_Model_FullFeatures.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
labels = ["measured yield\n(real model)", "measured yield\n(exhaustive search)",
          "predicted_Yield\n(cultivar+climate\nfloor)", "predicted_Yield\n(exhaustive search,\nall features+pairs)"]
vals = [MEASURED_YIELD_MODEL_CVR2, MEASURED_YIELD_FULL_SEARCH_CVR2, cv_floor, cv_main]
colors = ["#16A085", "#16A085", "#E67E22", "#C0392B"]
ax.bar(range(len(labels)), vals, color=colors)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("CV R^2")
ax.set_title("Real (measured) vs modelled (predicted_Yield)\nunder the SAME exhaustive search", fontsize=10)
ax.axhline(MEASURED_YIELD_MODEL_CVR2, color="#16A085", ls="--", lw=1, alpha=0.6)

ax = axes[1]
steps = [h["step"] for h in hist_main]; cvh = [h["cvR2"] for h in hist_main]
ax.plot(steps, cvh, "o-", color="#C0392B")
for h in hist_main:
    if h["added"]: ax.annotate(h["added"], (h["step"], h["cvR2"]), fontsize=6.5, rotation=25, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("forward selection step"); ax.set_ylabel("CV R^2 vs predicted_Yield")
ax.set_title("predicted_Yield: forward selection across\nall 78 candidates (seed 42)", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Predicted_Yield_Model_FullFeatures.png"), dpi=130)
print("saved figure")
