#!/usr/bin/env python3
"""
predicted_yield_model_check.py: per request, builds the same style of forward-selected,
cross-validated yield model, but with the TARGET set to predicted_Yield_2022/2023 (the
Yield_with_clustering GPKGs' clustering-model output, already in the master as
predicted_Yield_2022/predicted_Yield_2023), instead of measured yield.

This is explicitly NOT a measured-yield accuracy result. Project convention (non-
negotiable 1): predicted_Yield is a MODELLED value, not ground truth, and section 24
established exactly how it is built: an RF model predicts Yield "using predicted
physiological measurements", i.e. it takes SWP_April, Growth_April, SWP_May.June,
Growth_May.June, CNC_June, SWP_June, Growth_June (the same 7 fields tested in sections
21-23) as direct regressors. So fitting a model with those same fields against
predicted_Yield is close to reconstructing the clustering model's own formula, not
learning anything new about the orchard, this script exists to demonstrate that
explicitly and quantify how much of the apparent "accuracy" is circular.

Three settings, all pooled 2022-2023 (n=167, same sample as the measured-yield model,
section 22/25), all labelled "modelled/predicted yield" throughout:
  A. Same feature set as the measured-yield recommended model (cultivar+EBIxcultivar+
     climate+Growth_April), for a direct, apples-to-apples CV R^2 comparison.
  B. cultivar+EBIxcultivar+climate only, no physiology, the cleanest (least circular)
     check of whether EBI/cultivar/climate track the modelled yield on their own.
  C. Full forward selection across all 7 raw physiology fields (+ cultivar interactions),
     to show how completely the clustering model's own formula can be reconstructed once
     its own direct inputs are made available again.
-> Results_Analysis/08_UEF53_Rerun/Predicted_Yield_Model_Check.json + figure.
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
CV_SEED = 42
PHYS_FIELDS = ["SWP_April", "Growth_April", "SWP_MayJune", "Growth_MayJune", "CNC_June", "SWP_June", "Growth_June"]
PHYS_GPKG_COLS = ["SWP_April", "Growth__April", "SWP_May.June", "Growth_May.June", "CNC_June", "SWP_June", "Growth_June"]
MATCH_THRESHOLD_M = 3.0

M = load_master()
cultivar = [r.get("cultivar") for r in M]


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
        plot = row_[1]
        vals = row_[3:3 + len(PHYS_FIELDS)]
        blob = row_[-1]
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
print(f"physiology matched: 2022={len(phys_match[2022])} 2023={len(phys_match[2023])}")

# ---------------- pooled sample: EBI + physiology + master's own predicted_Yield_{year} ----------------
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
gaP = zc([phys["Growth_April"] for j, y, phys, pv in rowsP])
oneP = np.ones(nP)

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
    rmse_cv = float(np.sqrt(np.mean((y - pred) ** 2)))
    return round(float(r2), 4), round(float(cv), 4), round(rmse_cv, 4)


# ---- A: same feature set as the measured-yield recommended model ----
XA = np.column_stack([oneP, uefP, ebiP, ebiP * uefP, climP, gaP])
r2A, cvA, rmseA = r2_cv(XA, yP)
print(f"\nA. cultivar+EBIxcultivar+climate+Growth_April -> predicted_Yield: R2={r2A} cvR2={cvA} RMSE={rmseA}")

# ---- B: no physiology at all, cleanest check ----
XB = np.column_stack([oneP, uefP, ebiP, ebiP * uefP, climP])
r2B, cvB, rmseB = r2_cv(XB, yP)
print(f"B. cultivar+EBIxcultivar+climate (no physiology) -> predicted_Yield: R2={r2B} cvR2={cvB} RMSE={rmseB}")

# ---- C: full forward selection across all 7 raw physiology fields ----
base_cols_C = {"intercept": oneP, "cultivar": uefP, "EBI": ebiP, "EBIxcultivar": ebiP * uefP, "climate": climP}
phys_cols = {}
for f in PHYS_FIELDS:
    v = zc([phys[f] for j, y, phys, pv in rowsP])
    phys_cols[f] = v
    phys_cols[f + "_x_cultivar"] = v * uefP


def forward_select(base_cols, candidate_cols, y, seed=None):
    selected = []; history = []
    Xd = np.column_stack(list(base_cols.values()))
    cur_r2, cur_cv, _ = r2_cv(Xd, y, seed=seed)
    history.append({"step": 0, "added": None, "cvR2": cur_cv})
    remaining = dict(candidate_cols)
    while remaining:
        best_name, best_cv, best_r2 = None, cur_cv, cur_r2
        for name, col in remaining.items():
            Xtry = np.column_stack(list(base_cols.values()) + [col])
            r2t, cvt, _ = r2_cv(Xtry, y, seed=seed)
            if cvt > best_cv + 1e-6:
                best_name, best_cv, best_r2 = name, cvt, r2t
        if best_name is None: break
        selected.append(best_name); base_cols[best_name] = remaining.pop(best_name)
        cur_cv, cur_r2 = best_cv, best_r2
        history.append({"step": len(selected), "added": best_name, "cvR2": cur_cv})
    return selected, history, cur_r2, cur_cv


base_for_search = dict(base_cols_C)
selected, history, final_r2, final_cv = forward_select(base_for_search, phys_cols, yP)
print(f"C. full forward selection across all 7 physiology fields -> predicted_Yield:")
print(f"   selected: {selected}")
print(f"   final: R2={final_r2} cvR2={final_cv}")

# raw correlations of predicted_Yield with each physiology field (pooled), to show the mechanical link directly
raw_corrs = {}
for f in PHYS_FIELDS:
    v = np.array([phys[f] for j, y, phys, pv in rowsP])
    raw_corrs[f] = round(float(np.corrcoef(v, yP)[0, 1]), 3)
print("\nraw correlations, predicted_Yield vs each physiology field (pooled):")
for f, r in raw_corrs.items(): print(f"  {f}: r={r:+.3f}")

RESULTS = {
    "note": "TARGET IS MODELLED/PREDICTED YIELD (predicted_Yield_2022/2023 from the "
            "Yield_with_clustering GPKGs), NOT measured yield. This is not a real "
            "yield-accuracy result, it demonstrates how much of the apparent fit is "
            "circular reconstruction of the clustering model's own generating formula. "
            "See project convention: predicted_Yield is a modelled value, never ground truth.",
    "n": nP,
    "measured_yield_model_for_comparison": {
        "features": "cultivar+EBIxcultivar+climate+Growth_April",
        "cvR2": 0.4563, "note": "from section 22/25, the real, honest, non-circular result"
    },
    "A_same_features_vs_predicted_yield": {"R2": r2A, "cvR2": cvA, "RMSE_kg": rmseA},
    "B_no_physiology_vs_predicted_yield": {"R2": r2B, "cvR2": cvB, "RMSE_kg": rmseB},
    "C_full_physiology_forward_selection_vs_predicted_yield": {
        "selected_features": selected, "forward_selection_history": history,
        "final_R2": final_r2, "final_cvR2": final_cv,
    },
    "raw_correlations_predicted_yield_vs_physiology_fields": raw_corrs,
}
with open(os.path.join(OUT, "Predicted_Yield_Model_Check.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
labels = ["measured yield\n(cultivar+EBIxcv+\nclimate+Growth_April)",
          "predicted_Yield\n(same features, A)",
          "predicted_Yield\n(no physiology, B)",
          "predicted_Yield\n(full physiology\nforward select, C)"]
vals = [0.4563, cvA, cvB, final_cv]
colors = ["#16A085", "#C0392B", "#E67E22", "#C0392B"]
ax.bar(range(len(labels)), vals, color=colors)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("CV R^2"); ax.set_title("Measured yield vs modelled predicted_Yield\n(circularity check)", fontsize=10)
ax.axhline(0.4563, color="#16A085", ls="--", lw=1, alpha=0.6)

ax = axes[1]
fs = list(raw_corrs.keys()); rs = list(raw_corrs.values())
ax.barh(range(len(fs)), rs, color="#8E44AD")
ax.set_yticks(range(len(fs))); ax.set_yticklabels(fs, fontsize=8)
ax.axvline(0, color="k", lw=0.8)
ax.set_title("predicted_Yield vs each physiology field\n(raw pooled r, the mechanical link)", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Predicted_Yield_Model_Check.png"), dpi=130)
print("saved figure")
