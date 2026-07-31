#!/usr/bin/env python3
"""
gpkg_plus_climate_comparison.py: per request, extends gpkg_only_comparison.py by
adding CLIMATE (Chill Portions) back into the base for both targets (still no
EBI), giving a middle-ground comparison between "GPKG files only, no EBI, no
climate" (section 47's Yield_Model... no wait, this project's
GPKG_Only_Comparison.json) and the full recommended model (which also has EBI).
Also extracts, for every 2022/2023 tree-year with all three values available,
measured yield, predicted_Yield (modelled), and the recommended model's own
in-sample fitted prediction, for a single side-by-side chart.
-> Results_Analysis/08_UEF53_Rerun/GPKG_Plus_Climate_Comparison.json
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

BASE = find_thesis_root()
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
PHYS_FIELDS = ["SWP_April", "Growth_April", "SWP_MayJune", "Growth_MayJune", "CNC_June", "SWP_June", "Growth_June"]
PHYS_GPKG_COLS = ["SWP_April", "Growth__April", "SWP_May.June", "Growth_May.June", "CNC_June", "SWP_June", "Growth_June"]
MATCH_THRESHOLD_M = 3.0
CV_SEED = 42

M = load_master(); cultivar = [r.get("cultivar") for r in M]

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
    cols = ",".join(f'"{c}"' for c in PHYS_GPKG_COLS)
    cur.execute(f'SELECT "Row","Plot",cultivar,{cols},geom FROM "{tbl}"')
    rows = cur.fetchall(); con.close(); out = []
    for row_ in rows:
        plot = row_[1]; vals = row_[3:3 + len(PHYS_FIELDS)]; blob = row_[-1]
        if plot != "A" or any(v is None for v in vals): continue
        x, y = gpkg_geom_centroid(blob); d = {"x": x, "y": y}
        for f, v in zip(PHYS_FIELDS, vals): d[f] = float(v)
        out.append(d)
    return out

PHYS = {y: load_phys(y) for y in [2022, 2023]}
mx = np.array([r.get("X_UTM") for r in M], float); my = np.array([r.get("Y_UTM") for r in M], float)
phys_match = {}
for y in [2022, 2023]:
    cand = []
    for i, r in enumerate(PHYS[y]):
        d = np.sqrt((mx - r["x"]) ** 2 + (my - r["y"]) ** 2); j = int(np.argmin(d))
        if d[j] <= MATCH_THRESHOLD_M: cand.append((d[j], i, j))
    cand.sort(); ui = set(); uj = set(); m = {}
    for d, i, j in cand:
        if i in ui or j in uj: continue
        m[j] = PHYS[y][i]; ui.add(i); uj.add(j)
    phys_match[y] = m

CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}

def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)

def r2_cv(Xd, y, k=5, seed=CV_SEED):
    n = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n); folds = np.array_split(idx, k)
    pred = np.full(n, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None); pred[f] = Xd[f] @ beta
    ss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum(); cv = 1 - ss / tss
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None); ins = Xd @ beta
    r2 = 1 - ((y - ins) ** 2).sum() / tss
    return round(float(r2), 4), round(float(cv), 4)

def forward_select(base_cols_, candidate_cols, y, seed=None):
    selected = []; history = []
    Xd = np.column_stack(list(base_cols_.values()))
    cur_r2, cur_cv = r2_cv(Xd, y, seed=seed if seed is not None else CV_SEED)
    history.append({"step": 0, "added": None, "cvR2": cur_cv})
    remaining = dict(candidate_cols)
    while remaining:
        best_name, best_cv, best_r2 = None, cur_cv, cur_r2
        for name, col in remaining.items():
            Xtry = np.column_stack(list(base_cols_.values()) + [col])
            r2t, cvt = r2_cv(Xtry, y, seed=seed if seed is not None else CV_SEED)
            if cvt > best_cv + 1e-6:
                best_name, best_cv, best_r2 = name, cvt, r2t
        if best_name is None: break
        selected.append(best_name); base_cols_[best_name] = remaining.pop(best_name)
        cur_cv, cur_r2 = best_cv, best_r2
        history.append({"step": len(selected), "added": best_name, "cvR2": cur_cv})
    return selected, history, cur_r2, cur_cv

def run_for_target(rowsP, yP, uefP, climP, label):
    n = len(yP)
    phys_cols = {}
    for f in PHYS_FIELDS:
        v = zc([phys[f] for phys in rowsP])
        phys_cols[f] = v
        phys_cols[f + "_x_cultivar"] = v * uefP
    one = np.ones(n)
    base = {"intercept": one, "cultivar": uefP, "climate": climP}
    r2_base, cv_base = r2_cv(np.column_stack(list(base.values())), yP)
    sel_main, hist_main, r2_main, cv_main = forward_select(dict(base), phys_cols, yP)
    robust = []
    for seed in [1, 7, 13, 99, 2024]:
        sel_s, hist_s, r2_s, cv_s = forward_select(dict(base), dict(phys_cols), yP, seed=seed)
        robust.append({"seed": seed, "selected": sel_s, "final_cvR2": cv_s})
    print(f"\n[{label}] n={n}")
    print(f"  cultivar+climate floor: R2={r2_base} cvR2={cv_base}")
    print(f"  forward selection (seed 42): selected={sel_main}")
    print(f"  final: R2={r2_main} cvR2={cv_main}")
    for r in robust: print(f"  seed {r['seed']}: final_cvR2={r['final_cvR2']} selected={r['selected']}")
    return {"n": n, "cultivar_climate_floor": {"R2": r2_base, "cvR2": cv_base},
            "forward_selection_main_seed42": {"selected_features": sel_main, "final_R2": r2_main, "final_cvR2": cv_main},
            "robustness_across_5_seeds": robust}

# --- Target A: real measured yield ---
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

rowsA_meta = [(t, j, yr) for t, j in ymap.items() for yr in (2022, 2023)
              if ymeas[t].get(yr) is not None and j in phys_match[yr]]
rowsA = [phys_match[yr][j] for t, j, yr in rowsA_meta]
yA = np.array([ymeas[t][yr] for t, j, yr in rowsA_meta])
uefA = np.array([1.0 if cultivar[j] == "UEF" else 0.0 for t, j, yr in rowsA_meta])
climA = zc([CP[yr] for t, j, yr in rowsA_meta])
result_measured = run_for_target(rowsA, yA, uefA, climA, "REAL measured yield, GPKG physiology + cultivar + CLIMATE (no EBI)")

# --- Target B: predicted_Yield ---
rowsB_meta = []
for y in [2022, 2023]:
    for j, phys in phys_match[y].items():
        pv = M[j].get(f"predicted_Yield_{y}")
        if pv is None: continue
        rowsB_meta.append((j, y, phys, pv))
rowsB = [phys for j, y, phys, pv in rowsB_meta]
yB = np.array([pv for j, y, phys, pv in rowsB_meta])
uefB = np.array([1.0 if cultivar[j] == "UEF" else 0.0 for j, y, phys, pv in rowsB_meta])
climB = zc([CP[y] for j, y, phys, pv in rowsB_meta])
result_predicted = run_for_target(rowsB, yB, uefB, climB, "MODELLED predicted_Yield, GPKG physiology + cultivar + CLIMATE (no EBI)")

RESULTS = {
    "note": "Adds climate (Chill Portions) back into the GPKG-only base (cultivar+climate, "
            "still no EBI) for both targets, a middle ground between GPKG_Only_Comparison.json "
            "(no climate) and the full recommended model (which also has EBI).",
    "real_measured_yield_gpkg_plus_climate": result_measured,
    "modelled_predicted_yield_gpkg_plus_climate": result_predicted,
}
with open(os.path.join(OUT, "GPKG_Plus_Climate_Comparison.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")
