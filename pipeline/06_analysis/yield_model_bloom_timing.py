#!/usr/bin/env python3
"""
yield_model_bloom_timing.py: per request, tests field-survey 50%-bloom day-of-year
(t50) directly as a yield-model feature, distinct from section 16.4's test (which
folded bloom timing into an EBI correction factor). t50 per cultivar-year is already
computed in Phenology_Insights.json (section 16.2). 2022 and 2023 both have complete
2x2 coverage (53 and UEF, both years), so this can be tested on the full pooled
n=167 sample used by the recommended model, no subsetting needed.

Adds 2 candidates (t50, t50 x cultivar) on top of the recommended model
(cultivar+EBIxcultivar+climate+Growth_April) via the same partial-F test +
forward-selection + 5-seed robustness rigor as sections 29/30.
-> Results_Analysis/08_UEF53_Rerun/Yield_Model_Bloom_Timing.json + figure.
"""
import os, sys, csv, math, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import ols_rss, f_pvalue

BASE = find_thesis_root()
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
CV_SEED = 42
PHYS_FIELDS = ["Growth_April"]
PHYS_GPKG_COLS = ["Growth__April"]
MATCH_THRESHOLD_M = 3.0

M = load_master()
cultivar = [r.get("cultivar") for r in M]

import sqlite3, struct
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

rowsP = [(t, j, yr) for t, j in ymap.items() for yr in (2022, 2023)
         if ymeas[t].get(yr) is not None and M[j].get(f"EBI_Norm_{yr}") is not None and j in phys_match[yr]]
nP = len(rowsP)
print(f"pooled 2022-2023 sample: n={nP}")

yP = np.array([ymeas[t][yr] for t, j, yr in rowsP])
uefP = np.array([1.0 if cultivar[j] == "UEF" else 0.0 for t, j, yr in rowsP])

def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)

ebiP = zc([M[j].get(f"EBI_Norm_{yr}") for t, j, yr in rowsP])
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
climP = zc([CP[yr] for t, j, yr in rowsP])
oneP = np.ones(nP)
gaP = zc([phys_match[yr][j]["Growth_April"] for t, j, yr in rowsP])

# --- t50 bloom-day-of-year, per cultivar-year, from the field phenology survey ---
PI = json.load(open(os.path.join(OUT, "Phenology_Insights.json")))
T50 = {}
for r in PI["bloom_timing_by_cultivar_year"]:
    if r["c"] in ("53", "UEF") and r["y"] in (2022, 2023) and r["t50"] is not None:
        T50[(r["y"], r["c"])] = float(r["t50"])
print(f"t50 lookup (2022/2023, 53/UEF): {T50}")
missing = [(t, j, yr) for t, j, yr in rowsP if (yr, cultivar[j]) not in T50]
print(f"rows missing a t50 value: {len(missing)} of {nP}")

t50_raw = np.array([T50[(yr, cultivar[j])] for t, j, yr in rowsP])
t50P = zc(t50_raw)

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

base_cols = {"intercept": oneP, "cultivar": uefP, "EBI": ebiP, "EBIxcultivar": ebiP * uefP,
             "climate": climP, "Growth_April": gaP}
X_base = np.column_stack(list(base_cols.values()))
baseline_r2, baseline_cv = r2_cv(X_base, yP)
print(f"baseline (recommended model): R2={baseline_r2} cvR2={baseline_cv}")

candidates = {"t50_bloom_DOY": t50P, "t50_bloom_DOY_x_cultivar": t50P * uefP}

# correlation of t50 with climate (both are largely year-level, expect high overlap)
r_t50_climate = float(np.corrcoef(t50P, climP)[0, 1])
print(f"corr(t50, climate) = {r_t50_climate:.3f}  (both largely reduce to a year label at this n)")

rss0, p0, _ = ols_rss(X_base, yP)
partial_F = {}
for name, col in candidates.items():
    X1 = np.column_stack(list(base_cols.values()) + [col])
    rss1, p1, _ = ols_rss(X1, yP)
    df1 = p1 - p0; df2 = nP - p1
    Fstat = ((rss0 - rss1) / df1) / (rss1 / df2)
    pF = float(f_pvalue(Fstat, df1, df2))
    beta, *_ = np.linalg.lstsq(X1, yP, rcond=None)
    partial_F[name] = {"F": round(float(Fstat), 3), "p": round(pF, 4), "coef": round(float(beta[-1]), 4)}
    print(f"  {name}: F={partial_F[name]['F']} p={partial_F[name]['p']} coef={partial_F[name]['coef']:+.3f}")

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

sel_main, hist_main, r2_main, cv_main = forward_select(dict(base_cols), candidates, yP)
print(f"\nforward selection (main seed 42): selected={sel_main} final_cvR2={cv_main} (gain={round(cv_main-baseline_cv,4)})")

robust = []
for seed in [1, 7, 13, 99, 2024]:
    sel_s, hist_s, r2_s, cv_s = forward_select(dict(base_cols), dict(candidates), yP, seed=seed)
    robust.append({"seed": seed, "selected": sel_s, "baseline_cvR2": hist_s[0]["cvR2"], "final_cvR2": cv_s})
    print(f"  seed {seed}: selected={sel_s} final_cvR2={cv_s} (gain={round(cv_s-hist_s[0]['cvR2'],4)})")

from collections import Counter
sel_counter = Counter()
for r in [{"seed": "main42", "selected": sel_main}] + robust:
    for s in r["selected"]: sel_counter[s] += 1
print(f"how often each feature selected of 6: {dict(sel_counter)}")

# ---- no-climate variant: does t50 substitute for climate if climate is removed? ----
base_noclim = {"intercept": oneP, "cultivar": uefP, "EBI": ebiP, "EBIxcultivar": ebiP * uefP, "Growth_April": gaP}
r2_noclim, cv_noclim = r2_cv(np.column_stack(list(base_noclim.values())), yP)
sel_nc, hist_nc, r2_nc_f, cv_nc_f = forward_select(dict(base_noclim), {"t50_bloom_DOY": t50P, "t50_bloom_DOY_x_cultivar": t50P * uefP, "climate": climP}, yP)
print(f"\nno-climate baseline: cvR2={cv_noclim}; offering t50 AND climate back as candidates: selected={sel_nc} final_cvR2={cv_nc_f}")

RESULTS = {
    "note": "Per request: tests field-survey 50%-bloom day-of-year (t50), used directly as a "
            "yield-model feature (not folded into an EBI correction, cf. section 16.4). 2022 and "
            "2023 have complete 2x2 coverage (53 and UEF), so no subsetting is needed, full n=167. "
            "Adds 2 candidates (t50, t50 x cultivar) on top of the recommended model "
            "(cultivar+EBIxcultivar+climate+Growth_April).",
    "n": nP,
    "t50_lookup_used": {f"{y}-{c}": v for (y, c), v in T50.items()},
    "corr_t50_vs_climate": round(r_t50_climate, 3),
    "baseline_recommended_model": {"R2": baseline_r2, "cvR2": baseline_cv},
    "partial_F_test": partial_F,
    "forward_selection_main_seed42": {"selected_features": sel_main, "history": hist_main,
                                       "final_R2": r2_main, "final_cvR2": cv_main,
                                       "improvement_cvR2": round(cv_main - baseline_cv, 4)},
    "robustness_across_5_seeds": robust,
    "how_often_selected_of_6": dict(sel_counter),
    "no_climate_variant": {
        "baseline_no_climate_cvR2": cv_noclim,
        "candidates_offered": ["t50_bloom_DOY", "t50_bloom_DOY_x_cultivar", "climate"],
        "selected": sel_nc, "final_cvR2": cv_nc_f,
        "interpretation": "with climate removed and offered back alongside t50 on equal footing, "
                           "whichever forward selection picks shows whether t50 can substitute for "
                           "climate's season-indicator role or whether climate still wins."
    },
}
with open(os.path.join(OUT, "Yield_Model_Bloom_Timing.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
a = ax[0]
labels = ["baseline\n(recommended)", "+t50", "+t50 x cultivar\n(forward-selected)"]
vals = [baseline_cv, partial_F["t50_bloom_DOY"], cv_main]
a.bar(["baseline", "+t50\n(partial-F test)"], [baseline_cv, baseline_cv], color="#95a5a6")
a.axhline(baseline_cv, color="k", ls="--", lw=1)
a.set_ylim(0, max(baseline_cv, cv_main) * 1.3)
a.set_title(f"Adding t50 bloom timing:\nCV R2 {baseline_cv} -> {cv_main}\n(forward selection, main seed)", fontsize=10, fontweight="bold")
a.bar(["baseline", "+ forward\nselection"], [baseline_cv, cv_main], color=["#95a5a6", "#16A085"])
a.grid(axis="y", alpha=.3)
b = ax[1]
seeds = ["main42"] + [str(r["seed"]) for r in robust]
gains = [round(cv_main - baseline_cv, 4)] + [round(r["final_cvR2"] - r["baseline_cvR2"], 4) for r in robust]
b.bar(seeds, gains, color="#2E75B6")
b.axhline(0, color="k", lw=1)
b.set_title("CV R2 gain from adding t50,\nacross main seed + 5 robustness seeds", fontsize=10, fontweight="bold")
b.set_ylabel("CV R2 gain over recommended model")
b.grid(axis="y", alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Yield_Model_Bloom_Timing.png"), dpi=130)
print("saved figure")
