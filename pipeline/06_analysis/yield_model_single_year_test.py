#!/usr/bin/env python3
"""
yield_model_single_year_test.py: per request, tests whether dropping to a single
measured-yield year (instead of pooling 2022+2023) raises CV R^2. Reuses the exact
section-29/30/32 data pipeline (n=167 pooled sample), then splits it by year and
refits cultivar+EBIxcultivar+Growth_April (climate dropped, it is constant within
one year and cannot be fit) on each year alone, comparing to the pooled baseline.
-> Results_Analysis/08_UEF53_Rerun/Yield_Model_Single_Year_Test.json
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict, Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

BASE = find_thesis_root()
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
PHYS_FIELDS = ["Growth_April"]
PHYS_GPKG_COLS = ["Growth__April"]
MATCH_THRESHOLD_M = 3.0
CV_SEED = 42

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
by_year = Counter(yr for t, j, yr in rowsP)
print(f"pooled n={nP}, by year: {dict(by_year)}")

def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)

def r2_cv(Xd, y, k=5, seed=CV_SEED):
    n = len(y)
    if n < 2 * k:
        k = max(2, n // 5)
    rng = np.random.default_rng(seed); idx = rng.permutation(n); folds = np.array_split(idx, k)
    pred = np.full(n, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        if len(tr) <= Xd.shape[1]: continue
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None)
        pred[f] = Xd[f] @ beta
    valid = ~np.isnan(pred)
    ss = ((y[valid] - pred[valid]) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum()
    cv = 1 - ss / tss
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None); ins = Xd @ beta
    r2 = 1 - ((y - ins) ** 2).sum() / tss
    return round(float(r2), 4), round(float(cv), 4), int(valid.sum())

CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}

def fit_year(yr_filter, label):
    rows = [(t, j, yr) for t, j, yr in rowsP if yr in yr_filter]
    n = len(rows)
    yv = np.array([ymeas[t][yr] for t, j, yr in rows])
    uef = np.array([1.0 if cultivar[j] == "UEF" else 0.0 for t, j, yr in rows])
    ebi = zc([M[j].get(f"EBI_Norm_{yr}") for t, j, yr in rows])
    ga = zc([phys_match[yr][j]["Growth_April"] for t, j, yr in rows])
    one = np.ones(n)
    # cultivar-only floor
    X0 = np.column_stack([one, uef])
    r2_0, cv_0, k_used0 = r2_cv(X0, yv)
    # + EBIxcultivar
    X1 = np.column_stack([one, uef, ebi, ebi * uef])
    r2_1, cv_1, k1 = r2_cv(X1, yv)
    # + Growth_April too (same feature set as pooled model minus climate)
    X2 = np.column_stack([one, uef, ebi, ebi * uef, ga])
    r2_2, cv_2, k2 = r2_cv(X2, yv)
    print(f"\n[{label}] n={n}")
    print(f"  cultivar-only floor: R2={r2_0} cvR2={cv_0}")
    print(f"  +EBIxcultivar: R2={r2_1} cvR2={cv_1}")
    print(f"  +EBIxcultivar+Growth_April: R2={r2_2} cvR2={cv_2}")
    return {"n": n, "cultivar_only": {"R2": r2_0, "cvR2": cv_0},
            "plus_EBIxcultivar": {"R2": r2_1, "cvR2": cv_1},
            "plus_EBIxcultivar_plus_GrowthApril": {"R2": r2_2, "cvR2": cv_2}}

res_2023 = fit_year({2023}, "2023 ONLY (drop 2022)")
res_2022 = fit_year({2022}, "2022 ONLY (drop 2023, directional, tiny n)")
res_pooled = fit_year({2022, 2023}, "POOLED 2022+2023 (same rows as recommended model, climate excluded here for fair comparison)")

RESULTS = {
    "note": "Per request: does dropping to a single measured-yield year raise CV R^2, versus "
            "pooling 2022+2023? Climate is excluded from all three fits here (it is constant "
            "within a single year and cannot be fit there), so this isolates the effect of "
            "sample composition/pooling itself, holding the feature set fixed at "
            "cultivar+EBIxcultivar+Growth_April. The full recommended model (which adds climate "
            "back for the pooled case) reaches CV R^2=0.456, reported separately in section 22/29.",
    "pooled_sample_year_breakdown": dict(by_year),
    "year_2023_only": res_2023,
    "year_2022_only": res_2022,
    "pooled_2022_2023_no_climate": res_pooled,
}
with open(os.path.join(OUT, "Yield_Model_Single_Year_Test.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")
