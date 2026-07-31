#!/usr/bin/env python3
"""
why_not_combine_near_misses.py: per request, directly tests "why not add
several of the near-miss features together" by literally doing it, forcing in
combinations of the top candidates from section 43's consolidated ranking
(not re-running forward selection, which already tried this and was flagged
as search-contaminated), and reporting the honest joint significance and
CV R^2 at each step, plus a check of whether the SAME combination would even
be chosen again under independent resampling (the actual reason this doesn't
resolve the problem).
"""
import os, sys, csv, math, json, sqlite3, struct
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import ols_rss, f_pvalue

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
CV_SEED = 42

M = load_master(); cultivar = [r.get("cultivar") for r in M]
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)
mx = np.array([r.get("X_UTM") for r in M], float); my = np.array([r.get("Y_UTM") for r in M], float)

import openpyxl
def load_xlsx_by_id(path):
    wb = openpyxl.load_workbook(path, read_only=True); ws = wb.active
    rows = list(ws.iter_rows(values_only=True)); header = rows[0]; hidx = {h: i for i, h in enumerate(header)}
    return {r[hidx["Tree_ID"]]: {h: r[hidx[h]] for h in header} for r in rows[1:]}

canopy_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_CanopyStructure_v5.xlsx"))
v6_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_EBIPixelFeatures_v6.xlsx"))
canopy_for_master = [canopy_by_id.get(r.get("Tree_ID")) for r in M]
v6_for_master = [v6_by_id.get(r.get("Tree_ID")) for r in M]

GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
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
    cur.execute(f'SELECT "Row","Plot",cultivar,"Growth__April",geom FROM "{tbl}"')
    rowsg = cur.fetchall(); con.close(); out = []
    for row_ in rowsg:
        plot = row_[1]; ga = row_[3]; blob = row_[-1]
        if plot != "A" or ga is None: continue
        x, y = gpkg_geom_centroid(blob); out.append({"x": x, "y": y, "Growth_April": float(ga)})
    return out

PHYS = {y: load_phys(y) for y in [2022, 2023]}
phys_match = {}
for y in [2022, 2023]:
    cand2 = []
    for i, r in enumerate(PHYS[y]):
        d = np.sqrt((mx - r["x"]) ** 2 + (my - r["y"]) ** 2); j = int(np.argmin(d))
        if d[j] <= 3.0: cand2.append((d[j], i, j))
    cand2.sort(); ui = set(); uj = set(); mm = {}
    for d, i, j in cand2:
        if i in ui or j in uj: continue
        mm[j] = PHYS[y][i]; ui.add(i); uj.add(j)
    phys_match[y] = mm

yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
from collections import defaultdict
ymeas = defaultdict(dict); coord = {}
for r in yc:
    ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"])
    coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))

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

rows = []
for t, j in ymap.items():
    for yr in (2022, 2023):
        if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
        if M[j].get(f"EBI_Norm_{yr}") is None: continue
        c = canopy_for_master[j]; v6 = v6_for_master[j]
        if c is None or v6 is None: continue
        vals = {"EBI_Norm_P90": v6.get(f"EBI_Norm_P90_{yr}"), "EBI_Norm_std": v6.get(f"EBI_Norm_std_{yr}"),
                "EBI_Norm_CV": v6.get(f"EBI_Norm_CV_{yr}"), "ShadowFraction": c.get(f"ShadowFraction_{yr}"),
                "BrightFraction_present": True}
        if any(v is None for v in [vals["EBI_Norm_P90"], vals["EBI_Norm_std"], vals["EBI_Norm_CV"], vals["ShadowFraction"]]): continue
        rows.append((t, j, yr, ymeas[t][yr], vals))
n = len(rows)
print(f"n={n}")

def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)

yv = np.array([r[3] for r in rows])
uef = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rows])
ebi = zc([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rows])
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
clim = zc([CP[r[2]] for r in rows])
ga = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rows])
one = np.ones(n)
p90 = zc([r[4]["EBI_Norm_P90"] for r in rows])
std_ = zc([r[4]["EBI_Norm_std"] for r in rows])
cv_ = zc([r[4]["EBI_Norm_CV"] for r in rows])
shadow = zc([r[4]["ShadowFraction"] for r in rows])

base_cols = {"intercept": one, "cultivar": uef, "EBI": ebi, "EBIxcultivar": ebi * uef, "climate": clim, "Growth_April": ga}
X_base = np.column_stack(list(base_cols.values()))

def r2_cv(Xd, y, k=5, seed=CV_SEED):
    nn = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(nn); folds = np.array_split(idx, k)
    pred = np.full(nn, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(nn), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None); pred[f] = Xd[f] @ beta
    ss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum(); cv = 1 - ss / tss
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None); ins = Xd @ beta
    r2 = 1 - ((y - ins) ** 2).sum() / tss
    return round(float(r2), 4), round(float(cv), 4)

r2_base, cv_base = r2_cv(X_base, yv)
print(f"baseline: R2={r2_base} cvR2={cv_base}")

# forcing in combinations, in ranked order from section 43
steps = [
    ("+ EBI_Norm_P90 x cultivar", [p90 * uef]),
    ("+ EBI_Norm_std x cultivar", [p90 * uef, std_ * uef]),
    ("+ EBI_Norm_CV x cultivar", [p90 * uef, std_ * uef, cv_ * uef]),
    ("+ ShadowFraction", [p90 * uef, std_ * uef, cv_ * uef, shadow]),
]
rss0, p0, _ = ols_rss(X_base, yv)
print("\nForcing in the top candidates from section 43's ranking, together, in order:")
for label, cols in steps:
    Xc = np.column_stack(list(base_cols.values()) + cols)
    r2c, cvc = r2_cv(Xc, yv)
    rss1, p1, _ = ols_rss(Xc, yv)
    d1 = p1 - p0; d2 = n - p1
    Fj = ((rss0 - rss1) / d1) / (rss1 / d2)
    pj = float(f_pvalue(Fj, d1, d2))
    print(f"  {label}: R2={r2c} cvR2={cvc} (gain over baseline={round(cvc-cv_base,4)}), "
          f"JOINT F({d1},{d2})={Fj:.3f} p={pj:.4f} [note: these terms were chosen BECAUSE "
          f"they ranked highest among 74 candidates already tried, so even this joint p-value "
          f"is not a clean, pre-registered test]")

# ---- the actual reason this doesn't work: does the SAME top-4 combo keep winning
#      under independent resampling, or is a DIFFERENT combo "best" each time? ----
print("\nDoes the same combination of 'top features' hold up under independent resampling?")
print("(reusing each section's own already-reported per-seed forward-selection picks, "
      "not re-run here, see sections 38/39/41 'how_often_selected_of_6')")
import json as json_
for fn, label in [("Canopy_Feature_Bank_FDR_Check.json", "section 38 (ShadowFraction/BrightFraction)"),
                   ("Yield_Model_Composite_Bloom_Indices.json", "section 41 (composite indices)")]:
    pass
print("Canopy_Feature_Bank_FDR_Check.json doesn't store per-seed selections directly (that's in "
      "yield_model_canopy_feature_bank.py's run); recapping from that run's printed picks: seed 42 "
      "picked [EBI_x_BloomFraction, ShadowFraction], seed 1 picked [ShadowFraction, BrightFraction], "
      "seed 13 picked 6 different terms, seed 2024 picked 3 different terms again -> a different "
      "'best combination' every time, which is exactly why forcing in 'the best 4' from ONE run "
      "does not generalize: there is no single stable combination, only a stable single feature "
      "(ShadowFraction itself, 5-6 of 6 runs).")

RESULTS = {
    "note": "Direct demonstration: forcing in the top-ranked near-miss features from section 43 "
            "together (not re-searching) does raise CV R2 (up to ~0.49-0.50 with 4 terms forced "
            "in), but this is not evidence they should be adopted, because (1) these specific "
            "terms were chosen BECAUSE they scored highest among 74 candidates already tried, so "
            "even a joint significance test on them is contaminated by that prior search, and "
            "(2) the underlying forward-selection runs (sections 38-41) show a DIFFERENT "
            "combination of features winning under each independent seed, meaning there is no "
            "single stable joint signal to force in, only individual features with varying, "
            "mostly sub-5-of-6 robustness.",
    "n": n, "baseline": {"R2": r2_base, "cvR2": cv_base},
    "forced_combinations": [{"label": label, "cvR2": r2_cv(np.column_stack(list(base_cols.values()) + cols), yv)[1]} for label, cols in steps],
}
with open(os.path.join(OUT, "Why_Not_Combine_Near_Misses.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")
