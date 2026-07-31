#!/usr/bin/env python3
"""
four_way_comparison_no_outliers.py: repeats the four-way yield comparison
(predicted_Yield vs actual, our model vs actual, both vs actual, our model vs
predicted_Yield) after flagging and removing actual-yield anomalies.

Outlier rule: modified Z-score on measured_yield_kg, threshold 3.5
(Iglewicz & Hoaglin), the same robust method already used elsewhere in this
project for per-canopy outlier removal. Applied only to the actual-yield
column, since that is what was flagged as anomalous; predicted_Yield and our
model's inputs are not filtered directly.

The recommended model (cultivar + EBI x cultivar + climate + Growth_April) is
RE-FIT on the outlier-free subsample, not just re-evaluated with the old
coefficients, since a handful of extreme actual-yield points can otherwise
distort the fitted coefficients themselves (leverage), not just the residuals.

-> Results_Analysis/08_UEF53_Rerun/Four_Way_Comparison_No_Outliers.json
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
import stats_utils as su

BASE = find_thesis_root()
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
M = load_master(); cultivar = [r.get("cultivar") for r in M]
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)
mx = np.array([r.get("X_UTM") for r in M], float); my = np.array([r.get("Y_UTM") for r in M], float)

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
        dd = np.sqrt((mx - r["x"]) ** 2 + (my - r["y"]) ** 2); j = int(np.argmin(dd))
        if dd[j] <= 3.0: cand2.append((dd[j], i, j))
    cand2.sort(); ui = set(); uj = set(); mm = {}
    for dd, i, j in cand2:
        if i in ui or j in uj: continue
        mm[j] = PHYS[y][i]; ui.add(i); uj.add(j)
    phys_match[y] = mm

yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
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
        pv = M[j].get(f"predicted_Yield_{yr}")
        if pv is None: continue
        rows.append((t, j, yr, ymeas[t][yr], pv))
n_all = len(rows)

def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)

yv_all = np.array([r[3] for r in rows])
pred_all = np.array([r[4] for r in rows])
uef_all = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rows])
ebi_all = zc([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rows])
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
clim_all = zc([CP[r[2]] for r in rows])
ga_all = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rows])

# ---------- outlier flagging on actual yield only: modified Z-score, MAD-based ----------
med = float(np.median(yv_all))
mad = float(np.median(np.abs(yv_all - med)))
modz = 0.6745 * (yv_all - med) / mad
THRESH = 3.5
keep = np.abs(modz) <= THRESH
n_removed = int((~keep).sum())
removed_info = [
    {"tree_id": rows[i][0], "year": rows[i][2], "cultivar": cultivar[rows[i][1]],
     "measured_yield_kg": round(float(yv_all[i]), 4), "modified_z": round(float(modz[i]), 3)}
    for i in range(n_all) if not keep[i]
]

yv = yv_all[keep]; pred = pred_all[keep]; uef = uef_all[keep]
ebi = ebi_all[keep]; clim = clim_all[keep]; ga = ga_all[keep]
n = int(keep.sum())
one = np.ones(n)
X = np.column_stack([one, uef, ebi, ebi * uef, clim, ga])
beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
our_model_fitted = X @ beta

def stats_pair(x, y):
    r = float(np.corrcoef(x, y)[0, 1])
    p = su.r_pvalue(r, len(x))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    mae = float(np.mean(np.abs(y - x)))
    return dict(n=len(x), r=round(r, 4), r2=round(r ** 2, 4), p=p,
                slope=round(float(slope), 4), intercept=round(float(intercept), 4), mae=round(mae, 4))

s1 = stats_pair(yv, pred)
s2 = stats_pair(yv, our_model_fitted)
s4 = stats_pair(pred, our_model_fitted)

print(f"n_all={n_all}  n_removed(|modz|>{THRESH})={n_removed}  n_kept={n}")
print("removed:", removed_info)
print("predicted_vs_measured (no outliers):", s1)
print("ours_vs_measured (no outliers, refit):", s2)
print("ours_vs_predicted (no outliers, refit):", s4)

order = np.argsort(yv)
out = {
    "outlier_rule": f"modified Z-score on measured_yield_kg, |modZ| > {THRESH} (Iglewicz & Hoaglin)",
    "n_all": n_all, "n_removed": n_removed, "n_kept": n,
    "removed_trees": removed_info,
    "tree_id": [rows[i][0] for i in np.where(keep)[0][order]],
    "year": [rows[i][2] for i in np.where(keep)[0][order]],
    "cultivar": [cultivar[rows[i][1]] for i in np.where(keep)[0][order]],
    "measured_yield_kg": [round(float(v), 4) for v in yv[order]],
    "predicted_Yield_modelled_kg": [round(float(v), 4) for v in pred[order]],
    "our_model_fitted_kg": [round(float(v), 4) for v in our_model_fitted[order]],
    "predicted_vs_measured": s1,
    "ours_vs_measured": s2,
    "ours_vs_predicted": s4,
}
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "Four_Way_Comparison_No_Outliers.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)
print("saved json")
