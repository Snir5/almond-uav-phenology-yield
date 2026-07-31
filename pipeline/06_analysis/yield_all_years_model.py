#!/usr/bin/env python3
"""
yield_all_years_model.py — join ALL measured yield (2022, 2023, 2024) to trees and
fit one pooled tree-year model. Builds the full long table (one row per tree-year with
that tree's features), saves it, then fits nested OLS with 5-fold cross-validated R^2.

Features per tree-year: cultivar, same-year EBI (and EBI x cultivar), NGRDI, spatial
X/Y, season-level climate (Chill Portions, GDD_Jan_Feb) and field 50%-bloom day
(t50, per cultivar-year). Season enters via CLIMATE, not a year label.
-> Results_Analysis/Yield_AllYears_Joined.xlsx + yield_all_years_model.json + figure.
"""
import os, sys, csv, math, json
from datetime import datetime
from collections import defaultdict
import numpy as np, openpyxl
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import ols_rss
BASE = find_thesis_root(); OUT = os.path.join(BASE, "Results_Analysis")
FIG = os.path.join(OUT, "08_UEF53_Rerun"); SC = os.environ.get("SCRATCH_DIR", FIG)
MEAS = [2022, 2023, 2024]; rng = np.random.default_rng(42)

M = load_master()
def col(c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in M], float)
mlat, mlon = col("Latitude"), col("Longitude"); mcult = [r.get("cultivar") for r in M]
CP = {y: float(np.nanmean(col(f"Chill_Portions_{y}"))) for y in [2021, 2022, 2023, 2024]}
GDD = {y: float(np.nanmean(col(f"GDD_Jan_Feb_{y}"))) for y in [2021, 2022, 2023, 2024]}

# field 50%-bloom day (t50) per cultivar-year
def doy(s): return datetime.strptime(s, "%d/%m/%Y").timetuple().tm_yday
def nc(c): return "UEF" if ("UEF" in c or "Um" in c) else c
curve = {}
for r in csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "Phenology_Surveys", "Kadma_Phenology_by_CultivarDate_2022_2024.csv"), encoding="utf-8-sig")):
    c = nc(r["Cultivar"])
    if c in ("53", "UEF"): curve.setdefault((int(r["Year"]), c), []).append((doy(r["Date"]), float(r["Mean_Percent_Open"])))
def t50(pts):
    pts = sorted(pts); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if max(ys) < 50: return np.nan
    for i in range(1, len(xs)):
        if ys[i - 1] < 50 <= ys[i]: return xs[i - 1] + (50 - ys[i - 1]) / (ys[i] - ys[i - 1]) * (xs[i] - xs[i - 1])
    return np.nan
T50 = {k: t50(v) for k, v in curve.items()}

# yield join (greedy 1:1, <=8 m)
yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
coord = {}; ym = {}
for r in yc:
    coord.setdefault(r["tree_id"], (float(r["latitude"]), float(r["longitude"])))
    ym[(r["tree_id"], int(r["year"]))] = float(r["net_kernel_yield_per_tree_kg"])
def hav(a, b, d, e):
    R = 6371000; p = math.pi / 180; x = math.sin((d - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(d * p) * math.sin((e - b) * p / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0, x)))
cand = []
for t, (la, lo) in coord.items():
    ds = np.array([hav(la, lo, a, b) for a, b in zip(mlat, mlon)])
    for j in np.where(ds <= 8)[0]: cand.append((float(ds[j]), t, int(j)))
cand.sort(); uy = set(); um = set(); mp = {}
for d, t, j in cand:
    if t in uy or j in um: continue
    mp[t] = (j, d); uy.add(t); um.add(j)

# build long table
rows = []
for t, (j, d) in mp.items():
    c = mcult[j]
    if c not in ("UEF", "53"): continue
    for y in MEAS:
        yv = ym.get((t, y)); e = M[j].get(f"EBI_Norm_{y}")
        if yv is None or e is None: continue
        rows.append({"tree_id": t, "Zone_Value": M[j]["Zone_Value"], "year": y, "cultivar": c,
                     "yield_kg": yv, "EBI": float(e), "NGRDI": M[j].get(f"NGRDI_Norm_{y}"),
                     "X_UTM": M[j].get("X_UTM"), "Y_UTM": M[j].get("Y_UTM"),
                     "ChillPortions": CP[y], "GDD_JanFeb": GDD[y], "bloom_t50": T50.get((y, c), np.nan),
                     "match_dist_m": round(d, 2)})
# save joined table
wb = openpyxl.Workbook(); ws = wb.active; ws.title = "yield_allyears"
hdr = list(rows[0].keys()); ws.append(hdr)
for r in rows: ws.append([r[k] for k in hdr])
wb.save(os.path.join(BASE, "Trees_Data_Survey", "Yield_AllYears_Joined.xlsx"))
n = len(rows)
print(f"joined tree-year rows: {n}  (trees {len(set(r['tree_id'] for r in rows))}; by year "
      f"{ {y: sum(1 for r in rows if r['year']==y) for y in MEAS} })")

# ---- model fitting with 5-fold CV R^2 ----
y = np.array([r["yield_kg"] for r in rows]); one = np.ones(n)
def z(v): v = np.array(v, float); s = v.std(); return (v - v.mean()) / (s if s else 1)
uef = np.array([1.0 if r["cultivar"] == "UEF" else 0.0 for r in rows])
ebi = z([r["EBI"] for r in rows]); ngrdi = z([r["NGRDI"] for r in rows])
xu = z([r["X_UTM"] for r in rows]); yu = z([r["Y_UTM"] for r in rows])
cp = z([r["ChillPortions"] for r in rows]); gdd = z([r["GDD_JanFeb"] for r in rows])
# bloom t50: impute missing (cv-53 2024) with column mean so the feature is usable
t50v = np.array([r["bloom_t50"] for r in rows], float); t50v[np.isnan(t50v)] = np.nanmean(t50v); t50z = z(t50v)

def cvR2(X):
    idx = rng.permutation(n); folds = np.array_split(idx, 5); pred = np.full(n, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f); b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None); pred[f] = X[f] @ b
    cv = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return round(float(r2), 3), round(float(cv), 3)
def D(cols): return np.column_stack([one] + cols)
models = {
    "cultivar": D([uef]),
    "+ EBI x cultivar": D([uef, ebi, ebi * uef]),
    "+ climate (CP+GDD)": D([uef, ebi, ebi * uef, cp, gdd]),
    "+ bloom timing (t50)": D([uef, ebi, ebi * uef, cp, gdd, t50z]),
    "+ spatial (X,Y)": D([uef, ebi, ebi * uef, cp, gdd, t50z, xu, yu]),
    "+ NGRDI": D([uef, ebi, ebi * uef, cp, gdd, t50z, xu, yu, ngrdi]),
}
res = {k: dict(zip(["R2", "cvR2"], cvR2(X))) for k, X in models.items()}
# season ceiling = free year factor
d22 = np.array([1.0 if r["year"] == 2022 else 0 for r in rows]); d24 = np.array([1.0 if r["year"] == 2024 else 0 for r in rows])
ceil = cvR2(D([uef, ebi, ebi * uef, d22, d24]))
out = {"n_tree_year_rows": n, "n_trees": len(set(r["tree_id"] for r in rows)),
       "by_year": {y: sum(1 for r in rows if r["year"] == y) for y in MEAS},
       "models_cvR2": res, "season_ceiling_year_factor_cvR2": ceil[1],
       "saved_table": "Trees_Data_Survey/Yield_AllYears_Joined.xlsx"}
json.dump(out, open(os.path.join(OUT, "yield_all_years_model.json"), "w"), indent=2)
json.dump(out, open(os.path.join(SC, "yield_all_years_model.json"), "w"), indent=2)
print(json.dumps(out, indent=1))

# figure
fig, ax = plt.subplots(figsize=(11, 5.6))
mk = list(res); cv = [res[k]["cvR2"] for k in mk]; ins = [res[k]["R2"] for k in mk]
xb = np.arange(len(mk)); w = .38
ax.bar(xb - w / 2, ins, w, color="#bdc3c7", label="in-sample R²")
ax.bar(xb + w / 2, cv, w, color="#E67E22", label="5-fold CV R²")
for x, c in zip(xb, cv): ax.text(x + w / 2, c + .004, f"{c:.2f}", ha="center", fontsize=9)
ax.axhline(ceil[1], color="#7f8c8d", ls="--", lw=1.5); ax.text(len(mk) - 1, ceil[1] + .01, f"season ceiling {ceil[1]:.2f} (year factor)", ha="right", fontsize=8, color="#555")
ax.set_xticks(xb); ax.set_xticklabels(mk, rotation=18, ha="right", fontsize=9); ax.set_ylabel("R²  (measured yield, all years pooled)")
ax.set_title(f"All yield data fitted to trees, pooled 2022-2024 (n={n} tree-years)\nclimate captures most; EBI x cultivar and bloom timing add a little; NGRDI/spatial do not generalize", fontweight="bold", fontsize=10.5)
ax.legend(); fig.tight_layout()
for d in (FIG, SC): fig.savefig(os.path.join(d, "Yield_AllYears_Model.png"), dpi=140, bbox_inches="tight")
print("saved figure + Yield_AllYears_Joined.xlsx")
