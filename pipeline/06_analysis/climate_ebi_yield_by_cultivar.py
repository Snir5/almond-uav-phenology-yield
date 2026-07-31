#!/usr/bin/env python3
"""
climate_ebi_yield_by_cultivar.py — the climate -> EBI -> yield triangle tested
WITHIN each cultivar (UEF, 53), year by year, same-season only.

Per cultivar it reports:
  (1) climate -> EBI   : Chill Portions vs annual mean EBI          (year level, n=4)
  (2) climate -> yield : Chill Portions vs annual mean measured yield(year level, n=3)
  (3) EBI -> yield     : same-year tree-level r/slope for 2022,2023,2024
  (4) all three together: tree-level OLS  yield ~ EBI + year(=climate proxy),
      pooled over the measured years, with a nested F-test for the EBI effect
      adjusted for the between-year (climate) shift.

Chilling = Chill Portions (Dynamic Model). No lag. from-scratch stats.
-> Results_Analysis/climate_ebi_yield_by_cultivar.json
"""
import os, sys, csv, math, json
import numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import f_pvalue, t_pvalue, r_pvalue, ols_rss, sig_stars
BASE = find_thesis_root(); OUT = os.path.join(BASE, "Results_Analysis")
SC = os.environ.get("SCRATCH_DIR", OUT)
MEAS = [2022, 2023, 2024]; ALLY = [2021, 2022, 2023, 2024]


def star(p): return sig_stars(p) if (p == p) else ""


def pear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = ~np.isnan(x) & ~np.isnan(y); x, y = x[m], y[m]; n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0: return dict(r=None, p=None, n=n)
    r = float(np.corrcoef(x, y)[0, 1]); return dict(r=round(r, 3), p=round(r_pvalue(r, n), 4), n=n, sig=star(r_pvalue(r, n)))


def linreg(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = ~np.isnan(x) & ~np.isnan(y); x, y = x[m], y[m]; n = len(x)
    if n < 3: return dict(n=n)
    b1, b0 = np.polyfit(x, y, 1); pred = b1 * x + b0; rss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum()
    se = math.sqrt(rss / (n - 2) / ((x - x.mean()) ** 2).sum()); t = b1 / se
    return dict(slope=round(float(b1), 2), r=round(float(np.corrcoef(x, y)[0, 1]), 3),
                p=round(float(t_pvalue(abs(t), n - 2)), 4), n=n, sig=star(t_pvalue(abs(t), n - 2)))


M = load_master()
def col(rs, c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in rs], float)
CP = {y: float(np.nanmean(col(M, f"Chill_Portions_{y}"))) for y in ALLY}
# yield join
yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc: ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"]); coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))
mlat, mlon = col(M, "Latitude"), col(M, "Longitude")
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
    mp[t] = j; uy.add(t); um.add(j)

RES = {"chilling": "Chill Portions (Dynamic Model)", "chill_portions_by_year": CP,
       "design": "within cultivar; same-season EBI and yield; no lag"}

for C in ["UEF", "53"]:
    idxset = set(i for i, r in enumerate(M) if r.get("cultivar") == C)
    D = {}
    # (1) climate -> EBI  (annual mean EBI over all 4 years)
    meanEBI = {}
    for y in ALLY:
        e = np.array([M[i].get(f"EBI_Norm_{y}") for i in idxset], float); e = e[~np.isnan(e)]
        meanEBI[y] = float(e.mean())
    D["climate_to_EBI_year"] = {**pear([CP[y] for y in ALLY], [meanEBI[y] for y in ALLY]),
                                "note": "n=4 years, directional", "annual_mean_EBI": {y: round(meanEBI[y], 4) for y in ALLY}}
    # (2) climate -> yield (annual mean measured yield over 3 years)
    meanY = {}
    for y in MEAS:
        vals = [ymeas[t].get(y) for t, j in mp.items() if M[j].get("cultivar") == C and ymeas[t].get(y) is not None]
        meanY[y] = float(np.mean(vals)) if vals else np.nan
    D["climate_to_yield_year"] = {**pear([CP[y] for y in MEAS], [meanY[y] for y in MEAS]),
                                  "note": "n=3 years, descriptive", "annual_mean_yield": {y: round(meanY[y], 3) for y in MEAS}}
    # (3) EBI -> yield tree-level per year
    ey = {}
    for y in MEAS:
        E = []; Y = []
        for t, j in mp.items():
            if M[j].get("cultivar") != C: continue
            v = ymeas[t].get(y); e = M[j].get(f"EBI_Norm_{y}")
            if v is not None and e is not None: E.append(e); Y.append(v)
        ey[y] = linreg(E, Y)
    D["EBI_to_yield_tree_by_year"] = ey
    # (4) all three together: yield ~ EBI + year(climate), pooled tree-level
    E = []; Y = []; yr = []
    for t, j in mp.items():
        if M[j].get("cultivar") != C: continue
        for y in MEAS:
            v = ymeas[t].get(y); e = M[j].get(f"EBI_Norm_{y}")
            if v is not None and e is not None: E.append(e); Y.append(v); yr.append(y)
    E = np.array(E); Y = np.array(Y); yr = np.array(yr); n = len(Y)
    d22 = (yr == 2022).astype(float); d24 = (yr == 2024).astype(float)  # 2023 = reference
    def dm(cols): return np.column_stack([np.ones(n)] + cols)
    rss_full, kf, coef = ols_rss(dm([E, d22, d24]), Y)          # EBI + year
    rss_yr, ky, _ = ols_rss(dm([d22, d24]), Y)                  # year only
    rss_ebi, ke, _ = ols_rss(dm([E]), Y)                        # EBI only
    rss_null, kn, _ = ols_rss(dm([]), Y)
    def Ft(rss_r, dr, rss_f, dfk):
        F = ((rss_r - rss_f) / (dfk - dr)) / (rss_f / (n - dfk)); return round(float(F), 2), round(float(f_pvalue(F, dfk - dr, n - dfk)), 4)
    F_ebi, p_ebi = Ft(rss_yr, ky, rss_full, kf)     # EBI effect | year(climate)
    F_yr, p_yr = Ft(rss_ebi, ke, rss_full, kf)      # year(climate) effect | EBI
    D["all_three_together"] = {"model": "yield ~ EBI + year(climate proxy), pooled tree-level", "n": n,
        "EBI_slope_adj": round(float(coef[1]), 2), "EBI_given_climate": {"F": F_ebi, "p": p_ebi, "sig": star(p_ebi)},
        "climate_given_EBI": {"F": F_yr, "p": p_yr, "sig": star(p_yr)},
        "R2_full": round(1 - rss_full / rss_null, 3), "R2_year_only": round(1 - rss_yr / rss_null, 3)}
    RES[C] = D

json.dump(RES, open(os.path.join(OUT, "climate_ebi_yield_by_cultivar.json"), "w"), indent=2, default=str)
json.dump(RES, open(os.path.join(SC, "climate_ebi_yield_by_cultivar.json"), "w"), indent=2, default=str)
print(json.dumps(RES, indent=1, default=str))
