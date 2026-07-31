#!/usr/bin/env python3
"""
dormancy_models_from_climate.py — reproduce (at our scale) the paper's climate step:
compute the dormancy models from our station hourly temperature and relate them to our
bloom-timing benchmark. Paper (Oren et al.) uses Chill Hours (CH), Utah Chill Units
(CU), Dynamic Chill Portions (CP) and the mechanistic CT model, then a Random Forest
ensemble to predict satellite bloom date over 3840 grid-cell-years (leave-one-year-out
CV MAE 2.98 d, R2 0.80). We have ONE station and FOUR seasons, so an RF is not
identifiable; instead we compute CH/CU/CP from the 10-min temperature and evaluate each
against the field 50%-bloom day (the paper's per-model evaluation that precedes the RF).
-> Results_Analysis/dormancy_models.json + figure
"""
import os, sys, csv, glob, math, json
from datetime import datetime
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import r_pvalue
BASE = find_thesis_root(); FIG = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
SC = os.environ.get("SCRATCH_DIR", FIG)

# ---- load hourly temperature per SEASON (Oct 15 prev-year -> spring), file per year ----
# columns: station, datetime(dd/mm/YYYY HH:MM), RH, Temp(C), wind, rain
def load_temp(path):
    out = []
    for r in csv.reader(open(path, encoding="utf-8-sig")):
        if len(r) < 4 or r[0].startswith("תחנה") or not r[1]: continue
        try:
            dt = datetime.strptime(r[1].strip(), "%d/%m/%Y %H:%M"); T = float(r[2 + 1])
        except Exception:
            continue
        out.append((dt, T))
    return out


# map each file to the bloom SEASON year (season spans Oct(y-1)-Mar(y))
files = {}
for f in glob.glob(os.path.join(BASE, "Climate_Data", "*.csv")):
    rows = load_temp(f)
    if not rows: continue
    # season year = year of the spring (Jan-Mar) records
    springyrs = [dt.year for dt, T in rows if dt.month in (1, 2, 3)]
    if not springyrs: continue
    y = max(set(springyrs), key=springyrs.count)
    files.setdefault(y, f)
print("season files:", {y: os.path.basename(f) for y, f in sorted(files.items())})


def hourly(rows):
    # average 10-min temps to hourly
    buckets = {}
    for dt, T in rows:
        key = dt.replace(minute=0)
        buckets.setdefault(key, []).append(T)
    return sorted((k, float(np.mean(v))) for k, v in buckets.items())


def chill_hours(hrs, end):
    return sum(1 for dt, T in hrs if dt <= end and 0.0 <= T <= 7.2)


def utah(hrs, end):
    def cu(T):
        if T <= 1.4: return 0.0
        if T <= 2.4: return 0.5
        if T <= 9.1: return 1.0
        if T <= 12.4: return 0.5
        if T <= 15.9: return 0.0
        if T <= 18.0: return -0.5
        return -1.0
    return sum(cu(T) for dt, T in hrs if dt <= end)


def dynamic_cp(hrs, end):
    # Fishman/Erez dynamic model (standard chillR constants), hourly
    e0, e1, a0, a1, slp, tetmlt = 4153.5, 12888.8, 139500.0, 2.567e18, 1.6, 277.0
    aa = a0 / a1; ee = e1 - e0; inter = 0.0; cp = 0.0
    for dt, T in hrs:
        if dt > end: break
        TK = T + 273.0
        flmprt = slp * tetmlt * (TK - tetmlt) / TK
        sr = math.exp(flmprt); xi = sr / (1 + sr)
        xs = aa * math.exp(ee / TK); ak1 = a1 * math.exp(-e1 / TK)
        x = xs - (xs - inter) * math.exp(-ak1)
        if x >= 1.0:
            delt = x * xi; inter = x - delt; cp += delt
        else:
            inter = x
    return cp


M = load_master()
def col(c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in M], float)
CPm = {y: float(np.nanmean(col(f"Chill_Portions_{y}"))) for y in [2021, 2022, 2023, 2024]}
CHm = {y: float(np.nanmean(col(f"Chill_Hrs_{y}"))) for y in [2021, 2022, 2023, 2024]}

# bloom timing t50 (cultivar-mean per year) from phenology
from datetime import datetime as _d
def doy(s): return _d.strptime(s, "%d/%m/%Y").timetuple().tm_yday
def nc(c): return "UEF" if ("UEF" in c or "Um" in c) else c
curve = {}
for r in csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "Phenology_Surveys", "Kadma_Phenology_by_CultivarDate_2022_2024.csv"), encoding="utf-8-sig")):
    c = nc(r["Cultivar"])
    if c in ("53", "UEF"): curve.setdefault((int(r["Year"]), c), []).append((doy(r["Date"]), float(r["Mean_Percent_Open"])))
from collections import defaultdict
p21 = defaultdict(list)
for r in csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "Bloom_Survey_2021", "Phenology_Survey_2021_20trees.csv"), encoding="utf-8-sig")):
    c = "53" if "53" in r["Cultivar"] else "UEF"; p21[(c, doy(r["Date"]))].append(float(r["Flowers"]))
for (c, d), v in p21.items(): curve.setdefault((2021, c), []).append((d, float(np.mean(v))))
def t50(pts):
    pts = sorted(pts); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if max(ys) < 50: return np.nan
    for i in range(1, len(xs)):
        if ys[i - 1] < 50 <= ys[i]: return xs[i - 1] + (50 - ys[i - 1]) / (ys[i] - ys[i - 1]) * (xs[i] - xs[i - 1])
    return np.nan
bloom = {}
for y in [2021, 2022, 2023, 2024]:
    ts = [t50(curve[(y, c)]) for c in ("53", "UEF") if (y, c) in curve and t50(curve[(y, c)]) == t50(curve[(y, c)])]
    if ts: bloom[y] = float(np.mean(ts))

# compute models to Feb 28 (DOY 59, pre-bloom) each season
res = {"note": "Dormancy models computed from station 10-min temperature to Feb 28 each season; "
               "related to field 50%-bloom day. n=4 seasons -> per-model evaluation only (no RF)."}
rows = []
for y in sorted(files):
    hrs = hourly(load_temp(files[y]))
    end = datetime(y, 2, 28, 23, 0)
    CH = chill_hours(hrs, end); CU = utah(hrs, end); CP = dynamic_cp(hrs, end)
    rows.append({"year": y, "CH_computed": round(CH, 1), "CU_computed": round(CU, 1), "CP_computed": round(CP, 2),
                 "CP_master": round(CPm.get(y, np.nan), 1), "CH_master": round(CHm.get(y, np.nan), 1),
                 "bloom_t50": round(bloom.get(y, np.nan), 1)})
res["by_season"] = rows
print(json.dumps(res, indent=1))

def corr(a, b):
    a = np.array(a, float); b = np.array(b, float); m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 3: return None
    r = float(np.corrcoef(a[m], b[m])[0, 1]); return round(r, 3), round(r_pvalue(r, int(m.sum())), 3), int(m.sum())
bl = [r["bloom_t50"] for r in rows]
res["model_vs_bloom_timing"] = {
    "CH_computed": corr([r["CH_computed"] for r in rows], bl),
    "CU_computed": corr([r["CU_computed"] for r in rows], bl),
    "CP_computed": corr([r["CP_computed"] for r in rows], bl),
    "CP_master": corr([r["CP_master"] for r in rows], bl)}
res["CP_computed_vs_master_r"] = corr([r["CP_computed"] for r in rows], [r["CP_master"] for r in rows])
json.dump(res, open(os.path.join(FIG, "dormancy_models.json"), "w"), indent=2, default=str)
json.dump(res, open(os.path.join(SC, "dormancy_models.json"), "w"), indent=2, default=str)
print("\nmodel vs bloom timing:", json.dumps(res["model_vs_bloom_timing"], default=str))
print("CP computed vs master:", res["CP_computed_vs_master_r"])

# figure — the paper's Fig.10 compares four dormancy models (CT, CH, CU, Dynamic-Model CP)
# against remote-sensing bloom date over 3840 grid-cell-years. We have one station and n=4
# seasons, so this is the same comparison at our scale for the three models computable from
# hourly temperature (CH, CU, CP); the Carbohydrate-Temperature model (CT) is not implemented
# here. Panel D is our own addition: validating the computed Dynamic-Model CP against the
# value already stored in the master.
fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
models = [("CH_computed", "Chill Hours (CH)", "#E67E22", ax[0], "A"),
          ("CU_computed", "Utah Chill Units (CU)", "#8E44AD", ax[1], "B"),
          ("CP_computed", "Dynamic Model, Chill Portions (CP)", "#4aa3df", ax[2], "C")]
for key, label, color, a, tag in models:
    xs = [r[key] for r in rows]; ys = [r["bloom_t50"] for r in rows]
    a.scatter(xs, ys, s=110, color=color, edgecolors="k", zorder=3)
    for r in rows:
        a.annotate(str(r["year"]), (r[key], r["bloom_t50"]), xytext=(5, 4), textcoords="offset points", fontsize=9)
    cc = res["model_vs_bloom_timing"][key]
    a.set_title(f"{tag}. {label} vs bloom timing\nr={cc[0] if cc else 'na'} (n=4, directional)", fontweight="bold", fontsize=9.5)
    a.set_xlabel(f"{label} to Feb 28 (computed)"); a.grid(alpha=.3)
ax[0].set_ylabel("field 50% bloom DOY")
a = ax[3]
cpc = [r["CP_computed"] for r in rows]; cpm = [r["CP_master"] for r in rows]
a.scatter(cpm, cpc, s=110, color="#16A085", edgecolors="k")
lim = [min(cpm + cpc) - 2, max(cpm + cpc) + 2]; a.plot(lim, lim, "k--", lw=1)
for r in rows: a.annotate(str(r["year"]), (r["CP_master"], r["CP_computed"]), xytext=(5, 4), textcoords="offset points", fontsize=9)
a.set_title(f"D. Our Dynamic-Model CP vs master CP (validation)\nr={res['CP_computed_vs_master_r'][0] if res['CP_computed_vs_master_r'] else 'na'}", fontweight="bold", fontsize=9.5)
a.set_xlabel("master Chill Portions"); a.set_ylabel("computed Chill Portions"); a.grid(alpha=.3); a.set_aspect("equal")
fig.suptitle("Dormancy models from station temperature (paper's climate step, at our scale, n=4 seasons)", fontweight="bold", fontsize=12)
fig.tight_layout()
for d in (FIG, SC): fig.savefig(os.path.join(d, "Dormancy_Models.png"), dpi=140, bbox_inches="tight")
print("saved figure")
