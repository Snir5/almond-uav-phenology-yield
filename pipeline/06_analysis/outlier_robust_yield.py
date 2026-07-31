#!/usr/bin/env python3
"""
outlier_robust_yield.py: quantify how much per-canopy outlier removal shifts
EBI, and whether it changes the EBI-yield finding. Companion to
apply_outlier_robust_v4.py (rasterio, run on the user's machine; this script
needs no rasterio and runs in the analysis sandbox).

Two questions, in order:
  1. How much does outlier removal actually change EBI? (diagnostic first,
     before touching yield: if method A/B barely move EBI, the yield test
     result will necessarily look like EBI's, and that is the point, not a
     bug.) Reports mean/median outlier fraction per year and the
     plain-vs-robust EBI correlation and mean absolute shift.
  2. Does the robust EBI change the same-year, within-cultivar yield battery
     (section 12.1 / 19 style) relative to the existing (non-robust) EBI?

Requires 4band_mosaic/Master_Trees_OutlierRobust_v4.xlsx (produced by
apply_outlier_robust_v4.py). If missing, prints instructions and exits.
-> Results_Analysis/08_UEF53_Rerun/outlier_robust_yield.json + Outlier_Robust_EBI.png
"""
import os, sys, csv, math, json
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import r_pvalue, f_pvalue, ols_rss, sig_stars

BASE = find_thesis_root()
V4_PATH = os.path.join(BASE, "4band_mosaic", "Master_Trees_OutlierRobust_v4.xlsx")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
YEARS = [2021, 2022, 2023, 2024]

if not os.path.exists(V4_PATH):
    print(f"NOT FOUND: {V4_PATH}")
    print("Run this on a machine with rasterio first:")
    print("  python3 pipeline/03_ebi_extraction/apply_outlier_robust_v4.py")
    print("then copy the output xlsx back into 4band_mosaic/ and re-run this script.")
    sys.exit(0)


def star(p): return sig_stars(p) if p == p else ""


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y); x, y = x[m], y[m]; n = len(x)
    if n < 4: return {"r": None, "n": n}
    r = float(np.corrcoef(x, y)[0, 1]); p = float(r_pvalue(r, n))
    return {"r": round(r, 3), "p": round(p, 4), "n": n, "sig": star(p)}


import openpyxl
wb = openpyxl.load_workbook(V4_PATH, read_only=True, data_only=True)
ws = wb.active; raw = list(ws.iter_rows(values_only=True)); H = raw[0]
v4rows = {int(r[H.index("Zone_Value")]): dict(zip(H, r)) for r in raw[1:] if r[H.index("Zone_Value")] is not None}
wb.close()
print(f"v4 outlier-robust rows: {len(v4rows)}")

RESULTS = {"note": "Per-canopy outlier removal (method A: modified Z-score |z|>3.5 any channel; "
                   "method B: brightness 5-95 percentile trim), applied before averaging R,G,B "
                   "per crown, then EBI recomputed from the cleaned means. Compares to the "
                   "existing (non-robust) EBI_ofMeans already validated in section 13.5."}

# ---------------- Q1: how much does outlier removal move EBI, at all (diagnostic) ----------------
diag = {}
for y in YEARS:
    outlier_pct = [v.get(f"pct_outlier_A_{y}") for v in v4rows.values() if v.get(f"pct_outlier_A_{y}") is not None]
    trim_pct = [v.get(f"pct_trimmed_B_{y}") for v in v4rows.values() if v.get(f"pct_trimmed_B_{y}") is not None]
    plain = np.array([v.get(f"EBI_ofMeans_{y}") for v in v4rows.values()], float)
    rob = np.array([v.get(f"EBI_ofMeans_robust_{y}") for v in v4rows.values()], float)
    trim = np.array([v.get(f"EBI_ofMeans_trim_{y}") for v in v4rows.values()], float)
    m = ~np.isnan(plain) & ~np.isnan(rob)
    diag[str(y)] = {
        "mean_pct_pixels_flagged_outlier_A": round(float(np.mean(outlier_pct)), 2) if outlier_pct else None,
        "mean_pct_pixels_trimmed_B": round(float(np.mean(trim_pct)), 2) if trim_pct else None,
        "plain_vs_robust_EBI": pearson(plain[m], rob[m]),
        "mean_abs_shift_robust": round(float(np.nanmean(np.abs(rob[m] - plain[m]))), 4),
        "mean_abs_shift_trim": round(float(np.nanmean(np.abs(trim[m] - plain[m]))), 4) if np.isfinite(trim[m]).any() else None,
    }
RESULTS["Q1_how_much_does_it_move_EBI"] = diag
print("\nQ1 diagnostic (how much outlier removal moves EBI):")
print(json.dumps(diag, indent=2, default=str))

# ---------------- Q2: does robust EBI change the yield battery? ----------------
KEEP = {"UEF", "53"}
rows = [r for r in load_master() if r.get("cultivar") in KEEP]
mlat = np.array([r.get("Latitude") for r in rows], float)
mlon = np.array([r.get("Longitude") for r in rows], float)
zone = [r.get("Zone_Value") for r in rows]
cultivar = [r.get("cultivar") for r in rows]

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
cand.sort(); uy = set(); um = set(); mp = {}
for d, t, j in cand:
    if t in uy or j in um: continue
    mp[t] = j; uy.add(t); um.add(j)
print(f"\nyield trees matched to master (<=8m): {len(mp)}")

FIELDS = {"plain": "EBI_ofMeans", "robust": "EBI_ofMeans_robust", "trim": "EBI_ofMeans_trim"}
by_cultivar = {}
for C in ["UEF", "53"]:
    by_cultivar[C] = {}
    for yr in [2022, 2023, 2024]:
        cell = {}
        for label, field in FIELDS.items():
            xs = []; ys = []
            for t, j in mp.items():
                if cultivar[j] != C: continue
                z = zone[j]; v4 = v4rows.get(int(z)) if z is not None else None
                yv = ymeas[t].get(yr)
                if v4 is None or yv is None: continue
                val = v4.get(f"{field}_{yr}")
                if val is None: continue
                xs.append(float(val)); ys.append(yv)
            cell[label] = pearson(xs, ys)
        by_cultivar[C][str(yr)] = cell
RESULTS["Q2_yield_battery_plain_vs_robust_vs_trim"] = by_cultivar
print("\nQ2 same-year, within-cultivar yield correlations (plain vs robust vs trimmed EBI):")
print(json.dumps(by_cultivar, indent=2, default=str))


def ancova(field, yr=2023):
    E = []; Yv = []; CU = []
    for t, j in mp.items():
        z = zone[j]; v4 = v4rows.get(int(z)) if z is not None else None
        yv = ymeas[t].get(yr)
        if v4 is None or yv is None: continue
        val = v4.get(f"{field}_{yr}")
        if val is None: continue
        E.append(float(val)); Yv.append(yv); CU.append(cultivar[j])
    E = np.array(E); Yv = np.array(Yv); CU = np.array(CU); n = len(Yv)
    def design(cols): return np.column_stack([np.ones(n)] + cols)
    dum = [(CU == "UEF").astype(float)]; inter = [((CU == "UEF").astype(float)) * E]
    rss_full, kf, _ = ols_rss(design([E] + dum), Yv)
    rss_null, kn, _ = ols_rss(design([]), Yv)
    rss_int, ki, _ = ols_rss(design([E] + dum + inter), Yv)
    def Ft(rss_r, dr, rss_f, dfk):
        F = ((rss_r - rss_f) / (dfk - dr)) / (rss_f / (n - dfk))
        return round(float(F), 2), round(float(f_pvalue(F, dfk - dr, n - dfk)), 4)
    Fi, pi = Ft(rss_full, kf, rss_int, ki)
    return {"n": n, "EBIxcultivar_interaction": {"F": Fi, "p": pi, "sig": star(pi)},
            "R2": round(1 - rss_full / rss_null, 3), "R2_with_interaction": round(1 - rss_int / rss_null, 3)}


RESULTS["ANCOVA_2023"] = {label: ancova(field) for label, field in FIELDS.items()}
print("\nANCOVA 2023 (plain vs robust vs trimmed):")
print(json.dumps(RESULTS["ANCOVA_2023"], indent=2, default=str))

json.dump(RESULTS, open(os.path.join(OUT, "outlier_robust_yield.json"), "w"), indent=2, default=str)

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
a = ax[0]
years_x = YEARS
a.bar([y - 0.15 for y in years_x], [diag[str(y)]["mean_pct_pixels_flagged_outlier_A"] for y in years_x], width=0.3, label="method A (modified Z)", color="#4aa3df", edgecolor="k")
a.bar([y + 0.15 for y in years_x], [diag[str(y)]["mean_pct_pixels_trimmed_B"] for y in years_x], width=0.3, label="method B (5-95% trim, fixed)", color="#E67E22", edgecolor="k")
a.set_xticks(years_x); a.set_ylabel("mean % pixels removed per crown"); a.legend(fontsize=8.5); a.grid(alpha=.3)
a.set_title("A. How much gets removed per crown, by year", fontsize=10, fontweight="bold")

a = ax[1]
x = np.arange(2); w = 0.35
for i, (C, color) in enumerate([("UEF", "#2E86C1"), ("53", "#C0392B")]):
    rs = [by_cultivar[C]["2023"]["plain"]["r"], by_cultivar[C]["2023"]["robust"]["r"]]
    a.bar(x + (i - 0.5) * w, rs, width=w, color=color, edgecolor="k", label=f"cv {C}")
a.axhline(0, color="k", lw=.8); a.set_xticks(x); a.set_xticklabels(["plain EBI", "robust EBI (method A)"])
a.set_ylabel("Pearson r vs measured yield 2023"); a.legend(fontsize=9); a.grid(alpha=.3)
a.set_title("B. Yield correlation: plain vs outlier-robust EBI", fontsize=10, fontweight="bold")

a = ax[2]
plain2023 = np.array([v4rows.get(int(z), {}).get("EBI_ofMeans_2023") for z in zone], float)
rob2023 = np.array([v4rows.get(int(z), {}).get("EBI_ofMeans_robust_2023") for z in zone], float)
m = ~np.isnan(plain2023) & ~np.isnan(rob2023)
a.scatter(plain2023[m], rob2023[m], s=8, alpha=.35, color="#4aa3df")
lim = [np.nanmin(plain2023[m]), np.nanmax(plain2023[m])]
a.plot(lim, lim, "k--", lw=1)
a.set_xlabel("EBI_ofMeans (plain), 2023"); a.set_ylabel("EBI_ofMeans (outlier-robust), 2023")
r = diag["2023"]["plain_vs_robust_EBI"]["r"]
a.set_title(f"C. Per-tree EBI, plain vs robust, 2023\n(r={r}, all trees)", fontsize=10, fontweight="bold")
a.grid(alpha=.3)
fig.suptitle("Does per-canopy outlier removal change EBI or the yield finding?", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Outlier_Robust_EBI.png"), dpi=140, bbox_inches="tight")
print("\nsaved figure")
