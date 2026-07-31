#!/usr/bin/env python3
"""
jul2024_spectral_yield.py — test the July-2024 multispectral acquisition
(already in the master: NDVI, NDRE, EVI, SAVI, MSAVI, GNDI, OSAVI, CI, GI, BAI,
DVI, CWSI, Tc) against measured yield. Never tested before (grep of the
analysis scripts and results JSONs turns up zero prior use of these columns
against yield; only coord_audit.py touches them, for registration QA).

Motivation (literature): Jin et al. 2023 (Front. Plant Sci., CNN tree-level
almond yield from SUMMER 4-band aerial imagery, ~2000 trees, R2=0.96, red edge
the most important band) and the broader canopy light-interception literature
(Lampinen et al. 2012; Zarate-Valdez et al. 2012, 2015) both point to SUMMER
canopy vigor / red-edge signal, not spring bloom color, as the strongest
imagery-based yield predictor in almonds. This project has never tested its
own July-2024 acquisition (which includes NDRE, a red-edge index, and CWSI, a
thermal water-stress index) against measured yield, only bloom-season EBI.

Design honesty: July 2024 is ONE date.
  - vs measured 2024 yield: same-season, causally sensible (canopy state before
    harvest predicting that harvest). This is the primary test.
  - vs measured 2022/2023 yield: the imagery POSTDATES the yield (2024 canopy
    state cannot cause a 2022/2023 harvest). Reported only as a descriptive
    "is this generally a more/less vigorous tree" check, explicitly not causal,
    and not used for any predictive claim.
Small n throughout (n=18-20 per year, same trees as the H2b repeated panel), so
every result here is directional, not inferential, consistent with project
convention for small samples.
-> Results_Analysis/08_UEF53_Rerun/jul2024_spectral_yield.json + .png
"""
import os, sys, csv, math, json
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import r_pvalue, sig_stars

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")

FIELDS = ["NDVI_Jul2024", "NDRE_Jul2024", "EVI_Jul2024", "SAVI_Jul2024", "MSAVI_Jul2024",
          "GNDI_Jul2024", "OSAVI_Jul2024", "CI_Jul2024", "GI_Jul2024", "BAI_Jul2024",
          "DVI_Jul2024", "CWSI_Jul2024", "Tc_Jul2024"]


def star(p): return sig_stars(p) if p == p else ""


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y); x, y = x[m], y[m]; n = len(x)
    if n < 4: return {"r": None, "n": n}
    r = float(np.corrcoef(x, y)[0, 1]); p = float(r_pvalue(r, n))
    return {"r": round(r, 3), "p": round(p, 4), "n": n, "sig": star(p)}


def bh_fdr(pvals):
    idx = [i for i, p in enumerate(pvals) if p == p]
    ps = sorted((pvals[i], i) for i in idx); m = len(ps)
    adj = {}; prev = 1.0
    for rank in range(m - 1, -1, -1):
        p, i = ps[rank]; val = min(prev, p * m / (rank + 1)); adj[i] = round(val, 4); prev = val
    return [adj.get(i, np.nan) for i in range(len(pvals))]


KEEP = {"UEF", "53"}
rows = [r for r in load_master() if r.get("cultivar") in KEEP]
mlat = np.array([r.get("Latitude") for r in rows], float)
mlon = np.array([r.get("Longitude") for r in rows], float)
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
print(f"yield trees matched to master (<=8m): {len(mp)}")

RESULTS = {"note": "July-2024 multispectral (single date) vs measured yield. 2024 is same-season "
                   "(causal, primary test); 2022/2023 are descriptive only (imagery postdates the "
                   "yield, not a predictive claim). FDR (Benjamini-Hochberg) applied across the 13 "
                   "candidate fields within each cultivar's 2024 test (n=9-10, directional given n)."}

by_cultivar_year = {}
for C in ["UEF", "53"]:
    by_cultivar_year[C] = {}
    for yr in [2022, 2023, 2024]:
        cell = {}; pvals = []
        for f in FIELDS:
            xs = []; ys = []
            for t, j in mp.items():
                if cultivar[j] != C: continue
                yv = ymeas[t].get(yr); xv = rows[j].get(f)
                if yv is None or xv is None: continue
                xs.append(float(xv)); ys.append(yv)
            res = pearson(xs, ys); cell[f] = res; pvals.append(res.get("p", np.nan))
        if yr == 2024:
            adj = bh_fdr(pvals)
            for f, a in zip(FIELDS, adj): cell[f]["p_fdr"] = a
        by_cultivar_year[C][str(yr)] = cell
RESULTS["by_cultivar_year"] = by_cultivar_year
print(json.dumps({C: by_cultivar_year[C]["2024"] for C in ["UEF", "53"]}, indent=2, default=str))

# pooled (both cultivars), all three years, for a quick overview too
pooled = {}
for yr in [2022, 2023, 2024]:
    cell = {}
    for f in FIELDS:
        xs = []; ys = []
        for t, j in mp.items():
            yv = ymeas[t].get(yr); xv = rows[j].get(f)
            if yv is None or xv is None: continue
            xs.append(float(xv)); ys.append(yv)
        cell[f] = pearson(xs, ys)
    pooled[str(yr)] = cell
RESULTS["pooled_both_cultivars"] = pooled

json.dump(RESULTS, open(os.path.join(OUT, "jul2024_spectral_yield.json"), "w"), indent=2, default=str)

# ---------------- figure ----------------
best2024 = {C: max(FIELDS, key=lambda f: abs(by_cultivar_year[C]["2024"][f]["r"]) if by_cultivar_year[C]["2024"][f]["r"] is not None else 0) for C in ["UEF", "53"]}
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
a = ax[0]
x = np.arange(len(FIELDS)); w = 0.35
for i, (C, color) in enumerate([("UEF", "#2E86C1"), ("53", "#C0392B")]):
    rs = [by_cultivar_year[C]["2024"][f]["r"] if by_cultivar_year[C]["2024"][f]["r"] is not None else 0 for f in FIELDS]
    a.bar(x + (i - 0.5) * w, rs, width=w, color=color, edgecolor="k", label=f"cv {C}")
a.axhline(0, color="k", lw=.8); a.set_xticks(x); a.set_xticklabels([f.replace("_Jul2024", "") for f in FIELDS], rotation=45, ha="right", fontsize=8)
a.set_ylabel("Pearson r vs measured yield 2024 (n=9-10)"); a.legend(fontsize=9); a.grid(alpha=.3)
a.set_title("A. July-2024 multispectral vs SAME-SEASON yield\n(causal test, n small, directional)", fontsize=10, fontweight="bold")

for k, C in enumerate(["UEF", "53"]):
    a = ax[k + 1]; f = best2024[C]
    xs = []; ys = []
    for t, j in mp.items():
        if cultivar[j] != C: continue
        yv = ymeas[t].get(2024); xv = rows[j].get(f)
        if yv is None or xv is None: continue
        xs.append(float(xv)); ys.append(yv)
    a.scatter(xs, ys, s=60, color="#2E86C1" if C == "UEF" else "#C0392B", edgecolors="k")
    r = by_cultivar_year[C]["2024"][f]
    if len(xs) > 2:
        b1, b0 = np.polyfit(xs, ys, 1); xr = np.linspace(min(xs), max(xs), 20); a.plot(xr, b0 + b1 * xr, "k--", lw=1.2)
    a.set_xlabel(f.replace("_Jul2024", "")); a.set_ylabel("measured yield 2024 (kg)")
    a.set_title(f"{'B' if k == 0 else 'C'}. {C} best: {f.replace('_Jul2024','')}\nr={r['r']} (n={r['n']}, p={r.get('p')})", fontsize=9.5, fontweight="bold")
    a.grid(alpha=.3)
fig.suptitle("Does the July-2024 multispectral acquisition predict measured yield?", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Jul2024_Spectral_Yield.png"), dpi=140, bbox_inches="tight")
print("\nsaved figure")
print(f"\nBest same-season (2024) candidate per cultivar: {best2024}")
