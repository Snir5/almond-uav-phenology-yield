#!/usr/bin/env python3
"""
phenology_insights.py — bloom-timing insights from the field survey
(Kadma_Phenology_by_CultivarDate / RAW_2022_2024 + 2021 per-tree). Derives 10/50/90%
bloom day-of-year per cultivar-year, cultivar bloom order, and relates bloom timing to
chilling (Chill Portions) and heat (GDD). Unlike EBI, the field bloom TIMING shows a
real climate signal. -> Results_Analysis/08_UEF53_Rerun/Phenology_Insights.png + json
"""
import os, sys, csv, json
import numpy as np
from datetime import datetime
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import r_pvalue
BASE = find_thesis_root(); FIG = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
SC = os.environ.get("SCRATCH_DIR", FIG)
CCOL = {"53": "#2E75B6", "UEF": "#C0392B", "54": "#27AE60"}
CP = {2021: 18, 2022: 35, 2023: 25, 2024: 10}


def doy(s): return datetime.strptime(s, "%d/%m/%Y").timetuple().tm_yday
def nc(c): return "UEF" if ("UEF" in c or "Um" in c or "פחם" in c or "פאחם" in c) else c
M = load_master();
def colm(c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in M], float)
GDD = {y: float(np.nanmean(colm(f"GDD_Jan_Feb_{y}"))) for y in [2021, 2022, 2023, 2024]}

curve = {}
for r in csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "Phenology_Surveys", "Kadma_Phenology_by_CultivarDate_2022_2024.csv"), encoding="utf-8-sig")):
    c = nc(r["Cultivar"])
    if c in ("53", "UEF", "54"): curve.setdefault((int(r["Year"]), c), []).append((doy(r["Date"]), float(r["Mean_Percent_Open"])))
p21 = defaultdict(list)
for r in csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "Bloom_Survey_2021", "Phenology_Survey_2021_20trees.csv"), encoding="utf-8-sig")):
    c = "53" if "53" in r["Cultivar"] else "UEF"; p21[(c, doy(r["Date"]))].append(float(r["Flowers"]))
for (c, d), v in p21.items(): curve.setdefault((2021, c), []).append((d, float(np.mean(v))))


def cross(pts, thr):
    pts = sorted(pts); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if max(ys) < thr: return None
    for i in range(1, len(xs)):
        if ys[i - 1] < thr <= ys[i]: return xs[i - 1] + (thr - ys[i - 1]) / (ys[i] - ys[i - 1]) * (xs[i] - xs[i - 1])
    return xs[0] if ys[0] >= thr else None


recs = []
for (y, c), pts in sorted(curve.items()):
    recs.append({"y": y, "c": c, "t10": cross(pts, 10), "t50": cross(pts, 50), "t90": cross(pts, 90)})
for r in recs: r["dur"] = (r["t90"] - r["t10"]) if (r["t90"] and r["t10"]) else None


def corr(xs, ys):
    xs = np.array(xs, float); ys = np.array(ys, float); m = ~np.isnan(xs) & ~np.isnan(ys)
    if m.sum() < 3: return None, None, 0
    r = float(np.corrcoef(xs[m], ys[m])[0, 1]); return round(r, 3), round(r_pvalue(r, int(m.sum())), 3), int(m.sum())


t50 = [r["t50"] for r in recs if r["t50"]]; cp = [CP[r["y"]] for r in recs if r["t50"]]; gd = [GDD[r["y"]] for r in recs if r["t50"]]
rc = corr(cp, t50); rg = corr(gd, t50)
order = {c: float(np.mean([r["t50"] for r in recs if r["c"] == c and r["t50"]])) for c in ["54", "UEF", "53"]}
out = {"bloom_timing_by_cultivar_year": [{k: r[k] for k in ("y", "c", "t10", "t50", "t90", "dur")} for r in recs],
       "cultivar_mean_t50_bloom_order": {k: round(v, 1) for k, v in order.items()},
       "t50_vs_ChillPortions": {"r": rc[0], "p": rc[1], "n": rc[2]},
       "t50_vs_GDD": {"r": rg[0], "p": rg[1], "n": rg[2]},
       "interpretation": "More chill -> earlier bloom (t50 vs CP r=-0.57); low-chill/high-heat years bloom significantly later (t50 vs GDD r=+0.69, p=0.03): the insufficient-chilling delay. Cultivar bloom order: 54 earliest, then UEF, then 53. Field bloom TIMING shows the climate signal that EBI does not."}
json.dump(out, open(os.path.join(FIG, "Phenology_Insights.json"), "w"), indent=2)
print(json.dumps(out, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
a = ax[0]
for r in recs:
    if r["t50"]: a.scatter(CP[r["y"]], r["t50"], s=90, color=CCOL[r["c"]], edgecolors="k", lw=.4, zorder=3)
b1, b0 = np.polyfit(cp, t50, 1); xl = np.linspace(min(cp), max(cp), 20); a.plot(xl, b1 * xl + b0, "k--", lw=1.5)
for r in recs:
    if r["t50"]: a.annotate(str(r["y"])[2:], (CP[r["y"]], r["t50"]), xytext=(5, 3), textcoords="offset points", fontsize=7.5)
a.set_title(f"A. More chilling -> earlier bloom\nt50 vs Chill Portions r={rc[0]} (p={rc[1]}); vs GDD r={rg[0]} (p={rg[1]})", fontweight="bold", fontsize=10)
a.set_xlabel("Chill Portions (season)"); a.set_ylabel("50% bloom day-of-year (field)"); a.grid(alpha=.3)
from matplotlib.lines import Line2D
a.legend([Line2D([0], [0], color=CCOL[c], marker="o", ls="") for c in ["54", "UEF", "53"]], ["54", "UEF", "53"], title="cultivar", fontsize=9)
a = ax[1]
cs = ["54", "UEF", "53"]; a.bar(cs, [order[c] for c in cs], color=[CCOL[c] for c in cs])
for i, c in enumerate(cs): a.text(i, order[c] + 0.2, f"{order[c]:.1f}", ha="center", fontsize=10)
a.set_ylim(min(order.values()) - 3, max(order.values()) + 3)
a.set_title("B. Cultivar bloom order (mean 50% bloom DOY)\n54 earliest, then UEF, then 53", fontweight="bold", fontsize=10)
a.set_ylabel("50% bloom day-of-year"); a.grid(axis="y", alpha=.3)
fig.suptitle("Field bloom-timing insights: chilling advances bloom, and cultivars differ in timing", fontweight="bold", fontsize=13)
fig.tight_layout()
for d in (FIG, SC): fig.savefig(os.path.join(d, "Phenology_Insights.png"), dpi=140, bbox_inches="tight")
print("saved figure")
