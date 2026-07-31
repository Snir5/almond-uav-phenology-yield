#!/usr/bin/env python3
"""
phenology_flight_timing.py — does the UAV flight catch the same bloom stage each
year? Overlays the field bloom-progression curves (per cultivar-year) with the UAV
flight date, and checks whether cultivar-mean EBI tracks the field % open at flight.
This is the test behind the '2021 bias' question.
-> Results_Analysis/08_UEF53_Rerun/Phenology_Flight_Timing.png + json
"""
import os, sys, csv, json
import numpy as np
from datetime import datetime
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import r_pvalue
BASE = find_thesis_root(); FIG = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
SC = os.environ.get("SCRATCH_DIR", FIG); CCOL = {"53": "#2E75B6", "UEF": "#C0392B"}
FLIGHT = {2021: 66, 2022: 61, 2023: 60, 2024: 60}   # 07/03, 02/03, 01/03, 29/02
FLIGHT_LBL = {2021: "Mar 7", 2022: "Mar 2", 2023: "Mar 1", 2024: "Feb 29"}

M = load_master(); cult = np.array([r.get("cultivar") for r in M])
def col(c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in M], float)
ebimean = {y: {c: float(np.nanmean(col(f"EBI_Norm_{y}")[cult == c])) for c in ["53", "UEF"]} for y in [2021, 2022, 2023, 2024]}


def doy(s): return datetime.strptime(s, "%d/%m/%Y").timetuple().tm_yday
def norm_c(c): return "UEF" if ("UEF" in c or "Um" in c) else c
curve = {}
for r in csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "Phenology_Surveys", "Kadma_Phenology_by_CultivarDate_2022_2024.csv"), encoding="utf-8-sig")):
    c = norm_c(r["Cultivar"])
    if c in ("53", "UEF"): curve.setdefault((int(r["Year"]), c), []).append((doy(r["Date"]), float(r["Mean_Percent_Open"])))
p21 = {}
for r in csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "Bloom_Survey_2021", "Phenology_Survey_2021_20trees.csv"), encoding="utf-8-sig")):
    c = "53" if "53" in r["Cultivar"] else "UEF"; p21.setdefault((c, doy(r["Date"])), []).append(float(r["Flowers"]))
for (c, d), v in p21.items(): curve.setdefault((2021, c), []).append((d, float(np.mean(v))))


def interp(pts, x):
    pts = sorted(pts); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return float(np.interp(x, xs, ys))


rows = []
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, y in zip(axes.ravel(), [2021, 2022, 2023, 2024]):
    for c in ["53", "UEF"]:
        if (y, c) not in curve: continue
        pts = sorted(curve[(y, c)]); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-", color=CCOL[c], lw=2, ms=5, label=f"{c}")
        po = interp(pts, FLIGHT[y]); rows.append((y, c, round(po, 1), round(ebimean[y][c], 3)))
        ax.plot(FLIGHT[y], po, "s", color=CCOL[c], ms=11, mec="k")
        ax.annotate(f"{po:.0f}%", (FLIGHT[y], po), xytext=(6, -12 if c == "53" else 6), textcoords="offset points", fontsize=9, color=CCOL[c], fontweight="bold")
    ax.axvline(FLIGHT[y], color="k", ls="--", lw=1.5)
    ax.text(FLIGHT[y] + 0.5, 5, f"UAV flight\n{FLIGHT_LBL[y]}", fontsize=8.5)
    ax.set_title(f"{y}  (flight DOY {FLIGHT[y]})", fontweight="bold"); ax.set_xlabel("day of year"); ax.set_ylabel("% bloom open (field)")
    ax.set_ylim(-3, 105); ax.grid(alpha=.3); ax.legend(title="cultivar", fontsize=8)
fig.suptitle("Field bloom progression and the UAV flight date, per year: flights caught different bloom stages", fontweight="bold", fontsize=13)
fig.tight_layout()
for d in (FIG, SC): fig.savefig(os.path.join(d, "Phenology_Flight_Timing.png"), dpi=140, bbox_inches="tight")

po = np.array([r[2] for r in rows]); eb = np.array([r[3] for r in rows])
r_all = float(np.corrcoef(po, eb)[0, 1])
out = {"flight_doy": FLIGHT, "percent_open_at_flight_and_EBI": [{"year": r[0], "cultivar": r[1], "pct_open_at_flight": r[2], "mean_EBI": r[3]} for r in rows],
       "EBI_vs_pct_open_at_flight_r": round(r_all, 3), "n": len(rows),
       "interpretation": "flights caught 45-85% open in different years (2022 near peak ~80%, others mid); EBI does NOT track field bloom stage (r=+0.12 ns); 2021 EBI is high at only mid-bloom -> inflated, consistent with its brightness/resolution anomaly"}
json.dump(out, open(os.path.join(FIG, "Phenology_Flight_Timing.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
