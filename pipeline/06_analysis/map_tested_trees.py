#!/usr/bin/env python3
"""
map_tested_trees.py — how many yield trees were joined per year, and where they sit.
Maps the whole orchard (grey) with the yield-tested trees for each measured year
highlighted by cultivar. -> Results_Analysis/08_UEF53_Rerun/Tested_Trees_By_Year.png + json
"""
import os, sys, csv, math, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master, field_boundary
BASE = find_thesis_root(); FIG = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
SC = os.environ.get("SCRATCH_DIR", FIG); MEAS = [2022, 2023, 2024]; CCOL = {"53": "#2E75B6", "UEF": "#C0392B"}

M = load_master()
def col(c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in M], float)
mX, mY, mlat, mlon = col("X_UTM"), col("Y_UTM"), col("Latitude"), col("Longitude")
mcult = [r.get("cultivar") for r in M]
yc = [r for r in csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig"))]
coord = {}; yrs = defaultdict(set)
for r in yc:
    coord.setdefault(r["tree_id"], (float(r["latitude"]), float(r["longitude"]))); yrs[r["tree_id"]].add(int(r["year"]))
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

counts = {}; used = {}
for y in MEAS:
    tids = [t for t in coord if y in yrs[t]]
    matched = [t for t in tids if t in mp]
    withebi = [t for t in matched if M[mp[t][0]].get(f"EBI_Norm_{y}") is not None and mcult[mp[t][0]] in ("UEF", "53")]
    used[y] = withebi
    counts[y] = {"measured": len(tids), "matched": len(matched), "used": len(withebi),
                 "by_cultivar": dict(Counter(mcult[mp[t][0]] for t in withebi))}
json.dump(counts, open(os.path.join(FIG, "Tested_Trees_By_Year.json"), "w"), indent=2)
print(json.dumps(counts, indent=1))

fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
gx, gy = mX, mY
hx, hy = field_boundary(gx[~np.isnan(gx)], gy[~np.isnan(gy)])
for ax, y in zip(axes, MEAS):
    ax.scatter(mX, mY, s=7, color="#dddddd", zorder=1, label="all trees")
    ax.plot(hx, hy, color="#555", ls="--", lw=1, alpha=.6)
    for C in ["UEF", "53"]:
        js = [mp[t][0] for t in used[y] if mcult[mp[t][0]] == C]
        ax.scatter(mX[js], mY[js], s=42, color=CCOL[C], edgecolors="k", lw=.4, zorder=3, label=f"{C} (n={len(js)})")
    ax.set_aspect("equal"); ax.set_title(f"{y}  -  {len(used[y])} tested trees", fontweight="bold")
    ax.set_xlabel("X_UTM (m)"); ax.set_ylabel("Y_UTM (m)"); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.25)
fig.suptitle("Yield-tested trees per year (highlighted on the orchard)", fontweight="bold", fontsize=14)
fig.tight_layout()
for d in (FIG, SC): fig.savefig(os.path.join(d, "Tested_Trees_By_Year.png"), dpi=140, bbox_inches="tight")
print("saved Tested_Trees_By_Year.png")
