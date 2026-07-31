#!/usr/bin/env python3
"""figure_triangle_by_cultivar.py - climate -> EBI -> yield path diagram per cultivar.
Large, well-spaced nodes and readable fonts."""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root
BASE = find_thesis_root(); FIG = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
SC = os.environ.get("SCRATCH_DIR", FIG); os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(BASE, "Results_Analysis", "climate_ebi_yield_by_cultivar.json")))
CCOL = {"53": "#2E75B6", "UEF": "#C0392B"}

# node geometry (axis coords)
NW, NH = 0.17, 0.095          # node half-width / half-height
CP_ = (0.50, 0.80)            # Climate  (top center)
EB_ = (0.20, 0.23)            # EBI      (bottom left)
YL_ = (0.80, 0.23)            # Yield    (bottom right)


def node(ax, c, label, sub):
    x, y = c
    ax.add_patch(FancyBboxPatch((x - NW, y - NH), 2 * NW, 2 * NH, boxstyle="round,pad=0.015",
                                fc="#eef1fb", ec="#5b6cff", lw=2.2, zorder=3))
    ax.text(x, y + 0.022, label, ha="center", va="center", fontweight="bold", fontsize=19, zorder=4)
    ax.text(x, y - 0.042, sub, ha="center", va="center", fontsize=12, color="#555", zorder=4)


def edge(ax, p0, p1, text, color, lw, rad, lx, ly):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                 mutation_scale=26, lw=lw, color=color, zorder=2,
                                 shrinkA=2, shrinkB=2))
    ax.text(lx, ly, text, ha="center", va="center", fontsize=12.5, color=color, fontweight="bold",
            bbox=dict(fc="white", ec=color, lw=1.0, alpha=.95, boxstyle="round,pad=0.3"), zorder=6)


fig, axes = plt.subplots(1, 2, figsize=(20, 9.5))
for ax, C in zip(axes, ["UEF", "53"]):
    d = R[C]; col = CCOL[C]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    node(ax, CP_, "Climate", "Chill Portions (per year)")
    node(ax, EB_, "EBI", "bloom intensity (per tree)")
    node(ax, YL_, "Yield", "field-measured (per tree)")

    ce = d["climate_to_EBI_year"]
    edge(ax, (CP_[0] - 0.15, CP_[1] - NH), (EB_[0] + 0.03, EB_[1] + NH), f"r={ce['r']}  (n=4, ns)",
         "#9aa0a6", 2.0, 0.18, 0.235, 0.55)

    cy = d["climate_to_yield_year"]; clg = d["all_three_together"]["climate_given_EBI"]
    edge(ax, (CP_[0] + 0.15, CP_[1] - NH), (YL_[0] - 0.03, YL_[1] + NH),
         f"r={cy['r']} (n=3)\nyear effect F={clg['F']}***", "#16A085", 3.0, -0.18, 0.765, 0.55)

    y23 = d["EBI_to_yield_tree_by_year"]["2023"]; eg = d["all_three_together"]["EBI_given_climate"]
    edge(ax, (EB_[0] + NW, EB_[1]), (YL_[0] - NW, YL_[1]),
         f"2023 r={y23['r']}{'*' if y23['sig'] not in ('ns','') else ''}\nEBI | climate  F={eg['F']} {eg['sig']}",
         col, 3.4, 0.0, 0.50, 0.42)

    ey = d["EBI_to_yield_tree_by_year"]
    strip = "EBI → yield by year:   " + "     ".join(
        f"{y}: r={ey[str(y)]['r']:+.2f}{'*' if ey[str(y)]['sig'] not in ('ns','') else ''} (n={ey[str(y)]['n']})"
        for y in [2022, 2023, 2024])
    ax.text(0.5, 0.075, strip, ha="center", fontsize=12.5, color=col)
    at = d["all_three_together"]
    ax.text(0.5, 0.005, f"joint model  yield ~ EBI + climate:   R² {at['R2_year_only']} → {at['R2_full']}",
            ha="center", fontsize=12.5, color="#333")
    ax.set_title(C, fontweight="bold", fontsize=22, color=col, pad=14)

fig.suptitle("Climate → EBI → Yield, within each cultivar  (Chill Portions, Dynamic Model; same season)",
             fontweight="bold", fontsize=17, y=0.99)
fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.04, wspace=0.10)
for dd in (FIG, SC): fig.savefig(os.path.join(dd, "Climate_EBI_Yield_Triangle_byCultivar.png"), dpi=145, bbox_inches="tight")
print("saved triangle figure")
