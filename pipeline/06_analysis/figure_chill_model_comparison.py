#!/usr/bin/env python3
"""
figure_chill_model_comparison.py

Builds the chill-model comparison figure for the Results chapter, answering
Tarin Paz-Kagan's comment 466: "The thesis notes that the chill model is not
specifically parameterized for Israeli almonds. Consider comparing: Chill
Portions, Chill Hours, Utah Chill Units, heat accumulation after chill
fulfillment."

Values are taken from dormancy_models_from_climate.py, which computes all three
models from the station's 10-minute temperature record for each season and
evaluates them against the field 50 percent bloom date.

-> Results_Analysis/08_UEF53_Rerun/Chill_Model_Comparison.png
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root
from stats_utils import r_pvalue

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
SRC = os.path.join(OUT, "dormancy_models.json")

NAVY = "#1f3a5f"
BLUE = "#2f6db8"
ORANGE = "#c8641a"
GREEN = "#2f7d4f"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def main():
    d = json.load(open(SRC))
    rows = d["by_season"]
    years = [r["year"] for r in rows]
    bloom = np.array([r["bloom_t50"] for r in rows], float)

    models = [
        ("Chill Hours", "CH_computed", "hours at or below 7 " + chr(176) + "C", ORANGE),
        ("Utah Chill Units", "CU_computed", "Utah chill units", BLUE),
        ("Chill Portions (Dynamic Model)", "CP_computed", "chill portions", GREEN),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))
    for ax, (name, key, xlabel, colour) in zip(axes, models):
        x = np.array([r[key] for r in rows], float)
        r = float(np.corrcoef(x, bloom)[0, 1])
        p = r_pvalue(r, len(x))
        ax.scatter(x, bloom, s=90, color=colour, edgecolor="black", zorder=3)
        for xi, yi, yr in zip(x, bloom, years):
            ax.annotate(str(yr), (xi, yi), textcoords="offset points",
                        xytext=(7, 4), fontsize=8, color=NAVY)
        if len(x) > 2:
            b = np.polyfit(x, bloom, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.polyval(b, xs), "--", color="0.35", lw=1.2, zorder=2)
        ax.set_title(f"{name}\nr = {r:+.2f}, p = {p:.2f}", color=NAVY,
                     fontsize=9.5, fontweight="bold")
        ax.set_xlabel(f"accumulation to 28 February ({xlabel})", fontsize=8.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("field date of 50% bloom (day of year)", fontsize=8.5)

    fig.suptitle("Three chill models against observed bloom timing, four seasons",
                 color=NAVY, fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "Chill_Model_Comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)

    for name, key, _, _ in models:
        x = np.array([r[key] for r in rows], float)
        r = float(np.corrcoef(x, bloom)[0, 1])
        p = r_pvalue(r, len(x))
        print(f"  {name:32s} r = {r:+.3f}  p = {p:.3f}  range {x.min():.0f} to {x.max():.0f}")


if __name__ == "__main__":
    main()
