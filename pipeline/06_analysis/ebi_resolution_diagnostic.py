#!/usr/bin/env python3
"""
ebi_resolution_diagnostic.py — quantify the cross-year RESOLUTION / radiometry
confound in EBI, without needing rasterio (reads GSD from the .tfw world files and
EBI stats from the master). Motivation: the four orthomosaics differ in ground
sample distance (GSD) and exposure; finer/whiter years resolve more pure-bloom
pixels, which inflates EBI spread and the bloom-bright tail independently of climate.
The shared-scale affine correction in apply_fixed_zones_yearly.py aligns global
channel means/contrast but CANNOT remove a spatial-scale (pixel-mixing) effect.

Outputs: Results_Analysis/08_UEF53_Rerun/EBI_Resolution_Confound.png + a short report.
This is a within-master statistical diagnostic; it does not re-extract EBI.
"""
import os, sys, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
BASE = find_thesis_root()
FIG = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun"); os.makedirs(FIG, exist_ok=True)
SC = os.environ.get("SCRATCH_DIR", FIG)
YEARS = [2021, 2022, 2023, 2024]
EXPORT = {2021: "Final_Exports_2021_03_07", 2022: "Final_Exports_2022_03_02",
          2023: "Final_Exports_2023_03_01", 2024: "Final_Exports_2024_02_29"}

# --- GSD (m/px) from any .tfw in each year's export folder ---
gsd = {}
for y, d in EXPORT.items():
    tfws = glob.glob(os.path.join(BASE, "4band_mosaic", d, "*.tfw")) + glob.glob(os.path.join(BASE, "4band_mosaic", d, "*.TFW"))
    if tfws:
        with open(tfws[0]) as f:
            gsd[y] = abs(float(f.readline().strip())) * 100.0  # cm/px
gsd_cm = np.array([gsd[y] for y in YEARS])

# --- canopy fraction of Plot1 (from Diagnose_EBI_Deep_Check; hard values documented) ---
canopy_pct = {2021: 51.7, 2022: 37.2, 2023: 27.9, 2024: 30.3}
raw_ebi_uncorrected = {2021: 3.89, 2022: 3.43, 2023: 4.34, 2024: 3.99}  # pre-affine, from deep-check

# --- EBI stats (UEF+53) from the master ---
rows = [r for r in load_master() if r.get("cultivar") in {"UEF", "53"}]
def col(c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in rows], float)
ebi_mean = {}; ebi_sd = {}; bright65 = {}
for y in YEARS:
    a = col(f"EBI_Norm_{y}"); a = a[~np.isnan(a)]
    ebi_mean[y] = float(a.mean()); ebi_sd[y] = float(a.std(ddof=1)); bright65[y] = 100 * float((a > 0.65).mean())
CP = {y: float(np.nanmean(col(f"Chill_Portions_{y}"))) for y in YEARS}


def r_(a, b):
    a = np.array([a[y] for y in YEARS]); b = np.array([b[y] for y in YEARS])
    return float(np.corrcoef(a, b)[0, 1])


report = {
    "GSD_cm_per_px": {y: round(gsd[y], 4) for y in YEARS},
    "canopy_pct_of_plot1": canopy_pct,
    "raw_EBI_mean_uncorrected": raw_ebi_uncorrected,
    "EBI_Norm_mean_UEF53": {y: round(ebi_mean[y], 4) for y in YEARS},
    "EBI_Norm_sd_UEF53": {y: round(ebi_sd[y], 4) for y in YEARS},
    "bloom_bright_pct_gt065": {y: round(bright65[y], 2) for y in YEARS},
    "n4_correlations": {
        "GSD_vs_EBIsd": round(r_(gsd, ebi_sd), 3),
        "GSD_vs_EBImean": round(r_(gsd, ebi_mean), 3),
        "GSD_vs_bright065": round(r_(gsd, bright65), 3),
        "GSD_vs_canopyPct": round(r_(gsd, canopy_pct), 3),
        "ChillPortions_vs_EBIsd": round(r_(CP, ebi_sd), 3),
        "ChillPortions_vs_bright065": round(r_(CP, bright65), 3)},
    "verdict": ("Resolution (GSD) tracks EBI spread and the bloom-bright tail at least "
                "as strongly as Chill Portions does (n=4). Cross-year EBI level/spread/tail "
                "comparisons are confounded with resolution+exposure; WITHIN-year analyses "
                "(cultivar, spatial, EBI-yield) share one GSD and are NOT affected.")}
json.dump(report, open(os.path.join(FIG, "EBI_Resolution_Confound.json"), "w"), indent=2)
json.dump(report, open(os.path.join(SC, "EBI_Resolution_Confound.json"), "w"), indent=2)

# --- figure ---
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
a = ax[0]
a.bar([str(y) for y in YEARS], gsd_cm, color=["#C0392B", "#E67E22", "#16A085", "#4aa3df"])
for i, y in enumerate(YEARS): a.text(i, gsd[y] + .02, f"{gsd[y]:.2f}", ha="center", fontsize=9)
a.set_title("A. Pixel size (GSD) differs 1.8x across years\n2021 finest (1.22 cm), 2024 (reference) coarsest (2.24 cm)", fontweight="bold", fontsize=9.5)
a.set_ylabel("GSD (cm / pixel)")
a = ax[1]
a.scatter(gsd_cm, [ebi_sd[y] for y in YEARS], s=90, color="#e07b4a")
for y in YEARS: a.annotate(str(y), (gsd[y], ebi_sd[y]), xytext=(5, 4), textcoords="offset points", fontsize=9)
a.set_title(f"B. Coarser pixels -> lower EBI spread\nGSD vs EBI SD r={r_(gsd,ebi_sd):+.2f} | GSD vs bright>0.65 r={r_(gsd,bright65):+.2f} (n=4)", fontweight="bold", fontsize=9.5)
a.set_xlabel("GSD (cm/px)"); a.set_ylabel("EBI_Norm SD")
a = ax[2]
a.bar([str(y) for y in YEARS], [canopy_pct[y] for y in YEARS], color="#7f8c8d")
for i, y in enumerate(YEARS): a.text(i, canopy_pct[y] + .6, f"{canopy_pct[y]:.0f}%", ha="center", fontsize=9)
a.set_title("C. Canopy pixels detected per year\n2021 = 51.7% vs 28-30% later (resolution+brightness)", fontweight="bold", fontsize=9.5)
a.set_ylabel("% of Plot 1 flagged canopy")
fig.suptitle("EBI cross-year confound: resolution (GSD) + exposure, not corrected by global affine matching", fontweight="bold", fontsize=12.5)
fig.tight_layout()
for d in (FIG, SC): fig.savefig(os.path.join(d, "EBI_Resolution_Confound.png"), dpi=140, bbox_inches="tight")
print(json.dumps(report, indent=1))
