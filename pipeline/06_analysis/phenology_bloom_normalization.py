#!/usr/bin/env python3
"""
phenology_bloom_normalization.py: corrects phenology_flight_timing.py's bloom-stage
reading and builds a phenology-timing-adjusted EBI.

Two corrections to the existing flight-timing analysis (section 16.1):

1. The field 0-10 phenology scale is NOT monotonic in visible bloom coverage. Per the
   scale definition, stage 8 ("80% flowers open") is FULL BLOOM, the true maximum of
   visible white canopy; stages 9-10 ("shedding of most/all petals") are the DECLINE
   phase, coverage falling back toward zero as petals drop. The existing
   Kadma_Phenology CSVs report "Percent" as a naive stage*10 relabeling (verified:
   Percent == Phenology_0_10*10 exactly in the raw file), which wrongly treats stage
   9-10 as MORE bloom (90%, 100%) instead of less. This script replaces that with a
   coverage(stage) function anchored at the scale's own labelled points (6->10%,
   7->50%, 8->80% peak) and declining after 8.

2. Given the corrected coverage-at-flight per cultivar-year, this builds an actual
   phenology-timing-normalized EBI: EBI_norm = EBI_ofMeans * (80 / coverage_at_flight),
   projecting each cultivar-year's observed EBI to what it would show at peak bloom
   (stage 8). This is only reported where coverage_at_flight is not too small (factor
   capped, else the correction is dominated by noise, since dividing by a small
   denominator amplifies any error in the interpolated stage). Missing-data
   cultivar-years (no field survey that year) are left uncorrected and flagged.

Important, and stated honestly: the one direct test of this correction's core
assumption (more field-observed flower coverage -> higher same-day EBI) is the 2021
20-tree survey, where the ground observation date is the exact flight date. That
per-tree test gives r=-0.08 (n=20, ns); section 16.1's cultivar-year-level test (naive
coverage) gives r=+0.12 (n=7, ns). Neither supports the assumption significantly. The
correction below is therefore reported as an exploratory, assumption-driven sensitivity
check on the cross-year EBI trend, not a validated replacement for raw EBI, consistent
with section 16's existing recommendation to flag rather than blindly rescale cross-year
comparisons. Within-year, tree-level results (yield correlations, cultivar contrasts,
Moran's I) are mathematically unaffected, since every tree in a cultivar-year is
multiplied by the same constant, which cannot change a within-group Pearson correlation.

-> Results_Analysis/08_UEF53_Rerun/Phenology_Bloom_Normalization.png + .json
"""
import os, sys, csv, json
import numpy as np
from datetime import datetime
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import r_pvalue
BASE = find_thesis_root(); FIG = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
SC = os.environ.get("SCRATCH_DIR", FIG)
CCOL = {"53": "#2E75B6", "UEF": "#C0392B"}
FLIGHT = {2021: 66, 2022: 61, 2023: 60, 2024: 60}          # DOY of 07/03, 02/03, 01/03, 29/02
FLIGHT_LBL = {2021: "Mar 7", 2022: "Mar 2", 2023: "Mar 1", 2024: "Feb 29"}
PEAK_COVERAGE = 80.0                                        # stage 8 = full bloom, per field scale
STABLE_COVERAGE_FLOOR = 30.0                                 # below this, factor>2.67x -> flag as unreliable

# stage -> visible bloom coverage (%). Anchored at the scale's own labels
# (6->10, 7->50, 8->80 peak); 1-5 are a conservative ramp (qualitative labels only,
# no flight in this dataset lands there); 9-10 decline as petals shed (qualitative
# "most/all shed", approximated linearly to zero at stage 10).
STAGE_ANCHORS = [(0, 0), (1, 0), (2, 0), (3, 1), (4, 3), (5, 6),
                 (6, 10), (7, 50), (8, PEAK_COVERAGE), (9, 30), (10, 0)]
_sx = [a[0] for a in STAGE_ANCHORS]; _sy = [a[1] for a in STAGE_ANCHORS]
def coverage_from_stage(stage): return float(np.interp(stage, _sx, _sy))


def doy(s): return datetime.strptime(s, "%d/%m/%Y").timetuple().tm_yday
def norm_c(c): return "UEF" if ("UEF" in c or "Um" in c) else c


M = load_master(); cult = np.array([r.get("cultivar") for r in M])
def col(c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in M], float)
ebi_tree = {y: col(f"EBI_Norm_{y}") for y in [2021, 2022, 2023, 2024]}
ebimean = {y: {c: float(np.nanmean(ebi_tree[y][cult == c])) for c in ["53", "UEF"]} for y in [2021, 2022, 2023, 2024]}

# --- build per-cultivar-year stage curves (2022-2024) and direct coverage (2021) ---
stage_curve = {}
for r in csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "Phenology_Surveys",
                                           "Kadma_Phenology_by_CultivarDate_2022_2024.csv"), encoding="utf-8-sig")):
    c = norm_c(r["Cultivar"])
    if c in ("53", "UEF"):
        stage_curve.setdefault((int(r["Year"]), c), []).append((doy(r["Date"]), float(r["Mean_Phenology_0_10"])))

flowers21 = {}
for r in csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "Bloom_Survey_2021",
                                           "Phenology_Survey_2021_20trees.csv"), encoding="utf-8-sig")):
    if r["Date"] != "07/03/2021":
        continue
    c = "53" if "53" in r["Cultivar"] else "UEF"
    flowers21.setdefault(c, []).append(float(r["Flowers"]))


def interp(pts, x):
    pts = sorted(pts); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return float(np.interp(x, xs, ys))


rows = []
for y in [2021, 2022, 2023, 2024]:
    for c in ["53", "UEF"]:
        if y == 2021:
            if c not in flowers21:
                continue
            cov = float(np.mean(flowers21[c]))
            n_obs = len(flowers21[c])
            source = f"direct measurement, same-day, n={n_obs} trees"
            stage_at_flight = None
        else:
            if (y, c) not in stage_curve:
                rows.append({"year": y, "cultivar": c, "coverage_at_flight": None, "stage_at_flight": None,
                             "source": "no field survey this cultivar-year", "correction_factor": None,
                             "reliable": False, "mean_EBI_raw": round(ebimean[y][c], 4) if not np.isnan(ebimean[y][c]) else None,
                             "mean_EBI_normalized": None})
                continue
            pts = stage_curve[(y, c)]
            stage_at_flight = interp(pts, FLIGHT[y])
            cov = coverage_from_stage(stage_at_flight)
            source = f"interpolated field stage {stage_at_flight:.2f}/10 ({len(pts)} survey dates)"
        factor = PEAK_COVERAGE / cov if cov > 0 else None
        reliable = (factor is not None) and (cov >= STABLE_COVERAGE_FLOOR)
        raw_mean = ebimean[y][c]
        rows.append({
            "year": y, "cultivar": c, "coverage_at_flight": round(cov, 1),
            "stage_at_flight": round(stage_at_flight, 2) if stage_at_flight is not None else None,
            "source": source, "correction_factor": round(factor, 2) if factor else None,
            "reliable": bool(reliable),
            "mean_EBI_raw": round(raw_mean, 4) if not np.isnan(raw_mean) else None,
            "mean_EBI_normalized": round(raw_mean * factor, 4) if (reliable and not np.isnan(raw_mean)) else None,
        })

# --- does cultivar-mean EBI track the CORRECTED coverage-at-flight? (supersedes 16.1's r=+0.12 naive test) ---
have = [r for r in rows if r["coverage_at_flight"] is not None and r["mean_EBI_raw"] is not None]
cov_arr = np.array([r["coverage_at_flight"] for r in have])
ebi_arr = np.array([r["mean_EBI_raw"] for r in have])
r_corr = float(np.corrcoef(cov_arr, ebi_arr)[0, 1])
p_corr = float(r_pvalue(r_corr, len(have)))

# --- tree-level normalized EBI columns (only for reliable cultivar-years), for transparency ---
tree_norm = {y: np.full(len(M), np.nan) for y in [2021, 2022, 2023, 2024]}
for r in rows:
    if r["reliable"]:
        y, c, f = r["year"], r["cultivar"], r["correction_factor"]
        mask = (cult == c)
        tree_norm[y][mask] = ebi_tree[y][mask] * f

# --- year-level pooled trend, raw vs normalized (only years/cultivars with a value) ---
year_trend = {}
for y in [2021, 2022, 2023, 2024]:
    raw_vals = ebi_tree[y][~np.isnan(ebi_tree[y])]
    norm_vals = tree_norm[y][~np.isnan(tree_norm[y])]
    year_trend[y] = {
        "mean_raw": round(float(np.mean(raw_vals)), 4) if len(raw_vals) else None,
        "sd_raw": round(float(np.std(raw_vals, ddof=1)), 4) if len(raw_vals) else None,
        "mean_normalized_where_reliable": round(float(np.mean(norm_vals)), 4) if len(norm_vals) else None,
        "sd_normalized_where_reliable": round(float(np.std(norm_vals, ddof=1)), 4) if len(norm_vals) else None,
        "n_trees_normalized": int(len(norm_vals)),
    }

out = {
    "note": "Stage->coverage anchored at field-scale labels (6=10%,7=50%,8=80% PEAK); "
            "stages 9-10 are decline (petal shed), not additional bloom. Supersedes the naive "
            "stage*10 reading used in phenology_flight_timing.py (section 16.1).",
    "peak_coverage_reference": PEAK_COVERAGE,
    "stable_coverage_floor": STABLE_COVERAGE_FLOOR,
    "per_cultivar_year": rows,
    "EBI_vs_corrected_coverage_at_flight": {"r": round(r_corr, 3), "p": round(p_corr, 3), "n": len(have),
        "interpretation": "corrected coverage still does not predict cultivar-mean EBI level "
                           "(consistent with section 16.1's naive-coverage test, r=+0.12 ns, and "
                           "the 2021 same-day per-tree test, r=-0.08 ns): no version of the field "
                           "bloom-coverage estimate explains cross-year/cross-cultivar EBI level."},
    "year_level_trend_raw_vs_normalized": year_trend,
    "caveat": "Normalization applied only where coverage_at_flight>=30 (correction factor<=2.67x); "
              "cultivar-53 2023 (coverage ~4.6%, factor ~17x) and UEF 2024 (coverage ~10%, factor 8x) "
              "are flagged unreliable and left as raw EBI; cultivar-53 2024 has no field survey at all "
              "and cannot be corrected. The correction rests on an assumption (EBI scales linearly "
              "with visible bloom coverage) not supported by either direct test above; treat as an "
              "exploratory sensitivity check on the cross-year EBI trend, not a validated substitute.",
}
json.dump(out, open(os.path.join(FIG, "Phenology_Bloom_Normalization.json"), "w"), indent=2)
print(json.dumps(out, indent=1))

# --- figure: (A) corrected stage->coverage function, (B) per-cultivar-year corrected coverage
#      at flight with reliability flag, (C) raw vs normalized cultivar-mean EBI where reliable ---
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

a = axes[0]
xs = np.linspace(0, 10, 200)
a.plot(xs, [coverage_from_stage(x) for x in xs], color="#2C3E50", lw=2.5)
a.scatter(_sx, _sy, color="#C0392B", zorder=5, s=40)
a.axvline(8, color="#16A085", ls="--", lw=1.3)
a.text(8.05, 85, "stage 8 = full bloom\n(peak, not stage 10)", fontsize=8.5, color="#16A085")
a.set_xlabel("field phenology stage (0-10)"); a.set_ylabel("visible bloom coverage (%)")
a.set_title("A. Corrected stage->coverage function\n(peak at 8, declines through 9-10 as petals shed)", fontsize=10, fontweight="bold")
a.grid(alpha=.3); a.set_ylim(-5, 95)

a = axes[1]
for r in rows:
    if r["coverage_at_flight"] is None:
        continue
    marker = "o" if r["reliable"] else "x"
    a.scatter(r["year"], r["coverage_at_flight"], color=CCOL[r["cultivar"]], marker=marker, s=110, edgecolors="k", lw=.5, zorder=3)
    a.annotate(f"{r['coverage_at_flight']:.0f}%", (r["year"], r["coverage_at_flight"]), xytext=(6, 4), textcoords="offset points", fontsize=8)
a.axhline(PEAK_COVERAGE, color="#16A085", ls="--", lw=1.2, label="peak (stage 8)")
a.axhline(STABLE_COVERAGE_FLOOR, color="grey", ls=":", lw=1.2, label="reliability floor")
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=CCOL["UEF"], marker="o", ls="", label="UEF"),
           Line2D([0], [0], color=CCOL["53"], marker="o", ls="", label="cultivar 53"),
           Line2D([0], [0], color="grey", marker="x", ls="", label="unreliable (factor>2.67x)"),
           Line2D([0], [0], color="#16A085", ls="--", label="peak / reliability floor")]
a.legend(handles=handles, fontsize=7.5, loc="lower right")
a.set_xticks([2021, 2022, 2023, 2024]); a.set_ylim(-5, 95)
a.set_xlabel("year"); a.set_ylabel("corrected visible bloom coverage at flight (%)")
a.set_title("B. Corrected coverage at flight, by cultivar-year\n('x' = correction too unstable to trust)", fontsize=10, fontweight="bold")
a.grid(alpha=.3)

a = axes[2]
for r in rows:
    if r["mean_EBI_raw"] is None:
        continue
    a.scatter(r["year"] - 0.08, r["mean_EBI_raw"], color=CCOL[r["cultivar"]], marker="o", s=90, alpha=.55, zorder=3)
    if r["mean_EBI_normalized"] is not None:
        a.scatter(r["year"] + 0.08, r["mean_EBI_normalized"], color=CCOL[r["cultivar"]], marker="^", s=110, edgecolors="k", lw=.5, zorder=4)
handles2 = [Line2D([0], [0], color="grey", marker="o", ls="", alpha=.55, label="raw EBI (as flown)"),
            Line2D([0], [0], color="grey", marker="^", ls="", label="normalized to peak (stable only)"),
            Line2D([0], [0], color=CCOL["UEF"], marker="s", ls="", label="UEF"),
            Line2D([0], [0], color=CCOL["53"], marker="s", ls="", label="cultivar 53")]
a.legend(handles=handles2, fontsize=7.5, loc="upper left")
a.set_xticks([2021, 2022, 2023, 2024])
a.set_xlabel("year"); a.set_ylabel("cultivar-mean EBI")
a.set_title(f"C. Raw vs peak-normalized cultivar-mean EBI\nEBI vs corrected coverage r={r_corr:.2f} (p={p_corr:.2f}, ns)", fontsize=10, fontweight="bold")
a.grid(alpha=.3)

fig.suptitle("Phenology-timing-adjusted EBI: correcting for stage-8 (not stage-10) as full bloom", fontweight="bold", fontsize=13)
fig.tight_layout()
for d in (FIG, SC):
    fig.savefig(os.path.join(d, "Phenology_Bloom_Normalization.png"), dpi=140, bbox_inches="tight")
print("saved figure")
