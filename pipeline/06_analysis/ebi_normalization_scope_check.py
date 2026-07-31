#!/usr/bin/env python3
"""
ebi_normalization_scope_check.py: exploratory diagnostic (not a results-producing
analysis), per request. User's concern: the canonical EBI extraction
(pipeline/03_ebi_extraction/apply_fixed_zones_yearly.py) computes its cross-year
radiometric correction (the "shared-scale affine" that shifts+scales each year's
R,G,B to match the reference year, 2024) using statistics (this_mean_R, this_std_R,
etc.) taken ONLY from CANOPY pixels (mask!=0 & zones_dst>0, i.e. only tree-crown
pixels, confirmed by direct code read of lines 268-290 of that script). Oren et al.
(the paper this thesis's EBI/dormancy comparison is built on), by contrast, compute
their index "for each pixel" and average "across each almond orchard" (their section
2.4.2), i.e. over the WHOLE orchard footprint (soil, inter-row gaps, everything),
not restricted to a vegetation mask, since Sentinel-2's 10m pixels cannot even
resolve individual canopies.

Why this could matter: canopy pixels are not a radiometrically neutral population.
They ARE the bloom signal (white/pink flower pixels mixed with leaf/branch pixels),
and bloom intensity/fraction differs sharply by year (established: SD of EBI halves
2021->2024, bloom-bright fraction 20.8%->0.6%). If the correction's baseline
(mean_R, std_R, etc.) is computed only from this biologically-variable population,
a year with more/brighter bloom pixels will show a different canopy-only color
distribution for reasons that are NOT purely photographic/illumination artifacts,
so correcting to match the reference year's canopy-only statistics risks removing
some of the real cross-year bloom signal along with the intended exposure/lighting
correction. Whole-image statistics (dominated by soil/background, which does not
bloom) would give a more radiometrically neutral baseline.

This cannot be fully re-tested here (rebuilding the master needs rasterio on the
full-resolution mosaics, not available in this sandbox; see section 26's cv2
overview-frame technique and its registration limits). But the SPECIFIC question of
"how much does restricting the baseline statistics to canopy-only pixels change them,
relative to using the whole image" does NOT require precise per-tree registration,
only the global canopy mask (band 4) and R,G,B arrays from the SAME overview frame,
compared within one year at a time. This script performs exactly that bounded,
sandbox-safe comparison.

-> Results_Analysis/08_UEF53_Rerun/EBI_Normalization_Scope_Check.json + figure.
"""
import os, sys, json
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root

BASE = find_thesis_root()
MOS = os.path.join(BASE, "4band_mosaic")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")

YEAR_DIR = {2021: "Final_Exports_2021_03_07", 2022: "Final_Exports_2022_03_02",
            2023: "Final_Exports_2023_03_01", 2024: "Final_Exports_2024_02_29"}
YEAR_FRAME = {2021: 2, 2022: 2, 2023: 2, 2024: 1}
REFERENCE_YEAR = 2024


def load_year_bands(year):
    d = os.path.join(MOS, YEAR_DIR[year])
    tif = os.path.join(d, "Final_Orthomosaic_4Band.tif")
    ovr = tif + ".ovr"
    frame = YEAR_FRAME[year]
    ok, mats = cv2.imreadmulti(ovr, start=frame, count=1, flags=cv2.IMREAD_UNCHANGED)
    assert ok, f"failed to read frame {frame} for {year}"
    arr = mats[0]  # H x W x 4, BGR + band4 (cv2 order)
    return arr


results = {}
for year in [2021, 2022, 2023, 2024]:
    arr = load_year_bands(year)
    B_ = arr[:, :, 0].astype(np.float64)
    G_ = arr[:, :, 1].astype(np.float64)
    R_ = arr[:, :, 2].astype(np.float64)
    M_ = arr[:, :, 3]

    # true nodata / border: all three color channels exactly zero (no real pixel value)
    true_nodata = (R_ == 0) & (G_ == 0) & (B_ == 0)
    valid = ~true_nodata
    canopy = (M_ != 0) & valid
    background = (M_ == 0) & valid

    n_valid = int(valid.sum()); n_canopy = int(canopy.sum()); n_bg = int(background.sum())
    canopy_frac = n_canopy / max(n_valid, 1)

    def stats(mask_):
        return {"mean_R": round(float(R_[mask_].mean()), 3), "std_R": round(float(R_[mask_].std()), 3),
                "mean_G": round(float(G_[mask_].mean()), 3), "std_G": round(float(G_[mask_].std()), 3),
                "mean_B": round(float(B_[mask_].mean()), 3), "std_B": round(float(B_[mask_].std()), 3),
                "n": int(mask_.sum())}

    canopy_stats = stats(canopy)
    whole_stats = stats(valid)   # "whole image" = every valid (non-nodata) pixel, canopy+background together
    background_stats = stats(background) if n_bg > 100 else None

    results[year] = {
        "n_valid_pixels": n_valid, "n_canopy_pixels": n_canopy, "n_background_pixels": n_bg,
        "canopy_fraction_of_valid": round(canopy_frac, 4),
        "canopy_only_stats": canopy_stats,
        "whole_image_stats": whole_stats,
        "background_only_stats": background_stats,
    }
    print(f"{year}: canopy_frac={canopy_frac:.3f}  "
          f"canopy(R,G,B)=({canopy_stats['mean_R']},{canopy_stats['mean_G']},{canopy_stats['mean_B']}) "
          f"whole(R,G,B)=({whole_stats['mean_R']},{whole_stats['mean_G']},{whole_stats['mean_B']})")
    del arr, R_, G_, B_, M_, true_nodata, valid, canopy, background

# ---------------- compare the two candidate baselines' implied correction ----------------
ref = results[REFERENCE_YEAR]
comparison = {}
for year in [2021, 2022, 2023, 2024]:
    y = results[year]
    for scope, label in [("canopy_only_stats", "canopy_only"), ("whole_image_stats", "whole_image")]:
        this_std_avg = (y[scope]["std_R"] + y[scope]["std_G"] + y[scope]["std_B"]) / 3.0
        ref_std_avg = (ref[scope]["std_R"] + ref[scope]["std_G"] + ref[scope]["std_B"]) / 3.0
        shared_scale = ref_std_avg / max(this_std_avg, 1e-9)
        comparison.setdefault(year, {})[f"shared_scale_{label}"] = round(float(shared_scale), 4)
        comparison[year][f"mean_shift_R_{label}"] = round(ref[scope]["mean_R"] - y[scope]["mean_R"], 3)
        comparison[year][f"mean_shift_G_{label}"] = round(ref[scope]["mean_G"] - y[scope]["mean_G"], 3)
        comparison[year][f"mean_shift_B_{label}"] = round(ref[scope]["mean_B"] - y[scope]["mean_B"], 3)

print("\nshared_scale under each baseline (relative to 2024 reference):")
for year in [2021, 2022, 2023, 2024]:
    c = comparison[year]
    print(f"  {year}: canopy_only={c['shared_scale_canopy_only']}  whole_image={c['shared_scale_whole_image']}  "
          f"diff={round(c['shared_scale_canopy_only']-c['shared_scale_whole_image'],4)}")

# does the canopy-vs-whole-image divergence in std track bloom intensity (canopy_fraction, a bloom-coverage proxy)?
divergence_std = {}
for year in [2021, 2022, 2023, 2024]:
    y = results[year]
    c_std_avg = (y["canopy_only_stats"]["std_R"] + y["canopy_only_stats"]["std_G"] + y["canopy_only_stats"]["std_B"]) / 3.0
    w_std_avg = (y["whole_image_stats"]["std_R"] + y["whole_image_stats"]["std_G"] + y["whole_image_stats"]["std_B"]) / 3.0
    divergence_std[year] = {"canopy_std_avg": round(c_std_avg, 3), "whole_image_std_avg": round(w_std_avg, 3),
                             "ratio_canopy_over_whole": round(c_std_avg / max(w_std_avg, 1e-9), 4),
                             "canopy_fraction": y["canopy_fraction_of_valid"]}

RESULTS = {
    "note": "Sandbox-safe diagnostic (cv2 overview frames, no rasterio needed) testing whether "
            "restricting the canonical EBI extraction's radiometric-correction baseline statistics "
            "to CANOPY-only pixels (as apply_fixed_zones_yearly.py does) vs the WHOLE image "
            "(background+canopy together, the scope Oren et al. effectively use since Sentinel-2 "
            "cannot resolve individual canopies) gives materially different correction factors. "
            "Uses low-resolution pyramid overview frames (memory-safe), same frames as section 26. "
            "Does NOT require per-tree registration (the step that failed validation in section 26), "
            "only the year's own canopy mask (band 4) applied to the year's own R,G,B, within one "
            "year at a time.",
    "reference_year": REFERENCE_YEAR,
    "per_year_stats": results,
    "shared_scale_comparison_canopy_vs_whole_image": comparison,
    "canopy_vs_whole_image_std_divergence": divergence_std,
}
with open(os.path.join(OUT, "EBI_Normalization_Scope_Check.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")

# ---------------- figure ----------------
years = [2021, 2022, 2023, 2024]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
canopy_scale = [comparison[y]["shared_scale_canopy_only"] for y in years]
whole_scale = [comparison[y]["shared_scale_whole_image"] for y in years]
ax.plot(years, canopy_scale, "o-", label="baseline = canopy-only pixels (current pipeline)", color="#C0392B")
ax.plot(years, whole_scale, "o-", label="baseline = whole image (Oren-style scope)", color="#16A085")
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xlabel("Year"); ax.set_ylabel("Implied shared_scale (vs 2024 reference)")
ax.set_title("Radiometric correction factor:\ncanopy-only vs whole-image baseline", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
canopy_frac = [results[y]["canopy_fraction_of_valid"] for y in years]
ratio = [divergence_std[y]["ratio_canopy_over_whole"] for y in years]
ax2 = ax.twinx()
ax.bar([y - 0.15 for y in years], canopy_frac, width=0.3, color="#8E44AD", label="canopy fraction of image")
ax2.plot(years, ratio, "s-", color="#E67E22", label="canopy std / whole-image std")
ax.set_xlabel("Year"); ax.set_ylabel("Canopy fraction of image", color="#8E44AD")
ax2.set_ylabel("std ratio (canopy/whole)", color="#E67E22")
ax.set_title("Canopy coverage and color-spread\ndivergence, by year", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "EBI_Normalization_Scope_Check.png"), dpi=130)
print("saved figure")
