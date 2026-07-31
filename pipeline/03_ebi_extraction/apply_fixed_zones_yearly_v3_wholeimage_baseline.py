#!/usr/bin/env python3
# ============================================================
# apply_fixed_zones_yearly_v3_wholeimage_baseline.py
#
# Per thesis section 13.6: tests whether the cross-year radiometric
# correction's BASELINE STATISTICS should span the WHOLE image (background +
# canopy together, the scope Oren et al. effectively use, since Sentinel-2
# cannot resolve individual canopies) rather than CANOPY PIXELS ONLY, which
# is what the canonical pipeline (apply_fixed_zones_yearly.py, lines 288-290)
# actually does.
#
# THIS IS THE ONLY CHANGE FROM v1. Everything else is identical and kept
# because it is correct:
#   - the clipping polygon, the fixed Tree_Zones_Master, the georeferencing
#     offsets (all unchanged)
#   - the shared-scale affine correction FORM: one scale shared across R/G/B
#     (not a separate scale per channel), because EBI depends on (R-B) and a
#     per-channel scale would inject a brightness-dependent bias even for a
#     color-neutral (R==B) pixel -- see v1's own comment on this, still true
#     here
#   - the EBI epsilon fix (EBI_EPS=256 for 0-255 DN data, per Chen et al. 2019
#     Eq.1) and the 1-99 percentile winsorization of raw EBI
#   - the FINAL per-tree EBI/NGRDI aggregation still averages CANOPY pixels
#     only (a tree's own bloom intensity should still come from its own
#     canopy, not be diluted by soil) -- only the STATISTICS used to correct
#     each year's R,G,B onto the reference year change scope, from "canopy
#     pixels of this year" to "every valid pixel of this year (canopy +
#     background)"
#
# WHY THIS MIGHT MATTER: canopy pixels are not a radiometrically neutral
# population, they ARE the bloom signal (white/pink flower pixels mixed with
# leaf/branch pixels), and bloom fraction differs sharply by year. Basing the
# correction only on that population risks folding real cross-year bloom
# signal into what is supposed to be a pure exposure/illumination fix.
# Whole-image statistics, dominated by soil/background (which does not
# bloom), give a more radiometrically neutral baseline.
#
# DOES NOT touch, overwrite, or modify any existing file:
#   - Master_Trees_Time_Series_Plot1.xlsx (canonical, v1 output) untouched
#   - Master_Trees_Extended.xlsx (canonical master used everywhere) untouched
#   - Master_Trees_EBIofMeans_v2.xlsx (the section-13.5 v2 output) untouched
# Writes a NEW, separate workbook so the canopy-baseline (canonical) and the
# whole-image-baseline results exist SIDE BY SIDE, nothing is deleted. Every
# new column is suffixed `_wholeimg` so a future merge into the master (if
# ever wanted) cannot collide with the canonical EBI_Raw_*/EBI_Norm_*/
# EBI_Z_* columns.
#
# Requires rasterio -- run this on your own machine (the analysis sandbox has
# no rasterio and no internet to install it). After it finishes, run
# compare_ebi_wholeimage_baseline.py (sandbox-safe, no rasterio needed) to
# compare this output against the canonical master, and to test whether a
# combined variant (e.g. the average of both baselines) adds anything on top
# of either alone.
#
# Output: 4band_mosaic/Master_Trees_WholeImageBaseline_v3.xlsx
# ============================================================

import rasterio
from rasterio.warp import reproject, Resampling, transform as transform_coords
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine
from shapely.geometry import Polygon
import numpy as np
from scipy.ndimage import center_of_mass
import pandas as pd
import os
import gc

# ============================================================
# CONFIG -- identical paths to v1/v2, update if your folder differs
# ============================================================
BASE = "/Users/snirtahasa/Thesis/4band_mosaic"
zones_master_path = f"{BASE}/Final_Exports_2021_03_07/Tree_Zones_Master.tif"
mosaics = {
    "2021": f"{BASE}/Final_Exports_2021_03_07/Final_Orthomosaic_4Band.tif",
    "2022": f"{BASE}/Final_Exports_2022_03_02/Final_Orthomosaic_4Band.tif",
    "2023": f"{BASE}/Final_Exports_2023_03_01/Final_Orthomosaic_4Band.tif",
    "2024": f"{BASE}/Final_Exports_2024_02_29/Final_Orthomosaic_4Band.tif",
}
reference_year = "2024"
output_folder = BASE
master_excel = os.path.join(output_folder, "Master_Trees_WholeImageBaseline_v3.xlsx")

EBI_EPS = 256.0        # Chen et al. 2019 Eq.1, for 0-255 DN (keeps EBI bounded)
epsilon = 1e-6         # guard against this_std==0, not part of the EBI formula itself


def sample_percentile(arr, q, max_samples=5_000_000):
    """Percentile from a random sample -- avoids an internal full-array sort
    copy that can OOM on hundreds of millions of pixels."""
    if arr.size > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(arr.size, size=max_samples, replace=False)
        return np.percentile(arr[idx], q)
    return np.percentile(arr, q)


# ============================================================
# Plot-1 polygon + offsets -- identical to v1/v2
# ============================================================
PLOT1_COORDS = [
    (669846.69, 3509814.25),
    (670075.00, 3509755.50),
    (669973.83, 3509452.97),
    (669791.15, 3509556.32),
]
plot1_polygon = Polygon(PLOT1_COORDS)
POLYGON_OFFSET = {"2021": (-66.15, 0.0), "2022": (0.0, 0.0), "2023": (0.0, 0.0), "2024": (0.0, 0.0)}
ZONES_OFFSET_M = {"2021": (0.0, 0.0), "2022": (65.20, 42.65), "2023": (66.15, 43.05), "2024": (65.45, 42.70)}


def clip_raster(path, polygon_offset=(0.0, 0.0)):
    off_x, off_y = polygon_offset
    poly = plot1_polygon if (off_x == 0.0 and off_y == 0.0) else Polygon(
        [(x + off_x, y + off_y) for x, y in PLOT1_COORDS])
    with rasterio.open(path) as src:
        out_image, out_transform = rio_mask(src, [poly], crop=True, all_touched=True, filled=True)
        return out_image, out_transform, src.crs, src.nodatavals


# ============================================================
# Stage 0: zones master + base table -- identical to v1/v2
# ============================================================
print("Clipping Tree_Zones_Master to Plot 1...")
zm_image, zm_transform, zm_crs, zm_nodatavals = clip_raster(zones_master_path, polygon_offset=POLYGON_OFFSET["2021"])
zones_master_arr = zm_image[0]
zm_nodata = zm_nodatavals[0] if zm_nodatavals and zm_nodatavals[0] is not None else 0
valid_zone_mask = zones_master_arr > 0
if zm_nodata != 0:
    valid_zone_mask &= (zones_master_arr != zm_nodata)
tree_ids = np.unique(zones_master_arr[valid_zone_mask]); tree_ids = tree_ids[tree_ids > 0]
print(f"  -> {len(tree_ids)} tree zones inside Plot 1.")

weights = np.ones(zones_master_arr.shape, dtype=np.float32)
centers = center_of_mass(weights, labels=zones_master_arr, index=tree_ids)
base_rows = []
for tid, (row, col) in zip(tree_ids, centers):
    x_utm, y_utm = rasterio.transform.xy(zm_transform, row, col)
    lon, lat = transform_coords(zm_crs, "EPSG:4326", [x_utm], [y_utm])
    x_utm_true = x_utm + 66.15  # zones_master centroid is in the 2021 frame; correct to true frame
    base_rows.append({"Tree_ID": f"Tree_{int(tid):04d}", "Zone_Value": int(tid),
                       "X_UTM": x_utm_true, "Y_UTM": y_utm, "Latitude": lat[0], "Longitude": lon[0]})
df_master = pd.DataFrame(base_rows)
print(f"  -> {len(df_master)} tree zones found.")
del weights, valid_zone_mask
gc.collect()


# ============================================================
# Per-year: WHOLE-IMAGE baseline stats (NEW) + canopy-pixel EBI/NGRDI
# (canopy extraction/aggregation identical to v1; only the baseline scope for
# the affine correction changes)
# ============================================================
def compute_year_raw_metrics(mosaic_path, year, ref_stats_wholeimg=None):
    poly_off = POLYGON_OFFSET.get(year, (0.0, 0.0))
    img, dst_transform, dst_crs, nodatavals = clip_raster(mosaic_path, polygon_offset=poly_off)

    mask = img[3]
    mask_nodata = nodatavals[3] if len(nodatavals) > 3 else None

    zm_off_x, zm_off_y = ZONES_OFFSET_M.get(year, (0.0, 0.0))
    zm_transform_corrected = Affine.translation(zm_off_x, zm_off_y) * zm_transform
    dst_shape = mask.shape
    zones_dst = np.zeros(dst_shape, dtype=np.int32)
    reproject(source=zones_master_arr, destination=zones_dst,
              src_transform=zm_transform_corrected, src_crs=zm_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              src_nodata=zm_nodata, dst_nodata=0, resampling=Resampling.nearest)

    R_full = img[0].astype(np.float32)
    G_full = img[1].astype(np.float32)
    B_full = img[2].astype(np.float32)

    # ---- NEW: whole-image validity mask, for the BASELINE stats only ----
    # "True nodata" = outside the clipped plot polygon, or a genuine sensor
    # gap: all three color bands are exactly zero. This is NOT the same as
    # mask==0, which the canonical pipeline treats as "non-vegetation" (still
    # a real, valid soil/background reflectance value that SHOULD count here).
    true_nodata = (R_full == 0) & (G_full == 0) & (B_full == 0)
    if mask_nodata is not None:
        true_nodata |= (mask == mask_nodata)
    valid_whole = ~true_nodata
    R_whole = R_full[valid_whole]; G_whole = G_full[valid_whole]; B_whole = B_full[valid_whole]
    this_mean_R_w, this_std_R_w = float(R_whole.mean()), float(R_whole.std())
    this_mean_G_w, this_std_G_w = float(G_whole.mean()), float(G_whole.std())
    this_mean_B_w, this_std_B_w = float(B_whole.mean()), float(B_whole.std())
    n_whole = int(valid_whole.sum())
    del R_whole, G_whole, B_whole, valid_whole, true_nodata
    # -----------------------------------------------------------------------

    # canopy pixels -- identical definition to v1 (0=non-vegetation, any other
    # value=vegetation, AND inside a fixed tree zone)
    canopy = (mask != 0) & (zones_dst > 0)
    if mask_nodata is not None:
        canopy &= (mask != mask_nodata)
    zones_c = zones_dst[canopy]
    R_c = R_full[canopy].astype(np.float32)
    G_c = G_full[canopy].astype(np.float32)
    B_c = B_full[canopy].astype(np.float32)
    n_canopy = int(canopy.sum())

    del img, mask, zones_dst, canopy, R_full, G_full, B_full
    gc.collect()

    if ref_stats_wholeimg is not None:
        (ref_mean_R, ref_std_R, ref_mean_G, ref_std_G, ref_mean_B, ref_std_B) = ref_stats_wholeimg
        this_std_avg = (this_std_R_w + this_std_G_w + this_std_B_w) / 3.0
        ref_std_avg = (ref_std_R + ref_std_G + ref_std_B) / 3.0
        shared_scale = ref_std_avg / max(this_std_avg, epsilon)

        def affine(values, this_mean, ref_mean, scale):
            return (values - this_mean) * scale + ref_mean

        # The correction is still APPLIED to the canopy pixels (a tree's
        # bloom index should still come from its own canopy), only the
        # BASELINE (this_mean_*_w / this_std_*_w, computed above from the
        # WHOLE valid image) changes scope relative to v1.
        R_c = affine(R_c, this_mean_R_w, ref_mean_R, shared_scale)
        G_c = affine(G_c, this_mean_G_w, ref_mean_G, shared_scale)
        B_c = affine(B_c, this_mean_B_w, ref_mean_B, shared_scale)
        np.clip(R_c, 0, None, out=R_c); np.clip(G_c, 0, None, out=G_c); np.clip(B_c, 0, None, out=B_c)
        this_stats_wholeimg = None
        print(f"  {year}: n_whole={n_whole:,} n_canopy={n_canopy:,} shared_scale(whole-image baseline)={shared_scale:.4f}")
    else:
        this_stats_wholeimg = (this_mean_R_w, this_std_R_w, this_mean_G_w, this_std_G_w, this_mean_B_w, this_std_B_w)
        print(f"  {year} (reference): n_whole={n_whole:,} n_canopy={n_canopy:,} "
              f"whole-image mean(R,G,B)=({this_mean_R_w:.2f},{this_mean_G_w:.2f},{this_mean_B_w:.2f})")

    ebi_num = R_c + G_c + B_c
    ebi_den = (G_c / (B_c + EBI_EPS)) * (R_c - B_c + EBI_EPS)
    raw_ebi = np.divide(ebi_num, ebi_den, out=np.zeros_like(ebi_num), where=ebi_den != 0)
    ebi_lo = sample_percentile(raw_ebi, 1)
    ebi_hi = sample_percentile(raw_ebi, 99)
    raw_ebi = np.clip(raw_ebi, ebi_lo, ebi_hi)

    ngrdi_num = G_c - R_c
    ngrdi_den = G_c + R_c
    raw_ngrdi = np.divide(ngrdi_num, ngrdi_den, out=np.zeros_like(ngrdi_num), where=ngrdi_den != 0)

    del R_c, G_c, B_c, ebi_num, ebi_den, ngrdi_num, ngrdi_den
    gc.collect()

    return zones_c, raw_ebi, raw_ngrdi, this_stats_wholeimg


# ============================================================
# Stage 1: reference year -- whole-image baseline stats
# ============================================================
print(f"--- Processing reference year: {reference_year} (Plot 1 only) ---")
zones_ref, raw_ebi_ref, raw_ngrdi_ref, ref_stats_wholeimg = compute_year_raw_metrics(
    mosaics[reference_year], reference_year, ref_stats_wholeimg=None)

mean_ebi_ref = np.mean(raw_ebi_ref); std_ebi_ref = np.std(raw_ebi_ref)
min_ebi_ref = sample_percentile(raw_ebi_ref, 1); max_ebi_ref = sample_percentile(raw_ebi_ref, 99)
mean_ngrdi_ref = np.mean(raw_ngrdi_ref); std_ngrdi_ref = np.std(raw_ngrdi_ref)
min_ngrdi_ref = sample_percentile(raw_ngrdi_ref, 1); max_ngrdi_ref = sample_percentile(raw_ngrdi_ref, 99)

range_ebi = (max_ebi_ref - min_ebi_ref) or 1e-6
range_ngrdi = (max_ngrdi_ref - min_ngrdi_ref) or 1e-6


# ============================================================
# Stage 2: aggregate every year to the fixed Tree_ID -- identical logic to v1
# ============================================================
def aggregate_to_master(year, zones_c, raw_ebi, raw_ngrdi):
    z_score_ebi = (raw_ebi - mean_ebi_ref) / std_ebi_ref
    z_score_ngrdi = (raw_ngrdi - mean_ngrdi_ref) / std_ngrdi_ref
    norm_ebi = np.clip((raw_ebi - min_ebi_ref) / range_ebi, 0, 1)
    norm_ngrdi = np.clip((raw_ngrdi - min_ngrdi_ref) / range_ngrdi, 0, 1)

    n_bins = int(zones_c.max()) + 1
    counts = np.bincount(zones_c, minlength=n_bins).astype(np.float64)
    valid = counts > 0

    def zone_mean(values):
        sums = np.bincount(zones_c, weights=values, minlength=n_bins)
        out = np.full(n_bins, np.nan)
        out[valid] = sums[valid] / counts[valid]
        return out

    zone_values = np.where(valid)[0]
    df_year = pd.DataFrame({"Zone_Value": zone_values})
    df_year[f"EBI_Raw_wholeimg_{year}"] = np.round(zone_mean(raw_ebi)[valid], 4)
    df_year[f"NGRDI_Raw_wholeimg_{year}"] = np.round(zone_mean(raw_ngrdi)[valid], 4)
    df_year[f"EBI_Norm_wholeimg_{year}"] = np.round(zone_mean(norm_ebi)[valid], 4)
    df_year[f"NGRDI_Norm_wholeimg_{year}"] = np.round(zone_mean(norm_ngrdi)[valid], 4)
    df_year[f"EBI_Z_wholeimg_{year}"] = np.round(zone_mean(z_score_ebi)[valid], 4)
    df_year[f"NGRDI_Z_wholeimg_{year}"] = np.round(zone_mean(z_score_ngrdi)[valid], 4)
    return df_year


df_year = aggregate_to_master(reference_year, zones_ref, raw_ebi_ref, raw_ngrdi_ref)
df_master = pd.merge(df_master, df_year, on="Zone_Value", how="left")
del zones_ref, raw_ebi_ref, raw_ngrdi_ref, df_year
gc.collect()

for year, mosaic_path in mosaics.items():
    if year == reference_year:
        continue
    print(f"--- Processing {year} (Plot 1 only) ---")
    zones_c, raw_ebi, raw_ngrdi, _ = compute_year_raw_metrics(mosaic_path, year, ref_stats_wholeimg=ref_stats_wholeimg)
    df_year = aggregate_to_master(year, zones_c, raw_ebi, raw_ngrdi)
    df_master = pd.merge(df_master, df_year, on="Zone_Value", how="left")
    del zones_c, raw_ebi, raw_ngrdi, df_year
    gc.collect()

# ============================================================
# Stage 3: save -- NEW file, canonical master/v1/v2 files untouched
# ============================================================
df_master.to_excel(master_excel, index=False)
print("=" * 60)
print("SUCCESS -- nothing was overwritten.")
print(f"Canonical master (Master_Trees_Extended.xlsx) untouched.")
print(f"v1 output (Master_Trees_Time_Series_Plot1.xlsx) untouched.")
print(f"v2 output (Master_Trees_EBIofMeans_v2.xlsx) untouched.")
print(f"New file saved: {master_excel}")
print(f"{len(df_master)} trees x {len(mosaics)} years, columns suffixed _wholeimg.")
print("Next: run compare_ebi_wholeimage_baseline.py (sandbox-safe) to compare")
print("this against the canonical master and test a combined variant.")
print("=" * 60)
