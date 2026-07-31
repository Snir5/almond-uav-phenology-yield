# ============================================================
# apply_outlier_robust_v4.py
#
# Per-canopy OUTLIER-ROBUST version of the EBI-of-means extraction. Motivation
# (user question, 2026-07-04): EBI_ofMeans averages R, G, B across every pixel
# in a crown before computing one EBI value, so a handful of extreme pixels,
# sky glint / sensor saturation, deep shadow gaps between branches, or
# boundary pixels that are actually part soil / part neighbouring tree because
# the crown-mask segmentation is imperfect at the edge, can pull the crown mean
# away from what the bulk of the canopy actually looks like.
#
# This script removes such pixels PER CROWN, PER YEAR, before averaging, using
# two methods run side by side so their effect can be measured, not assumed:
#
#   Method A (primary): per-channel modified Z-score (Iglewicz & Hoya 1993),
#     z = 0.6745 * (x - median) / MAD, computed separately within EACH crown
#     (not globally), a pixel is dropped if |z| > 3.5 in ANY of R, G, B. This
#     is a robust, standard outlier rule that adapts to each crown's own pixel
#     distribution, so it should mainly catch genuine artefacts (saturation,
#     shadow, mixed boundary pixels) and leave ordinary within-crown bloom
#     heterogeneity (partial bloom, partial shade) alone, since that
#     heterogeneity is usually a graded, not extreme, deviation from the median.
#   Method B (comparator): brightness percentile trim, drop pixels below the
#     5th or above the 95th percentile of per-pixel brightness (R+G+B)/3
#     WITHIN each crown, then average the rest. Cruder (always removes ~10%
#     regardless of whether that 10% is noise or real signal), kept only as a
#     sanity check against method A.
#
# IMPORTANT STATISTICAL CAVEAT, read before using either output as ground
# truth: within-crown pixel spread is not pure noise. Some of it is real
# biological heterogeneity (a crown is rarely 100% uniformly in bloom; shaded
# vs sunlit sides differ). Trimming removes BOTH artefacts and some genuine
# tail structure, so a robust mean is a different quantity from the plain
# mean, not a strictly "more correct" one. Compare both to the existing
# (non-robust) EBI_ofMeans and let the data show whether it matters
# (bloom_fraction_yield.py-style comparison script provided separately).
#
# Built on apply_bloom_fraction_v3.py: identical clipping, zones alignment,
# and shared-scale affine radiometric correction. Adds, per crown per year:
#   meanR_robust_{y}, meanG_robust_{y}, meanB_robust_{y}   <- method A means
#   EBI_ofMeans_robust_{y}                                  <- EBI from method-A means
#   meanR_trim_{y..}, EBI_ofMeans_trim_{y}                  <- method B (percentile trim)
#   pct_outlier_A_{y}, pct_trimmed_B_{y}                    <- diagnostic: how much was removed
# plus keeps the original (non-robust) meanR/G/B and EBI_ofMeans for direct
# A/B/C comparison in one file.
#
# Requires rasterio (same environment as v2/v3). Output does NOT overwrite
# the canonical master, v2, or v3 files.
# ============================================================

import rasterio
from rasterio.warp import reproject, Resampling, transform as transform_coords
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import os, gc

# ---------------- CONFIG (identical paths to v2/v3) ----------------
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
master_excel = os.path.join(output_folder, "Master_Trees_OutlierRobust_v4.xlsx")

EBI_EPS = 256.0
epsilon = 1e-6
MOD_Z_THRESHOLD = 3.5     # Iglewicz & Hoya 1993 standard cutoff
TRIM_LO, TRIM_HI = 5, 95  # percentile trim bounds for method B
MIN_PIXELS_KEPT = 5       # never trim a crown down below this many pixels; fall back to full mean

# ---------------- Plot-1 polygon + offsets (identical to v2/v3) ----------------
PLOT1_COORDS = [
    (669846.69, 3509814.25), (670075.00, 3509755.50),
    (669973.83, 3509452.97), (669791.15, 3509556.32),
]
plot1_polygon = Polygon(PLOT1_COORDS)
POLYGON_OFFSET = {"2021": (-66.15, 0.0), "2022": (0.0, 0.0), "2023": (0.0, 0.0), "2024": (0.0, 0.0)}
ZONES_OFFSET_M = {"2021": (0.0, 0.0), "2022": (65.20, 42.65), "2023": (66.15, 43.05), "2024": (65.45, 42.70)}


def clip_raster(path, polygon_offset=(0.0, 0.0)):
    off_x, off_y = polygon_offset
    poly = plot1_polygon if (off_x == 0 and off_y == 0) else Polygon([(x + off_x, y + off_y) for x, y in PLOT1_COORDS])
    with rasterio.open(path) as src:
        out_image, out_transform = rio_mask(src, [poly], crop=True, all_touched=True, filled=True)
        return out_image, out_transform, src.crs, src.nodatavals


# ---------------- Stage 0: zones master + base table (identical to v2/v3) ----------------
print("Clipping Tree_Zones_Master to Plot 1...")
zm_image, zm_transform, zm_crs, zm_nodatavals = clip_raster(zones_master_path, POLYGON_OFFSET["2021"])
zones_master_arr = zm_image[0]
zm_nodata = zm_nodatavals[0] if zm_nodatavals and zm_nodatavals[0] is not None else 0
valid_zone_mask = zones_master_arr > 0
if zm_nodata != 0:
    valid_zone_mask &= (zones_master_arr != zm_nodata)
tree_ids = np.unique(zones_master_arr[valid_zone_mask]); tree_ids = tree_ids[tree_ids > 0]
print(f"  -> {len(tree_ids)} tree zones inside Plot 1.")
from scipy.ndimage import center_of_mass
centers = center_of_mass(np.ones(zones_master_arr.shape, np.float32), labels=zones_master_arr, index=tree_ids)
base_rows = []
for tid, (row, col) in zip(tree_ids, centers):
    x_utm, y_utm = rasterio.transform.xy(zm_transform, row, col)
    x_true = x_utm + 66.15
    lon, lat = transform_coords(zm_crs, "EPSG:4326", [x_true], [y_utm])
    base_rows.append({"Tree_ID": f"Tree_{int(tid):04d}", "Zone_Value": int(tid),
                      "X_UTM": x_true, "Y_UTM": y_utm, "Latitude": lat[0], "Longitude": lon[0]})
df_master = pd.DataFrame(base_rows)
del valid_zone_mask; gc.collect()


def per_group_robust_stats(zones_c, R_c, G_c, B_c):
    """Sort once by zone id, then walk group boundaries. Returns per-crown
    (Zone_Value order matches np.unique(zones_c)) robust means (method A),
    trimmed means (method B), plain means, and outlier/trim fractions."""
    order = np.argsort(zones_c, kind="stable")
    zs, Rs, Gs, Bs = zones_c[order], R_c[order], G_c[order], B_c[order]
    boundaries = np.flatnonzero(np.diff(zs)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(zs)]))
    zone_vals = zs[starts]

    n = len(zone_vals)
    meanR = np.full(n, np.nan); meanG = np.full(n, np.nan); meanB = np.full(n, np.nan)
    robR = np.full(n, np.nan); robG = np.full(n, np.nan); robB = np.full(n, np.nan)
    trimR = np.full(n, np.nan); trimG = np.full(n, np.nan); trimB = np.full(n, np.nan)
    pct_outlier_A = np.full(n, np.nan); pct_trimmed_B = np.full(n, np.nan)

    for i, (s, e) in enumerate(zip(starts, ends)):
        r, g, b = Rs[s:e], Gs[s:e], Bs[s:e]
        npix = e - s
        meanR[i], meanG[i], meanB[i] = r.mean(), g.mean(), b.mean()

        # ---- Method A: per-channel modified Z-score, drop if |z|>thr in ANY channel ----
        def modz(x):
            med = np.median(x); mad = np.median(np.abs(x - med))
            if mad < 1e-9: return np.zeros_like(x)
            return 0.6745 * (x - med) / mad
        keepA = (np.abs(modz(r)) <= MOD_Z_THRESHOLD) & (np.abs(modz(g)) <= MOD_Z_THRESHOLD) & (np.abs(modz(b)) <= MOD_Z_THRESHOLD)
        if keepA.sum() >= MIN_PIXELS_KEPT:
            robR[i], robG[i], robB[i] = r[keepA].mean(), g[keepA].mean(), b[keepA].mean()
            pct_outlier_A[i] = 100.0 * (1 - keepA.sum() / npix)
        else:
            robR[i], robG[i], robB[i] = meanR[i], meanG[i], meanB[i]; pct_outlier_A[i] = 0.0

        # ---- Method B: brightness percentile trim (consistent pixel set across channels) ----
        bright = (r + g + b) / 3.0
        lo, hi = np.percentile(bright, [TRIM_LO, TRIM_HI])
        keepB = (bright >= lo) & (bright <= hi)
        if keepB.sum() >= MIN_PIXELS_KEPT:
            trimR[i], trimG[i], trimB[i] = r[keepB].mean(), g[keepB].mean(), b[keepB].mean()
            pct_trimmed_B[i] = 100.0 * (1 - keepB.sum() / npix)
        else:
            trimR[i], trimG[i], trimB[i] = meanR[i], meanG[i], meanB[i]; pct_trimmed_B[i] = 0.0

    return (zone_vals, meanR, meanG, meanB, robR, robG, robB, trimR, trimG, trimB, pct_outlier_A, pct_trimmed_B)


def ebi_of(meanR, meanG, meanB):
    return (meanR + meanG + meanB) / ((meanG / (meanB + EBI_EPS)) * (meanR - meanB + EBI_EPS))


def compute_year(mosaic_path, year, ref_stats=None):
    poly_off = POLYGON_OFFSET.get(year, (0.0, 0.0))
    img, dst_transform, dst_crs, nodatavals = clip_raster(mosaic_path, polygon_offset=poly_off)
    mask = img[3]; mask_nodata = nodatavals[3] if len(nodatavals) > 3 else None
    zm_off_x, zm_off_y = ZONES_OFFSET_M.get(year, (0.0, 0.0))
    zm_transform_corrected = Affine.translation(zm_off_x, zm_off_y) * zm_transform
    zones_dst = np.zeros(mask.shape, dtype=np.int32)
    reproject(source=zones_master_arr, destination=zones_dst,
              src_transform=zm_transform_corrected, src_crs=zm_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              src_nodata=zm_nodata, dst_nodata=0, resampling=Resampling.nearest)
    canopy = (mask != 0) & (zones_dst > 0)
    if mask_nodata is not None:
        canopy &= (mask != mask_nodata)
    zones_c = zones_dst[canopy]
    R_c = img[0][canopy].astype(np.float32); G_c = img[1][canopy].astype(np.float32); B_c = img[2][canopy].astype(np.float32)
    del img, mask, zones_dst, canopy; gc.collect()

    # shared-scale affine radiometric correction (identical to v2/v3) -----------
    this_mean_R, this_std_R = float(R_c.mean()), float(R_c.std())
    this_mean_G, this_std_G = float(G_c.mean()), float(G_c.std())
    this_mean_B, this_std_B = float(B_c.mean()), float(B_c.std())
    if ref_stats is not None:
        (rmR, rsR, rmG, rsG, rmB, rsB) = ref_stats
        shared_scale = ((rsR + rsG + rsB) / 3.0) / max((this_std_R + this_std_G + this_std_B) / 3.0, epsilon)
        R_c = (R_c - this_mean_R) * shared_scale + rmR
        G_c = (G_c - this_mean_G) * shared_scale + rmG
        B_c = (B_c - this_mean_B) * shared_scale + rmB
        np.clip(R_c, 0, None, out=R_c); np.clip(G_c, 0, None, out=G_c); np.clip(B_c, 0, None, out=B_c)
        this_stats = None
    else:
        this_stats = (this_mean_R, this_std_R, this_mean_G, this_std_G, this_mean_B, this_std_B)

    (zv, meanR, meanG, meanB, robR, robG, robB, trimR, trimG, trimB, pctA, pctB) = per_group_robust_stats(zones_c, R_c, G_c, B_c)
    del R_c, G_c, B_c, zones_c; gc.collect()

    ebi_plain = ebi_of(meanR, meanG, meanB)
    ebi_rob = ebi_of(robR, robG, robB)
    ebi_trim = ebi_of(trimR, trimG, trimB)
    print(f"  {year}: mean %% pixels flagged outlier (method A) = {np.nanmean(pctA):.2f}%%, "
          f"median EBI shift (robust-plain) = {np.nanmedian(ebi_rob - ebi_plain):.4f}")

    out = pd.DataFrame({"Zone_Value": zv,
        f"meanR_{year}": np.round(meanR, 3), f"meanG_{year}": np.round(meanG, 3), f"meanB_{year}": np.round(meanB, 3),
        f"EBI_ofMeans_{year}": np.round(ebi_plain, 4),
        f"meanR_robust_{year}": np.round(robR, 3), f"meanG_robust_{year}": np.round(robG, 3), f"meanB_robust_{year}": np.round(robB, 3),
        f"EBI_ofMeans_robust_{year}": np.round(ebi_rob, 4),
        f"meanR_trim_{year}": np.round(trimR, 3), f"meanG_trim_{year}": np.round(trimG, 3), f"meanB_trim_{year}": np.round(trimB, 3),
        f"EBI_ofMeans_trim_{year}": np.round(ebi_trim, 4),
        f"pct_outlier_A_{year}": np.round(pctA, 2), f"pct_trimmed_B_{year}": np.round(pctB, 2)})
    return out, this_stats, ebi_plain


print(f"--- reference year {reference_year} ---")
ref_df, ref_stats, _ = compute_year(mosaics[reference_year], reference_year, ref_stats=None)
df_master = df_master.merge(ref_df, on="Zone_Value", how="left")
for year, path in mosaics.items():
    if year == reference_year:
        continue
    print(f"--- {year} ---")
    dfy, _, _ = compute_year(path, year, ref_stats=ref_stats)
    df_master = df_master.merge(dfy, on="Zone_Value", how="left")
    gc.collect()

df_master.to_excel(master_excel, index=False)
print("=" * 60, f"\nSaved: {master_excel}\n{len(df_master)} trees x {len(mosaics)} years", "\n" + "=" * 60)
print("Next: run pipeline/06_analysis/outlier_robust_yield.py (sandbox-safe, no")
print("rasterio needed) to see how much outlier removal shifts EBI and whether")
print("it changes the yield battery vs the existing (non-robust) EBI.")
