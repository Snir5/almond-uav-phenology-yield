# ============================================================
# plot_ebi_heatmaps.py
#
# Purpose: create 2 figures, each with 4 subplots (2021-2024) -
# the canopy layer (band 4) of Plot 1, colored as a heatmap by each
# tree's normalized EBI (fixed TreeID from Tree_Zones_Master).
#
#   Figure 1: EBI_Z    - Z-score normalization (mean + std, from baseline_..._stats.csv)
#   Figure 2: EBI_Norm - Min-Max normalization (0 to 1)
#
# The script reads Master_Trees_Time_Series_Plot1.xlsx (produced by
# apply_fixed_zones_yearly.py) plus Tree_Zones_Master.tif and the
# per-year mosaics, clips everything to Plot 1, and for each canopy
# pixel assigns the normalized EBI value of its TreeID.
# ============================================================

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIGURATION - same paths as apply_fixed_zones_yearly.py
# ============================================================
zones_master_path = "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2021_03_07/Tree_Zones_Master.tif"

mosaics = {
    "2021": "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2021_03_07/Final_Orthomosaic_4Band.tif",
    "2022": "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2022_03_02/Final_Orthomosaic_4Band.tif",
    "2023": "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2023_03_01/Final_Orthomosaic_4Band.tif",
    "2024": "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2024_02_29/Final_Orthomosaic_4Band.tif",
}

output_folder = "/Users/snirtahasa/Thesis/4band_mosaic"
master_excel = os.path.join(output_folder, "Master_Trees_Time_Series_Plot1.xlsx")

out_png_zscore = os.path.join(output_folder, "EBI_Zscore_Heatmaps_Plot1.png")
out_png_minmax = os.path.join(output_folder, "EBI_MinMax_Heatmaps_Plot1.png")
out_png_norm_peryear = os.path.join(output_folder, "EBI_Norm_PerYearScale_Plot1.png")

# Plot 1 polygon - same polygon as in apply_fixed_zones_yearly.py
PLOT1_COORDS = [
    (669782.83, 3509795.55),
    (670021.59, 3509735.96),
    (669907.39, 3509411.30),
    (669715.51, 3509525.26),
]
plot1_polygon = Polygon(PLOT1_COORDS)

# ============================================================
# Georeferencing offset correction between years - must match
# ZONES_OFFSET_M in apply_fixed_zones_yearly.py exactly. Without this,
# clipping and reprojecting zones_master for 2022-2024 would land on
# the wrong area (~65/43 m off from Plot 1's true physical location).
# ============================================================
ZONES_OFFSET_M = {
    "2021": (0.0, 0.0),
    "2022": (65.20, 42.65),
    "2023": (66.15, 43.05),
    "2024": (65.45, 42.70),
}


# ============================================================
# Helper: read a raster, clipped to the Plot 1 polygon, shifted by
# an offset (PLOT1_COORDS + offset = Plot 1's correct physical
# location within that year's coordinate frame)
# ============================================================
def clip_raster(path, offset=(0.0, 0.0)):
    off_x, off_y = offset
    if off_x == 0.0 and off_y == 0.0:
        poly = plot1_polygon
    else:
        poly = Polygon([(x + off_x, y + off_y) for x, y in PLOT1_COORDS])
    with rasterio.open(path) as src:
        out_image, out_transform = rio_mask(
            src, [poly], crop=True, all_touched=True, filled=True
        )
        crs = src.crs
        nodatavals = src.nodatavals
    return out_image, out_transform, crs, nodatavals


# ============================================================
# Load Tree_Zones_Master (clipped to Plot 1) + the per-year values table
# ============================================================
print("Loading Tree_Zones_Master (Plot 1)...")
zm_image, zm_transform, zm_crs, zm_nodatavals = clip_raster(zones_master_path)
zones_master_arr = zm_image[0]
zm_nodata = zm_nodatavals[0] if zm_nodatavals and zm_nodatavals[0] is not None else 0

print("Loading Master_Trees_Time_Series_Plot1.xlsx...")
df = pd.read_excel(master_excel)
max_zone = int(df["Zone_Value"].max())


def build_lut(value_col):
    """Lookup table: lut[zone_id] = the column's value for that TreeID. NaN = no data."""
    lut = np.full(max_zone + 1, np.nan, dtype=np.float32)
    for _, row in df.iterrows():
        zid = int(row["Zone_Value"])
        if zid <= max_zone:
            lut[zid] = row[value_col]
    return lut


# ============================================================
# Build a "value map" for a given year: for each canopy pixel ->
# the normalized EBI value of its TreeID. All other pixels = NaN
# (rendered as white/transparent)
# ============================================================
def build_value_map(year, lut):
    off_x, off_y = ZONES_OFFSET_M.get(year, (0.0, 0.0))
    img, dst_transform, dst_crs, nodatavals = clip_raster(mosaics[year], offset=(off_x, off_y))
    mask = img[3]
    mask_nodata = nodatavals[3] if len(nodatavals) > 3 else None

    zm_transform_corrected = Affine.translation(off_x, off_y) * zm_transform

    zones_dst = np.zeros(mask.shape, dtype=np.int32)
    reproject(
        source=zones_master_arr,
        destination=zones_dst,
        src_transform=zm_transform_corrected,
        src_crs=zm_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=zm_nodata,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

    canopy = (mask != 0) & (zones_dst > 0) & (zones_dst <= max_zone)
    if mask_nodata is not None:
        canopy &= (mask != mask_nodata)

    value_map = np.full(mask.shape, np.nan, dtype=np.float32)
    value_map[canopy] = lut[zones_dst[canopy]]

    return value_map


# ============================================================
# Plot: 2x2 subplots, heatmap + shared colorbar
#
# The colormap (cmap="RdYlBu_r") was chosen to match the convention
# used in the paper (Chen et al. 2019, "An enhanced bloom index..."):
# blue = low EBI (green vegetation, R<<B-ish - green dominates), red =
# high EBI (bright/whitish pixels where R≈B - typical of bloom).
# Yellow/orange = intermediate.
# ============================================================
def plot_grid(value_maps, title, out_path, cmap="RdYlBu_r", vmin=None, vmax=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    im = None
    for ax, (year, vmap) in zip(axes.ravel(), value_maps.items()):
        im = ax.imshow(vmap, cmap=cmap_obj, vmin=vmin, vmax=vmax)
        ax.set_title(year)
        ax.axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=title)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  -> Saved: {out_path}")
    plt.close(fig)


# ============================================================
# Run: build maps for each year, for both metrics
# ============================================================
years = list(mosaics.keys())

print("Building EBI value maps...")
maps_z = {}
maps_norm = {}
for year in years:
    print(f"  -> {year}")
    lut_z_year = build_lut(f"EBI_Z_{year}")
    lut_norm_year = build_lut(f"EBI_Norm_{year}")
    maps_z[year] = build_value_map(year, lut_z_year)
    maps_norm[year] = build_value_map(year, lut_norm_year)


# ============================================================
# Dynamic color range: based on the 1-99 percentile of all values
# (all years combined), so the colorbar reflects the data's real
# range instead of being "flat"
# ============================================================
def combined_range(value_maps, pct=1):
    all_vals = np.concatenate([v[~np.isnan(v)].ravel() for v in value_maps.values()])
    lo = np.percentile(all_vals, pct)
    hi = np.percentile(all_vals, 100 - pct)
    return float(lo), float(hi)


print("Plotting EBI Z-score heatmaps...")
vmin_z, vmax_z = combined_range(maps_z)
print(f"  -> color range (Z): {vmin_z:.3f} to {vmax_z:.3f}")
# Blue = low Z (EBI below the reference year's mean, relatively green vegetation) |
# Red = high Z (EBI above the mean, bright/whitish pixels - typical of bloom)
plot_grid(maps_z, "EBI (Z-score) | Blue=green vegetation, Red=bloom/bright", out_png_zscore, cmap="RdYlBu_r", vmin=vmin_z, vmax=vmax_z)

print("Plotting EBI Min-Max heatmaps...")
vmin_n, vmax_n = combined_range(maps_norm)
print(f"  -> color range (Norm): {vmin_n:.3f} to {vmax_n:.3f}")
# Blue = 0 (low EBI, relatively green vegetation) | Red = 1 (high EBI, bright/whitish - bloom)
plot_grid(maps_norm, "EBI (Min-Max Normalized) | Blue=green vegetation, Red=bloom/bright", out_png_minmax, cmap="RdYlBu_r", vmin=vmin_n, vmax=vmax_n)


# ============================================================
# Additional version: a separate color scale per year (1-99 percentile
# of that year only) - purpose: expose each year's true internal
# variation, without one year's range (combined_range) - e.g. 2021,
# whose variance is ~2x the others - compressing the rest into a
# narrow color band.
# Use: for understanding the spatial pattern WITHIN each year only -
# for quantitative comparison ACROSS years, use EBI_MinMax / EBI_Zscore
# (shared scale).
# ============================================================
def plot_grid_peryear(value_maps, title, out_path, cmap="RdYlBu_r", pct=1):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle(title, fontsize=16)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    for ax, (year, vmap) in zip(axes.ravel(), value_maps.items()):
        vals = vmap[~np.isnan(vmap)]
        lo, hi = np.percentile(vals, pct), np.percentile(vals, 100 - pct)
        im = ax.imshow(vmap, cmap=cmap_obj, vmin=lo, vmax=hi)
        ax.set_title(f"{year}  (own range: {lo:.2f} - {hi:.2f})")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  -> Saved: {out_path}")
    plt.close(fig)


print("Plotting EBI Min-Max heatmaps (per-year scale)...")
plot_grid_peryear(maps_norm, "EBI (Min-Max, per-year scale) | Blue=green vegetation, Red=bloom/bright", out_png_norm_peryear, cmap="RdYlBu_r")

print("Done.")
