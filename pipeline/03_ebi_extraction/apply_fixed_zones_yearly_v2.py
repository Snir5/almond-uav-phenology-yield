# ============================================================
# apply_fixed_zones_yearly_v2.py
#
# Improved EBI/NGRDI extraction that removes the cross-year RESOLUTION
# confound documented in THESIS_PROGRESS_LOG.md section 13.
#
# It is identical to apply_fixed_zones_yearly.py in clipping, the fixed
# Tree_Zones_Master, the georeferencing offsets, and the shared-scale affine
# radiometric correction (all kept because they are correct), and changes ONE
# thing plus adds an optional second fix:
#
#   FIX #2 (core, always on): EBI-of-means, not mean-of-EBIs.
#     v1 computed EBI per pixel and then averaged per crown. A per-crown mean of
#     a nonlinear index is dominated by the few pure-white bloom pixels, whose
#     FREQUENCY depends on ground sample distance (GSD). Finer years (2021, 1.22
#     cm) therefore inflate EBI spread and the bright tail vs coarse years (2024,
#     2.24 cm). v2 first averages the radiometrically-corrected R, G, B within
#     each crown, then computes ONE EBI and NGRDI from those crown-mean bands.
#     This is far less sensitive to pixel size.
#
#   FIX #1 (optional): RESAMPLE_TO_COMMON_GSD. If True, every mosaic is read at a
#     common target GSD (default = coarsest year) with average resampling, so all
#     years share identical pixel support before extraction. Off by default
#     because FIX #2 already neutralises most of the effect and this doubles I/O.
#
# Outputs a NEW workbook (does NOT overwrite the canonical master) with, per year:
#   EBI_ofMeans_{y}, NGRDI_ofMeans_{y}      <- new, resolution-robust (USE THESE)
#   EBI_Norm_ofMeans_{y}, NGRDI_Norm_ofMeans_{y}, EBI_Z_ofMeans_{y}
#   meanR_{y}, meanG_{y}, meanB_{y}         <- crown-mean corrected bands
#   EBI_pixmean_{y}                          <- old mean-of-pixel EBI, for A/B compare
#
# Requires rasterio (run in an environment that has it; the analysis sandbox does
# not). Everything else matches v1.
# ============================================================

import rasterio
from rasterio.warp import reproject, Resampling, transform as transform_coords
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine
from rasterio.enums import Resampling as RIO_Resampling
from shapely.geometry import Polygon
import numpy as np
from scipy.ndimage import center_of_mass
import pandas as pd
import os, gc

# ---------------- CONFIG (same paths as v1) ----------------
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
master_excel = os.path.join(output_folder, "Master_Trees_EBIofMeans_v2.xlsx")

EBI_EPS = 256.0        # Chen et al. 2019 Eq.1, for 0-255 DN (keeps EBI bounded)
epsilon = 1e-6

# FIX #1 toggle
RESAMPLE_TO_COMMON_GSD = False      # set True to also equalise pixel support
TARGET_GSD_M = 0.0224               # coarsest year (2024); used only if resampling

# ---------------- Plot-1 polygon + offsets (identical to v1) ----------------
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
        if RESAMPLE_TO_COMMON_GSD:
            scale = src.res[0] / TARGET_GSD_M
            out_h = max(1, int(src.height * scale)); out_w = max(1, int(src.width * scale))
            data = src.read(out_shape=(src.count, out_h, out_w), resampling=RIO_Resampling.average)
            t = src.transform * src.transform.scale(src.width / out_w, src.height / out_h)
            from rasterio.io import MemoryFile
            prof = src.profile; prof.update(height=out_h, width=out_w, transform=t)
            with MemoryFile() as mf:
                with mf.open(**prof) as tmp:
                    tmp.write(data)
                with mf.open() as tmp:
                    out_image, out_transform = rio_mask(tmp, [poly], crop=True, all_touched=True, filled=True)
                    nod = tmp.nodatavals; crs = tmp.crs
            return out_image, out_transform, crs, nod
        out_image, out_transform = rio_mask(src, [poly], crop=True, all_touched=True, filled=True)
        return out_image, out_transform, src.crs, src.nodatavals


# ---------------- Stage 0: zones master + base table (identical to v1) ----------------
print("Clipping Tree_Zones_Master to Plot 1...")
zm_image, zm_transform, zm_crs, zm_nodatavals = clip_raster(zones_master_path, POLYGON_OFFSET["2021"])
zones_master_arr = zm_image[0]
zm_nodata = zm_nodatavals[0] if zm_nodatavals and zm_nodatavals[0] is not None else 0
valid_zone_mask = zones_master_arr > 0
if zm_nodata != 0:
    valid_zone_mask &= (zones_master_arr != zm_nodata)
tree_ids = np.unique(zones_master_arr[valid_zone_mask]); tree_ids = tree_ids[tree_ids > 0]
print(f"  -> {len(tree_ids)} tree zones inside Plot 1.")
centers = center_of_mass(np.ones(zones_master_arr.shape, np.float32), labels=zones_master_arr, index=tree_ids)
base_rows = []
for tid, (row, col) in zip(tree_ids, centers):
    x_utm, y_utm = rasterio.transform.xy(zm_transform, row, col)
    # zones_master centroid is in the 2021 frame; correct X to the true frame
    # (+66.15 m) BEFORE deriving lat/lon so stored lat/lon stay consistent with
    # X_UTM (otherwise geo_guardrails.assert_coords_aligned trips a ~66 m shift).
    x_true = x_utm + 66.15
    lon, lat = transform_coords(zm_crs, "EPSG:4326", [x_true], [y_utm])
    base_rows.append({"Tree_ID": f"Tree_{int(tid):04d}", "Zone_Value": int(tid),
                      "X_UTM": x_true, "Y_UTM": y_utm, "Latitude": lat[0], "Longitude": lon[0]})
df_master = pd.DataFrame(base_rows)
del valid_zone_mask; gc.collect()


# ---------------- per-year: corrected crown-mean bands + both EBI variants ----------------
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

    # shared-scale affine radiometric correction (identical to v1) -----------
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

    n_bins = int(zones_c.max()) + 1
    counts = np.bincount(zones_c, minlength=n_bins).astype(np.float64)
    valid = counts > 0

    def zmean(v):
        s = np.bincount(zones_c, weights=v, minlength=n_bins)
        out = np.full(n_bins, np.nan); out[valid] = s[valid] / counts[valid]; return out

    # ----- FIX #2: crown-mean bands FIRST, then one EBI per crown -----
    meanR = zmean(R_c); meanG = zmean(G_c); meanB = zmean(B_c)
    ebi_ofmeans = (meanR + meanG + meanB) / ((meanG / (meanB + EBI_EPS)) * (meanR - meanB + EBI_EPS))
    ngrdi_ofmeans = (meanG - meanR) / (meanG + meanR + epsilon)

    # ----- old metric (mean of per-pixel EBI) for A/B comparison -----
    ebi_pix = (R_c + G_c + B_c) / ((G_c / (B_c + EBI_EPS)) * (R_c - B_c + EBI_EPS))
    lo, hi = np.percentile(ebi_pix, [1, 99]); ebi_pix = np.clip(ebi_pix, lo, hi)
    ebi_pixmean = zmean(ebi_pix)

    del R_c, G_c, B_c, ebi_pix; gc.collect()
    zv = np.where(valid)[0]
    out = pd.DataFrame({"Zone_Value": zv,
        f"meanR_{year}": np.round(meanR[valid], 3), f"meanG_{year}": np.round(meanG[valid], 3),
        f"meanB_{year}": np.round(meanB[valid], 3),
        f"EBI_ofMeans_{year}": np.round(ebi_ofmeans[valid], 4),
        f"NGRDI_ofMeans_{year}": np.round(ngrdi_ofmeans[valid], 4),
        f"EBI_pixmean_{year}": np.round(ebi_pixmean[valid], 4)})
    return out, this_stats, ebi_ofmeans[valid], ngrdi_ofmeans[valid]


# ---------------- reference year first (defines affine target + norm scale) ----------------
print(f"--- reference year {reference_year} ---")
ref_df, ref_stats, ref_ebi, ref_ngrdi = compute_year(mosaics[reference_year], reference_year, ref_stats=None)
mean_ebi_ref, std_ebi_ref = float(np.mean(ref_ebi)), float(np.std(ref_ebi))
min_ebi_ref, max_ebi_ref = np.percentile(ref_ebi, [1, 99])
min_ng_ref, max_ng_ref = np.percentile(ref_ngrdi, [1, 99])
range_ebi = (max_ebi_ref - min_ebi_ref) or 1e-6
range_ng = (max_ng_ref - min_ng_ref) or 1e-6


def add_norm(df, y):
    e = df[f"EBI_ofMeans_{y}"].to_numpy(float); n = df[f"NGRDI_ofMeans_{y}"].to_numpy(float)
    df[f"EBI_Norm_ofMeans_{y}"] = np.round(np.clip((e - min_ebi_ref) / range_ebi, 0, 1), 4)
    df[f"NGRDI_Norm_ofMeans_{y}"] = np.round(np.clip((n - min_ng_ref) / range_ng, 0, 1), 4)
    df[f"EBI_Z_ofMeans_{y}"] = np.round((e - mean_ebi_ref) / (std_ebi_ref or 1e-6), 4)
    return df


ref_df = add_norm(ref_df, reference_year)
df_master = df_master.merge(ref_df, on="Zone_Value", how="left")
for year, path in mosaics.items():
    if year == reference_year:
        continue
    print(f"--- {year} ---")
    dfy, _, _, _ = compute_year(path, year, ref_stats=ref_stats)
    dfy = add_norm(dfy, year)
    df_master = df_master.merge(dfy, on="Zone_Value", how="left")
    gc.collect()

df_master.to_excel(master_excel, index=False)
print("=" * 50, f"\nSaved: {master_excel}\n{len(df_master)} trees x {len(mosaics)} years", "\n" + "=" * 50)
print("Compare EBI_Norm_ofMeans_* (new) vs the old EBI_Norm_* in the canonical master,")
print("and EBI_pixmean_* (old metric recomputed) to quantify how much the resolution")
print("artefact shifts each year's mean/SD/bright-tail.")
