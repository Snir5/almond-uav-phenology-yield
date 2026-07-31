# ============================================================
# apply_bloom_fraction_v3.py
#
# Adds a PIXEL-LEVEL bloom-fraction feature per crown, the one thing the
# crown-mean color indices tested in THESIS_PROGRESS_LOG.md section 19 cannot
# provide. Motivation: Chen, Jin & Brown (2019), the paper that originates the
# EBI formula this thesis uses, validate EBI against a per-pixel SVM-classified
# "bloom coverage" (their Eq.2, BC = N_bloom / N_scene), R^2=0.72 vs their
# pixel-mean EBI. Every color index tried so far (EBI itself, and the 8 tested
# in section 19) is a crown MEAN of R,G,B, i.e. one averaged value per tree per
# year, which cannot recover that pixel-level bloom-fraction information no
# matter how the channels are recombined. This script computes the actual
# fraction, using an unsupervised (Otsu) threshold instead of a trained SVM,
# since there is no hand-labelled bloom/non-bloom ground truth for Kedma.
#
# It is built directly on apply_fixed_zones_yearly_v2.py: identical clipping,
# fixed Tree_Zones_Master, georeferencing offsets, and shared-scale affine
# radiometric correction (all unchanged, they are correct and this keeps the
# new feature comparable to the existing v2 crown means). It ADDS, per crown
# per year:
#   BloomFraction_{y}   = fraction of canopy pixels with high Brightness*(1-Saturation)
#                          ("whiteness-brightness", the best-performing crown-mean
#                          candidate from section 19) above a per-year Otsu threshold
#   BrightFraction_{y}  = fraction of canopy pixels with high Brightness alone,
#                          above its own per-year Otsu threshold (simpler comparator)
#   OtsuThr_WB_{y}, OtsuThr_Bright_{y} = the thresholds used, for transparency
# plus keeps meanR/G/B and EBI_ofMeans/NGRDI_ofMeans (identical to v2) so this
# single output file is self-sufficient.
#
# Requires rasterio (run in an environment that has it; the analysis sandbox
# does not). Run this on the same machine/environment used for
# apply_fixed_zones_yearly_v2.py. Output does NOT overwrite the canonical
# master or the v2 file.
# ============================================================

import rasterio
from rasterio.warp import reproject, Resampling, transform as transform_coords
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine
from shapely.geometry import Polygon
import numpy as np
from scipy.ndimage import center_of_mass
import pandas as pd
import os, gc

# ---------------- CONFIG (identical paths to v2) ----------------
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
master_excel = os.path.join(output_folder, "Master_Trees_BloomFraction_v3.xlsx")

EBI_EPS = 256.0
epsilon = 1e-6

# ---------------- Plot-1 polygon + offsets (identical to v2) ----------------
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


def otsu_threshold(values, nbins=256):
    """Classic Otsu (1979) threshold from scratch (no sklearn/cv2 needed)."""
    v = values[np.isfinite(values)]
    hist, edges = np.histogram(v, bins=nbins)
    hist = hist.astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(prob)
    centers = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.nanargmax(sigma_b2))
    return float(centers[idx])


# ---------------- Stage 0: zones master + base table (identical to v2) ----------------
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
    x_true = x_utm + 66.15
    lon, lat = transform_coords(zm_crs, "EPSG:4326", [x_true], [y_utm])
    base_rows.append({"Tree_ID": f"Tree_{int(tid):04d}", "Zone_Value": int(tid),
                      "X_UTM": x_true, "Y_UTM": y_utm, "Latitude": lat[0], "Longitude": lon[0]})
df_master = pd.DataFrame(base_rows)
del valid_zone_mask; gc.collect()


# ---------------- per-year: corrected bands, EBI-of-means, AND pixel-level bloom fraction ----------------
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

    # shared-scale affine radiometric correction (identical to v2) -----------
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

    def zfrac(bool_mask):
        s = np.bincount(zones_c, weights=bool_mask.astype(np.float64), minlength=n_bins)
        out = np.full(n_bins, np.nan); out[valid] = s[valid] / counts[valid]; return out

    # ----- crown-mean bands + EBI-of-means (identical to v2, kept for continuity) -----
    meanR = zmean(R_c); meanG = zmean(G_c); meanB = zmean(B_c)
    ebi_ofmeans = (meanR + meanG + meanB) / ((meanG / (meanB + EBI_EPS)) * (meanR - meanB + EBI_EPS))
    ngrdi_ofmeans = (meanG - meanR) / (meanG + meanR + epsilon)

    # ----- NEW: pixel-level whiteness-brightness + Otsu bloom fraction -----
    mx = np.maximum(np.maximum(R_c, G_c), B_c); mn = np.minimum(np.minimum(R_c, G_c), B_c)
    S = (mx - mn) / (mx + epsilon)
    Bright = (R_c + G_c + B_c) / 3.0
    WB = Bright * (1.0 - S)
    thr_wb = otsu_threshold(WB)
    thr_bright = otsu_threshold(Bright)
    bloom_frac = zfrac(WB > thr_wb)
    bright_frac = zfrac(Bright > thr_bright)
    print(f"  {year}: Otsu WB threshold={thr_wb:.1f}, Otsu Brightness threshold={thr_bright:.1f}, "
          f"scene bloom%={100*np.mean(WB > thr_wb):.1f}")

    del R_c, G_c, B_c, WB, Bright, S, mx, mn; gc.collect()
    zv = np.where(valid)[0]
    out = pd.DataFrame({"Zone_Value": zv,
        f"meanR_{year}": np.round(meanR[valid], 3), f"meanG_{year}": np.round(meanG[valid], 3),
        f"meanB_{year}": np.round(meanB[valid], 3),
        f"EBI_ofMeans_{year}": np.round(ebi_ofmeans[valid], 4),
        f"NGRDI_ofMeans_{year}": np.round(ngrdi_ofmeans[valid], 4),
        f"BloomFraction_{year}": np.round(bloom_frac[valid], 4),
        f"BrightFraction_{year}": np.round(bright_frac[valid], 4),
        f"OtsuThr_WB_{year}": thr_wb, f"OtsuThr_Bright_{year}": thr_bright})
    return out, this_stats, ebi_ofmeans[valid], ngrdi_ofmeans[valid]


# ---------------- reference year first (defines affine target + norm scale) ----------------
print(f"--- reference year {reference_year} ---")
ref_df, ref_stats, ref_ebi, ref_ngrdi = compute_year(mosaics[reference_year], reference_year, ref_stats=None)
mean_ebi_ref, std_ebi_ref = float(np.mean(ref_ebi)), float(np.std(ref_ebi))
min_ebi_ref, max_ebi_ref = np.percentile(ref_ebi, [1, 99])
range_ebi = (max_ebi_ref - min_ebi_ref) or 1e-6


def add_norm(df, y):
    e = df[f"EBI_ofMeans_{y}"].to_numpy(float)
    df[f"EBI_Norm_ofMeans_{y}"] = np.round(np.clip((e - min_ebi_ref) / range_ebi, 0, 1), 4)
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
print("=" * 60, f"\nSaved: {master_excel}\n{len(df_master)} trees x {len(mosaics)} years", "\n" + "=" * 60)
print("Next: copy/keep this file in 4band_mosaic/, then run")
print("  pipeline/06_analysis/bloom_fraction_yield.py")
print("(in the analysis sandbox, no rasterio needed there) to test")
print("BloomFraction_* and BrightFraction_* against measured yield,")
print("same battery as color_indices_yield.py (section 19 of the log).")
