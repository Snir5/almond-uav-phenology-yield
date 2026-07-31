# ============================================================
# apply_ebi_pixel_features_v6.py
#
# Per request: extracts a much richer bank of PIXEL-LEVEL EBI features per crown
# per year, not just the crown mean (EBI_Norm) already in the canonical master,
# nor the whiteness-brightness/brightness Otsu fractions already in
# Master_Trees_BloomFraction_v3.xlsx. This is the first extraction in this
# project to threshold on EBI ITSELF, using the exact same per-pixel EBI formula,
# radiometric correction, and reference-year normalization as the canonical
# apply_fixed_zones_yearly.py, so EBI_Norm_mean_{y} in this file's output should
# match EBI_Norm_{y} in the canonical master almost exactly (a numerical
# cross-check, not just a new feature).
#
# New per-crown, per-year features (all derived from the true per-pixel
# normalized EBI distribution within each canopy, not a crown mean of R/G/B):
#   EBI_Norm_mean_{y}     - cross-check against the canonical EBI_Norm_{y}
#   EBI_Norm_std_{y}      - requested: in-canopy EBI standard deviation
#   EBI_Norm_var_{y}      - requested: in-canopy EBI variance
#   EBI_Norm_median_{y}, EBI_Norm_P10/P25/P75/P90_{y}, EBI_Norm_IQR_{y}
#                         - distribution shape (percentiles), robust to outliers
#   EBI_Norm_skew_{y}, EBI_Norm_kurtosis_{y}
#                         - distribution shape: skew=asymmetric bloom, kurtosis=
#                           peaked (uniform bloom) vs heavy-tailed (few very
#                           bright patches) canopy
#   EBI_Norm_CV_{y}       - coefficient of variation (std/mean), a normalized
#                           patchiness measure independent of mean bloom level
#   HighEBI_frac_055_{y}, HighEBI_frac_065_{y}, HighEBI_frac_075_{y}
#                         - requested: literal EBI-threshold-crossing percentage,
#                           at 3 fixed absolute cutoffs; 0.65 matches the
#                           existing project convention ("bloom-bright fraction
#                           EBI>0.65", THESIS_PROGRESS_LOG.md key findings)
#   EBI_OtsuThr_{y}, EBI_OtsuFrac_{y}
#                         - an EBI-specific Otsu-optimal threshold (unlike
#                           BloomFraction_v3, which thresholds whiteness-
#                           brightness, not EBI) and the fraction above it
#   HighEBI_density_065_{y}
#                         - requested: "high EBI density" = count of
#                           EBI>0.65 pixels per m^2 of canopy (an areal density,
#                           not just a fraction, using each year's GSD)
#   CanopyPixelCount_v6_{y}, CanopyArea_m2_v6_{y}, GSD_m_v6_{y}
#                         - kept for self-sufficiency (density needs area) and
#                           as a cross-check against Master_Trees_CanopyStructure_v5.xlsx
#
# Built directly on apply_fixed_zones_yearly.py: IDENTICAL clipping, fixed
# Tree_Zones_Master, georeferencing offsets, EBI_EPS=256 formula, winsorizing,
# and shared-scale affine radiometric correction (all unchanged, proven correct
# in that script and load_master()'s coordinate guardrails depend on this being
# unchanged). Does NOT touch the canonical master or any v2-v5 file.
#
# Requires rasterio; the analysis sandbox does not have it. Run this on the
# same machine/environment used for apply_fixed_zones_yearly.py and the earlier
# apply_fixed_zones_yearly_v3_wholeimage_baseline.py / apply_bloom_fraction_v3.py
# extractions, then copy the output back for sandbox-side statistical testing.
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

# ---------------- CONFIG (identical paths/offsets to the canonical script) ----------------
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
master_excel = os.path.join(output_folder, "Master_Trees_EBIPixelFeatures_v6.xlsx")

EBI_EPS = 256.0
epsilon = 1e-6
HIGH_EBI_THRESHOLDS = [0.55, 0.65, 0.75]

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
        return out_image, out_transform, src.crs, src.nodatavals, src.res


def sample_percentile(arr, q, max_samples=5_000_000):
    if arr.size > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(arr.size, size=max_samples, replace=False)
        return np.percentile(arr[idx], q)
    return np.percentile(arr, q)


def otsu_threshold(values, nbins=256):
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


# ---------------- Stage 0: zones master + base table (identical to canonical) ----------------
print("Clipping Tree_Zones_Master to Plot 1...")
zm_image, zm_transform, zm_crs, zm_nodatavals, _ = clip_raster(zones_master_path, POLYGON_OFFSET["2021"])
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


def per_zone_moments_and_percentiles(zones_c, ebi_c, n_bins, valid):
    """Returns dict of per-zone arrays (mean/std/var/skew/kurt/percentiles/CV),
    all length n_bins, NaN where invalid. Percentiles computed via a single
    sort + searchsorted group split (fast, avoids a Python loop per pixel)."""
    counts = np.bincount(zones_c, minlength=n_bins).astype(np.float64)

    def zsum(v):
        return np.bincount(zones_c, weights=v, minlength=n_bins)

    mean_ = np.full(n_bins, np.nan)
    mean_[valid] = zsum(ebi_c)[valid] / counts[valid]
    mean_lookup = mean_[zones_c]
    dev = ebi_c - mean_lookup
    var_ = np.full(n_bins, np.nan)
    var_[valid] = zsum(dev ** 2)[valid] / counts[valid]
    std_ = np.sqrt(var_)
    m3 = np.full(n_bins, np.nan); m3[valid] = zsum(dev ** 3)[valid] / counts[valid]
    m4 = np.full(n_bins, np.nan); m4[valid] = zsum(dev ** 4)[valid] / counts[valid]
    with np.errstate(invalid="ignore", divide="ignore"):
        skew_ = m3 / np.where(std_ > 0, std_ ** 3, np.nan)
        kurt_ = m4 / np.where(std_ > 0, std_ ** 4, np.nan) - 3.0  # excess kurtosis
        cv_ = std_ / np.where(np.abs(mean_) > epsilon, mean_, np.nan)

    # percentiles via sort + group boundaries
    order = np.argsort(zones_c, kind="stable")
    zones_sorted = zones_c[order]; ebi_sorted = ebi_c[order]
    boundaries = np.searchsorted(zones_sorted, np.arange(n_bins + 1))
    p10 = np.full(n_bins, np.nan); p25 = np.full(n_bins, np.nan)
    p50 = np.full(n_bins, np.nan); p75 = np.full(n_bins, np.nan); p90 = np.full(n_bins, np.nan)
    for z in np.where(valid)[0]:
        seg = ebi_sorted[boundaries[z]:boundaries[z + 1]]
        if seg.size == 0:
            continue
        p10[z], p25[z], p50[z], p75[z], p90[z] = np.percentile(seg, [10, 25, 50, 75, 90])

    return {"mean": mean_, "std": std_, "var": var_, "skew": skew_, "kurtosis": kurt_, "cv": cv_,
            "p10": p10, "p25": p25, "median": p50, "p75": p75, "p90": p90, "iqr": p75 - p25}


def compute_year(mosaic_path, year, ref_stats=None):
    poly_off = POLYGON_OFFSET.get(year, (0.0, 0.0))
    img, dst_transform, dst_crs, nodatavals, res = clip_raster(mosaic_path, polygon_offset=poly_off)
    gsd = float(abs(res[0]) * abs(res[1]))
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

    # shared-scale affine radiometric correction (IDENTICAL to canonical script)
    this_mean_R, this_std_R = float(R_c.mean()), float(R_c.std())
    this_mean_G, this_std_G = float(G_c.mean()), float(G_c.std())
    this_mean_B, this_std_B = float(B_c.mean()), float(B_c.std())
    if ref_stats is not None:
        (rmR, rsR, rmG, rsG, rmB, rsB) = ref_stats
        this_std_avg = (this_std_R + this_std_G + this_std_B) / 3.0
        ref_std_avg = (rsR + rsG + rsB) / 3.0
        shared_scale = ref_std_avg / max(this_std_avg, epsilon)
        R_c = (R_c - this_mean_R) * shared_scale + rmR
        G_c = (G_c - this_mean_G) * shared_scale + rmG
        B_c = (B_c - this_mean_B) * shared_scale + rmB
        np.clip(R_c, 0, None, out=R_c); np.clip(G_c, 0, None, out=G_c); np.clip(B_c, 0, None, out=B_c)
        this_stats = None
    else:
        this_stats = (this_mean_R, this_std_R, this_mean_G, this_std_G, this_mean_B, this_std_B)

    # per-pixel raw EBI, IDENTICAL formula/winsorizing to the canonical script
    ebi_num = R_c + G_c + B_c
    ebi_den = (G_c / (B_c + EBI_EPS)) * (R_c - B_c + EBI_EPS)
    raw_ebi = np.divide(ebi_num, ebi_den, out=np.zeros_like(ebi_num), where=ebi_den != 0)
    ebi_lo = sample_percentile(raw_ebi, 1); ebi_hi = sample_percentile(raw_ebi, 99)
    raw_ebi = np.clip(raw_ebi, ebi_lo, ebi_hi)
    del R_c, G_c, B_c, ebi_num, ebi_den; gc.collect()

    n_bins = int(zones_c.max()) + 1
    counts = np.bincount(zones_c, minlength=n_bins).astype(np.float64)
    valid = counts > 0
    zv = np.where(valid)[0]

    return zones_c, raw_ebi, this_stats, gsd, counts, valid, zv, n_bins


def aggregate_year(year, zones_c, raw_ebi, gsd, counts, valid, zv, n_bins,
                    mean_ebi_ref, std_ebi_ref, min_ebi_ref, range_ebi):
    norm_ebi = np.clip((raw_ebi - min_ebi_ref) / range_ebi, 0, 1)

    moments = per_zone_moments_and_percentiles(zones_c, norm_ebi, n_bins, valid)

    thr_ebi_otsu = otsu_threshold(norm_ebi)

    def zfrac(bool_mask):
        s = np.bincount(zones_c, weights=bool_mask.astype(np.float64), minlength=n_bins)
        out = np.full(n_bins, np.nan); out[valid] = s[valid] / counts[valid]; return out

    def zcount(bool_mask):
        s = np.bincount(zones_c, weights=bool_mask.astype(np.float64), minlength=n_bins)
        out = np.full(n_bins, np.nan); out[valid] = s[valid]; return out

    area_m2 = np.full(n_bins, np.nan); area_m2[valid] = counts[valid] * gsd

    cols = {"Zone_Value": zv,
            f"EBI_Norm_mean_{year}": np.round(moments["mean"][valid], 4),
            f"EBI_Norm_std_{year}": np.round(moments["std"][valid], 4),
            f"EBI_Norm_var_{year}": np.round(moments["var"][valid], 5),
            f"EBI_Norm_median_{year}": np.round(moments["median"][valid], 4),
            f"EBI_Norm_P10_{year}": np.round(moments["p10"][valid], 4),
            f"EBI_Norm_P25_{year}": np.round(moments["p25"][valid], 4),
            f"EBI_Norm_P75_{year}": np.round(moments["p75"][valid], 4),
            f"EBI_Norm_P90_{year}": np.round(moments["p90"][valid], 4),
            f"EBI_Norm_IQR_{year}": np.round(moments["iqr"][valid], 4),
            f"EBI_Norm_skew_{year}": np.round(moments["skew"][valid], 4),
            f"EBI_Norm_kurtosis_{year}": np.round(moments["kurtosis"][valid], 4),
            f"EBI_Norm_CV_{year}": np.round(moments["cv"][valid], 4),
            f"EBI_OtsuThr_{year}": thr_ebi_otsu,
            f"EBI_OtsuFrac_{year}": np.round(zfrac(norm_ebi > thr_ebi_otsu)[valid], 4),
            f"CanopyPixelCount_v6_{year}": counts[valid].astype(int),
            f"CanopyArea_m2_v6_{year}": np.round(area_m2[valid], 4),
            f"GSD_m_v6_{year}": gsd}
    for thr in HIGH_EBI_THRESHOLDS:
        tag = str(thr).replace(".", "")
        frac = zfrac(norm_ebi > thr)
        cnt = zcount(norm_ebi > thr)
        cols[f"HighEBI_frac_{tag}_{year}"] = np.round(frac[valid], 4)
        with np.errstate(invalid="ignore", divide="ignore"):
            dens = np.where(area_m2 > 0, cnt / area_m2, np.nan)
        cols[f"HighEBI_density_{tag}_{year}"] = np.round(dens[valid], 4)
    print(f"  {year}: EBI Otsu threshold={thr_ebi_otsu:.3f}, scene EBI>0.65 frac={100*np.mean(norm_ebi>0.65):.1f}%, "
          f"mean std-within-canopy={np.nanmean(moments['std'][valid]):.4f}")
    return pd.DataFrame(cols)


print(f"--- reference year {reference_year} ---")
zones_ref, raw_ebi_ref, ref_stats, gsd_ref, counts_ref, valid_ref, zv_ref, nbins_ref = compute_year(mosaics[reference_year], reference_year, ref_stats=None)
mean_ebi_ref, std_ebi_ref = float(np.mean(raw_ebi_ref)), float(np.std(raw_ebi_ref))
min_ebi_ref, max_ebi_ref = sample_percentile(raw_ebi_ref, 1), sample_percentile(raw_ebi_ref, 99)
range_ebi = (max_ebi_ref - min_ebi_ref) or 1e-6

ref_df = aggregate_year(reference_year, zones_ref, raw_ebi_ref, gsd_ref, counts_ref, valid_ref, zv_ref, nbins_ref,
                         mean_ebi_ref, std_ebi_ref, min_ebi_ref, range_ebi)
df_master = df_master.merge(ref_df, on="Zone_Value", how="left")
del zones_ref, raw_ebi_ref; gc.collect()

for year, path in mosaics.items():
    if year == reference_year:
        continue
    print(f"--- {year} ---")
    zones_c, raw_ebi, _, gsd, counts, valid, zv, n_bins = compute_year(path, year, ref_stats=ref_stats)
    dfy = aggregate_year(year, zones_c, raw_ebi, gsd, counts, valid, zv, n_bins,
                          mean_ebi_ref, std_ebi_ref, min_ebi_ref, range_ebi)
    df_master = df_master.merge(dfy, on="Zone_Value", how="left")
    del zones_c, raw_ebi, dfy; gc.collect()

df_master.to_excel(master_excel, index=False)
print("=" * 60, f"\nSaved: {master_excel}\n{len(df_master)} trees x {len(mosaics)} years", "\n" + "=" * 60)
print("Next: copy this file into 4band_mosaic/, then it can be tested in the")
print("analysis sandbox (no rasterio needed there) against measured yield,")
print("standalone, pairwise, and combined with every other feature.")
print("Sanity check to run first: EBI_Norm_mean_{y} here should match the")
print("canonical master's EBI_Norm_{y} almost exactly (same formula/correction).")
