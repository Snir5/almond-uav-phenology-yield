# ============================================================
# apply_canopy_structure_v5.py
#
# Implements recommendations #1 and #2 from THESIS_PROGRESS_LOG.md section 20.4:
# canopy AREA and a canopy-shadow LIGHT-INTERCEPTION proxy, per tree per year.
# Both are colour-independent structural features, deliberately orthogonal to
# every EBI/colour-index variant tested in sections 19-19.2 (all of which
# failed to beat plain EBI).
#
# Literature motivation:
#   - Canopy area/volume from UAV RGB correlates with yield at R^2=0.71-0.98
#     across several fruit-tree studies (section 20.1.2 of the log).
#   - Lampinen et al. (2012) and Zarate-Valdez et al. (2015) link canopy LIGHT
#     INTERCEPTION directly to maximum potential almond yield, and show it can
#     be estimated from plain RGB photographs of canopy shadow (R^2=0.95 vs a
#     ceptometer light bar). This script implements two proxies for that idea:
#       (a) CanopyCoverFraction: canopy pixels / (canopy + ground) within each
#           tree's own footprint (the simpler, assumption-light proxy: canopy
#           cover fraction is a standard light-interception surrogate on its
#           own, independent of sun angle).
#       (b) ShadowFraction: within the ground pixels of that footprint, the
#           fraction classified as shadow (dark) vs sunlit (bright) by an
#           unsupervised Otsu threshold, closer to Zarate-Valdez's literal
#           method. CAVEAT, stated plainly: this assumes the acquisition was
#           near solar noon with compact shadows; flight time was not verified
#           for this project, so treat ShadowFraction as more experimental than
#           CanopyCoverFraction.
#
# Per-tree ground "footprint": since Kedma has no fixed regular grid spacing
# (median nearest-neighbour distance 5.75 m, but a substantial close-pair tail
# from interleaved pollinizer rows, computed from the master's X_UTM/Y_UTM),
# each tree gets an ADAPTIVE buffer radius = min(MAX_RADIUS_M, 0.5 * distance
# to its nearest neighbour), so buffers never overlap between adjacent trees
# and shrink automatically for tightly-spaced pairs instead of using one
# fixed radius for the whole orchard.
#
# Built on the same clip/zones-alignment machinery as v2/v3/v4. Ground-pixel
# assignment to the nearest tree uses each year's OWN mosaic georeferencing
# directly (rasterio's dst_transform from the clipped mosaic), not the
# zones-offset hack, so it needs no year-specific offset correction: the
# mosaic's own coordinates and the master's X_UTM/Y_UTM are both already in
# the true UTM frame.
#
# Requires rasterio AND scipy (scipy.spatial.cKDTree; scipy.ndimage is already
# used in v2-v4, so if those ran on this machine, scipy is already installed).
# Output does NOT overwrite the canonical master or any prior v2-v4 file.
# ============================================================

import rasterio
from rasterio.warp import reproject, Resampling, transform as transform_coords
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine
from shapely.geometry import Polygon
from scipy.ndimage import center_of_mass
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import os, gc

# ---------------- CONFIG (identical paths to v2/v3/v4) ----------------
BASE = "/Users/snirtahasa/Thesis/4band_mosaic"
zones_master_path = f"{BASE}/Final_Exports_2021_03_07/Tree_Zones_Master.tif"
mosaics = {
    "2021": f"{BASE}/Final_Exports_2021_03_07/Final_Orthomosaic_4Band.tif",
    "2022": f"{BASE}/Final_Exports_2022_03_02/Final_Orthomosaic_4Band.tif",
    "2023": f"{BASE}/Final_Exports_2023_03_01/Final_Orthomosaic_4Band.tif",
    "2024": f"{BASE}/Final_Exports_2024_02_29/Final_Orthomosaic_4Band.tif",
}
output_folder = BASE
master_excel = os.path.join(output_folder, "Master_Trees_CanopyStructure_v5.xlsx")

MAX_RADIUS_M = 3.0          # cap on the ground-footprint buffer radius
MIN_GROUND_PIXELS = 20      # below this, ShadowFraction/CanopyCoverFraction -> NaN (too few to trust)
GROUND_QUERY_CHUNK = 3_000_000   # batch size for the nearest-crown KD-tree query

# ---------------- Plot-1 polygon + offsets (identical to v2/v3/v4) ----------------
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
    v = values[np.isfinite(values)]
    hist, edges = np.histogram(v, bins=nbins)
    hist = hist.astype(np.float64); prob = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(prob); centers = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(prob * centers); mu_t = mu[-1]
    denom = omega * (1.0 - omega); denom[denom <= 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    return float(centers[int(np.nanargmax(sigma_b2))])


# ---------------- Stage 0: zones master + base table + adaptive buffer radius ----------------
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
    x_true = x_utm + 66.15   # same true-frame correction as v2-v4
    lon, lat = transform_coords(zm_crs, "EPSG:4326", [x_true], [y_utm])
    base_rows.append({"Tree_ID": f"Tree_{int(tid):04d}", "Zone_Value": int(tid),
                      "X_UTM": x_true, "Y_UTM": y_utm, "Latitude": lat[0], "Longitude": lon[0]})
df_master = pd.DataFrame(base_rows)
del valid_zone_mask; gc.collect()

# adaptive buffer radius per crown: half the distance to its nearest neighbour, capped
XY = df_master[["X_UTM", "Y_UTM"]].to_numpy(float)
kdt_centroids = cKDTree(XY)
nn_dist, _ = kdt_centroids.query(XY, k=2)
nn_dist = nn_dist[:, 1]
buffer_radius = np.minimum(MAX_RADIUS_M, 0.5 * nn_dist)
df_master["Buffer_Radius_m"] = np.round(buffer_radius, 3)
print(f"Buffer radius (m): median={np.median(buffer_radius):.2f}, min={buffer_radius.min():.2f}, "
      f"max={buffer_radius.max():.2f} (cap={MAX_RADIUS_M})")
zone_order = df_master["Zone_Value"].to_numpy(int)
radius_by_zone = dict(zip(zone_order, buffer_radius))
N_BINS = int(zone_order.max()) + 1   # global, so canopy/ground arrays are always the same length


def compute_year(mosaic_path, year):
    poly_off = POLYGON_OFFSET.get(year, (0.0, 0.0))
    img, dst_transform, dst_crs, nodatavals = clip_raster(mosaic_path, polygon_offset=poly_off)
    # Band 4 is documented (apply_fixed_zones_yearly.py header) as a VEGETATION MASK:
    # 0 = background/non-vegetation, non-zero = tree/canopy. It is NOT a generic
    # alpha/nodata channel, so "veg != 0" means vegetation, not "valid pixel".
    veg = img[3]
    # Real image extent (as opposed to the 0-filled area outside the clip polygon,
    # which rio_mask fills with 0 across every band including band 4): require at
    # least one RGB channel to be non-zero. This is the check that was missing
    # before, causing "ground pixels = 0" for every year (valid was accidentally
    # requiring vegetation==True, so non-vegetation ground pixels were excluded).
    has_data = (img[0] > 0) | (img[1] > 0) | (img[2] > 0)
    is_vegetation = (veg != 0)
    zm_off_x, zm_off_y = ZONES_OFFSET_M.get(year, (0.0, 0.0))
    zm_transform_corrected = Affine.translation(zm_off_x, zm_off_y) * zm_transform
    zones_dst = np.zeros(has_data.shape, dtype=np.int32)
    reproject(source=zones_master_arr, destination=zones_dst,
              src_transform=zm_transform_corrected, src_crs=zm_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              src_nodata=zm_nodata, dst_nodata=0, resampling=Resampling.nearest)
    # Tree_Zones_Master.tif carries a stray out-of-range sentinel value (observed: 65535,
    # a classic uint16 nodata placeholder) somewhere outside the real tree instances, never
    # filtered from zm_nodata because this file declares no explicit nodata metadata. Prior
    # scripts (v2-v4) never noticed because they only ever counted canopy pixels and merged
    # onto a table of real crown IDs, silently dropping any bogus row; treat any reprojected
    # value outside the real Zone_Value range as background so it can never reach bincount.
    zones_dst[zones_dst >= N_BINS] = 0
    canopy = has_data & is_vegetation & (zones_dst > 0)
    ground = has_data & (~is_vegetation)   # real, non-vegetation pixels: soil / bare ground / alleys

    # ---- pixel area (GSD^2) from this year's OWN transform ----
    px_w = abs(dst_transform.a); px_h = abs(dst_transform.e)
    pixel_area_m2 = px_w * px_h
    print(f"  {year}: GSD = {px_w*100:.2f} x {px_h*100:.2f} cm/px, pixel area = {pixel_area_m2:.6f} m2")

    # ---- canopy area per crown ----
    zones_c = zones_dst[canopy]
    n_bins = N_BINS   # global bin count (see above), NOT derived from this year's canopy zones only,
                       # otherwise ground_counts (which can reference any crown ID via nearest-neighbour
                       # assignment) could exceed canopy_counts's length and misalign the two arrays.
    canopy_counts = np.bincount(zones_c, minlength=n_bins).astype(np.float64)

    # ---- ground pixel -> true UTM coords (this year's own georeferencing, no offset hack needed) ----
    grows, gcols = np.where(ground)
    gX = dst_transform.a * (gcols + 0.5) + dst_transform.b * (grows + 0.5) + dst_transform.c
    gY = dst_transform.d * (gcols + 0.5) + dst_transform.e * (grows + 0.5) + dst_transform.f
    gR = img[0][ground].astype(np.float32); gG = img[1][ground].astype(np.float32); gB = img[2][ground].astype(np.float32)
    gBright = (gR + gG + gB) / 3.0
    del gR, gG, gB

    # ---- nearest crown for every ground pixel, chunked KD-tree query ----
    npts = len(gX)
    nearest_zone = np.full(npts, -1, dtype=np.int64)
    nearest_dist = np.full(npts, np.inf)
    for s in range(0, npts, GROUND_QUERY_CHUNK):
        e = min(s + GROUND_QUERY_CHUNK, npts)
        pts = np.column_stack([gX[s:e], gY[s:e]])
        d, idx = kdt_centroids.query(pts, k=1)
        nearest_dist[s:e] = d
        nearest_zone[s:e] = zone_order[idx]
    within_radius = nearest_dist <= np.array([radius_by_zone.get(z, 0.0) for z in nearest_zone])
    assigned_zone = np.where(within_radius, nearest_zone, -1)
    del gX, gY, nearest_dist

    # ---- Otsu shadow/sunlit split, one global threshold per year over assigned ground pixels ----
    assigned_mask = assigned_zone >= 0
    thr_shadow = otsu_threshold(gBright[assigned_mask]) if assigned_mask.sum() > MIN_GROUND_PIXELS else np.nan
    is_shadow = gBright < thr_shadow if thr_shadow == thr_shadow else np.zeros(npts, bool)
    print(f"  {year}: ground pixels={npts}, assigned-to-a-tree={assigned_mask.sum()}, "
          f"Otsu shadow threshold={thr_shadow:.1f}")

    ground_counts = np.bincount(assigned_zone[assigned_mask], minlength=n_bins).astype(np.float64)
    shadow_counts = np.bincount(assigned_zone[assigned_mask & is_shadow], minlength=n_bins).astype(np.float64)
    del gBright, is_shadow, assigned_zone, assigned_mask

    zv = np.arange(n_bins)
    canopy_area = canopy_counts * pixel_area_m2
    ground_pix = ground_counts
    shadow_frac = np.where(ground_pix >= MIN_GROUND_PIXELS, shadow_counts / np.maximum(ground_pix, 1), np.nan)
    cover_frac = np.where((canopy_counts + ground_pix) >= MIN_GROUND_PIXELS,
                          canopy_counts / np.maximum(canopy_counts + ground_pix, 1), np.nan)

    keep = canopy_counts > 0
    out = pd.DataFrame({"Zone_Value": zv[keep],
        f"CanopyPixelCount_{year}": canopy_counts[keep].astype(int),
        f"CanopyArea_m2_{year}": np.round(canopy_area[keep], 4),
        f"GSD_m_{year}": round(px_w, 5),
        f"GroundPixels_{year}": ground_pix[keep].astype(int),
        f"ShadowFraction_{year}": np.round(shadow_frac[keep], 4),
        f"CanopyCoverFraction_{year}": np.round(cover_frac[keep], 4)})
    del img, veg, has_data, is_vegetation, zones_dst, canopy, ground, zones_c; gc.collect()
    return out


for year, path in mosaics.items():
    print(f"--- {year} ---")
    dfy = compute_year(path, year)
    df_master = df_master.merge(dfy, on="Zone_Value", how="left")
    gc.collect()

df_master.to_excel(master_excel, index=False)
print("=" * 60, f"\nSaved: {master_excel}\n{len(df_master)} trees x {len(mosaics)} years", "\n" + "=" * 60)
print("Columns per year: CanopyPixelCount, CanopyArea_m2, GSD_m, GroundPixels,")
print("ShadowFraction (experimental, assumes near-solar-noon acquisition),")
print("CanopyCoverFraction (the more assumption-light light-interception proxy).")
print("Next: run pipeline/06_analysis/canopy_structure_yield.py (sandbox-safe, no")
print("rasterio needed) to test these against measured yield and whether they add")
print("anything beyond EBI.")
