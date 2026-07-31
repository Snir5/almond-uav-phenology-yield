# ============================================================
# extract_raw_crown_bands.py
#
# One-pass extraction that dumps, per tree per year, the RAW (uncorrected)
# crown-mean R, G, B, plus each year's canopy channel statistics (mean/std of
# R,G,B over all canopy pixels). From this small table ANY radiometric reference
# year can be applied offline (shift+scale) and EBI recomputed - so the
# "which reference year is best" experiment needs NO re-extraction.
#
# Same clipping / fixed Tree_Zones_Master / georeferencing offsets as
# apply_fixed_zones_yearly.py. Requires rasterio (run where it is installed).
# Output: 4band_mosaic/Raw_Crown_Bands.xlsx  and  Raw_Crown_ChannelStats.json
# ============================================================
import rasterio
from rasterio.warp import reproject, Resampling, transform as transform_coords
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine
from shapely.geometry import Polygon
import numpy as np, pandas as pd, os, gc, json

BASE = "/Users/snirtahasa/Thesis/4band_mosaic"
zones_master_path = f"{BASE}/Final_Exports_2021_03_07/Tree_Zones_Master.tif"
mosaics = {
    "2021": f"{BASE}/Final_Exports_2021_03_07/Final_Orthomosaic_4Band.tif",
    "2022": f"{BASE}/Final_Exports_2022_03_02/Final_Orthomosaic_4Band.tif",
    "2023": f"{BASE}/Final_Exports_2023_03_01/Final_Orthomosaic_4Band.tif",
    "2024": f"{BASE}/Final_Exports_2024_02_29/Final_Orthomosaic_4Band.tif",
}
PLOT1_COORDS = [(669846.69, 3509814.25), (670075.00, 3509755.50),
                (669973.83, 3509452.97), (669791.15, 3509556.32)]
plot1_polygon = Polygon(PLOT1_COORDS)
POLYGON_OFFSET = {"2021": (-66.15, 0.0), "2022": (0.0, 0.0), "2023": (0.0, 0.0), "2024": (0.0, 0.0)}
ZONES_OFFSET_M = {"2021": (0.0, 0.0), "2022": (65.20, 42.65), "2023": (66.15, 43.05), "2024": (65.45, 42.70)}


def clip_raster(path, off=(0.0, 0.0)):
    ox, oy = off
    poly = plot1_polygon if (ox == 0 and oy == 0) else Polygon([(x + ox, y + oy) for x, y in PLOT1_COORDS])
    with rasterio.open(path) as src:
        img, tr = rio_mask(src, [poly], crop=True, all_touched=True, filled=True)
        return img, tr, src.crs, src.nodatavals


# zones master + base table
zm_img, zm_tr, zm_crs, zm_nod = clip_raster(zones_master_path, POLYGON_OFFSET["2021"])
zones = zm_img[0]; zm_nodata = zm_nod[0] if zm_nod and zm_nod[0] is not None else 0
from scipy.ndimage import center_of_mass
valid = zones > 0
if zm_nodata != 0: valid &= (zones != zm_nodata)
tids = np.unique(zones[valid]); tids = tids[tids > 0]
cen = center_of_mass(np.ones(zones.shape, np.float32), labels=zones, index=tids)
base = []
for tid, (row, cc) in zip(tids, cen):
    x, y = rasterio.transform.xy(zm_tr, row, cc); xt = x + 66.15
    lon, lat = transform_coords(zm_crs, "EPSG:4326", [xt], [y])
    base.append({"Tree_ID": f"Tree_{int(tid):04d}", "Zone_Value": int(tid),
                 "X_UTM": xt, "Y_UTM": y, "Latitude": lat[0], "Longitude": lon[0]})
df = pd.DataFrame(base)
chan_stats = {}

for year, path in mosaics.items():
    print("year", year)
    img, dst_tr, dst_crs, nod = clip_raster(path, POLYGON_OFFSET.get(year, (0, 0)))
    mask = img[3]; mnod = nod[3] if len(nod) > 3 else None
    ox, oy = ZONES_OFFSET_M.get(year, (0, 0)); zt = Affine.translation(ox, oy) * zm_tr
    zdst = np.zeros(mask.shape, np.int32)
    reproject(source=zones, destination=zdst, src_transform=zt, src_crs=zm_crs,
              dst_transform=dst_tr, dst_crs=dst_crs, src_nodata=zm_nodata, dst_nodata=0,
              resampling=Resampling.nearest)
    canopy = (mask != 0) & (zdst > 0)
    if mnod is not None: canopy &= (mask != mnod)
    zc = zdst[canopy]
    R = img[0][canopy].astype(np.float32); G = img[1][canopy].astype(np.float32); B = img[2][canopy].astype(np.float32)
    del img, mask, zdst, canopy; gc.collect()
    # per-year canopy channel stats (for defining any reference affine)
    chan_stats[year] = {"mR": float(R.mean()), "sR": float(R.std()), "mG": float(G.mean()),
                        "sG": float(G.std()), "mB": float(B.mean()), "sB": float(B.std()), "n_px": int(len(R))}
    n_bins = int(zc.max()) + 1; counts = np.bincount(zc, minlength=n_bins).astype(np.float64); ok = counts > 0
    def zmean(v):
        s = np.bincount(zc, weights=v, minlength=n_bins); out = np.full(n_bins, np.nan); out[ok] = s[ok] / counts[ok]; return out
    mr, mg, mb = zmean(R), zmean(G), zmean(B)
    zv = np.where(ok)[0]
    dy = pd.DataFrame({"Zone_Value": zv, f"rawR_{year}": np.round(mr[ok], 3),
                       f"rawG_{year}": np.round(mg[ok], 3), f"rawB_{year}": np.round(mb[ok], 3)})
    df = df.merge(dy, on="Zone_Value", how="left")
    del R, G, B; gc.collect()

df.to_excel(f"{BASE}/Raw_Crown_Bands.xlsx", index=False)
json.dump(chan_stats, open(f"{BASE}/Raw_Crown_ChannelStats.json", "w"), indent=2)
print("Saved Raw_Crown_Bands.xlsx and Raw_Crown_ChannelStats.json")
print("channel stats:", json.dumps(chan_stats, indent=1))
