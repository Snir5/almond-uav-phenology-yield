"""
plot_cultivar_segments.py
מציג את סגמנטציית העצים (שורות 39-76) צבועה לפי זן — 4 שנים
"""
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
zones_master_path = "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2021_03_07/Tree_Zones_Master.tif"
master_excel      = "/Users/snirtahasa/Thesis/4band_mosaic/Master_Trees_Time_Series_Plot1.xlsx"
mosaics = {
    "2021": "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2021_03_07/Final_Orthomosaic_4Band.tif",
    "2022": "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2022_03_02/Final_Orthomosaic_4Band.tif",
    "2023": "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2023_03_01/Final_Orthomosaic_4Band.tif",
    "2024": "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2024_02_29/Final_Orthomosaic_4Band.tif",
}
out_path = "/Users/snirtahasa/Thesis/Results_Analysis/cultivar_segments_4years.png"

PLOT1_COORDS = [
    (669846.69, 3509814.25),  # NW
    (670075.00, 3509755.50),  # NE
    (669973.83, 3509452.97),  # SE
    (669791.15, 3509556.32),  # SW
]
POLYGON_OFFSET = {
    "2021": (-66.15, 0.0),
    "2022": (0.0, 0.0),
    "2023": (0.0, 0.0),
    "2024": (0.0, 0.0),
}
ZONES_OFFSET_M = {
    "2021": (0.0, 0.0),
    "2022": (65.20, 42.65),
    "2023": (66.15, 43.05),
    "2024": (65.45, 42.70),
}

CULT_COLOR = {
    'UEF': np.array([33,  102, 172], dtype=np.uint8),   # blue
    '53':  np.array([215,  48,  39], dtype=np.uint8),   # red
    '54':  np.array([26,  150,  65], dtype=np.uint8),   # green
}
BG_COLOR  = np.array([200, 200, 200], dtype=np.uint8)   # gray = non-canopy

# ============================================================
print("Loading master...")
df = pd.read_excel(master_excel)
# Only main block (already filtered to 39-76, but guard anyway)
df = df[(df['Row_2023'] >= 39) & (df['Row_2023'] <= 76)]
zone_to_cult = dict(zip(df['Zone_Value'].astype(int), df['cultivar']))
print(f"  {len(zone_to_cult)} zones, cultivars: {df['cultivar'].value_counts().to_dict()}")

def make_poly(year):
    off = POLYGON_OFFSET.get(year, (0.0, 0.0))
    return Polygon([(x+off[0], y+off[1]) for x,y in PLOT1_COORDS])

def render_year(year):
    mosaic_path = mosaics[year]
    zones_off   = ZONES_OFFSET_M[year]
    poly        = make_poly(year)
    poly_2021   = make_poly("2021")

    print(f"[{year}] clipping mosaic...")
    with rasterio.open(mosaic_path) as mos:
        mos_crs = mos.crs
        clip, clip_trans = rio_mask(mos, [poly.__geo_interface__], crop=True, nodata=0)
        h, w = clip.shape[1], clip.shape[2]
        veg = clip[3] > 0 if mos.count >= 4 else np.ones((h, w), bool)

    # build destination transform (shift zones to year frame)
    dst_trans = rasterio.transform.Affine(
        clip_trans.a, clip_trans.b, clip_trans.c - zones_off[0],
        clip_trans.d, clip_trans.e, clip_trans.f - zones_off[1],
    )

    print(f"[{year}] reprojecting zones master...")
    with rasterio.open(zones_master_path) as zm:
        zm_clip, zm_trans = rio_mask(zm, [poly_2021.__geo_interface__], crop=True, nodata=0)
        zm_data = zm_clip[0]
        zones_reproj = np.zeros((h, w), dtype=zm_data.dtype)
        reproject(
            source=zm_data, destination=zones_reproj,
            src_transform=zm_trans, src_crs=zm.crs,
            dst_transform=dst_trans, dst_crs=mos_crs,
            resampling=Resampling.nearest,
        )

    # paint RGB
    rgb = np.full((h, w, 3), BG_COLOR, dtype=np.uint8)
    for z in np.unique(zones_reproj):
        if z == 0:
            continue
        cult = zone_to_cult.get(int(z))
        if cult not in CULT_COLOR:
            continue
        rgb[(zones_reproj == z) & veg] = CULT_COLOR[cult]

    print(f"[{year}] done ({h}x{w}px)")
    return rgb

# ============================================================
years = ["2021", "2022", "2023", "2024"]
images = {yr: render_year(yr) for yr in years}

fig, axes = plt.subplots(1, 4, figsize=(22, 8))
fig.patch.set_facecolor('white')
for ax, yr in zip(axes, years):
    ax.imshow(images[yr], origin='upper')
    ax.set_title(yr, fontsize=14, fontweight='bold', pad=8)
    ax.axis('off')

patches = [
    mpatches.Patch(color=np.array(CULT_COLOR['UEF'])/255, label='אום אל פאחם (UEF)'),
    mpatches.Patch(color=np.array(CULT_COLOR['53'])/255,  label='זן 53'),
    mpatches.Patch(color=np.array(CULT_COLOR['54'])/255,  label='זן 54'),
]
fig.legend(handles=patches, loc='upper center', ncol=3, fontsize=12,
           bbox_to_anchor=(0.5, 1.01), framealpha=0.9, edgecolor='#ccc')
fig.suptitle('Plot 1 — סגמנטים צבועים לפי זן (2021–2024)', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {out_path}")
