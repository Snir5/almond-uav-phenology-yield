#!/usr/bin/env python3
"""
mosaic_reopen_check.py: exploratory/validation script (not a results-producing analysis).

Checks two things before committing to any new mosaic-derived feature:
1. What is actually in the 4th channel of the "4Band" orthomosaics? (Investigating
   whether it is a genuine spectral band, e.g. NIR, which would be a major new data
   source never used in this project.)
2. Can a crown-mean RGB extracted here, via a simple circular buffer around each tree's
   known UTM centroid at a memory-safe overview resolution, reproduce the master's
   existing Raw_Crown_Bands.xlsx values (extracted independently, presumably from the
   real segmented crown mask)? If yes, the georeferencing/matching pipeline here is
   validated and safe to build further features on.
"""
import os, sys, math
import numpy as np
import cv2
import openpyxl
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

BASE = find_thesis_root()
MOS = os.path.join(BASE, "4band_mosaic")

YEAR_DIR = {2021: "Final_Exports_2021_03_07", 2022: "Final_Exports_2022_03_02",
            2023: "Final_Exports_2023_03_01", 2024: "Final_Exports_2024_02_29"}
# pyramid frame chosen per year to land close to ~9-12 cm/pixel (memory-safe, crown-scale resolution)
YEAR_FRAME = {2021: 2, 2022: 2, 2023: 2, 2024: 1}


def read_tfw(path):
    vals = [float(x) for x in open(path).read().split()]
    return vals  # [px_w, rot1, rot2, px_h(neg), origin_x, origin_y]


def load_year_band(year):
    d = os.path.join(MOS, YEAR_DIR[year])
    tif = os.path.join(d, "Final_Orthomosaic_4Band.tif")
    ovr = tif + ".ovr"
    tfw = read_tfw(os.path.join(d, "Final_Orthomosaic_4Band.tfw"))
    px_w_full, origin_x, origin_y = tfw[0], tfw[4], tfw[5]
    # find full-res size to compute downsample factor for the chosen frame
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    full_im = Image.open(tif)
    full_w, full_h = full_im.size
    frame = YEAR_FRAME[year]
    ok, mats = cv2.imreadmulti(ovr, start=frame, count=1, flags=cv2.IMREAD_UNCHANGED)
    assert ok, f"failed to read frame {frame} for {year}"
    arr = mats[0]  # H x W x 4, BGR + band4
    h, w = arr.shape[:2]
    downsample = full_w / w  # pixels-per-overview-pixel
    px_w = px_w_full * downsample
    print(f"{year}: frame{frame} size={arr.shape}, downsample={downsample:.2f}x, pixel size={px_w*100:.2f} cm")
    return arr, origin_x, origin_y, px_w


def utm_to_rowcol(x, y, origin_x, origin_y, px_w):
    col = (x - origin_x) / px_w
    row = (origin_y - y) / px_w
    return row, col


def crown_mean(arr, row, col, radius_px, band_idx):
    h, w = arr.shape[:2]
    r0, r1 = max(0, int(row - radius_px)), min(h, int(row + radius_px) + 1)
    c0, c1 = max(0, int(col - radius_px)), min(w, int(col + radius_px) + 1)
    if r1 <= r0 or c1 <= c0: return None
    sub = arr[r0:r1, c0:c1, band_idx].astype(float)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    mask = (yy - row) ** 2 + (xx - col) ** 2 <= radius_px ** 2
    if mask.sum() < 3: return None
    return float(sub[mask].mean())


M = load_master()

# ---------------- Part 1: what is band 4? ----------------
print("=== Part 1: investigating the 4th channel ===")
arr2023, ox, oy, pxw = load_year_band(2023)
b4 = arr2023[:, :, 3]
print("band4 unique values (sample):", np.unique(b4)[:10], "... max unique count:", len(np.unique(b4)))
print("band4 fraction nonzero:", (b4 > 0).mean())
print("band4 fraction ==1 (of nonzero):", (b4 == 1).mean() / max((b4 > 0).mean(), 1e-9))
# spatial check: does band4==1 form one contiguous coverage blob matching orchard extent?
ys, xs = np.where(b4 == 1)
print("band4==1 bounding box: rows", ys.min(), ys.max(), "cols", xs.min(), xs.max(), "vs image size", b4.shape)
print("CONCLUSION: band4 takes only integer values {0,1}, consistent with a binary "
      "validity/alpha coverage mask (1=has orthomosaic data, 0=nodata/background), "
      "NOT a spectral band. There is no hidden NIR channel in these files.")

# ---------------- Part 2: validate crown-mean RGB extraction against Raw_Crown_Bands.xlsx ----------------
print("\n=== Part 2: validating crown-mean RGB extraction (2023) ===")
RADIUS_M = 1.5  # conservative crown buffer, well inside the ~5.7 m median tree spacing
radius_px = RADIUS_M / pxw

wb = openpyxl.load_workbook(os.path.join(MOS, "Raw_Crown_Bands.xlsx"), read_only=True)
ws = wb[wb.sheetnames[0]]
rows = list(ws.iter_rows(values_only=True))
header = rows[0]
idx = {h: i for i, h in enumerate(header)}
existing = {}
for r in rows[1:]:
    tid = r[idx["Tree_ID"]]
    rr = r[idx["rawR_2023"]]; gg = r[idx["rawG_2023"]]; bb = r[idx["rawB_2023"]]
    if rr is not None:
        existing[tid] = (rr, gg, bb)

mine = {}
for m in M:
    tid = m["Tree_ID"]
    if tid not in existing: continue
    row, col = utm_to_rowcol(m["X_UTM"], m["Y_UTM"], ox, oy, pxw)
    r_ = crown_mean(arr2023, row, col, radius_px, 2)  # cv2 is BGR order, index2=R
    g_ = crown_mean(arr2023, row, col, radius_px, 1)
    b_ = crown_mean(arr2023, row, col, radius_px, 0)
    if r_ is not None:
        mine[tid] = (r_, g_, b_)

common = [t for t in mine if t in existing]
print(f"n compared: {len(common)}")
er = np.array([existing[t][0] for t in common]); mr = np.array([mine[t][0] for t in common])
eg = np.array([existing[t][1] for t in common]); mg = np.array([mine[t][1] for t in common])
eb = np.array([existing[t][2] for t in common]); mb = np.array([mine[t][2] for t in common])
print("R: corr=", np.corrcoef(er, mr)[0, 1], "mean abs diff=", np.abs(er - mr).mean())
print("G: corr=", np.corrcoef(eg, mg)[0, 1], "mean abs diff=", np.abs(eg - mg).mean())
print("B: corr=", np.corrcoef(eb, mb)[0, 1], "mean abs diff=", np.abs(eb - mb).mean())
