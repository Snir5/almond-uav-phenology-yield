#!/usr/bin/env python3
"""
verify_yield_match_tree_position.py: per direct request, checks whether the
haversine-based coordinate matching (used throughout this entire project to
link kedma_plot_a_yield.csv trees to master trees, 8m threshold) agrees with an
INDEPENDENT identifier that was never used before: the yield CSV's
`tree_position` column (format A_r_{row}_c_{col}, a row/column grid position
within Plot A, populated for 148/202 rows) and the GPKG physiology files' own
`Row` field (which the physiology-matching scripts, e.g. gpkg_only_comparison.py,
also use via nearest-XY-match, not by Row number).
"""
import os, sys, csv, math, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
import sqlite3

BASE = find_thesis_root()
M = load_master()
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)
cultivar = [r.get("cultivar") for r in M]

yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
print(f"total yield rows: {len(yc)}")

def hav(a, b, d, e):
    p = math.pi / 180
    x = math.sin((d - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(d * p) * math.sin((e - b) * p / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(max(0, x)))

# ---- reproduce the exact matching used throughout the project (8m threshold, dedup) ----
coord = {}; ymeas_rows = []
for r in yc:
    coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))
    ymeas_rows.append(r)

cand = []
for t, (la, lo) in coord.items():
    ds = np.array([hav(la, lo, a, b) for a, b in zip(mlat, mlon)])
    for j in np.where(ds <= 8)[0]: cand.append((float(ds[j]), t, int(j)))
cand.sort(); uy = set(); um = set(); ymap = {}; ymap_dist = {}
for d, t, j in cand:
    if t in uy or j in um: continue
    ymap[t] = j; ymap_dist[t] = d; uy.add(t); um.add(j)
print(f"matched (haversine, <=8m, deduped): {len(ymap)} of {len(coord)} distinct tree_ids")

# ---- Row field available on the master (Row_2023, from the GPKG physiology nearest-XY match) ----
master_row2023 = [r.get("Row_2023") for r in M]

# ---- also load Row directly from the 2023 GPKG (independent of load_master's own join) ----
def gpkg_geom_centroid(blob):
    flags = blob[3]; envelope_ind = (flags >> 1) & 0x07
    env_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope_ind]
    wkb = blob[8 + env_bytes:]; endian = "<" if wkb[0] == 1 else ">"
    pos = 1
    import struct
    geom_type = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
    base_type = geom_type % 1000
    xs, ys = [], []
    if base_type == 6:
        npoly = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
        for _ in range(npoly):
            pos += 1; struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
            nrings = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
            for r in range(nrings):
                npts = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
                for _p in range(npts):
                    x, y = struct.unpack_from(endian + "dd", wkb, pos); pos += 16
                    if r == 0: xs.append(x); ys.append(y)
    elif base_type == 3:
        nrings = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
        for r in range(nrings):
            npts = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
            for _p in range(npts):
                x, y = struct.unpack_from(endian + "dd", wkb, pos); pos += 16
                if r == 0: xs.append(x); ys.append(y)
    return sum(xs) / len(xs), sum(ys) / len(ys)

con = sqlite3.connect(os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs", "Yield_with_clustering_2023.gpkg"))
cur = con.cursor()
cur.execute("SELECT table_name FROM gpkg_geometry_columns"); tbl = cur.fetchone()[0]
cur.execute(f'SELECT Row, Plot, cultivar, geom FROM "{tbl}"')
gpkg_rows = cur.fetchall(); con.close()
gx = []; gy = []; grow = []; gcult = []
for Row, Plot, cult, blob in gpkg_rows:
    if Plot != "A" or Row is None: continue
    x, y = gpkg_geom_centroid(blob)
    gx.append(x); gy.append(y); grow.append(Row); gcult.append(cult)
gx = np.array(gx); gy = np.array(gy); grow = np.array(grow)
mx = np.array([r.get("X_UTM") for r in M], float); my = np.array([r.get("Y_UTM") for r in M], float)
print(f"GPKG 2023 Plot A rows with Row+geom: {len(gx)}")

# ---- for every yield tree with a tree_position row number, compare it to: (a) master's Row_2023
#      (via load_master's own join), (b) an independent direct nearest-match to the GPKG Row ----
n_checked = 0; n_agree_a = 0; n_agree_b = 0; mismatches = []
for r in ymeas_rows:
    t = r["tree_id"]; tp = r["tree_position"].strip()
    m = re.match(r"A_r_(\d+)_c_(\d+)", tp)
    if not m or t not in ymap: continue
    csv_row = int(m.group(1))
    j = ymap[t]
    n_checked += 1
    # (a) master's own Row_2023 (already joined via geo_guardrails / earlier pipeline)
    mrow = master_row2023[j]
    agree_a = (mrow is not None and abs(float(mrow) - csv_row) <= 1)
    if agree_a: n_agree_a += 1
    # (b) independent direct nearest-XY match from master coords to the GPKG's own row/geom
    d = np.sqrt((gx - mx[j]) ** 2 + (gy - my[j]) ** 2)
    jj = int(np.argmin(d))
    grow_direct = grow[jj] if d[jj] <= 3.0 else None
    agree_b = (grow_direct is not None and abs(float(grow_direct) - csv_row) <= 1)
    if agree_b: n_agree_b += 1
    if not agree_a or not agree_b:
        mismatches.append({"tree_id": t, "csv_row": csv_row, "master_Row_2023": mrow,
                            "direct_gpkg_row": (float(grow_direct) if grow_direct is not None else None),
                            "haversine_dist_m": round(ymap_dist[t], 2)})

print(f"\nchecked {n_checked} yield trees with a tree_position row number and a haversine match")
print(f"  agree with master's Row_2023 (within +/-1): {n_agree_a} ({100*n_agree_a/max(n_checked,1):.1f}%)")
print(f"  agree with an independent direct GPKG nearest-match Row (within +/-1): {n_agree_b} ({100*n_agree_b/max(n_checked,1):.1f}%)")
print(f"  mismatches: {len(mismatches)}")
for mm in mismatches[:20]:
    print(f"    {mm}")
