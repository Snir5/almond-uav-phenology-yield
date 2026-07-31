"""
merge_gpkg_to_master.py
ממזג נתוני GPKG (Yield 2022/2023 + Spectral July 2024) למאסטר Excel
לפי שורה + NN בתוך שורה (לנתוני Yield) ו-NN מרחבי (לנתוני Spectral)
"""
import sqlite3
import numpy as np
import pandas as pd
import os

MASTER_PATH = "/Users/snirtahasa/Thesis/4band_mosaic/Master_Trees_Time_Series_Plot1.xlsx"
GPKG_DIR    = "/Users/snirtahasa/Thesis/Trees Data and Survay/נתוני עצים 22-23 יבול 24"

# ============================================================
def parse_gpkg(fname, table):
    con = sqlite3.connect(fname)
    rtree = f"rtree_{table}_geom"
    rows  = con.execute(f'SELECT id, minx, maxx, miny, maxy FROM "{rtree}"').fetchall()
    coords = {r[0]: ((r[1]+r[2])/2, (r[3]+r[4])/2) for r in rows}
    cols_info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
    attr_cols = [c[1] for c in cols_info if c[1] not in ('fid','geom')]
    quoted = ','.join(f'"{c}"' for c in attr_cols)
    data = []
    for row in con.execute(f'SELECT fid, {quoted} FROM "{table}"').fetchall():
        fid = row[0]
        if fid in coords:
            cx, cy = coords[fid]
            data.append({'cx': cx, 'cy': cy, **dict(zip(attr_cols, row[1:]))})
    con.close()
    return pd.DataFrame(data)

def nn_match_within_row(master_df, gpkg_df, row_col_master='Row_2023',
                         row_col_gpkg='Row', x_col='X_UTM', cx_col='cx',
                         max_dist=6.0):
    """Match master trees to GPKG trees row-by-row using nearest X."""
    result_idx = np.full(len(master_df), -1, dtype=int)
    result_dist = np.full(len(master_df), np.nan)

    for row_num in master_df[row_col_master].dropna().unique():
        m_mask = master_df[row_col_master] == row_num
        g_mask = gpkg_df[row_col_gpkg] == row_num
        if g_mask.sum() == 0:
            continue  # gap row — no GPKG data

        m_xs  = master_df.loc[m_mask, x_col].values
        m_idx = master_df.index[m_mask]
        g_xs  = gpkg_df.loc[g_mask, cx_col].values
        g_idx = gpkg_df.index[g_mask]

        for i, (midx, mx) in enumerate(zip(m_idx, m_xs)):
            d = np.abs(g_xs - mx)
            best = np.argmin(d)
            if d[best] <= max_dist:
                result_idx[master_df.index.get_loc(midx)] = g_idx[best]
                result_dist[master_df.index.get_loc(midx)] = d[best]

    return result_idx, result_dist

def nn_match_spatial(master_xy, src_xy, max_dist=5.0):
    """Global nearest-neighbour, returns matched row index or -1."""
    matched = np.full(len(master_xy), -1, dtype=int)
    dist_arr = np.full(len(master_xy), np.nan)
    for i, (mx, my) in enumerate(master_xy):
        d = np.sqrt((src_xy[:,0]-mx)**2 + (src_xy[:,1]-my)**2)
        idx = np.argmin(d)
        if d[idx] <= max_dist:
            matched[i] = idx
            dist_arr[i] = d[idx]
    return matched, dist_arr

# ============================================================
print("Loading master...")
master = pd.read_excel(MASTER_PATH)
print(f"  {len(master)} trees")

# ============================================================
# 1. Yield 2022
# ============================================================
print("\n[1/3] Merging Yield 2022...")
df_y22 = parse_gpkg(os.path.join(GPKG_DIR,'Yield_with_clustering_2022.gpkg'),
                     'Yield_with_clustering_2022')
df_y22a = df_y22[df_y22['Plot']=='A'].copy().reset_index(drop=True)
print(f"  Yield_2022 Plot A: {len(df_y22a)} trees")

yield_cols = ['predicted_Yield']

matched_idx, dists = nn_match_within_row(master, df_y22a)
print(f"  Matched: {(matched_idx>=0).sum()}/{len(master)} (max_dist=6m)")

for col in yield_cols:
    if col not in df_y22a.columns:
        continue
    new_col = col.replace('.','_').replace('__','_') + '_2022'
    vals = np.full(len(master), np.nan)
    mask = matched_idx >= 0
    vals[mask] = df_y22a.loc[matched_idx[mask], col].values
    master[new_col] = vals

# ============================================================
# 2. Yield 2023
# ============================================================
print("\n[2/3] Merging Yield 2023...")
df_y23 = parse_gpkg(os.path.join(GPKG_DIR,'Yield_with_clustering_2023.gpkg'),
                     'Yield_with_clustering_2023')
df_y23a = df_y23[df_y23['Plot']=='A'].copy().reset_index(drop=True)
print(f"  Yield_2023 Plot A: {len(df_y23a)} trees")

matched_idx, dists = nn_match_within_row(master, df_y23a)
print(f"  Matched: {(matched_idx>=0).sum()}/{len(master)} (max_dist=6m)")

for col in yield_cols:
    if col not in df_y23a.columns:
        continue
    new_col = col.replace('.','_').replace('__','_') + '_2023'
    vals = np.full(len(master), np.nan)
    mask = matched_idx >= 0
    vals[mask] = df_y23a.loc[matched_idx[mask], col].values
    master[new_col] = vals

# ============================================================
# 3. Spectral July 2024 (Plot A only)
# ============================================================
print("\n[3/3] Merging Spectral July 2024...")
df_sp = parse_gpkg(os.path.join(GPKG_DIR,'spectral_data_plot_A_31072024.gpkg'),
                    'spectral_data_plot_A_31072024')
print(f"  Spectral Plot A: {len(df_sp)} trees")

spectral_cols = ['CWSI','Tc','DEM','NDVI','NDRE','EVI','SAVI',
                 'MSAVI','GNDI','OSAVI','CI','GI','BAI','DVI',
                 'blue','green','red','rededge','IR']

master_xy = master[['X_UTM','Y_UTM']].values
src_xy    = df_sp[['cx','cy']].values
matched_idx, dists = nn_match_spatial(master_xy, src_xy, max_dist=5.0)
print(f"  Matched: {(matched_idx>=0).sum()}/{len(master)} (max_dist=5m)")

for col in spectral_cols:
    if col not in df_sp.columns:
        continue
    new_col = col + '_Jul2024'
    vals = np.full(len(master), np.nan)
    mask = matched_idx >= 0
    vals[mask] = df_sp.loc[matched_idx[mask], col].values
    master[new_col] = vals

# ============================================================
print(f"\nFinal master: {len(master)} trees, {len(master.columns)} columns")
print(f"Columns added:")
new_cols = [c for c in master.columns if any(c.endswith(s) for s in ['_2022','_2023','Jul2024'])]
print(f"  {new_cols}")

master.to_excel(MASTER_PATH, index=False)
print(f"\nSaved: {MASTER_PATH}")

# Quick summary
print("\n=== Data coverage ===")
for col in ['predicted_Yield_2022','predicted_Yield_2023','NDVI_Jul2024','CWSI_Jul2024']:
    if col in master.columns:
        n = master[col].notna().sum()
        print(f"  {col}: {n}/{len(master)} ({100*n/len(master):.1f}%)")
