import rasterio
from rasterio.warp import transform as transform_coords
import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.spatial import cKDTree
from skimage.exposure import match_histograms
import pandas as pd
import os
import gc

# ==========================================
# CONFIGURATION
# ==========================================
current_year = "2023"  # Change to "2024" for the base run, or "2025", "2026" etc.
reference_year = "2024"

ref_mosaic_path = "/Users/snirtahasa/Thesis/4band_mosaic/29_02_2024/kdm_mosaic_export.tif"

if current_year == reference_year:
    input_mosaic = ref_mosaic_path
else:
    input_mosaic = f"/Users/snirtahasa/Thesis/4band_mosaic/01_03_2023/Final_Orthomosaic_4Band.tif"

output_folder = "/Users/snirtahasa/Thesis/4band_mosaic"
master_excel = os.path.join(output_folder, "Master_Trees_Time_Series.xlsx")
stats_path = os.path.join(output_folder, "baseline_2024_stats.csv")

MAX_MATCH_DISTANCE = 3.0 
epsilon = 1e-6 

print(f"--- Processing Data for Year: {current_year} (Reference Anchor: {reference_year}) ---")

# ==========================================
# 1. READ CURRENT YEAR DATA
# ==========================================
with rasterio.open(input_mosaic) as src:
    R = src.read(1).astype(np.float32)
    G = src.read(2).astype(np.float32)
    B = src.read(3).astype(np.float32)
    mask = src.read(4)
    img_transform = src.transform
    img_crs = src.crs

# Extract tree segmentation objects
labeled_trees, num_trees = label(mask == 1)
tree_indices = np.arange(1, num_trees + 1)
centers_in_pixels = center_of_mass(mask, labeled_trees, tree_indices)

# ==========================================
# 2. EXACT RADIOMETRIC NORMALIZATION (CANOPY-ONLY)
# ==========================================
if current_year != reference_year:
    print("Performing EXACT Histogram Matching against 2024 (Canopy-Only Mode to save RAM)...")
    
    with rasterio.open(ref_mosaic_path) as ref_src:
        mask_ref = ref_src.read(4)
        
        print("  -> Exact Matching Red band...")
        R_ref = ref_src.read(1).astype(np.float32)
        R_ref_canopy = R_ref[mask_ref == 1]
        del R_ref
        gc.collect()
        
        R_canopy = R[mask == 1]
        R_canopy_matched = match_histograms(R_canopy, R_ref_canopy)
        R[mask == 1] = R_canopy_matched
        del R_ref_canopy, R_canopy, R_canopy_matched
        gc.collect()

        print("  -> Exact Matching Green band...")
        G_ref = ref_src.read(2).astype(np.float32)
        G_ref_canopy = G_ref[mask_ref == 1]
        del G_ref
        gc.collect()
        
        G_canopy = G[mask == 1]
        G_canopy_matched = match_histograms(G_canopy, G_ref_canopy)
        G[mask == 1] = G_canopy_matched
        del G_ref_canopy, G_canopy, G_canopy_matched
        gc.collect()

        print("  -> Exact Matching Blue band...")
        B_ref = ref_src.read(3).astype(np.float32)
        B_ref_canopy = B_ref[mask_ref == 1]
        del B_ref
        gc.collect()
        
        B_canopy = B[mask == 1]
        B_canopy_matched = match_histograms(B_canopy, B_ref_canopy)
        B[mask == 1] = B_canopy_matched
        del B_ref_canopy, B_canopy, B_canopy_matched
        del mask_ref
        gc.collect()
        
    print("Exact Histogram matching complete.")

# ==========================================
# 3. CALCULATE RAW VEGETATION INDICES (EXTREME 1D RAM SAVING)
# ==========================================
print("Calculating EBI and NGRDI (Strictly on canopy pixels)...")
tree_mask = (mask == 1)

# Extract 1D arrays of tree pixels only
R_trees = R[tree_mask]
G_trees = G[tree_mask]
B_trees = B[tree_mask]

# Free full 2D images immediately
del R, G, B
gc.collect()

# Calculate indices in 1D space
ebi_num = R_trees + G_trees + B_trees
ebi_den = (G_trees / (B_trees + epsilon)) * (R_trees - B_trees + epsilon)
raw_ebi_1d = np.divide(ebi_num, ebi_den, out=np.zeros_like(ebi_num), where=ebi_den!=0)

ngrdi_num = G_trees - R_trees
ngrdi_den = G_trees + R_trees
raw_ngrdi_1d = np.divide(ngrdi_num, ngrdi_den, out=np.zeros_like(ngrdi_num), where=ngrdi_den!=0)

del R_trees, G_trees, B_trees, ebi_num, ebi_den, ngrdi_num, ngrdi_den
gc.collect()

# Flatten the tree labels grid to match 1D tree arrays for grouping
labeled_trees_1d = labeled_trees[tree_mask]

# Free the 2D labels mask to keep memory low
del labeled_trees
gc.collect()

# ==========================================
# 4. SCALING & STANDARDIZATION BASED ON 2024 BASELINE (ROBUST PERCENTILE MODE)
# ==========================================
if current_year == reference_year:
    # Calculate robust statistics from the reference year using percentiles to eliminate extreme outliers
    mean_ebi_ref = np.nanmean(raw_ebi_1d)
    std_ebi_ref = np.nanstd(raw_ebi_1d)
    min_ebi_ref = np.nanpercentile(raw_ebi_1d, 1)   # 1st percentile filters out dark/error pixels
    max_ebi_ref = np.nanpercentile(raw_ebi_1d, 99)  # 99th percentile filters out extreme glare/bright outliers
    
    mean_ngrdi_ref = np.nanmean(raw_ngrdi_1d)
    std_ngrdi_ref = np.nanstd(raw_ngrdi_1d)
    min_ngrdi_ref = np.nanpercentile(raw_ngrdi_1d, 1)
    max_ngrdi_ref = np.nanpercentile(raw_ngrdi_1d, 99)
    
    # Save robust 2024 baseline statistics for future cross-year runs
    stats_df = pd.DataFrame({
        'Metric': ['mean_ebi', 'std_ebi', 'min_ebi', 'max_ebi', 'mean_ngrdi', 'std_ngrdi', 'min_ngrdi', 'max_ngrdi'],
        'Value': [mean_ebi_ref, std_ebi_ref, min_ebi_ref, max_ebi_ref, mean_ngrdi_ref, std_ngrdi_ref, min_ngrdi_ref, max_ngrdi_ref]
    })
    stats_df.to_csv(stats_path, index=False)
else:
    # Load locked robust 2024 baseline statistics for cross-year standardization
    if os.path.exists(stats_path):
        stats_df = pd.read_csv(stats_path).set_index('Metric')
        mean_ebi_ref = stats_df.loc['mean_ebi', 'Value']
        std_ebi_ref = stats_df.loc['std_ebi', 'Value']
        min_ebi_ref = stats_df.loc['min_ebi', 'Value']
        max_ebi_ref = stats_df.loc['max_ebi', 'Value']
        
        mean_ngrdi_ref = stats_df.loc['mean_ngrdi', 'Value']
        std_ngrdi_ref = stats_df.loc['std_ngrdi', 'Value']
        min_ngrdi_ref = stats_df.loc['min_ngrdi', 'Value']
        max_ngrdi_ref = stats_df.loc['max_ngrdi', 'Value']
    else:
        raise FileNotFoundError("Baseline 2024 stats missing! Run the script for 2024 first.")

# Process 1D scaling arrays using the locked robust 2024 reference metrics
z_score_ebi_1d = (raw_ebi_1d - mean_ebi_ref) / std_ebi_ref
z_score_ngrdi_1d = (raw_ngrdi_1d - mean_ngrdi_ref) / std_ngrdi_ref

range_ebi = (max_ebi_ref - min_ebi_ref) if (max_ebi_ref - min_ebi_ref) != 0 else 1e-6
range_ngrdi = (max_ngrdi_ref - min_ngrdi_ref) if (max_ngrdi_ref - min_ngrdi_ref) != 0 else 1e-6

# Normalization (0 to 1) based on the robust 2024 scale (values outside are clipped safely)
norm_ebi_1d = np.clip((raw_ebi_1d - min_ebi_ref) / range_ebi, 0, 1)
norm_ngrdi_1d = np.clip((raw_ngrdi_1d - min_ngrdi_ref) / range_ngrdi, 0, 1)

# ==========================================
# 5. EXTRACT PER-TREE DATA & COORDINATES
# ==========================================
print("Extracting per-tree data metrics...")
current_year_data = []

# Optimization: Group 1D pixels by tree ID efficiently using advanced numpy indexing
for i, temp_id in enumerate(tree_indices):
    pixel_selector = (labeled_trees_1d == temp_id)
    
    if not np.any(pixel_selector):
        continue
        
    tree_ebi_raw = np.nanmean(raw_ebi_1d[pixel_selector])
    tree_ngrdi_raw = np.nanmean(raw_ngrdi_1d[pixel_selector])
    
    tree_ebi_norm = np.nanmean(norm_ebi_1d[pixel_selector])
    tree_ngrdi_norm = np.nanmean(norm_ngrdi_1d[pixel_selector])
    
    tree_ebi_z = np.nanmean(z_score_ebi_1d[pixel_selector])
    tree_ngrdi_z = np.nanmean(z_score_ngrdi_1d[pixel_selector])
    
    row, col = centers_in_pixels[i]
    x_utm, y_utm = rasterio.transform.xy(img_transform, row, col)
    lon, lat = transform_coords(img_crs, 'EPSG:4326', [x_utm], [y_utm])
    
    current_year_data.append({
        "Temp_ID": temp_id,
        "X_UTM": x_utm,
        "Y_UTM": y_utm,
        "Latitude": lat[0],
        "Longitude": lon[0],
        f"EBI_Raw_{current_year}": round(tree_ebi_raw, 4),
        f"NGRDI_Raw_{current_year}": round(tree_ngrdi_raw, 4),
        f"EBI_Norm_{current_year}": round(tree_ebi_norm, 4),
        f"NGRDI_Norm_{current_year}": round(tree_ngrdi_norm, 4),
        f"EBI_Z_{current_year}": round(tree_ebi_z, 4),
        f"NGRDI_Z_{current_year}": round(tree_ngrdi_z, 4)
    })

df_current = pd.DataFrame(current_year_data)

del raw_ebi_1d, raw_ngrdi_1d, norm_ebi_1d, norm_ngrdi_1d, z_score_ebi_1d, z_score_ngrdi_1d, labeled_trees_1d
gc.collect()

# ==========================================
# 6. SPATIAL MATCHING & TIME-SERIES MERGE (GLOBAL GREEDY 1:1)
# ==========================================
if os.path.exists(master_excel):
    print(f"Master Excel found. Spatially matching {current_year} data to Master baseline...")
    df_master = pd.read_excel(master_excel)
    
    master_coords = np.c_[df_master['X_UTM'], df_master['Y_UTM']]
    kdtree = cKDTree(master_coords)
    current_coords = np.c_[df_current['X_UTM'], df_current['Y_UTM']]
    
    distances, indices = kdtree.query(current_coords, k=4)
    
    potential_matches = []
    for curr_idx, (dist_array, master_idx_array) in enumerate(zip(distances, indices)):
        dist_array = np.atleast_1d(dist_array)
        master_idx_array = np.atleast_1d(master_idx_array)
        
        for d, m_idx in zip(dist_array, master_idx_array):
            if d <= MAX_MATCH_DISTANCE: 
                potential_matches.append({
                    'current_idx': curr_idx,
                    'master_idx': m_idx,
                    'distance': d
                })
                
    potential_matches.sort(key=lambda x: x['distance'])
    
    assigned_masters = set()
    assigned_currents = set()
    matched_data = []
    
    for match in potential_matches:
        c_idx = match['current_idx']
        m_idx = match['master_idx']
        
        if m_idx not in assigned_masters and c_idx not in assigned_currents:
            assigned_masters.add(m_idx)
            assigned_currents.add(c_idx)
            
            real_tree_id = df_master.iloc[m_idx]['Tree_ID']
            matched_data.append({
                "Tree_ID": real_tree_id,
                f"EBI_Raw_{current_year}": df_current.iloc[c_idx][f"EBI_Raw_{current_year}"],
                f"NGRDI_Raw_{current_year}": df_current.iloc[c_idx][f"NGRDI_Raw_{current_year}"],
                f"EBI_Norm_{current_year}": df_current.iloc[c_idx][f"EBI_Norm_{current_year}"],
                f"NGRDI_Norm_{current_year}": df_current.iloc[c_idx][f"NGRDI_Norm_{current_year}"],
                f"EBI_Z_{current_year}": df_current.iloc[c_idx][f"EBI_Z_{current_year}"],
                f"NGRDI_Z_{current_year}": df_current.iloc[c_idx][f"NGRDI_Z_{current_year}"]
            })
            
    df_matched = pd.DataFrame(matched_data)
    df_final = pd.merge(df_master, df_matched, on="Tree_ID", how="outer")

else:
    print("Master Excel not found. Creating brand new Master baseline file...")
    df_current['Tree_ID'] = [f"Tree_{i:04d}" for i in range(1, len(df_current) + 1)]
    cols = [
        'Tree_ID', 'Latitude', 'Longitude', 'X_UTM', 'Y_UTM', 
        f'EBI_Raw_{current_year}', f'NGRDI_Raw_{current_year}',
        f'EBI_Norm_{current_year}', f'NGRDI_Norm_{current_year}',
        f'EBI_Z_{current_year}', f'NGRDI_Z_{current_year}'
    ]
    df_final = df_current[cols]
    
# Save tracking history to Excel
df_final.to_excel(master_excel, index=False)

print("="*50)
print(f"SUCCESS! {current_year} data processed and saved.")
print(f"Saved to: {master_excel}")
print("="*50)