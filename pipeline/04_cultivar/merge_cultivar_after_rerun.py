"""
merge_cultivar_after_rerun.py
הרצה לאחר apply_fixed_zones_yearly.py — ממזג חזרה את שיוכי הזן לפי Tree_ID
"""
import pandas as pd

master_path  = "/Users/snirtahasa/Thesis/4band_mosaic/Master_Trees_Time_Series_Plot1.xlsx"
backup_path  = "/Users/snirtahasa/Thesis/4band_mosaic/cultivar_assignments_backup.csv"

print("Reading new master...")
master = pd.read_excel(master_path)
print(f"  New master: {len(master)} trees, columns: {list(master.columns[:8])}")

print("Reading cultivar backup...")
backup = pd.read_csv(backup_path)
print(f"  Backup: {len(backup)} rows, cultivar dist: {backup['cultivar'].value_counts().to_dict()}")

# Merge on Tree_ID
before_cols = set(master.columns)
master = master.merge(backup[['Tree_ID','cultivar','Row_2023']], on='Tree_ID', how='left')

matched = master['cultivar'].notna().sum()
unmatched = master['cultivar'].isna().sum()
print(f"\nMerge results:")
print(f"  Matched: {matched} trees")
print(f"  Unmatched (new trees not in backup): {unmatched}")

if unmatched > 0:
    print("  NOTE: Unmatched trees will have cultivar=NaN — review manually")

print(f"\nFinal cultivar distribution:")
print(master['cultivar'].value_counts())

master.to_excel(master_path, index=False)
print(f"\nSaved: {master_path}")
print("Done.")
