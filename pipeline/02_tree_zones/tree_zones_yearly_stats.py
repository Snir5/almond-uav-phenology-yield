# ==========================================================================
# tree_zones_yearly_stats.py
#
# מטרה: להשתמש בחלוקה הקבועה (Tree_Zones_Master.tif, שנוצרה ע"י
# tree_separation_grid.py) ולחשב, לכל שנה, כמה פיקסלי-צמרת יש בכל
# "אזור עץ" (TreeID). בסוף נבנית טבלת השוואה אחת (CSV) - שורה לכל
# TreeID, עמודה לכל שנה - שאפשר לראות בה את השינוי בצמרת של כל עץ
# בנפרד בין שנים.
#
# אין צורך להריץ שוב detect_grid / dynamic centers / seed generation -
# כל זה כבר "קפוא" בתוך Tree_Zones_Master.tif.
#
# לכל שנה צריך רק:
#   1. בינריזציה של Band_4 לאותה שנה -> מסכת צמחייה (1=עץ, 0=רקע)
#   2. Zonal Statistics as Table עם Tree_Zones_Master.tif כראסטר-אזורים
# ==========================================================================

import arcpy
from arcpy.sa import *
import os
import csv

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False

# ==========================================
# נתיבים - לעדכן בהתאם
# ==========================================

# הראסטר הקבוע שנוצר בריצה הראשונה (אל תשנו אותו!)
tree_zones_master = r"C:\Users\snirt\2021_03_07\Final_Exports\Tree_Zones_Master.tif"

# תיקייה לפלטים של הסקריפט הזה
output_folder = r"C:\Users\snirt\2021_03_07\Final_Exports\YearlyStats"

# רשימת השנים להשוואה: (תווית, נתיב למוזאיקת 4Band של אותה שנה)
# השנה הראשונה (זו שיצרה את Tree_Zones_Master) אפשר/רצוי לכלול גם אותה
# כאן, כך שתקבלו בסיס השוואה (שנה 0).
years = [
    ("2021", r"C:\Users\snirt\2021_03_07\Final_Exports\Final_Orthomosaic_4Band.tif"),
    ("2022", r"C:\Users\snirt\2022_..\Final_Exports\Final_Orthomosaic_4Band.tif"),
    ("2023", r"C:\Users\snirt\2023_..\Final_Exports\Final_Orthomosaic_4Band.tif"),
    ("2024", r"C:\Users\snirt\2024_..\Final_Exports\Final_Orthomosaic_4Band.tif"),
]

# ==========================================
# הגדרות עיבוד
# ==========================================
os.makedirs(output_folder, exist_ok=True)

# חשוב: ליישר את כל החישובים לרשת התאים של Tree_Zones_Master,
# כדי שכל פיקסל בכל שנה ייפול בדיוק על אותו "תא-אזור".
arcpy.env.snapRaster = tree_zones_master
arcpy.env.cellSize = tree_zones_master

try:
    # -----------------------------------------------------
    # PART 1: עבור כל שנה - בינריזציה + Zonal Statistics
    # -----------------------------------------------------
    for year_label, path_4band in years:
        print(f"=== שנה {year_label} ===")

        band4 = Raster(path_4band + r"\Band_4")

        # מסכת צמחייה: 1=צמרת/עץ, 0=רקע (היפוך של binary_mask ב-PART 1
        # של tree_separation_grid.py, שם 1=רקע)
        veg_mask = Con(IsNull(band4), 0, Con(band4 == 0, 0, 1))

        temp_veg = os.path.join(output_folder, f"veg_mask_{year_label}.tif")
        veg_mask.save(temp_veg)

        out_table = os.path.join(output_folder, f"zone_stats_{year_label}.dbf")
        ZonalStatisticsAsTable(tree_zones_master, "VALUE", temp_veg, out_table, "DATA", "SUM")
        print(f"  -> נשמרה טבלה: {out_table}")

        try:
            arcpy.management.Delete(temp_veg)
        except Exception:
            pass

    # -----------------------------------------------------
    # PART 2: מיזוג כל הטבלות לטבלת השוואה אחת (CSV)
    # שורה לכל TreeID, עמודה לכל שנה = מספר פיקסלי-צמרת
    # -----------------------------------------------------
    print("Merging into comparison table...")
    all_ids = set()
    year_data = {}

    for year_label, _ in years:
        out_table = os.path.join(output_folder, f"zone_stats_{year_label}.dbf")
        data = {}
        with arcpy.da.SearchCursor(out_table, ["VALUE", "SUM"]) as cursor:
            for value, sum_px in cursor:
                tid = int(value)
                data[tid] = sum_px
                all_ids.add(tid)
        year_data[year_label] = data

    comparison_csv = os.path.join(output_folder, "tree_canopy_comparison.csv")
    with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["TreeID"] + [f"canopy_px_{y}" for y, _ in years]
        writer.writerow(header)
        for tid in sorted(all_ids):
            row = [tid]
            for y, _ in years:
                row.append(year_data[y].get(tid, 0))
            writer.writerow(row)

    print(f"✅ טבלת השוואה נשמרה: {comparison_csv}")
    print("   כל שורה = עץ אחד (TreeID), כל עמודה = מספר פיקסלי-צמרת באותה שנה.")

except Exception as e:
    print(f"❌ Error during processing: {e}")

finally:
    arcpy.CheckInExtension("Spatial")
    arcpy.env.addOutputsToMap = True
