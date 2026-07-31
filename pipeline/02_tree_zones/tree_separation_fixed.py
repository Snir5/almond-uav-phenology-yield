import arcpy
from arcpy.sa import *
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False

# נתיבים
input_4band      = r"C:\Users\snirt\2021_03_07\Final_Exports\Final_Orthomosaic_4Band.tif"
output_folder    = r"C:\Users\snirt\2021_03_07\Final_Exports"
output_separated = os.path.join(output_folder, "Trees_Final_Separated.tif")            # מסכה בינארית בלבד
output_mosaic_4band = os.path.join(output_folder, "Final_Mosaic_Trees_Separated.tif")   # מוזאיקה 4 שכבות (RGB + מסכה מופרדת)
arcpy.env.scratchWorkspace = output_folder
arcpy.env.compression = "LZW"

# מיישר את כל פעולות הרסטר (cell size / extent / snap) לפי המוזאיקה המקורית,
# כדי שאפשר יהיה לחבר את התוצאה חזרה ל-RGB ללא בעיות התאמה.
arcpy.env.cellSize    = input_4band
arcpy.env.extent      = input_4band
arcpy.env.snapRaster  = input_4band

# ---------------------------------------------------------
# SCALED PARAMETERS
# ---------------------------------------------------------
# רדיוס (בפיקסלים) לחיפוש "מקסימום מקומי" = מרכז עץ.
# צריך להיות קצת קטן מהמרחק בין מרכזי שני עצים שכנים, כדי שלכל עץ יתגלה שיא נפרד.
# לעץ ברוחב ~300px (רדיוס ~150px), נסה להתחיל מ-100 ולכוון לפי התוצאה.
local_max_radius_px = 100

# עומק מינימלי (בפיקסלים, מהרקע) שפיקסל "שיא" צריך להגיע אליו כדי להיחשב מרכז עץ אמיתי
# (מסנן שיאים זעירים שנובעים מרעש בקצוות).
min_peak_depth_px = 3

# גודל מינימלי (בפיקסלים²) לבלוב "עץ" בשלב הבינריזציה — מתחתיו זה רעש שמתבטל לרקע.
# 20 פיקסל זה זעיר מאוד (גרגיר רעש), לא פוגע בעצים אמיתיים (~300px רוחב).
min_tree_blob_px = 20

# גודל "עץ" אמיתי קטן ביותר, לפי קוטר במטרים/פיקסלים — לניקוי סופי של בלובי-עץ
# שנותרו בתוצאה אחרי ה-watershed אבל הם קטנים מכדי להיות עץ ממשי.
# כוון לפי העץ הקטן ביותר שאתה מצפה לראות בשטח (כולל שתילים צעירים).
min_final_tree_diameter_px = 50

print("Starting Full Pipeline (Silent & Safe Mode)...")
try:
    # =========================================================
    # PART 1: BINARIZATION
    # =========================================================
    print("Step 1: Extracting Band 4 and standardizing mask...")
    band4_path = input_4band + r"\Band_4"
    band4 = Raster(band4_path)

    # binary_mask: 1 = רקע/שבילים, 0 = עץ/צמרת
    binary_mask = Con(IsNull(band4) | (band4 == 0), 1, 0)

    # =========================================================
    # PART 1.5: NOISE REMOVAL (תיקון: שומר את ערך הבלוב המקורי)
    # =========================================================
    print(f"Step 1.5: Removing tree-blobs smaller than {min_tree_blob_px} px (noise)...")

    # מקבץ אזורים תוך שמירה על הקשר לערך המקורי (LINK) ולגודל (COUNT)
    all_blobs = RegionGroup(binary_mask, "EIGHT", "WITHIN")
    blob_value = Lookup(all_blobs, "LINK")    # 0 (עץ) או 1 (רקע) — הערך המקורי של הבלוב
    blob_count = Lookup(all_blobs, "COUNT")   # שטח הבלוב בפיקסלים

    # רק בלובי "עץ" (value==0) שהם קטנים מהסף -> הופכים לרקע (1).
    # בלובי רקע נשארים רקע בכל מקרה, ולא נוגעים בהם בשלב הזה.
    binary_mask = Con((blob_value == 0) & (blob_count < min_tree_blob_px), 1, binary_mask)

    # =========================================================
    # PART 2: DISTANCE TRANSFORM — מרחק כל פיקסל-עץ מהרקע הקרוב ביותר
    # =========================================================
    print("Step 2: Computing distance-from-background transform...")
    cell_size = float(arcpy.management.GetRasterProperties(input_4band, "CELLSIZEX").getOutput(0))
    min_peak_depth_map = min_peak_depth_px * cell_size

    background = Con(binary_mask == 1, 1)  # NoData בתוך העצים
    canopy     = Con(binary_mask == 0, 1)  # NoData ברקע

    # מרחק (ביחידות המפה) מכל פיקסל לפיקסל הרקע הקרוב ביותר.
    # בתוך כל "גוש עצים" מאוחד זה יוצר "גבעות" — שיא לכל עץ בנפרד, גם אם אין רקע ביניהם.
    dist_from_bg = EucDistance(background)

    # =========================================================
    # PART 3: LOCAL MAXIMA — מציאת מרכזי העצים (זרעים ל-watershed)
    # =========================================================
    print(f"Step 3: Finding local maxima (tree centers), radius={local_max_radius_px}px...")
    focal_max = FocalStatistics(dist_from_bg, NbrCircle(local_max_radius_px, "CELL"), "MAXIMUM")

    local_maxima = Con(
        (canopy == 1) & (dist_from_bg >= focal_max) & (dist_from_bg > min_peak_depth_map),
        1
    )

    temp_seeds_path = os.path.join(output_folder, "temp_seeds_safe.tif")
    print("        > Saving seed raster to disk quietly...")
    local_maxima_int = Int(local_maxima)
    local_maxima_int.save(temp_seeds_path)

    print("        > Calculating statistics for safety check...")
    try:
        arcpy.management.CalculateStatistics(temp_seeds_path)
        max_val = arcpy.management.GetRasterProperties(temp_seeds_path, "MAXIMUM").getOutput(0)
    except Exception:
        raise ValueError(
            "\n" + "=" * 50 +
            "\n🚨 NO TREE CENTERS FOUND! 🚨\n"
            "No local maxima detected. Try lowering 'local_max_radius_px'\n"
            "or 'min_peak_depth_px'.\n"
            + "=" * 50
        )

    print("Step 4: Assigning unique IDs to each tree center...")
    seed_ids = RegionGroup(temp_seeds_path, "EIGHT", "WITHIN")

    print("Step 5: Allocating each canopy pixel to its nearest tree center...")
    arcpy.env.mask = canopy
    grown_trees = EucAllocation(seed_ids)
    arcpy.env.mask = ""

    print("Step 6: Analysis of intersections and drawing separation lines...")
    borders = FocalStatistics(grown_trees, NbrRectangle(3, 3, "CELL"), "VARIETY")

    # =========================================================
    # PART 4: FINAL ASSEMBLY
    # =========================================================
    print("Step 7: Compiling final raster output...")
    # 0 = פנים העץ, 255 = רקע / קווי הפרדה בין עצים
    final_mask = Con(IsNull(grown_trees), 255,
                      Con(borders > 1, 255, 0))

    # =========================================================
    # PART 4.5: FINAL CLEANUP — הסרת בלובי-"עץ" קטנים מדי מהתוצאה
    # =========================================================
    import math
    min_final_area_px = math.pi * (min_final_tree_diameter_px / 2.0) ** 2
    print(f"Step 7.5: Removing leftover tree-blobs smaller than "
          f"{min_final_area_px:.0f} px² (diameter < {min_final_tree_diameter_px}px)...")

    final_blobs = RegionGroup(final_mask, "EIGHT", "WITHIN")
    final_value = Lookup(final_blobs, "LINK")
    final_count = Lookup(final_blobs, "COUNT")

    final_mask = Con((final_value == 0) & (final_count < min_final_area_px), 255, final_mask)

    final_mask.save(output_separated)
    print(f"✅ Success! Surgically separated mask saved to: {output_separated}")

    # =========================================================
    # PART 5: REBUILD 4-BAND MOSAIC (RGB + מסכת הסגמנטציה המופרדת)
    # =========================================================
    print("Step 8: Compositing final 4-band mosaic (RGB + separated binary mask)...")

    # final_mask כרגע: 0=עץ, 255=רקע/קו הפרדה (לתצוגה ב-output_separated).
    # ל-Band 4 החדש מחזירים לקידוד המקורי: 1=עץ (מופרד), 0=רקע/קו הפרדה.
    final_band4 = Con(final_mask == 0, 1, 0)

    # band 4 חדש חייב להיות קובץ על הדיסק כדי לשמש ב-CompositeBands
    temp_band4_path = os.path.join(output_folder, "temp_band4_separated.tif")
    final_band4.save(temp_band4_path)

    band1_path = input_4band + r"\Band_1"
    band2_path = input_4band + r"\Band_2"
    band3_path = input_4band + r"\Band_3"

    arcpy.management.CompositeBands(
        [band1_path, band2_path, band3_path, temp_band4_path],
        output_mosaic_4band
    )
    print(f"✅ 4-band mosaic saved to: {output_mosaic_4band}")
    print("   Band 1-3 = RGB (original)  |  Band 4 = separated segmentation mask")

    # ניקיון קבצי ביניים
    try:
        arcpy.management.Delete(temp_seeds_path)
        arcpy.management.Delete(temp_band4_path)
    except Exception:
        pass

except Exception as e:
    print(f"❌ Error during processing: {e}")

finally:
    arcpy.CheckInExtension("Spatial")
    arcpy.env.addOutputsToMap = True
