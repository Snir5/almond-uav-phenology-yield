# ==========================================
# build_mosaics_combined.py
#
# שילוב של שני הסקריפטים:
#  - מ-code 2: סינון חפיפות מרחבי (spatial_thinning) - מסיר תמונות
#    שצולמו קרוב מאוד אחת לשנייה (overlap גבוה).
#  - מ-code 1: הגדרות בנייה שהפיקו את הפסיפס באיכות גבוהה ל-2021 -
#    BuildFootprints בלי shrink_distance, default_mosaic_method=SEAMLINE.
#  - מ-code 2 (רק עבור RGB): CalculateStatistics + BuildPyramidsAndStatistics
#    בסוף, לתיקון שקיפות בתצוגה.
#
# הערה: ייצוא ה-mosaic datasets לקבצי GeoTIFF סופיים (Final_Orthomosaic_*.tif)
# נמצא בסקריפט נפרד: export_mosaics_to_tif.py - הרץ אותו אחרי הסקריפט הזה.
#
# שימוש: הרץ על תיקיית ה-output (output_folder) של השנה הרצויה
# (למשל 2024 ואילך) - עדכן את הנתיבים למטה בהתאם.
# ==========================================

import arcpy
import os
import shutil
import math

# ==========================================
# CONFIGURATION
# ==========================================
output_folder     = r"C:\Users\snirt\01_03_23\Output_Work_Final"
mosaic_folder     = r"C:\Users\snirt\01_03_23\Mosaics"
gdb_path          = r"C:\Users\snirt\01_03_23\Mosaics\Mosaics.gdb"
mosaic_4band_name = "Mosaic_4Band"
mosaic_rgb_name   = "Mosaic_RGB"
spatial_ref       = arcpy.SpatialReference(32636)  # WGS 1984 UTM Zone 36N

# מרחק דילול חפיפות במטרים (מ-code 2). שנה ל-20.0 אם עדיין צפוף מדי.
THINNING_DISTANCE_METERS = 15.0

arcpy.env.overwriteOutput = True

os.makedirs(mosaic_folder, exist_ok=True)

# Clean temp folders (מ-code 1)
for temp_dir in [os.path.join(mosaic_folder, "temp_rgb"),
                  os.path.join(mosaic_folder, "temp_nir")]:
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Deleted temp folder: {temp_dir}")

if not arcpy.Exists(gdb_path):
    arcpy.CreateFileGDB_management(mosaic_folder, "Mosaics.gdb")
    print(f"Created GDB: {gdb_path}\n")


# ==========================================
# SPATIAL THINNING (דילול חפיפות חכם) - מ-code 2
# ==========================================
def spatial_thinning(folder, min_distance_meters=15.0):
    print(f"Scanning '{folder}' for spatial thinning (min distance = {min_distance_meters} m)...")
    image_list = sorted([f for f in os.listdir(folder) if f.endswith(".tif")])

    thinned_images = []
    accepted_centers = []

    for img_name in image_list:
        path = os.path.join(folder, img_name)
        try:
            # שליפת מרכז התמונה במטרים
            ext = arcpy.Describe(path).extent
            center_x = (ext.XMax + ext.XMin) / 2.0
            center_y = (ext.YMax + ext.YMin) / 2.0

            # בדיקת מרחק מול תמונות שכבר אושרו
            is_too_close = False
            for (acc_x, acc_y) in accepted_centers:
                distance = math.sqrt((center_x - acc_x) ** 2 + (center_y - acc_y) ** 2)
                if distance < min_distance_meters:
                    is_too_close = True
                    break

            if not is_too_close:
                accepted_centers.append((center_x, center_y))
                thinned_images.append(img_name)
        except Exception as e:
            print(f"  ⚠️ Skipping '{img_name}' (could not read extent: {e})")

    print(f"  > Total images found in folder: {len(image_list)}")
    print(f"  > Images kept after thinning:   {len(thinned_images)}")
    print(f"  > Discarded due to overlap:      {len(image_list) - len(thinned_images)}\n")
    return thinned_images


# ==========================================
# HELPER: ADD RASTERS IN BATCHES (מ-code 1)
# ==========================================
def add_rasters_in_batches(mosaic_path, full_paths, batch_size=200):
    if not full_paths:
        print("  ⚠️ Warning: No images found to add.")
        return False
    for i in range(0, len(full_paths), batch_size):
        batch = full_paths[i:i + batch_size]
        arcpy.AddRastersToMosaicDataset_management(
            in_mosaic_dataset=mosaic_path, raster_type="Raster Dataset",
            input_path=";".join(batch), update_cellsize_ranges="UPDATE_CELL_SIZES",
            update_boundary="NO_BOUNDARY", update_overviews="NO_OVERVIEWS",
            duplicate_items_action="ALLOW_DUPLICATES"
        )
        print(f"  Batch {i // batch_size + 1}: {len(batch)} images added")

    try:
        count = int(arcpy.management.GetCount(mosaic_path)[0])
        print(f"  ✅ Total items sitting inside mosaic: {count}")
        return count > 0
    except Exception:
        return False


# ==========================================
# BUILD 4-BAND MOSAIC (הגדרות code 1 - SEAMLINE, בלי shrink_distance)
# ==========================================
def build_4band_mosaic(mosaic_name, image_list, spatial_ref, gdb_path):
    mosaic_path = os.path.join(gdb_path, mosaic_name)
    if arcpy.Exists(mosaic_path):
        arcpy.Delete_management(mosaic_path)
        print(f"Deleted existing: {mosaic_name}")

    print(f"Creating {mosaic_name} - 4 bands TIF ({len(image_list)} images)...")
    arcpy.CreateMosaicDataset_management(
        in_workspace=gdb_path, in_mosaicdataset_name=mosaic_name,
        coordinate_system=spatial_ref, num_bands=4, pixel_type="8_BIT_UNSIGNED"
    )

    if not add_rasters_in_batches(mosaic_path, [os.path.join(output_folder, f) for f in image_list]):
        print(f"  ⚠️ No images added to {mosaic_name}, skipping rest of build.\n")
        return mosaic_path

    print(f"  Rebuilding Footprints to fix geometry errors...")
    arcpy.BuildFootprints_management(
        in_mosaic_dataset=mosaic_path,
        reset_footprint="GEOMETRY",
        maintain_edges="NO_MAINTAIN_EDGES"
    )

    print(f"  Building boundary...")
    arcpy.BuildBoundary_management(mosaic_path)

    print(f"  Building smart Seamlines (Voronoi) to remove distorted edges...")
    arcpy.BuildSeamlines_management(
        in_mosaic_dataset=mosaic_path,
        computation_method="VORONOI"
    )

    arcpy.SetMosaicDatasetProperties_management(
        in_mosaic_dataset=mosaic_path,
        default_mosaic_method="SEAMLINE",
        mosaic_operator="BLEND",
        blend_width=10
    )

    print(f"  Building overviews...")
    arcpy.BuildOverviews_management(
        in_mosaic_dataset=mosaic_path,
        define_missing_tiles="DEFINE_MISSING_TILES",
        generate_overviews="GENERATE_OVERVIEWS"
    )
    print(f"  Done: {mosaic_name}\n")
    return mosaic_path


# ==========================================
# BUILD RGB MOSAIC (הגדרות code 1 - SEAMLINE, בלי shrink_distance,
# + Statistics/Pyramids מ-code 2 בסוף)
# ==========================================
def build_rgb_jpeg_mosaic(mosaic_name, image_list, spatial_ref, gdb_path):
    mosaic_path = os.path.join(gdb_path, mosaic_name)
    if arcpy.Exists(mosaic_path):
        arcpy.Delete_management(mosaic_path)
        print(f"Deleted existing: {mosaic_name}")

    print(f"Creating {mosaic_name} - 3 bands JPEG ({len(image_list)} images)...")

    # num_bands=3 -> ArcGIS מוריד אוטומטית את הערוץ הרביעי בזמן הייבוא
    arcpy.CreateMosaicDataset_management(
        in_workspace=gdb_path, in_mosaicdataset_name=mosaic_name,
        coordinate_system=spatial_ref, num_bands=3, pixel_type="8_BIT_UNSIGNED"
    )

    if not add_rasters_in_batches(mosaic_path, [os.path.join(output_folder, f) for f in image_list]):
        print(f"  ⚠️ No images added to {mosaic_name}, skipping rest of build.\n")
        return mosaic_path

    print(f"  Rebuilding Footprints to fix geometry errors...")
    arcpy.BuildFootprints_management(
        in_mosaic_dataset=mosaic_path,
        reset_footprint="GEOMETRY",
        maintain_edges="NO_MAINTAIN_EDGES"
    )

    print(f"  Building boundary...")
    arcpy.BuildBoundary_management(mosaic_path)

    print(f"  Building smart Seamlines (Voronoi) to remove distorted edges...")
    arcpy.BuildSeamlines_management(
        in_mosaic_dataset=mosaic_path,
        computation_method="VORONOI"
    )

    arcpy.SetMosaicDatasetProperties_management(
        in_mosaic_dataset=mosaic_path,
        default_mosaic_method="SEAMLINE",
        mosaic_operator="BLEND",
        blend_width=10,
        JPEG_quality=85
    )

    # תיקון שקיפות בתצוגת ה-RGB (מ-code 2): חישוב סטטיסטיקה + פירמידות
    try:
        print("  🔄 Forcing Render: Calculating Statistics and Pyramids for RGB display...")
        arcpy.management.CalculateStatistics(mosaic_path)
        arcpy.management.BuildPyramidsAndStatistics_management(mosaic_path)
    except Exception as e:
        print(f"  ⚠️ Statistics note: {e}")

    print(f"  Done: {mosaic_name}\n")
    return mosaic_path


# ==========================================
# EXECUTION
# ==========================================
final_thinned_images = spatial_thinning(output_folder, THINNING_DISTANCE_METERS)

path_4band = build_4band_mosaic(mosaic_4band_name, final_thinned_images, spatial_ref, gdb_path)
path_rgb   = build_rgb_jpeg_mosaic(mosaic_rgb_name, final_thinned_images, spatial_ref, gdb_path)

print(f"{'='*50}")
print(f"DONE")
print(f"  4-Band mosaic dataset: {path_4band}  ({len(final_thinned_images)} images)")
print(f"  RGB mosaic dataset:    {path_rgb}   ({len(final_thinned_images)} images)")
print(f"{'='*50}")
print("Next step: run export_mosaics_to_tif.py to export these to Final_Orthomosaic_*.tif")
