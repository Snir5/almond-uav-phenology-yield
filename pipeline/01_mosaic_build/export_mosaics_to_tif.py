# ==========================================
# export_mosaics_to_tif.py
#
# מטרה: לייצא (CopyRaster) את ה-mosaic datasets שכבר נבנו
# (Mosaic_4Band, Mosaic_RGB ב-Mosaics.gdb, ע"י build_mosaics_combined.py)
# לקבצי GeoTIFF סטנדרטיים, במבנה כמו 2021:
#   4band_mosaic/Final_Exports_<year>_<date>/Final_Orthomosaic_4Band.tif
#   4band_mosaic/Final_Exports_<year>_<date>/Final_Orthomosaic_RGB.tif
#
# כדי שצנרת ה-EBI (apply_fixed_zones_yearly.py, plot_ebi_heatmaps.py וכו')
# תוכל לקרוא את המוזאיקה החדשה.
#
# הרץ אחרי שבניית ה-mosaic datasets הסתיימה בהצלחה.
# ==========================================

import arcpy
import os

# ==========================================
# CONFIGURATION - עדכן לפי הצורך
# ==========================================
gdb_path          = r"C:\Users\snirt\01_03_23\Mosaics\Mosaics.gdb"
mosaic_4band_name = "Mosaic_4Band"
mosaic_rgb_name   = "Mosaic_RGB"

# תיקיית ייצוא סופית - מבנה כמו 2021
# (.../4band_mosaic/Final_Exports_<year>_<date>/Final_Orthomosaic_*.tif)
EXPORT_DATE_SUFFIX = "2024_XX_XX"  # לדוגמה "2024_02_29" כמו בשנים קודמות
final_export_folder = os.path.join(
    r"C:\Users\snirt\Thesis\4band_mosaic", f"Final_Exports_{EXPORT_DATE_SUFFIX}"
)
final_4band_tif = os.path.join(final_export_folder, "Final_Orthomosaic_4Band.tif")
final_rgb_tif   = os.path.join(final_export_folder, "Final_Orthomosaic_RGB.tif")

arcpy.env.overwriteOutput = True
os.makedirs(final_export_folder, exist_ok=True)


# ==========================================
# EXPORT: mosaic dataset -> GeoTIFF סטנדרטי (CopyRaster)
# ==========================================
def export_mosaic_to_tif(mosaic_path, out_tif, pixel_type="8_BIT_UNSIGNED"):
    if not arcpy.Exists(mosaic_path):
        print(f"  ❌ Mosaic dataset not found: {mosaic_path}")
        return None
    print(f"  Exporting {mosaic_path} -> {out_tif} ...")
    try:
        arcpy.management.CopyRaster(
            in_raster=mosaic_path,
            out_rasterdataset=out_tif,
            pixel_type=pixel_type,
            format="TIFF"
        )
        print(f"  ✅ Exported: {out_tif}\n")
        return out_tif
    except Exception as e:
        print(f"  ❌ Export failed for {mosaic_path}: {e}\n")
        return None


# ==========================================
# EXECUTION
# ==========================================
path_4band = os.path.join(gdb_path, mosaic_4band_name)
path_rgb   = os.path.join(gdb_path, mosaic_rgb_name)

print("Exporting mosaic datasets to final GeoTIFFs...")
exported_4band = export_mosaic_to_tif(path_4band, final_4band_tif, pixel_type="8_BIT_UNSIGNED")
exported_rgb   = export_mosaic_to_tif(path_rgb, final_rgb_tif, pixel_type="8_BIT_UNSIGNED")

print(f"{'='*50}")
print(f"DONE")
print(f"  4-Band GeoTIFF: {exported_4band}")
print(f"  RGB GeoTIFF:    {exported_rgb}")
print(f"{'='*50}")
