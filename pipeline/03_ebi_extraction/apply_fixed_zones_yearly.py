# ============================================================
# apply_fixed_zones_yearly.py
#
# מטרה: להחיל את החלוקה הקבועה לעצים (Tree_Zones_Master.tif, שנוצרה
# מתוך מוזאיקת 2021 ע"י tree_separation_grid.py) על מוזאיקות 4-band
# של שנים נוספות (2022, 2023, 2024 וכו'), ולחשב לכל "עץ" (TreeID קבוע)
# את אינדקסי הצבע EBI / NGRDI (גולמי, מנורמל 0-1, ו-Z-score) בכל שנה.
#
# שינוי מהותי מהסקריפט המקורי (Master_Trees_Time_Series):
#  - אין יותר scipy.ndimage.label() לכל שנה (מספר עצים/IDs שמשתנה משנה לשנה)
#  - אין יותר התאמה מרחבית עם cKDTree בין שנים
#  - כל פיקסל-צמרת מקבל TreeID ישירות מ-Tree_Zones_Master (קבוע!),
#    ולכן "Tree_ID" באקסל הוא אותו עץ בדיוק בכל השנים -> השוואה ישירה.
#
# הגבלה לאזור-עניין (חלקה 1):
#  - כל קריאה מהדיסק (גם Tree_Zones_Master וגם כל מוזאיקת-שנה) נגזרת
#    (clip) לפוליגון PLOT1_COORDS לפני שנכנסת לזיכרון - rasterio.mask
#    קורא רק את החלון (window) שמכיל את הפוליגון, ומאפס פיקסלים
#    שמחוץ לפוליגון עצמו (לא רק מחוץ לתיבת-הגבול שלו).
#
# פורמט קלט נדרש לכל מוזאיקה: GeoTIFF 4-band, band1-3=RGB, band4=מסכת
# צמחייה (0=רקע/לא-צמחייה, כל ערך אחר=עץ/צמרת). זה כולל את
# Final_Orthomosaic_4Band.tif המקורי של כל שנה (גם 2021).
# ============================================================

import rasterio
from rasterio.warp import reproject, Resampling, transform as transform_coords
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine
from shapely.geometry import Polygon
import numpy as np
from scipy.ndimage import center_of_mass
import pandas as pd
import os
import gc

# ============================================================
# CONFIGURATION - לעדכן בהתאם לנתיבים שלכם
# ============================================================

# הראסטר הקבוע (TreeID לכל פיקסל), נוצר ע"י tree_separation_grid.py
# מתוך מוזאיקת 2021 (Tree_Zones_Master.tif)
zones_master_path = "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2021_03_07/Tree_Zones_Master.tif"

# מוזאיקות 4-band לכל שנה (band1-3=RGB, band4=מסכת צמחייה בינארית 0/1)
# 2021 = המוזאיקה המקורית (Final_Orthomosaic_4Band.tif), עם מסכת הצמחייה
# המקורית (לא ה"מופרדת") - כך ההגדרה של "פיקסל צמרת" עקבית עם שאר השנים
mosaics = {
    "2021": "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2021_03_07/Final_Orthomosaic_4Band.tif",
    "2022": "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2022_03_02/Final_Orthomosaic_4Band.tif",
    "2023": "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2023_03_01/Final_Orthomosaic_4Band.tif",
    "2024": "/Users/snirtahasa/Thesis/4band_mosaic/Final_Exports_2024_02_29/Final_Orthomosaic_4Band.tif",
}

# שנת ייחוס לנרמול רדיומטרי (תיקון לינארי לכל ערוץ) ולסטטיסטיקות ה-baseline.
# כל שאר השנים מתוקנות (shift+scale per channel) כך שה-RGB שלהן יתאים
# לשנה הזו, וה-Z-score/Min-Max של EBI/NGRDI מחושבים יחסית לסטטיסטיקות שלה.
reference_year = "2024"

output_folder = "/Users/snirtahasa/Thesis/4band_mosaic"
master_excel = os.path.join(output_folder, "Master_Trees_Time_Series_Plot1.xlsx")
stats_path = os.path.join(output_folder, f"baseline_{reference_year}_plot1_stats.csv")

# --- FIX: epsilon ל-EBI/NGRDI מותאם לטווח ה-DN (0-255) ---
# epsilon=1e-6 (הישן) היה קטן מכדי לייצב את (B+eps) ו-(R-B+eps) במכנה
# של EBI: כאשר R≈B (פיקסלים אפרפרים/בהירים), (R-B+1e-6) הופך לזניח, וה-EBI
# "מתפוצץ" לערכים עצומים (במקרה הקיצוני - מיליוני/מיליארדי). זה זוהה
# ב-diagnose_ebi_deep_check.py: 2023 קיבל raw_ebi עם mean=5,534,695(!),
# ו-2021 קיבל std גבוה פי ~4-8 מ-2022/2024 - שתי תוצאות של אותה
# אי-יציבות נומרית, לא הבדל ביולוגי אמיתי.
#
# הפתרון (כמו במאמר Chen et al. 2019, Eq. 1): EBI_EPS=256 לנתוני DN
# גולמיים בטווח 0-255 - זה מבטיח ש-(B+EBI_EPS) ו-(R-B+EBI_EPS) נשארים
# בטווח [1,511] לעולם, כך ש-EBI נשאר חסום (~0-3) ולא מתפוצץ.
EBI_EPS = 256.0

# epsilon "טכני" - guard נגד חלוקה ב-0 כש-this_std==0 (לא קשור לנוסחת
# EBI/NGRDI עצמה - לכן נשאר קטן מאוד)
epsilon = 1e-6

# --- FIX (memory): percentile מתוך sample אקראי ---
# np.percentile/np.nanpercentile יוצר פנימית עותק-מיון של המערך כולו.
# למערך של מאות מיליוני פיקסלים זה יכול לתפוס כמה ג'יגה-בייט נוספים
# ולגרום ל-OOM. sample אקראי של כמה מיליוני פיקסלים נותן הערכת אחוזון
# כמעט זהה, בעלות זיכרון זניחה.
def sample_percentile(arr, q, max_samples=5_000_000):
    if arr.size > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(arr.size, size=max_samples, replace=False)
        return np.percentile(arr[idx], q)
    return np.percentile(arr, q)

# ============================================================
# פוליגון חלקה 1 - פינות ב-UTM Zone 36N (EPSG:32636), בסדר היקפי.
#
# ** פריים ייחוס: 2022/2023/2024 (הפריים ה"אמיתי") **
# הקואורדינטות המקוריות היו בפריים 2021 (מוזז). תוקנו בתוספת
# DX=+65.60m, DY=+42.80m (ממוצע ה-offsets של 2022/2023/2024)
# כדי שהפוליגון יישב בדיוק על גבולות החלקה במוזאיקות 2022-2024.
# ============================================================
PLOT1_COORDS = [
    (669846.69, 3509814.25),   # NW — נמדד ישירות במוזאיק 2023 (פריים אמיתי)
    (670075.00, 3509755.50),   # NE — נמדד ישירות במוזאיק 2023
    (669973.83, 3509452.97),   # SE — נמדד ישירות במוזאיק 2023
    (669791.15, 3509556.32),   # SW — נמדד ישירות במוזאיק 2023
    # הערה: פינת SE כמעט זהה לפוליגון הישן (הפרש <2m).
    # פינות NW/NE שונות ב-~23m דרומה מהפוליגון שנגזר מ-2021+הזזה אחידה,
    # כי ל-2021 יש עיוות גיאומטרי (לא רק הזזה פשוטה).
]
plot1_polygon = Polygon(PLOT1_COORDS)


# ============================================================
# תיקון הסטה גיאוגרפית (georeferencing offset) בין שנים
#
# ** שינוי פריים ייחוס **
# הפריים הקודם היה 2021 (ZONES_OFFSET_M["2021"]=(0,0)) אבל 2021
# הוא המוזז — 2022/2023/2024 יושבים בדיוק אחד על השני (פריים
# ה"אמיתי"). לכן שינינו את פריים הייחוס:
#   - PLOT1_COORDS עכשיו בפריים האמיתי (2022/2023/2024)
#   - POLYGON_OFFSET מגדיר כמה להזיז את הפוליגון לפריים של כל שנה
#   - ZONES_OFFSET_M מגדיר כמה להזיז את zones_master (שנשאר בפריים
#     2021) כדי ליישר אותו עם המוזאיקה של כל שנה
#
# ערכי ZONES_OFFSET_M (cross-correlation, מ'):
#   2022: dx=+65.20, dy=+42.65
#   2023: dx=+66.15, dy=+43.05
#   2024: dx=+65.45, dy=+42.70
# ============================================================

# כמה להזיז את PLOT1_COORDS (פריים אמיתי) לקבלת הפוליגון במסגרת
# הקואורדינטות של כל מוזאיקה
POLYGON_OFFSET = {
    # X: 2021-frame X = true-frame X - 66.15  → dx = -66.15
    # Y: 2021 raster Y is empirically already in true frame (no Y correction needed)
    #    → dy = 0.0  (using -43.05 incorrectly shifted the clip polygon 43m south,
    #      causing rows 77-79 to be clipped out of the master)
    "2021": (-66.15, 0.0),
    "2022": (0.0, 0.0),
    "2023": (0.0, 0.0),
    "2024": (0.0, 0.0),
}

# כמה להזיז את zones_master (תמיד בפריים 2021) כדי ליישר אותו
# עם מסגרת-הקואורדינטות של מוזאיקת כל שנה
ZONES_OFFSET_M = {
    "2021": (0.0, 0.0),
    "2022": (65.20, 42.65),
    "2023": (66.15, 43.05),
    "2024": (65.45, 42.70),
}


# ============================================================
# עזר: קריאת ראסטר עם גזירה (clip) לפוליגון חלקה 1.
# rasterio.mask קורא רק את ה-window שמכיל את הפוליגון (לא את כל
# הראסטר), ומאפס (fill=0) פיקסלים שמחוץ לפוליגון עצמו.
# ============================================================
def clip_raster(path, polygon_offset=(0.0, 0.0)):
    """גוזר ראסטר לפוליגון חלקה 1.
    polygon_offset: כמה להזיז את PLOT1_COORDS (פריים אמיתי) לקבלת
    הפוליגון במסגרת-הקואורדינטות של הראסטר המבוקש.
    """
    off_x, off_y = polygon_offset
    if off_x == 0.0 and off_y == 0.0:
        poly = plot1_polygon
    else:
        poly = Polygon([(x + off_x, y + off_y) for x, y in PLOT1_COORDS])
    with rasterio.open(path) as src:
        out_image, out_transform = rio_mask(
            src, [poly], crop=True, all_touched=True, filled=True
        )
        crs = src.crs
        nodatavals = src.nodatavals
    return out_image, out_transform, crs, nodatavals


# ============================================================
# שלב 0: גזירת Tree_Zones_Master לחלקה 1 + טבלת בסיס
# (Tree_ID, מרכז כל אזור-עץ - Lat/Lon, X/Y), קבוע וזהה לכל השנים
# ============================================================
print("Clipping Tree_Zones_Master to Plot 1...")
# zones_master נוצר מפריים 2021 -> גוזרים עם הפוליגון בפריים 2021
zm_image, zm_transform, zm_crs, zm_nodatavals = clip_raster(zones_master_path,
                                                              polygon_offset=POLYGON_OFFSET["2021"])
zones_master_arr = zm_image[0]
zm_nodata = zm_nodatavals[0] if zm_nodatavals and zm_nodatavals[0] is not None else 0

valid_zone_mask = zones_master_arr > 0
if zm_nodata != 0:
    valid_zone_mask &= (zones_master_arr != zm_nodata)

tree_ids = np.unique(zones_master_arr[valid_zone_mask])
tree_ids = tree_ids[tree_ids > 0]
print(f"  -> {len(tree_ids)} tree zones inside Plot 1.")

weights = np.ones(zones_master_arr.shape, dtype=np.float32)
centers = center_of_mass(weights, labels=zones_master_arr, index=tree_ids)

base_rows = []
for tid, (row, col) in zip(tree_ids, centers):
    x_utm, y_utm = rasterio.transform.xy(zm_transform, row, col)
    lon, lat = transform_coords(zm_crs, "EPSG:4326", [x_utm], [y_utm])
    # zones_master centroid is in 2021 frame; X_true = X_2021 + 66.15m
    # (based on ZONES_OFFSET_M["2023"] dx = 66.15, the 2023-reference offset)
    # Y is already in the true frame — confirmed empirically (no Y correction)
    x_utm_true = x_utm + 66.15
    base_rows.append({
        "Tree_ID": f"Tree_{int(tid):04d}",
        "Zone_Value": int(tid),
        "X_UTM": x_utm_true,
        "Y_UTM": y_utm,
        "Latitude": lat[0],
        "Longitude": lon[0],
    })

df_master = pd.DataFrame(base_rows)
print(f"  -> {len(df_master)} tree zones found.")

del weights, valid_zone_mask
gc.collect()


# ============================================================
# פונקציה: עיבוד שנה אחת -> EBI/NGRDI גולמיים לכל פיקסל-צמרת + ה-TreeID שלו
# (הכל מוגבל לתיבת-הגבול + פוליגון של חלקה 1)
# ============================================================
def compute_year_raw_metrics(mosaic_path, year, ref_stats=None):
    # גזירת המוזאיקה לפי הפוליגון במסגרת-הקואורדינטות של אותה שנה
    poly_off = POLYGON_OFFSET.get(year, (0.0, 0.0))
    img, dst_transform, dst_crs, nodatavals = clip_raster(mosaic_path, polygon_offset=poly_off)

    mask = img[3]
    mask_nodata = nodatavals[3] if len(nodatavals) > 3 else None

    # הזזת zones_master (פריים 2021) אל מסגרת-הקואורדינטות של מוזאיקת השנה
    zm_off_x, zm_off_y = ZONES_OFFSET_M.get(year, (0.0, 0.0))
    zm_transform_corrected = Affine.translation(zm_off_x, zm_off_y) * zm_transform

    dst_shape = mask.shape
    zones_dst = np.zeros(dst_shape, dtype=np.int32)
    reproject(
        source=zones_master_arr,
        destination=zones_dst,
        src_transform=zm_transform_corrected,
        src_crs=zm_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=zm_nodata,
        dst_nodata=0,
        resampling=Resampling.nearest,  # קריטי! ה-IDs קטגוריאליים
    )

    # פיקסל נחשב "צמרת" אם band4 != 0 (וגם לא NoData) - תואם להגדרה
    # המקורית "0=לא-צמחייה, כל ערך אחר=צמחייה" - וגם נמצא בתוך אזור-עץ קבוע
    # (זה כולל אוטומטית גם את ההגבלה לפוליגון חלקה 1, כי zones_dst==0
    # מחוץ לפוליגון)
    canopy = (mask != 0) & (zones_dst > 0)
    if mask_nodata is not None:
        canopy &= (mask != mask_nodata)

    zones_c = zones_dst[canopy]

    # --- FIX (memory): לחלץ פיקסלי-צמרת לפני המרה ל-float32 ---
    # ממירים ל-float32 רק את תת-הקבוצה של פיקסלי-הצמרת (canopy), לא את כל
    # המלבן של חלקה 1. כך נחסכים כמה עותקים בגודל מלא (R,G,B float32 על כל
    # הראסטר הגזור) שיכולים לתפוס כמה ג'יגה-בייט ולהביא ל-OOM ("Killed").
    R_c = img[0][canopy].astype(np.float32)
    G_c = img[1][canopy].astype(np.float32)
    B_c = img[2][canopy].astype(np.float32)

    del img, mask, zones_dst, canopy
    gc.collect()

    # --- FIX: תיקון רדיומטרי לינארי (affine) לכל ערוץ בנפרד, במקום
    # Histogram Matching לא-לינארי ---
    # Histogram Matching (לכל ערוץ R,G,B בנפרד) מיישר את כל ה-CDF של הערוץ,
    # אבל בעקומה לא-לינארית שונה לכל ערוץ. EBI תלוי קריטית ב-(R-B) וב-G/B,
    # ועקומות-תיקון לא-לינאריות שונות יכולות "לבלבל" את הסדר היחסי בין
    # R ל-B מפיקסל לפיקסל - וזו הסיבה שה-EBI לא תאם את התמונה בשנים
    # שה-RGB הגולמי שלהן שונה הרבה מ-2021 (במיוחד 2024).
    #
    # התיקון החדש: shift+scale לינארי לכל ערוץ (מיישר ממוצע וסטיית-תקן
    # של הערוץ בשנה הנוכחית לאלו של אותו ערוץ בשנת הייחוס). זה תיקון
    # מונוטוני (לא משנה את הסדר היחסי של ערכים בתוך אותו ערוץ), אבל בגלל
    # ש-aR/aG/aB (יחסי הסקייל בין הערוצים) אינם זהים, EBI כן "מגיב" לתיקון -
    # בניגוד לתיקון בהירות גלובלי אחיד (gray-world) שהוא NO-OP מתמטי ל-EBI.
    this_mean_R, this_std_R = float(R_c.mean()), float(R_c.std())
    this_mean_G, this_std_G = float(G_c.mean()), float(G_c.std())
    this_mean_B, this_std_B = float(B_c.mean()), float(B_c.std())

    if ref_stats is not None:
        (ref_mean_R, ref_std_R, ref_mean_G, ref_std_G, ref_mean_B, ref_std_B) = ref_stats

        # --- FIX: shared scale factor across R/G/B (instead of per-channel) ---
        # Using a separate scale per channel (scale = ref_std/this_std for each
        # of R,G,B independently) makes scale_R != scale_B whenever the channels'
        # std ratios differ. Since EBI depends critically on (R-B), this injects
        # a brightness-dependent, year-specific bias even for color-neutral
        # (R==B) pixels - confirmed empirically (diagnose_ebi_deep_check.py PART 3,
        # 2022 was worst affected). Using ONE shared scale (avg of the three
        # per-channel std ratios) still corrects each channel's mean/offset and
        # overall contrast to match the reference year, but keeps (R-B) for a
        # gray pixel close to 0 in every year - removing that EBI artifact.
        this_std_avg = (this_std_R + this_std_G + this_std_B) / 3.0
        ref_std_avg = (ref_std_R + ref_std_G + ref_std_B) / 3.0
        shared_scale = ref_std_avg / max(this_std_avg, epsilon)

        def affine(values, this_mean, ref_mean, scale):
            return (values - this_mean) * scale + ref_mean

        R_c = affine(R_c, this_mean_R, ref_mean_R, shared_scale)
        G_c = affine(G_c, this_mean_G, ref_mean_G, shared_scale)
        B_c = affine(B_c, this_mean_B, ref_mean_B, shared_scale)

        # אחרי shift+scale ייתכנו ערכים שליליים קלים (פיקסלים קיצוניים) -
        # נחתכים ל-0 כדי לשמור על stability של (B+epsilon) ב-EBI/NGRDI
        np.clip(R_c, 0, None, out=R_c)
        np.clip(G_c, 0, None, out=G_c)
        np.clip(B_c, 0, None, out=B_c)

        this_stats = None
    else:
        # שנת הייחוס - מחזירים את הסטטיסטיקות שלה (mean/std לכל ערוץ)
        # כדי שכל שאר השנים יתוקנו אליה
        this_stats = (this_mean_R, this_std_R, this_mean_G, this_std_G, this_mean_B, this_std_B)

    ebi_num = R_c + G_c + B_c
    ebi_den = (G_c / (B_c + EBI_EPS)) * (R_c - B_c + EBI_EPS)
    raw_ebi = np.divide(ebi_num, ebi_den, out=np.zeros_like(ebi_num), where=ebi_den != 0)

    # --- FIX: winsorize EBI ---
    # ebi_den מתאפס (R_c ≈ B_c) על פיקסלים אפורים/בצל, מה שגורם ל-raw_ebi
    # "להתפוצץ" לערכים של מיליונים ואף עשרות מיליונים. פיקסל בודד כזה
    # מספיק כדי להשתלט על הממוצע של כל אזור-עץ (ולכן גם על mean_ebi_ref/
    # std_ebi_ref ועל ה-Z-score וה-Norm). פותרים ע"י "winsorizing" - קיצוץ
    # כל ערך שמעבר לאחוזון 1-99 של אותה שנה לגבולות האחוזון עצמם.
    ebi_lo = sample_percentile(raw_ebi, 1)
    ebi_hi = sample_percentile(raw_ebi, 99)
    raw_ebi = np.clip(raw_ebi, ebi_lo, ebi_hi)

    ngrdi_num = G_c - R_c
    ngrdi_den = G_c + R_c
    raw_ngrdi = np.divide(ngrdi_num, ngrdi_den, out=np.zeros_like(ngrdi_num), where=ngrdi_den != 0)

    del R_c, G_c, B_c, ebi_num, ebi_den, ngrdi_num, ngrdi_den
    gc.collect()

    return zones_c, raw_ebi, raw_ngrdi, this_stats


# ============================================================
# שלב 1: שנת הייחוס - חישוב וטעינת/שמירת סטטיסטיקות baseline
# ============================================================
print(f"--- Processing reference year: {reference_year} (Plot 1 only) ---")
zones_ref, raw_ebi_ref, raw_ngrdi_ref, ref_stats = compute_year_raw_metrics(
    mosaics[reference_year], reference_year, ref_stats=None
)

mean_ebi_ref = np.mean(raw_ebi_ref)
std_ebi_ref = np.std(raw_ebi_ref)
min_ebi_ref = sample_percentile(raw_ebi_ref, 1)
max_ebi_ref = sample_percentile(raw_ebi_ref, 99)

mean_ngrdi_ref = np.mean(raw_ngrdi_ref)
std_ngrdi_ref = np.std(raw_ngrdi_ref)
min_ngrdi_ref = sample_percentile(raw_ngrdi_ref, 1)
max_ngrdi_ref = sample_percentile(raw_ngrdi_ref, 99)

pd.DataFrame({
    "Metric": ["mean_ebi", "std_ebi", "min_ebi", "max_ebi",
               "mean_ngrdi", "std_ngrdi", "min_ngrdi", "max_ngrdi"],
    "Value": [mean_ebi_ref, std_ebi_ref, min_ebi_ref, max_ebi_ref,
              mean_ngrdi_ref, std_ngrdi_ref, min_ngrdi_ref, max_ngrdi_ref],
}).to_csv(stats_path, index=False)
print(f"  -> Baseline stats saved: {stats_path}")

range_ebi = (max_ebi_ref - min_ebi_ref) if (max_ebi_ref - min_ebi_ref) != 0 else 1e-6
range_ngrdi = (max_ngrdi_ref - min_ngrdi_ref) if (max_ngrdi_ref - min_ngrdi_ref) != 0 else 1e-6


# ============================================================
# שלב 2: עיבוד כל השנים (כולל שנת הייחוס) - אגרגציה לפי TreeID קבוע
# ============================================================
def aggregate_to_master(year, zones_c, raw_ebi, raw_ngrdi):
    z_score_ebi = (raw_ebi - mean_ebi_ref) / std_ebi_ref
    z_score_ngrdi = (raw_ngrdi - mean_ngrdi_ref) / std_ngrdi_ref

    norm_ebi = np.clip((raw_ebi - min_ebi_ref) / range_ebi, 0, 1)
    norm_ngrdi = np.clip((raw_ngrdi - min_ngrdi_ref) / range_ngrdi, 0, 1)

    # --- FIX (memory): אגרגציה לפי TreeID עם np.bincount במקום
    # pandas DataFrame+groupby ---
    # groupby על DataFrame בגודל "מספר פיקסלי-צמרת" (יכול להיות מאות
    # מיליוני שורות) הוא יקר מאוד בזיכרון. bincount נותן תוצאה זהה
    # (ממוצע פר-TreeID), אבל הפלט שלו בגודל "מספר עצים" (כמה אלפים) בלבד.
    n_bins = int(zones_c.max()) + 1
    counts = np.bincount(zones_c, minlength=n_bins).astype(np.float64)
    valid = counts > 0

    def zone_mean(values):
        sums = np.bincount(zones_c, weights=values, minlength=n_bins)
        out = np.full(n_bins, np.nan)
        out[valid] = sums[valid] / counts[valid]
        return out

    zone_values = np.where(valid)[0]
    df_year = pd.DataFrame({"Zone_Value": zone_values})
    df_year[f"EBI_Raw_{year}"] = np.round(zone_mean(raw_ebi)[valid], 4)
    df_year[f"NGRDI_Raw_{year}"] = np.round(zone_mean(raw_ngrdi)[valid], 4)
    df_year[f"EBI_Norm_{year}"] = np.round(zone_mean(norm_ebi)[valid], 4)
    df_year[f"NGRDI_Norm_{year}"] = np.round(zone_mean(norm_ngrdi)[valid], 4)
    df_year[f"EBI_Z_{year}"] = np.round(zone_mean(z_score_ebi)[valid], 4)
    df_year[f"NGRDI_Z_{year}"] = np.round(zone_mean(z_score_ngrdi)[valid], 4)
    return df_year


df_year = aggregate_to_master(reference_year, zones_ref, raw_ebi_ref, raw_ngrdi_ref)
df_master = pd.merge(df_master, df_year, on="Zone_Value", how="left")
del zones_ref, raw_ebi_ref, raw_ngrdi_ref, df_year
gc.collect()

for year, mosaic_path in mosaics.items():
    if year == reference_year:
        continue
    print(f"--- Processing {year} (Plot 1 only) ---")
    zones_c, raw_ebi, raw_ngrdi, _ = compute_year_raw_metrics(mosaic_path, year, ref_stats=ref_stats)
    df_year = aggregate_to_master(year, zones_c, raw_ebi, raw_ngrdi)
    df_master = pd.merge(df_master, df_year, on="Zone_Value", how="left")
    del zones_c, raw_ebi, raw_ngrdi, df_year
    gc.collect()

# ============================================================
# שלב 3: שמירה
# ============================================================
df_master.to_excel(master_excel, index=False)
print("=" * 50)
print("SUCCESS!")
print(f"Saved to: {master_excel}")
print(f"{len(df_master)} trees x {len(mosaics)} years (Plot 1 only).")
print("=" * 50)
