"""
tree_separation_grid.py
========================
הפרדת עצים — שיטה משולבת: גריד + watershed allocation.

הרעיון:
  1. מזהים אוטומטית את הגריד (זווית + מרווח + פאזה) מתוך תבנית הרקע
     (השבילים בין השורות/הטורים) — בדיוק כמו לפני.
  2. ממיר את מרכזי-השבילים (phase) למרכזי-תאים (phase + period/2) —
     אלו הם מיקומי "מרכז עץ" צפויים, לפי הגריד, **לכל תא**, כולל עצים
     שכלואים בין 4 שכנים ולכן אין להם רכס/שיא נפרד ב-distance transform.
  3. בכל הצטלבות של שתי סבילות-המרכז (u ו-v) שמים "זרע" (seed) — אחד
     לכל עץ צפוי, ללא תלות בפיקסלים.
  4. EucAllocation, ממוסך לצמרת (canopy), מקצה לכל פיקסל-עץ את הזרע
     הקרוב אליו — כך שהגבולות מתעצבים לפי צורת הצמרת בפועל (לא קווים
     ישרים), אבל **כל עץ צפוי לפי הגריד מקבל אזור משלו**, גם אם הוא
     מוקף שכנים מכל עבר.

שני מצבי הרצה:
  PREVIEW_ONLY = True   -> מזהה זווית/מרווח/פאזה על תמונה מוקטנת, שומר תצוגה
                           מקדימה (temp_grid_preview.tif) ועוצר.
                           פתח ב-ArcGIS Pro עם סימול קטגוריאלי:
                           0=עץ, 1=רקע, 2=קו גריד (לבדיקת התיישרות),
                           3=זרע (מרכז עץ צפוי).
                           בדוק שקווי הגריד מתיישרים עם השבילים האמיתיים,
                           ושהזרעים (3) נופלים בערך במרכז כל עץ.
  PREVIEW_ONLY = False  -> מריץ את כל הפייפליין המלא: בינריזציה -> ניקוי רעש
                           -> זיהוי גריד -> זריעה לפי מרכזי-תאים ->
                           EucAllocation -> קווי הפרדה -> ניקוי בלובים
                           קטנים -> הרכבת מוזאיקה 4 שכבות.

אם הזיהוי האוטומטי לא נראה נכון בתצוגה המקדימה, אפשר לדרוס אותו עם הערכים
הקבועים ב-MANUAL OVERRIDES למטה (זווית במעלות, מרווחים בפיקסלים, פאזות
בפיקסלים — הכל ביחס לתמונה המוקטנת לפי DOWNSAMPLE_FACTOR).
"""

import arcpy
from arcpy.sa import *
import numpy as np
import os
import math
import json

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False

# ==========================================
# נתיבים
# ==========================================
input_4band         = r"C:\Users\snirt\2021_03_07\Final_Exports\Final_Orthomosaic_4Band.tif"
output_folder       = r"C:\Users\snirt\2021_03_07\Final_Exports"
output_separated    = os.path.join(output_folder, "Trees_Final_Separated.tif")
output_mosaic_4band = os.path.join(output_folder, "Final_Mosaic_Trees_Separated.tif")
output_tree_zones   = os.path.join(output_folder, "Tree_Zones_Master.tif")
output_grid_params  = os.path.join(output_folder, "tree_grid_params.json")

arcpy.env.scratchWorkspace = output_folder
arcpy.env.compression = "LZW"
arcpy.env.cellSize    = input_4band
arcpy.env.extent      = input_4band
arcpy.env.snapRaster  = input_4band

# ==========================================
# פרמטרים
# ==========================================
PREVIEW_ONLY = True   # שלב 1: True כדי לבדוק זיהוי גריד + זרעים. שלב 2: False להרצה מלאה.

min_tree_blob_px            = 100   # שטח (px²) - מתחתיו בלוב "עץ" הוא רעש
min_final_tree_diameter_px  = 100   # קוטר (px) - עץ סופי קטן מזה מוסר

DOWNSAMPLE_FACTOR = 8       # פי כמה מקטינים את הרזולוציה לזיהוי הגריד
GRID_LINE_WIDTH_PX = 5       # רוחב קו הגריד בתצוגה המקדימה (פיקסלים, רזולוציה מלאה)
SEED_SIZE_PX       = 7       # קוטר "כתם" הזרע במרכז כל תא (פיקסלים, רזולוציה מלאה)

# תחום חיפוש המרווח בין שורות/טורים, ביחידות פיקסל ברזולוציה המלאה.
# עץ ~300px -> נסה טווח שמכיל ערכים סבירים מסביב לזה.
SPACING_SEARCH_MIN_PX = 150
SPACING_SEARCH_MAX_PX = 600

# ---- MANUAL OVERRIDES (ביחידות התמונה המוקטנת, px) ----
# אם הזיהוי האוטומטי לא נכון, מלא ערכים פה (לא None) כדי לדרוס אותו.
MANUAL_ANGLE_DEG   = None   # למשל 17.5
MANUAL_PERIOD_U    = None   # מרווח בכיוון u (px, בתמונה המוקטנת)
MANUAL_PHASE_U     = None
MANUAL_PERIOD_V    = None   # מרווח בכיוון v (px, בתמונה המוקטנת)
MANUAL_PHASE_V     = None

# ---- מרווח דינמי (per-strip) ----
# באחד הכיוונים (u או v) המרווח בין עצים אינו קבוע (גודל/מרחק עצים משתנה
# לאורך השורה). לכיוון הזה, במקום מרווח גלובלי קבוע, מאתרים בכל "שורה"
# (strip לפי הכיוון השני, הקבוע) את מיקומי השבילים בפועל (שיאי צפיפות-רקע
# מקומיים), וקובעים מרכזי-עצים = אמצעים בין שבילים עוקבים.
# "u" = הכיוון עם period_u, "v" = הכיוון עם period_v (לפי ההדפסה ב-Step 3).
# לפי התצוגה המקדימה: קווי ה-"רוחב" (השורות) טובים -> הם הציר הקבוע;
# קווי ה-"אורך" (בתוך השורה) צריכים להיות דינמיים.
DYNAMIC_AXIS = "u"

# מספר מינימלי של פיקסלי-רקע (בתמונה המוקטנת) בתוך שורה כדי לנתח אותה.
DYNAMIC_MIN_BG_PER_STRIP = 20

# מרחק מינימלי בין שני "שבילים" (gap_centers) עוקבים בציר הדינמי, כיחס
# מתוך המרווח הקבוע (period) בציר השני (מרווח בין שורות) -- מונע זיהוי
# שני קווי הפרדה/טורים צמודים מדי. למשל 0.8 = לפחות 80% מהמרחק בין שורות.
DYNAMIC_MIN_SPACING_RATIO = 0.8

# "רוחב עץ" צפוי בציר הדינמי = period (גודל שורה, בציר הקבוע) * הפקטור הזה.
# מקטע (בין שני שבילים עוקבים) שרחב יותר מהצפוי -- מפוצל למספר עצים שווים:
# 150% -> 2 עצים, 250% -> 3 עצים, וכו' (round(width / expected_width)).
TREE_WIDTH_REF_FACTOR = 1.0


# ==========================================
# פונקציות עזר לזיהוי הגריד
# ==========================================
def best_period_and_phase(values, min_period, max_period):
    """
    מקבל וקטור של קואורדינטות (פיקסלים של רקע, מוטלות על כיוון מסוים),
    בונה היסטוגרמה, ומחפש את התדירות הדומיננטית (= המרווח של הגריד)
    ואת הפאזה (היכן ממוקם "מרכז" הפס המחזורי, כלומר מרכז שביל).
    מחזיר (period, phase, score). score גבוה = מחזוריות חזקה יותר.
    """
    if values.size < 200:
        return None, None, 0.0

    lo, hi = float(values.min()), float(values.max())
    span = hi - lo
    if span < min_period * 2:
        return None, None, 0.0

    nbins = max(int(span), 32)
    hist, edges = np.histogram(values, bins=nbins, range=(lo, hi))
    hist = hist.astype(float)
    hist -= hist.mean()

    fft = np.abs(np.fft.rfft(hist))
    freqs = np.fft.rfftfreq(len(hist))
    bin_width = span / nbins

    with np.errstate(divide='ignore'):
        periods = np.where(freqs > 0, (1.0 / np.maximum(freqs, 1e-12)) * bin_width, 0)

    valid = (periods >= min_period) & (periods <= max_period)
    if not np.any(valid):
        return None, None, 0.0

    fft_valid = np.where(valid, fft, 0)
    idx = int(np.argmax(fft_valid))
    if fft_valid[idx] <= 0:
        return None, None, 0.0

    period = periods[idx]

    baseline = fft[valid]
    baseline_mean = baseline.mean() if baseline.size else 1.0
    score = fft[idx] / (baseline_mean + 1e-9)

    peak_bin = int(np.argmax(hist))
    phase = lo + (peak_bin + 0.5) * bin_width

    return float(period), float(phase), float(score)


def block_reduce_any(arr, f):
    """
    Downsampling לפי בלוקים f x f: כל בלוק נהיה True אם יש בו לפחות
    פיקסל True אחד. בניגוד לדילוג סטרידי (arr[::f, ::f]), זה לא
    "מפסיד" תבניות דקות (כמו שבילים/פערים ברוחב 1-3px).
    """
    rows, cols = arr.shape
    r2 = (rows // f) * f
    c2 = (cols // f) * f
    cropped = arr[:r2, :c2]
    reshaped = cropped.reshape(r2 // f, f, c2 // f, f)
    return reshaped.any(axis=(1, 3))


def detect_grid(bg_small, spacing_min_small, spacing_max_small):
    """
    מחפש את זווית הגריד (0-90 מעלות) שממקסמת את חוזק המחזוריות
    בשני הכיוונים הניצבים זה לזה (u ו-v).
    מחזיר dict עם angle_deg, period_u, phase_u, period_v, phase_v, score.
    phase_u/phase_v הם מרכזי-השבילים (הרקע) בכיוונים u/v.
    """
    ys, xs = np.where(bg_small)

    best = None
    for angle_deg in np.arange(0, 90, 1.0):
        theta = math.radians(angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        u = xs * cos_t + ys * sin_t
        v = -xs * sin_t + ys * cos_t

        period_u, phase_u, score_u = best_period_and_phase(u, spacing_min_small, spacing_max_small)
        period_v, phase_v, score_v = best_period_and_phase(v, spacing_min_small, spacing_max_small)

        if period_u is None or period_v is None:
            continue

        total_score = score_u + score_v
        if best is None or total_score > best["score"]:
            best = {
                "angle_deg": angle_deg,
                "period_u": period_u, "phase_u": phase_u,
                "period_v": period_v, "phase_v": phase_v,
                "score": total_score,
            }

    return best


def _rotated_coords(shape, angle_deg):
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    u = xx * cos_t + yy * sin_t
    v = -xx * sin_t + yy * cos_t
    return u, v


def _near(coord, period, phase, width):
    d = np.mod(coord - phase, period)
    d = np.minimum(d, period - d)
    return d < (width / 2.0)


def build_grid_mask(shape, angle_deg, period_u, phase_u, period_v, phase_v, line_width):
    """מסכת בוליאן: True על קווי הגריד (מרכזי-שבילים), בזווית angle_deg."""
    u, v = _rotated_coords(shape, angle_deg)
    return _near(u, period_u, phase_u, line_width) | _near(v, period_v, phase_v, line_width)


def build_seed_mask(shape, angle_deg, period_u, center_u, period_v, center_v, seed_size):
    """
    מסכת בוליאן: True בהצטלבויות של מרכזי-תאים (center_u/center_v) —
    אלו מיקומי "מרכז עץ" צפויים, אחד לכל תא בגריד.
    """
    u, v = _rotated_coords(shape, angle_deg)
    return _near(u, period_u, center_u, seed_size) & _near(v, period_v, center_v, seed_size)


def _near_abs(coord, center, width):
    """True בטווח ±width/2 מ-center (לא מחזורי, מיקום ספציפי אחד)."""
    return np.abs(coord - center) < (width / 2.0)


def smooth_1d(arr, sigma):
    """החלקה גאוסיאנית פשוטה (ללא scipy)."""
    radius = max(1, int(round(sigma * 3)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


def find_local_maxima(arr, min_distance):
    """אינדקסים של מקסימה מקומית, עם מרחק מינימלי בין שני שיאים עוקבים."""
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1] and arr[i] > 0:
            if peaks and (i - peaks[-1]) < min_distance:
                if arr[i] > arr[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    return np.array(peaks, dtype=int)


def detect_dynamic_centers(bg_small, angle_deg, dynamic_axis,
                            strip_period, strip_phase,
                            scan_min, scan_max, min_bg_per_strip,
                            min_spacing_ratio=0.0,
                            tree_width_ref_factor=1.0):
    """
    מחלק את התמונה ל"שורות" (strips) לאורך הציר הקבוע (הניצב ל-dynamic_axis),
    ברוחב strip_period, כך שמרכז כל שורה = center בציר הקבוע
    (strip_phase + period/2 + k*period, k שלם).
    בכל שורה מאתר לאורך dynamic_axis את מיקומי השבילים בפועל (שיאי צפיפות
    רקע מקומיים, אחרי החלקה) ומחשב מרכזי-עצים = אמצעים בין שבילים עוקבים,
    עם פיצול מקטעים רחבים מדי (ראה tree_width_ref_factor).

    min_spacing_ratio: מרחק מינימלי בין שני gap_centers עוקבים, כיחס מתוך
    strip_period (מרווח השורות) -- מונע זיהוי שני "טורים" צמודים מדי.

    tree_width_ref_factor: "רוחב עץ" צפוי = strip_period * factor הזה.
    מקטע (בין שני gap_centers עוקבים) שרחב יותר -- מפוצל למספר עצים שווים
    (round(width/expected_width)), כלומר 150% -> 2 עצים, 250% -> 3 וכו'.

    מחזיר רשימת dict: strip_center, gap_centers (מיקומי שבילים שזוהו),
    tree_centers (מרכזי-עצים צפויים, כולל פיצולים). כל הערכים ביחידות bg_small.
    """
    u, v = _rotated_coords(bg_small.shape, angle_deg)
    if dynamic_axis == "u":
        scan, strip = u, v
    else:
        scan, strip = v, u

    ys, xs = np.where(bg_small)
    scan_bg = scan[ys, xs]
    strip_bg = strip[ys, xs]

    scan_lo, scan_hi = float(scan.min()), float(scan.max())
    bin_w = max(scan_min / 4.0, 1.0)
    nbins = max(int((scan_hi - scan_lo) / bin_w), 8)
    edges = np.linspace(scan_lo, scan_hi, nbins + 1)

    # מרחק מינימלי בין שני שיאים (gap_centers) עוקבים: המקסימום בין
    # scan_min (טווח החיפוש הכללי) לבין min_spacing_ratio * strip_period
    # (יחס מתוך מרווח השורות) -- מונע "טורים" צמודים מדי.
    min_spacing = max(scan_min, min_spacing_ratio * strip_period)
    min_dist_bins = max(int(round(min_spacing / bin_w)), 1)

    results = []
    # מיישרים את נקודת ההתחלה לרשת המחזורית (...,strip_phase-2*period,
    # strip_phase-period, strip_phase, strip_phase+period,...) הקרובה
    # ביותר ל-strip.min(), כדי לעבור על כל הטווח (לא רק בסביבת strip_phase).
    strip_min = float(strip.min())
    s_max = float(strip.max())
    offset = strip_phase - strip_period
    n_start = math.floor((strip_min - offset) / strip_period)
    s0 = offset + n_start * strip_period
    while s0 < s_max:
        s1 = s0 + strip_period
        sel = (strip_bg >= s0) & (strip_bg < s1)
        scan_vals = scan_bg[sel]
        strip_center = (s0 + s1) / 2.0
        if scan_vals.size >= min_bg_per_strip:
            hist, _ = np.histogram(scan_vals, bins=edges)
            hist_s = smooth_1d(hist.astype(float), sigma=2.0)
            peak_idx = find_local_maxima(hist_s, min_dist_bins)
            if peak_idx.size >= 2:
                gap_centers = edges[peak_idx] + bin_w / 2.0
                gap_centers.sort()

                # רוחב "עץ" מצופה = מרחק בין שני שבילים עוקבים. אם הוא גדול
                # בהרבה מ-strip_period (גודל שורה) -- כנראה חסר שביל בנתיים
                # ויש שם כמה עצים -- מחלקים את המקטע ל-n חלקים שווים
                # (n = round(width / (TREE_WIDTH_REF_FACTOR*strip_period)),
                # למשל 150% -> 2 עצים, 250% -> 3 עצים).
                ref_width = strip_period * tree_width_ref_factor
                tree_centers = []
                for i in range(len(gap_centers) - 1):
                    width = gap_centers[i + 1] - gap_centers[i]
                    n = max(1, int(width / ref_width + 0.5))
                    for j in range(n):
                        tree_centers.append(gap_centers[i] + width * (j + 0.5) / n)

                results.append({
                    "strip_center": strip_center,
                    "gap_centers": gap_centers.tolist(),
                    "tree_centers": tree_centers,
                })
        s0 = s1

    return results


def build_dynamic_masks(shape, angle_deg, dynamic_axis, strip_period,
                         strip_results, line_width, seed_size):
    """
    בונה משתי מסכות בוליאניות מתוך תוצאות detect_dynamic_centers:
    - grid_mask: קווי גריד דינמיים (במיקומי gap_centers שזוהו בכל שורה).
    - seed_mask: זרעים במרכזי-עצים (tree_centers), בתוך כל שורה.
    """
    u, v = _rotated_coords(shape, angle_deg)
    if dynamic_axis == "u":
        scan, strip = u, v
    else:
        scan, strip = v, u

    grid_mask = np.zeros(shape, dtype=bool)
    seed_mask = np.zeros(shape, dtype=bool)

    for entry in strip_results:
        # קווי הפרדה דינמיים: צריכים להימשך על פני כל גובה השורה (strip_period)
        line_band = _near_abs(strip, entry["strip_center"], strip_period)
        for gap_c in entry["gap_centers"]:
            grid_mask |= line_band & _near_abs(scan, gap_c, line_width)

        # זרעים: כתם קטן (seed_size x seed_size) במרכז כל עץ -- לא קו
        # שמתפרש על כל גובה השורה (אחרת זרעים משורות שכנות מתחברים לקו)
        seed_band = _near_abs(strip, entry["strip_center"], seed_size)
        for tree_c in entry["tree_centers"]:
            seed_mask |= seed_band & _near_abs(scan, tree_c, seed_size)

    return grid_mask, seed_mask


def place_seeds_full(shape, angle_deg, dynamic_axis, strip_results, seed_size):
    """
    בונה seed_mask ברזולוציה מלאה בלי לבנות מערכי u/v גלובליים (אלו היו
    מערכים בגודל כל-התמונה -- ~8GB כל אחד -- ולא ניתנים להקצאה).
    במקום זה, לכל "עץ צפוי" (tree_center בתוך strip) מחשבים את מיקומו
    (x,y) במרחב התמונה (טרנספורמציה הופכית מ-u,v), ומציבים כתם קטן
    (seed_size x seed_size ביחידות u/v) רק בתיבת-תיחום מקומית קטנה סביבו.
    """
    rows, cols = shape
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    half = seed_size / 2.0

    # תיבת-תיחום ב-(x,y) שמכילה ריבוע seed_size x seed_size ב-(u,v)
    r = int(math.ceil(half * (abs(cos_t) + abs(sin_t)))) + 1

    seed_mask = np.zeros(shape, dtype=bool)

    for entry in strip_results:
        strip_center = entry["strip_center"]
        for tree_c in entry["tree_centers"]:
            if dynamic_axis == "u":
                u0, v0 = tree_c, strip_center
            else:
                u0, v0 = strip_center, tree_c

            # טרנספורמציה הופכית: (u,v) -> (x,y)
            x0 = u0 * cos_t - v0 * sin_t
            y0 = u0 * sin_t + v0 * cos_t

            x_lo = max(int(math.floor(x0 - r)), 0)
            x_hi = min(int(math.ceil(x0 + r)) + 1, cols)
            y_lo = max(int(math.floor(y0 - r)), 0)
            y_hi = min(int(math.ceil(y0 + r)) + 1, rows)
            if x_lo >= x_hi or y_lo >= y_hi:
                continue

            yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
            u = xx * cos_t + yy * sin_t
            v = -xx * sin_t + yy * cos_t
            local_mask = (np.abs(u - u0) < half) & (np.abs(v - v0) < half)
            seed_mask[y_lo:y_hi, x_lo:x_hi] |= local_mask

    return seed_mask


# ==========================================
# פייפליין
# ==========================================
print("Starting Grid-Seeded Watershed Tree Separation...")
try:
    # ---------------------------------------------------------
    # PART 1: BINARIZATION
    # ---------------------------------------------------------
    print("Step 1: Extracting Band 4 and standardizing mask...")
    band4 = Raster(input_4band + r"\Band_4")
    # הערה: לא להשתמש ב- IsNull(band4) | (band4==0) -- אופרטור ה-Or מפיץ NoData,
    # כך שבכל פיקסל שבו band4 הוא NoData הביטוי כולו הופך ל-NoData (לא 1)!
    # קינון Con נפרד לכל מקרה פותר את זה.
    binary_mask = Con(IsNull(band4), 1, Con(band4 == 0, 1, 0))  # 1=רקע, 0=עץ

    # ---------------------------------------------------------
    # PART 1.5: NOISE REMOVAL (שומר את ערך הבלוב המקורי)
    # ---------------------------------------------------------
    print(f"Step 1.5: Removing tree-blobs smaller than {min_tree_blob_px} px (noise)...")
    all_blobs = RegionGroup(binary_mask, "EIGHT", "WITHIN")
    blob_value = Lookup(all_blobs, "LINK")
    blob_count = Lookup(all_blobs, "COUNT")
    binary_mask = Con((blob_value == 0) & (blob_count < min_tree_blob_px), 1, binary_mask)

    temp_binary_path = os.path.join(output_folder, "temp_binary_mask.tif")
    print("        > Saving binary mask to disk...")
    binary_mask.save(temp_binary_path)

    desc = arcpy.Describe(temp_binary_path)
    extent = desc.extent
    cellsize_x = desc.meanCellWidth
    cellsize_y = desc.meanCellHeight
    sr = desc.spatialReference
    lower_left = arcpy.Point(extent.XMin, extent.YMin)

    # ---------------------------------------------------------
    # PART 2: DETECT GRID (זווית, מרווחים, פאזות) מתוך הרקע
    # ---------------------------------------------------------
    print("Step 2: Loading binary mask into numpy...")
    bg_full = (arcpy.RasterToNumPyArray(temp_binary_path) == 1)
    print(f"        > גודל מלא: {bg_full.shape[1]}x{bg_full.shape[0]} פיקסלים")

    f = DOWNSAMPLE_FACTOR
    bg_small = block_reduce_any(bg_full, f)
    print(f"        > רקע (block-reduce, any): {int(bg_small.sum())} פיקסלים "
          f"מתוך {bg_small.size} בתמונה המוקטנת ({bg_small.shape[1]}x{bg_small.shape[0]})")
    spacing_min_small = SPACING_SEARCH_MIN_PX / f
    spacing_max_small = SPACING_SEARCH_MAX_PX / f

    if MANUAL_ANGLE_DEG is not None:
        print("Step 3: Using MANUAL grid parameters (override)...")
        grid_params = {
            "angle_deg": MANUAL_ANGLE_DEG,
            "period_u": MANUAL_PERIOD_U, "phase_u": MANUAL_PHASE_U,
            "period_v": MANUAL_PERIOD_V, "phase_v": MANUAL_PHASE_V,
            "score": float("nan"),
        }
    else:
        print("Step 3: Detecting grid angle & spacing (auto)...")
        grid_params = detect_grid(bg_small, spacing_min_small, spacing_max_small)

    if grid_params is None:
        raise ValueError(
            "\n" + "=" * 50 +
            "\n🚨 לא נמצא גריד מחזורי! 🚨\n"
            "נסה לשנות את SPACING_SEARCH_MIN_PX / SPACING_SEARCH_MAX_PX,\n"
            "או למלא MANUAL_* ידנית.\n"
            + "=" * 50
        )

    angle_deg = grid_params["angle_deg"]
    period_u, phase_u = grid_params["period_u"], grid_params["phase_u"]
    period_v, phase_v = grid_params["period_v"], grid_params["phase_v"]

    # מרכזי-תאים = מרכזי-שבילים + חצי מרווח (האמצע שבין שני שבילים עוקבים)
    center_u = phase_u + period_u / 2.0
    center_v = phase_v + period_v / 2.0

    print("        > תוצאות זיהוי הגריד (ביחידות התמונה המוקטנת):")
    print(f"          angle_deg = {angle_deg:.2f}")
    print(f"          period_u  = {period_u:.2f}  (path_phase_u={phase_u:.2f}, center_u={center_u:.2f})")
    print(f"          period_v  = {period_v:.2f}  (path_phase_v={phase_v:.2f}, center_v={center_v:.2f})")
    print(f"          score     = {grid_params['score']:.2f}")
    print(f"        > במלא-רזולוציה: period_u≈{period_u*f:.0f}px, period_v≈{period_v*f:.0f}px")

    # ---------------------------------------------------------
    # מרווח דינמי: לכיוון DYNAMIC_AXIS מאתרים מרכזי-עצים בפועל לכל שורה
    # (per-strip), במקום מרווח גלובלי קבוע. הציר השני (הקבוע) ממשיך
    # להשתמש ב-period/phase הגלובליים שזוהו למעלה.
    # ---------------------------------------------------------
    if DYNAMIC_AXIS == "u":
        fixed_period, fixed_phase = period_v, phase_v
    else:
        fixed_period, fixed_phase = period_u, phase_u

    print(f"        > DYNAMIC_AXIS = '{DYNAMIC_AXIS}' "
          f"(מרווח בכיוון זה יחושב per-strip, לא גלובלי)")
    strip_results_small = detect_dynamic_centers(
        bg_small, angle_deg, DYNAMIC_AXIS,
        strip_period=fixed_period, strip_phase=fixed_phase,
        scan_min=spacing_min_small, scan_max=spacing_max_small,
        min_bg_per_strip=DYNAMIC_MIN_BG_PER_STRIP,
        min_spacing_ratio=DYNAMIC_MIN_SPACING_RATIO,
        tree_width_ref_factor=TREE_WIDTH_REF_FACTOR,
    )
    n_trees_dynamic = sum(len(e["tree_centers"]) for e in strip_results_small)
    print(f"        > זוהו {len(strip_results_small)} שורות עם דפוס שבילים תקין, "
          f"סה\"כ {n_trees_dynamic} מרכזי-עץ דינמיים")

    # ---------------------------------------------------------
    # PREVIEW MODE: שומר תצוגה מקדימה קטנה (גריד + זרעים) ועוצר
    # ---------------------------------------------------------
    if PREVIEW_ONLY:
        print("Step 4 (PREVIEW): Building grid + seed overlay for visual check...")
        u_small, v_small = _rotated_coords(bg_small.shape, angle_deg)
        fixed_coord_small = v_small if DYNAMIC_AXIS == "u" else u_small

        line_width_small = max(GRID_LINE_WIDTH_PX / f, 1.0)
        seed_size_small = max(SEED_SIZE_PX / f, 1.0) * 2  # קצת יותר גדול לתצוגה

        # קווי הגריד הקבועים (התדירות הגלובלית בציר שאינו דינמי)
        fixed_grid_small = _near(fixed_coord_small, fixed_period, fixed_phase, line_width_small)
        # קווי גריד + זרעים דינמיים (per-strip, בציר DYNAMIC_AXIS)
        dyn_grid_small, seed_small = build_dynamic_masks(
            bg_small.shape, angle_deg, DYNAMIC_AXIS, fixed_period,
            strip_results_small, line_width=line_width_small, seed_size=seed_size_small,
        )
        grid_small = fixed_grid_small | dyn_grid_small

        # 0=עץ, 1=רקע, 2=קו גריד, 3=זרע (קטגוריאלי לתצוגה)
        preview = np.where(bg_small, 1, 0).astype(np.uint8)
        preview[grid_small] = 2
        preview[seed_small] = 3

        preview_path = os.path.join(output_folder, "temp_grid_preview.tif")
        preview_lower_left = arcpy.Point(extent.XMin, extent.YMin)
        preview_raster = arcpy.NumPyArrayToRaster(
            preview, preview_lower_left, cellsize_x * f, cellsize_y * f
        )
        preview_raster.save(preview_path)
        arcpy.management.DefineProjection(preview_path, sr)

        print(f"\n✅ תצוגה מקדימה נשמרה: {preview_path}")
        print("   פתח ב-ArcGIS Pro, סמל קטגוריאלי (Unique Values):")
        print("   0=עץ (שחור), 1=רקע/שביל (לבן), 2=קו גריד (אדום), 3=זרע/מרכז עץ צפוי (כחול).")
        print("   בדוק: (א) הקווים האדומים מתיישרים עם השבילים האמיתיים,")
        print("         (ב) הנקודות הכחולות נופלות בערך במרכז כל עץ (כולל עצים כלואים).")
        print("   אם לא — כוון MANUAL_* בראש הקובץ והרץ שוב עם PREVIEW_ONLY=True.")
        print("   כשזה נראה טוב — שנה PREVIEW_ONLY=False והרץ את הפייפליין המלא.")

    else:
        # -----------------------------------------------------
        # PART 3: בניית מסכת הזרעים ברזולוציה מלאה
        # -----------------------------------------------------
        print("Step 4: Building full-resolution seed mask (one seed per expected tree)...")
        fixed_period_full = fixed_period * f
        strip_results_full = [
            {
                "strip_center": e["strip_center"] * f,
                "gap_centers": [g * f for g in e["gap_centers"]],
                "tree_centers": [t * f for t in e["tree_centers"]],
            }
            for e in strip_results_small
        ]

        seed_full = place_seeds_full(
            bg_full.shape, angle_deg, DYNAMIC_AXIS,
            strip_results_full, seed_size=SEED_SIZE_PX,
        )
        n_seeds_est = int(seed_full.sum())
        print(f"        > seed pixels: {n_seeds_est}")

        print("Step 5: Saving seed raster...")
        seed_arr = seed_full.astype(np.uint8)
        seed_raster = arcpy.NumPyArrayToRaster(seed_arr, lower_left, cellsize_x, cellsize_y)
        temp_seed_path = os.path.join(output_folder, "temp_seed_mask.tif")
        seed_raster.save(temp_seed_path)
        arcpy.management.DefineProjection(temp_seed_path, sr)

        del bg_full, seed_full, seed_arr  # שחרור זיכרון

        # -----------------------------------------------------
        # PART 4: EucAllocation מהזרעים, ממוסך לצמרת
        # -----------------------------------------------------
        print("Step 6: Assigning unique IDs to each seed...")
        seed_ids_raw = RegionGroup(Raster(temp_seed_path), "EIGHT", "WITHIN")
        # רק תאים שבאמת היו seed (value==1) מקבלים ID; השאר NoData
        seed_ids = Con(Lookup(seed_ids_raw, "LINK") == 1, seed_ids_raw)

        print("Step 7: Allocating each canopy pixel to its nearest tree seed...")
        binary_raster_loaded = Raster(temp_binary_path)
        canopy = Con(binary_raster_loaded == 0, 1)  # NoData ברקע

        arcpy.env.mask = canopy
        grown_trees = EucAllocation(seed_ids)
        arcpy.env.mask = ""

        # -----------------------------------------------------
        # PART 4.4: חלוקה קבועה (Tree_Zones_Master) - לשימוש חוזר בשנים הבאות
        # -----------------------------------------------------
        # EucAllocation בלי מסכת צמרת -> מכסה את כל שטח התמונה (כמו Voronoi),
        # כך שכל פיקסל (לא רק צמרת) מקבל TreeID של הזרע הקרוב אליו.
        # מכיוון שהחלוקה הזו תלויה רק במיקומי הזרעים (לא בצורת הצמרת),
        # היא קבועה וניתן להשתמש בה כל שנה (Zonal Statistics) כדי להשוות
        # את אותו עץ (אותו Value=TreeID) בין שנים שונות.
        print("Step 7.5: Building FIXED tree-zone partition (Tree_Zones_Master)...")
        tree_zones_master = EucAllocation(seed_ids)
        tree_zones_master.save(output_tree_zones)
        print(f"✅ חלוקה קבועה לעצים נשמרה: {output_tree_zones}")
        print("   כל פיקסל מקודד ב-Value = TreeID (זהה ל-seed_ids).")
        print("   לשנים הבאות: בצעו בינריזציה (PART 1) על אותה שנה,")
        print("   ואז Zonal Statistics as Table עם zone raster=Tree_Zones_Master.tif,")
        print("   zone field=VALUE, על מסכת הצמחייה של אותה שנה -> שטח/סכום לכל TreeID.")

        print("Step 8: Drawing separation lines at allocation borders...")
        borders = FocalStatistics(grown_trees, NbrRectangle(3, 3, "CELL"), "VARIETY")

        # 0 = פנים עץ, 255 = רקע / קו הפרדה בין עצים
        final_mask = Con(IsNull(grown_trees), 255, Con(borders > 1, 255, 0))

        # -----------------------------------------------------
        # PART 4.5: ניקוי בלובי-עץ קטנים מדי
        # -----------------------------------------------------
        min_final_area_px = math.pi * (min_final_tree_diameter_px / 2.0) ** 2
        print(f"Step 8.5: Removing leftover tree-blobs smaller than "
              f"{min_final_area_px:.0f} px² (diameter < {min_final_tree_diameter_px}px)...")

        final_blobs = RegionGroup(final_mask, "EIGHT", "WITHIN")
        final_value = Lookup(final_blobs, "LINK")
        final_count = Lookup(final_blobs, "COUNT")
        final_mask = Con((final_value == 0) & (final_count < min_final_area_px), 255, final_mask)

        final_mask.save(output_separated)
        print(f"✅ Success! Grid-seeded separated mask saved to: {output_separated}")

        # -----------------------------------------------------
        # PART 5: הרכבת מוזאיקה 4 שכבות (RGB + מסכה מופרדת, קידוד 0/1)
        # -----------------------------------------------------
        print("Step 9: Compositing final 4-band mosaic...")
        final_band4 = Con(final_mask == 0, 1, 0)  # 1=עץ, 0=רקע/הפרדה (כמו במקור)

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
        print("   Band 1-3 = RGB (original)  |  Band 4 = separated segmentation mask (0/1)")

        # -----------------------------------------------------
        # PART 6: שמירת פרמטרי הרשת - לתיעוד/שחזור (לא נדרש כדי
        # להשתמש בחלוקה הקבועה, אבל שימושי לדיבוג/QA)
        # -----------------------------------------------------
        grid_params = {
            "angle_deg": angle_deg,
            "period_u_small": period_u, "phase_u_small": phase_u,
            "period_v_small": period_v, "phase_v_small": phase_v,
            "downsample_factor": f,
            "dynamic_axis": DYNAMIC_AXIS,
            "n_trees": n_seeds_est,
        }
        with open(output_grid_params, "w", encoding="utf-8") as fjson:
            json.dump(grid_params, fjson, ensure_ascii=False, indent=2)
        print(f"✅ פרמטרי הרשת נשמרו: {output_grid_params}")

        # ניקיון קבצי ביניים
        # שימו לב: temp_seed_path (מיקומי הזרעים) נשמר -
        # הוא ה"מקור" של Tree_Zones_Master ושימושי אם תרצו לבנות
        # מחדש את החלוקה הקבועה בעתיד.
        try:
            arcpy.management.Delete(temp_band4_path)
            arcpy.management.Delete(temp_binary_path)
        except Exception:
            pass

except Exception as e:
    print(f"❌ Error during processing: {e}")

finally:
    arcpy.CheckInExtension("Spatial")
    arcpy.env.addOutputsToMap = True
