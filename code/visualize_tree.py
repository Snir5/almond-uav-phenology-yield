import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import label, center_of_mass

# ==========================================
# הגדרות
# ==========================================
# נתיב לקובץ של 2023 (יש לוודא שהנתיב מדויק)
tif_2023 = "/Users/snirtahasa/Thesis/4band_mosaic/01_03_2023/Final_Orthomosaic_4Band.tif"

# הקואורדינטות המדויקות של עץ Tree_0731 (מהטבלה ששלחת)
master_x = 670069.6701
master_y = 3509614.952

search_radius = 3.0  # רדיוס החיפוש שלנו (במטרים)
view_buffer = 8.0    # כמה מטרים מסביב לעץ נרצה לראות (זום-אין)

print("Loading specific area for visualization...")

# ==========================================
# חיתוך ועיבוד מרחבי
# ==========================================
with rasterio.open(tif_2023) as src:
    # הגדרת "חלון" (Window) קטן סביב העץ כדי לא לטעון את כל הקובץ הכבד
    window = rasterio.windows.from_bounds(
        master_x - view_buffer, master_y - view_buffer,
        master_x + view_buffer, master_y + view_buffer,
        src.transform
    )
    
    # קריאת הפיקסלים רק של אזור העץ
    R = src.read(1, window=window).astype(float)
    G = src.read(2, window=window).astype(float)
    B = src.read(3, window=window).astype(float)
    mask = src.read(4, window=window)
    win_transform = src.window_transform(window)
    
    # הגדרת גבולות התמונה (Extent) ב-UTM כדי שהגרף יהיה בקואורדינטות אמיתיות
    extent = [
        win_transform.c, win_transform.c + win_transform.a * window.width,
        win_transform.f + win_transform.e * window.height, win_transform.f
    ]

# מציאת כל הסגמנטים (כתמים) שהמסיכה של 2023 זיהתה בתוך הריבוע הזה
labeled_trees, num_trees = label(mask == 1)
tree_indices = np.arange(1, num_trees + 1)
centers = center_of_mass(mask, labeled_trees, tree_indices)

# חיבור ויזואלי של צבעי ה-RGB (עם נרמול קל כדי שהתמונה תהיה בהירה ויפה)
rgb = np.dstack((R, G, B))
rgb_max = np.percentile(rgb[rgb > 0], 98) # מניעת סנוור
rgb_normalized = np.clip(rgb / rgb_max, 0, 1)

# ==========================================
# יצירת הגרף והתמונה
# ==========================================
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(rgb_normalized, extent=extent)

# 1. ציור מעגל החיפוש (3 מטר)
search_area = Circle((master_x, master_y), search_radius, color='white', fill=False, linestyle='--', linewidth=2, label='3m Search Radius')
ax.add_patch(search_area)

# 2. ציור נקודת המאסטר הקבועה (2024)
ax.plot(master_x, master_y, 'r*', markersize=20, label='Master Tree_0731 (2024)')

# 3. ציור הכתמים שהתגלו ב-2023
for row, col in centers:
    if not np.isnan(row) and not np.isnan(col):
        # המרה מפיקסלים של החלון לקואורדינטות UTM אמיתיות
        x_utm, y_utm = rasterio.transform.xy(win_transform, row, col)
        
        # נצייר רק את מה שנמצא בתוך החלון
        ax.plot(x_utm, y_utm, 'co', markersize=10, markeredgecolor='black', label='2023 Detected Segment')

# סידור המקרא שלא ישכפל את עצמו
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)

ax.set_title("Why Data Duplicated: Tree_0731 Visualization", fontsize=18, fontweight='bold')
ax.set_xlabel("Longitude (X UTM)", fontsize=12)
ax.set_ylabel("Latitude (Y UTM)", fontsize=12)

plt.show()