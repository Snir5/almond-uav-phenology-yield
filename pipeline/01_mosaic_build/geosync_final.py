"""
GeoSync — DJI Flight Log → Sony EXIF GPS Injector
===================================================
Core innovations in this version:

1. AUTO-CALIBRATION of CAMERA_CLOCK_OFFSET
   Uses the drone's own isPhoto=1 events as GPS ground truth.
   Scans candidate offsets and finds the one that minimises
   the distance between each Sony photo's interpolated GPS
   and the nearest DJI trigger position.

2. SPEED-DISTANCE CORRELATION (fallback calibration)
   For each candidate offset: measures how well drone speed
   at photo-time correlates with distance to the next photo.
   Peaks sharply at the correct offset.

3. VELOCITY CORRECTION using xSpeed / ySpeed (world frame)
   Compensates for camera shutter latency using the actual
   East/North velocity vector — no heading-trig approximation.

4. POSITION UNCERTAINTY per photo in exported CSV.
"""

import os
import math
import shutil
import piexif
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# USER SETTINGS
# ============================================================

LOG_FILES = [
    r"/Users/snirtahasa/Thesis/DJI LOG/March-7-2021-12-56-05-Flight-Airdata.csv",
    r"/Users/snirtahasa/Thesis/DJI LOG/March-7-2021-13-15-59-Flight-Airdata.csv",
]
IMAGE_FOLDER  = r"/Users/snirtahasa/Thesis/2021-03-07/photos"
BACKUP_FOLDER = r"/Users/snirtahasa/Thesis/2021-03-07/photos_ORIGINAL_BACKUP"

# Starting guess (seconds). Auto-calibration will refine this.
CAMERA_CLOCK_OFFSET_SECONDS = -19.0

# Per-flight offset overrides (log filename → offset in seconds).
# Use when the camera clock changed between flights (battery swap, manual reset).
# These BYPASS auto-calibration for that specific flight — use exact value.
# To find the correct value: do manual georeferencing of one flight-2 image
# in ArcGIS, read the world file, and run the calc_offset_from_worldfile()
# helper below. Leave empty {} to use the global offset for all flights.
CAMERA_CLOCK_OFFSET_PER_FILE = {
    # "March-7-2021-13-15-59-Flight-Airdata.csv": -49.0,   # ← uncomment + tune
}

# Set only if camera is in local time (e.g. Israel UTC+2 → set 2).
# If camera is on UTC (common when synced to DJI GPS), leave 0.
CAMERA_TIMEZONE_OFFSET_HOURS = 0

# Camera mechanical shutter latency (ms). Drone keeps moving during this.
# Typical Sony mirrorless: 50–150 ms.
CAMERA_SHUTTER_LATENCY_MS = 100

# Auto-calibration search window around the starting guess (seconds).
CALIBRATION_SEARCH_WINDOW = 15.0   # scan ±15 s
CALIBRATION_STEP_COARSE   = 0.5    # coarse pass resolution
CALIBRATION_STEP_FINE     = 0.05   # fine pass resolution (around coarse best)

# Photos with positional uncertainty above this are flagged in the CSV.
# At 5 m/s cruise + 0.5s EXIF precision → 2.5m is the expected floor.
# Set to 5.0 so HIGH flags only appear for genuinely fast segments (>10 m/s).
UNCERTAINTY_FLAG_M = 5.0


# ============================================================
# UTILITIES
# ============================================================

def backup_images():
    if os.path.exists(BACKUP_FOLDER):
        print(f"  [backup] already exists — skipping.")
        return
    print(f"  [backup] copying to {BACKUP_FOLDER} ...")
    shutil.copytree(IMAGE_FOLDER, BACKUP_FOLDER)
    print(f"  [backup] done.")


def float_to_rational(value: float) -> tuple:
    """Decimal degrees → EXIF rational. 1e-6 denominator ≈ 1 cm precision."""
    if pd.isna(value):
        return ((0, 1), (0, 1), (0, 1))
    value   = abs(float(value))
    deg     = int(value)
    min_f   = (value - deg) * 60
    minutes = int(min_f)
    sec_f   = (min_f - minutes) * 60
    return ((deg, 1), (minutes, 1), (int(round(sec_f * 1_000_000)), 1_000_000))


def inject_gps_to_jpg(image_path: str, lat: float, lon: float, alt: float) -> bool:
    try:
        exif_dict = piexif.load(image_path)
    except Exception as e:
        print(f"    [ERROR] cannot read EXIF {os.path.basename(image_path)}: {e}")
        return False

    gps_ifd = {
        piexif.GPSIFD.GPSVersionID:    (2, 3, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef:  b'N' if lat >= 0 else b'S',
        piexif.GPSIFD.GPSLatitude:     float_to_rational(abs(lat)),
        piexif.GPSIFD.GPSLongitudeRef: b'E' if lon >= 0 else b'W',
        piexif.GPSIFD.GPSLongitude:    float_to_rational(abs(lon)),
        piexif.GPSIFD.GPSAltitudeRef:  b'\x00' if alt >= 0 else b'\x01',
        piexif.GPSIFD.GPSAltitude:     (int(abs(alt) * 100), 100),
        piexif.GPSIFD.GPSMapDatum:     b'WGS-84',
    }
    try:
        exif_dict['GPS'] = gps_ifd
        exif_bytes = piexif.dump(exif_dict)
        piexif.remove(image_path)
        piexif.insert(exif_bytes, image_path)
        return True
    except Exception as e:
        print(f"    [ERROR] EXIF write failed {os.path.basename(image_path)}: {e}")
        return False


# ============================================================
# GPS INTERPOLATION HELPER
# ============================================================

def detect_flight_gaps(df_logs: pd.DataFrame,
                       gap_threshold_s: float = 30.0) -> list[tuple]:
    """
    Finds gaps between separate flights within the merged log.
    A gap is any interval between consecutive GPS points longer than
    gap_threshold_s seconds (typically happens between battery swaps).
    Returns a list of (gap_start, gap_end) as datetime64 pairs.
    """
    times  = df_logs['exact_time'].sort_values().values
    deltas = pd.Series(times[1:]) - pd.Series(times[:-1])
    gaps   = []
    for i, d in enumerate(deltas):
        if d > np.timedelta64(int(gap_threshold_s * 1e9), 'ns'):
            gaps.append((times[i], times[i + 1]))
    return gaps


def interpolate_gps(df_images_timed: pd.DataFrame,
                    df_logs: pd.DataFrame,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Merge image timestamps into the GPS timeline and interpolate
    lat/lon/alt/xSpeed/ySpeed/speed for each image.

    Gap-aware: images that fall inside a between-flight gap
    (drone on ground between battery swaps) are excluded so they
    don't silently inherit the home-point coordinates.

    Window-safe: images outside [gps_start, gps_end] are removed
    BEFORE interpolation. This is more reliable than limit_area='inside'
    which silently extrapolates in some pandas versions when limit= is unset.
    """
    gps_start = df_logs['exact_time'].iloc[0]
    gps_end   = df_logs['exact_time'].iloc[-1]

    # Hard clip: drop images outside the GPS window before interpolating
    in_window = (
        (df_images_timed['exact_time'] >= gps_start) &
        (df_images_timed['exact_time'] <= gps_end)
    )
    n_outside = (~in_window).sum()
    if n_outside and verbose:
        print(f"  ✂  {n_outside} images clipped — outside GPS window "
              f"({gps_start.strftime('%H:%M:%S')}–{gps_end.strftime('%H:%M:%S')} UTC)")
    df_images_timed = df_images_timed[in_window].copy()

    # Detect gaps between separate flights (e.g. battery swap)
    gaps = detect_flight_gaps(df_logs, gap_threshold_s=30.0)

    cols_gps = ['exact_time', 'latitude', 'longitude',
                'altitude_above_seaLevel(meters)',
                'xSpeed(m/s)', 'ySpeed(m/s)', 'speed(m/s)', 'flycState']

    logs_s   = df_logs[cols_gps].copy();  logs_s['_type'] = 'GPS'
    images_s = df_images_timed[['exact_time', 'image_name']].copy()
    images_s['_type'] = 'IMAGE'

    combined = (
        pd.concat([logs_s, images_s])
          .sort_values('exact_time')
          .set_index('exact_time')
    )
    for col in ['latitude', 'longitude', 'altitude_above_seaLevel(meters)',
                'xSpeed(m/s)', 'ySpeed(m/s)', 'speed(m/s)']:
        # limit=len(combined) ensures all interior gaps are filled;
        # images outside the window were already removed above so no
        # extrapolation is possible here.
        combined[col] = combined[col].interpolate(method='time',
                                                   limit=len(combined))

    # Forward-fill flycState so each image row inherits the drone's flight
    # phase at its interpolated time (Waypoint / Go_Home / AutoLanding …)
    combined['flycState'] = combined['flycState'].ffill()

    # Null-out any image that falls inside a between-flight gap
    # so it is excluded by the subsequent dropna() call
    for gap_start, gap_end in gaps:
        mask = (
            (combined.index > gap_start) &
            (combined.index < gap_end) &
            (combined['_type'] == 'IMAGE')
        )
        combined.loc[mask, ['latitude', 'longitude']] = np.nan

    result = (
        combined[combined['_type'] == 'IMAGE']
          .dropna(subset=['latitude', 'longitude'])
          .reset_index()
    )
    return result, gaps


# ============================================================
# AUTO-CALIBRATION
# ============================================================

def _apply_offset(df_images_base: pd.DataFrame, offset_s: float) -> pd.DataFrame:
    df = df_images_base.copy()
    df['exact_time'] = (
        df['base_time']
        + pd.to_timedelta(df['subsec'], unit='s')
        + pd.to_timedelta(offset_s, unit='s')
    ).astype('datetime64[ns]')
    return df


def score_isphoto(offset_s: float,
                  df_images_base: pd.DataFrame,
                  df_logs: pd.DataFrame,
                  isphoto_rows: pd.DataFrame) -> float:
    """
    For each DJI isPhoto=1 event we know:
      - exact UTC time the drone trigger fired
      - exact GPS position at that moment

    For a given clock offset, find the Sony photo closest in time
    to each trigger event and measure how far its interpolated GPS
    is from the trigger GPS.

    Returns mean error in meters (lower = better offset).
    Only valid when len(isphoto_rows) >= 2.
    """
    if len(isphoto_rows) < 2:
        return np.inf

    df_img = _apply_offset(df_images_base, offset_s)
    result, _ = interpolate_gps(df_img, df_logs, verbose=False)
    if len(result) < 3:
        return np.inf

    # Build a lookup: image_name → interpolated GPS
    gps_lookup = result.set_index('image_name')[['latitude', 'longitude']].to_dict('index')

    # For each isPhoto event, find the nearest Sony photo (by time)
    img_times = df_img.set_index('exact_time')['image_name']
    errors = []
    for _, ev in isphoto_rows.iterrows():
        # Find closest Sony image in time to this DJI trigger
        t_dji = ev['exact_time']
        idx   = (img_times.index - t_dji).abs().argmin()
        img   = img_times.iloc[idx]
        if img not in gps_lookup:
            continue
        g     = gps_lookup[img]
        # Haversine-lite error (metres)
        dlat  = (g['latitude']  - ev['latitude'])  * 111_111
        dlon  = (g['longitude'] - ev['longitude']) * 111_111 * math.cos(math.radians(ev['latitude']))
        errors.append(math.sqrt(dlat**2 + dlon**2))

    return float(np.mean(errors)) if errors else np.inf


def score_speed_distance(offset_s: float,
                         df_images_base: pd.DataFrame,
                         df_logs: pd.DataFrame) -> float:
    """
    Fallback calibration: measures correlation between drone speed at
    photo-capture time and the GPS distance to the NEXT photo.

    When the offset is correct:
      fast drone  → large inter-photo gap
      slow drone  → small inter-photo gap
    → Pearson correlation is maximised.

    Returns negative correlation (so minimising == finding best offset).
    """
    df_img = _apply_offset(df_images_base, offset_s)
    result, _ = interpolate_gps(df_img, df_logs, verbose=False)
    if len(result) < 5:
        return 0.0

    # Use only mid-flight photos: drone airborne at mission altitude
    # and actually moving. Ground photos corrupt the correlation signal.
    ground_elev  = df_logs['altitude_above_seaLevel(meters)'].iloc[0]  # takeoff altitude
    min_alt_asl  = ground_elev + 50.0    # at least 50 m above ground
    min_speed    = 1.5                   # m/s — ignore hover/landing
    flight_mask  = (
        (result['altitude_above_seaLevel(meters)'] > min_alt_asl) &
        (result['speed(m/s)'] > min_speed)
    )
    result = result[flight_mask].reset_index(drop=True)

    if len(result) < 10:
        return 0.0

    lats   = result['latitude'].values
    lons   = result['longitude'].values
    speeds = result['speed(m/s)'].values

    d_lat  = np.diff(lats) * 111_111
    d_lon  = np.diff(lons) * 111_111 * np.cos(np.radians(np.mean(lats)))
    dists  = np.sqrt(d_lat**2 + d_lon**2)
    avg_sp = (speeds[:-1] + speeds[1:]) / 2

    if np.std(dists) < 1e-6 or np.std(avg_sp) < 1e-6:
        return 0.0
    corr = np.corrcoef(avg_sp, dists)[0, 1]
    return -float(corr) if not np.isnan(corr) else 0.0


def auto_calibrate(df_images_base: pd.DataFrame,
                   df_logs: pd.DataFrame,
                   isphoto_rows: pd.DataFrame,
                   manual_offset: float) -> float:
    """
    Two-pass offset search:
      Pass 1 (coarse): scan ±CALIBRATION_SEARCH_WINDOW in 0.5 s steps
      Pass 2 (fine):   scan ±2 s around coarse best in 0.05 s steps

    Uses isPhoto score when available (≥2 triggers), otherwise
    falls back to speed-distance correlation.
    """
    use_isphoto = len(isphoto_rows) >= 2
    method_name = "isPhoto anchor" if use_isphoto else "speed-distance correlation"
    print(f"  calibration method : {method_name}")

    # Python ternary + lambda requires explicit parentheses on each branch
    if use_isphoto:
        score_fn = lambda o: score_isphoto(o, df_images_base, df_logs, isphoto_rows)
    else:
        score_fn = lambda o: score_speed_distance(o, df_images_base, df_logs)

    # ── coarse pass ──────────────────────────────────────
    coarse_offsets = np.arange(
        manual_offset - CALIBRATION_SEARCH_WINDOW,
        manual_offset + CALIBRATION_SEARCH_WINDOW + CALIBRATION_STEP_COARSE,
        CALIBRATION_STEP_COARSE
    )
    coarse_scores = [score_fn(o) for o in coarse_offsets]
    best_coarse   = coarse_offsets[int(np.argmin(coarse_scores))]

    # ── fine pass ────────────────────────────────────────
    fine_offsets = np.arange(
        best_coarse - 2.0,
        best_coarse + 2.0 + CALIBRATION_STEP_FINE,
        CALIBRATION_STEP_FINE
    )
    fine_scores = [score_fn(o) for o in fine_offsets]
    best_fine   = fine_offsets[int(np.argmin(fine_scores))]

    delta = best_fine - manual_offset
    print(f"  manual offset      : {manual_offset:.1f} s")
    print(f"  auto-calibrated    : {best_fine:.2f} s  (Δ = {delta:+.2f} s)")
    if use_isphoto:
        print(f"  best anchor error  : {min(fine_scores):.1f} m  "
              f"(across {len(isphoto_rows)} DJI trigger events)")

    # Safety check: speed-distance correlation works poorly for distance-triggered
    # surveys (inter-photo distance is constant → correlation signal is weak →
    # calibration may find a random local minimum far from the true offset).
    # If the result deviates more than 8 s from the manual guess without isPhoto
    # anchors, the signal is unreliable — fall back to the manual offset.
    MAX_AUTO_DEVIATION_S = 8.0
    if not use_isphoto and abs(delta) > MAX_AUTO_DEVIATION_S:
        print(f"\n  ⚠  Auto-calibration deviated {delta:+.2f} s from manual guess.")
        print(f"     Speed-distance correlation may be unreliable for this flight")
        print(f"     pattern (distance-triggered waypoint survey).")
        print(f"     ➜  Falling back to manual offset: {manual_offset:.1f} s")
        return manual_offset

    return best_fine


# ============================================================
# PER-FLIGHT HELPERS
# ============================================================

def _load_single_log(log_path: str) -> pd.DataFrame:
    """Load one DJI Airdata CSV and compute exact_time."""
    df = pd.read_csv(log_path)
    df.columns = df.columns.str.strip()
    df['exact_time'] = (
        pd.to_datetime(df['datetime(utc)'])
        + pd.to_timedelta(df['time(millisecond)'] % 1000, unit='ms')
    )
    return (df.sort_values('exact_time')
              .dropna(subset=['latitude', 'longitude'])
              .astype({'exact_time': 'datetime64[ns]'})
              .reset_index(drop=True))


def _filter_images_for_log(df_images_base: pd.DataFrame,
                            gps_start, gps_end,
                            manual_offset: float,
                            margin_s: float = 120.0,
                            assigned: set = None) -> pd.DataFrame:
    """
    Return images whose camera-time + offset falls within the GPS window ± margin.
    'assigned' is a set of image_names already claimed by an earlier flight.
    """
    df = df_images_base.copy()
    df['exact_time'] = (
        df['base_time']
        + pd.to_timedelta(df['subsec'], unit='s')
        + pd.to_timedelta(manual_offset, unit='s')
    ).astype('datetime64[ns]')

    lo = pd.Timestamp(gps_start) - pd.Timedelta(seconds=margin_s)
    hi = pd.Timestamp(gps_end)   + pd.Timedelta(seconds=margin_s)
    mask = (df['exact_time'] >= lo) & (df['exact_time'] <= hi)
    if assigned:
        mask = mask & (~df['image_name'].isin(assigned))
    return df[mask].reset_index(drop=True)


def calc_offset_from_worldfile(tfw_path: str,
                                log_file: str,
                                image_name: str,
                                img_width_px: int = 6000,
                                img_height_px: int = 4000) -> None:
    """
    Helper: given a manually-georeferenced .tfw world file (UTM Zone 36N),
    find when the drone was over that image's centre and print the
    correct CAMERA_CLOCK_OFFSET_PER_FILE entry.

    Usage:
        calc_offset_from_worldfile(
            "/path/to/DSC08600.tfw",
            "/path/to/March-7-2021-13-15-59-Flight-Airdata.csv",
            "DSC08600.JPG",
        )
    """
    import math

    # Parse world file
    with open(tfw_path) as f:
        vals = [float(ln.strip()) for ln in f if ln.strip()]
    px_x, _, _, px_y, ul_e, ul_n = vals  # px_y is negative (north-up)

    center_e = ul_e + (img_width_px  / 2) * px_x
    center_n = ul_n - (img_height_px / 2) * abs(px_y)

    # Inverse UTM Zone 36N → WGS84 (series expansion, accurate to ~1 m)
    a  = 6_378_137.0
    f  = 1 / 298.257_223_563
    b  = a * (1 - f)
    e2 = 1 - (b / a) ** 2
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    k0 = 0.9996

    x = center_e - 500_000
    y = center_n
    M = y / k0
    mu = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))

    phi1 = (mu
            + (3*e1/2   - 27*e1**3/32)  * math.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu)
            + (151*e1**3/96)              * math.sin(6*mu)
            + (1097*e1**4/512)            * math.sin(8*mu))

    N1 = a / math.sqrt(1 - e2 * math.sin(phi1)**2)
    T1 = math.tan(phi1)**2
    C1 = (e2/(1-e2)) * math.cos(phi1)**2
    R1 = a * (1-e2) / (1 - e2*math.sin(phi1)**2)**1.5
    D  = x / (N1 * k0)

    lat = math.degrees(
        phi1 - (N1*math.tan(phi1)/R1) *
        (D**2/2 - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*e2/(1-e2)) * D**4/24)
    )
    lon = 33.0 + math.degrees(
        (D - (1 + 2*T1 + C1) * D**3/6) / math.cos(phi1)
    )
    print(f"  World-file image centre: lat={lat:.6f}, lon={lon:.6f}")

    # Load GPS log and find nearest drone position
    df_log = _load_single_log(log_file)

    dlat = (df_log['latitude']  - lat) * 111_111
    dlon = (df_log['longitude'] - lon) * 111_111 * math.cos(math.radians(lat))
    dist = (dlat**2 + dlon**2).apply(math.sqrt)
    nearest = df_log.loc[dist.idxmin()]
    gps_time = nearest['exact_time']
    print(f"  Nearest drone GPS time : {gps_time}  (dist={dist.min():.1f} m)")

    # Read EXIF time of the image
    img_path = os.path.join(IMAGE_FOLDER, image_name)
    exif = piexif.load(img_path)
    dt_bytes = exif['Exif'].get(piexif.ExifIFD.DateTimeOriginal, b'')
    cam_dt = datetime.strptime(dt_bytes.decode().strip('\x00'), "%Y:%m:%d %H:%M:%S")
    cam_dt += timedelta(hours=CAMERA_TIMEZONE_OFFSET_HOURS)
    ss_bytes = exif['Exif'].get(piexif.ExifIFD.SubSecTimeOriginal, b'0')
    subsec = float("0." + ss_bytes.decode().strip('\x00')) if ss_bytes else 0.0
    cam_time = cam_dt + timedelta(seconds=subsec)

    computed_offset = (pd.Timestamp(gps_time) - pd.Timestamp(cam_time)).total_seconds()
    log_basename = os.path.basename(log_file)
    print(f"\n  Camera EXIF time       : {cam_time}")
    print(f"  Computed offset        : {computed_offset:.1f} s")
    print(f"\n  ➜  Add to CAMERA_CLOCK_OFFSET_PER_FILE:")
    print(f'     "{log_basename}": {computed_offset:.1f},')


def calc_offset_from_metashape_csv(metashape_csv: str,
                                    log_file: str,
                                    max_dist_m: float = 30.0,
                                    min_speed_mps: float = 1.5) -> None:
    """
    Use Metashape's camera-position CSV (lat, lon, alt, filename — no header)
    to compute the camera clock offset for a given flight log.

    For every image in metashape_csv:
      1. Read camera EXIF time from the photo file.
      2. Find the GPS log row nearest (in space) to the Metashape position.
      3. Compute candidate_offset = gps_time - exif_time.

    Takes the median over all images that are:
      - Moving (drone speed > min_speed_mps, filters parked/hovering frames)
      - Within max_dist_m of the nearest log point (filters Metashape errors)
      - Inside this log's time window (after applying the global manual guess)

    Usage:
        calc_offset_from_metashape_csv(
            "/path/to/RGB_coordinates_information.csv",
            "/path/to/March-7-2021-12-56-05-Flight-Airdata.csv",
        )
    """
    import math

    print(f"\n{'='*60}")
    print(f"  Metashape-CSV calibration")
    print(f"  log   : {os.path.basename(log_file)}")
    print(f"  csv   : {os.path.basename(metashape_csv)}")
    print(f"{'='*60}")

    # Load GPS log
    df_log = _load_single_log(log_file)
    gps_start = df_log['exact_time'].iloc[0]
    gps_end   = df_log['exact_time'].iloc[-1]

    # Compute per-row speed in log (m/s) for filtering stationary images
    dlat_m = df_log['latitude'].diff().fillna(0)  * 111_111
    dlon_m = df_log['longitude'].diff().fillna(0) * 111_111 * math.cos(math.radians(df_log['latitude'].mean()))
    dt_s   = df_log['exact_time'].diff().dt.total_seconds().fillna(1)
    df_log['_speed'] = ((dlat_m**2 + dlon_m**2)**0.5 / dt_s.replace(0, 1))

    # Load Metashape CSV (lat, lon, alt, filename — no header)
    df_ms = pd.read_csv(metashape_csv, header=None,
                        names=['lat', 'lon', 'alt', 'image_name'])
    df_ms['image_name'] = df_ms['image_name'].str.strip()

    results = []
    skipped_dist, skipped_speed, skipped_exif, skipped_window = 0, 0, 0, 0

    for _, row in df_ms.iterrows():
        img = row['image_name']
        img_path = os.path.join(IMAGE_FOLDER, img)
        if not os.path.exists(img_path):
            skipped_exif += 1
            continue

        # Find nearest GPS log point to Metashape position
        dlat = (df_log['latitude']  - row['lat']) * 111_111
        dlon = (df_log['longitude'] - row['lon']) * 111_111 * math.cos(math.radians(row['lat']))
        dist = (dlat**2 + dlon**2) ** 0.5
        idx  = dist.idxmin()
        nearest     = df_log.loc[idx]
        nearest_dist = dist.loc[idx]

        # Filter: Metashape point too far from any log point → likely Metashape error
        if nearest_dist > max_dist_m:
            skipped_dist += 1
            continue

        # Filter: drone was nearly stationary at that log point → noisy offset
        if nearest['_speed'] < min_speed_mps:
            skipped_speed += 1
            continue

        # Read EXIF time
        try:
            exif = piexif.load(img_path)
            dt_bytes = exif['Exif'].get(piexif.ExifIFD.DateTimeOriginal, b'')
            cam_dt = datetime.strptime(dt_bytes.decode().strip('\x00'), "%Y:%m:%d %H:%M:%S")
            cam_dt += timedelta(hours=CAMERA_TIMEZONE_OFFSET_HOURS)
            ss = exif['Exif'].get(piexif.ExifIFD.SubSecTimeOriginal, b'0')
            subsec = float("0." + ss.decode().strip('\x00')) if ss and ss.decode().strip('\x00').isdigit() else 0.0
            cam_time = cam_dt + timedelta(seconds=subsec)
        except Exception:
            skipped_exif += 1
            continue

        gps_time = nearest['exact_time']
        offset   = (pd.Timestamp(gps_time) - pd.Timestamp(cam_time)).total_seconds()

        # Filter: candidate offset puts this image outside the log window
        adjusted = pd.Timestamp(cam_time) + pd.Timedelta(seconds=offset)
        if adjusted < gps_start or adjusted > gps_end:
            skipped_window += 1
            continue

        results.append({'image': img, 'offset': offset,
                        'dist_m': nearest_dist, 'gps_time': gps_time,
                        'cam_time': cam_time, 'speed': nearest['_speed']})

    print(f"\n  Images processed : {len(df_ms)}")
    print(f"  Used for median  : {len(results)}")
    print(f"  Skipped (dist>{max_dist_m:.0f}m)  : {skipped_dist}")
    print(f"  Skipped (speed<{min_speed_mps:.1f})  : {skipped_speed}")
    print(f"  Skipped (no EXIF)  : {skipped_exif}")
    print(f"  Skipped (window)   : {skipped_window}")

    if not results:
        print("\n  ❌  No usable images found. Check max_dist_m or log file.")
        return

    df_r = pd.DataFrame(results)
    median_off = df_r['offset'].median()
    std_off    = df_r['offset'].std()
    p10, p90   = df_r['offset'].quantile(0.10), df_r['offset'].quantile(0.90)

    print(f"\n  Offset estimate  : {median_off:.1f} s  (median)")
    print(f"  Std deviation    : {std_off:.2f} s")
    print(f"  10th–90th pct    : [{p10:.1f} s, {p90:.1f} s]")

    # Show sample of best matches (closest Metashape points)
    top = df_r.nsmallest(5, 'dist_m')[['image', 'dist_m', 'offset', 'speed']]
    print(f"\n  Best 5 matches (smallest Metashape→log distance):")
    for _, r in top.iterrows():
        print(f"    {r['image']:15s}  dist={r['dist_m']:5.1f}m  offset={r['offset']:+.1f}s  speed={r['speed']:.1f}m/s")

    log_basename = os.path.basename(log_file)
    print(f"\n  ➜  Add to CAMERA_CLOCK_OFFSET_PER_FILE:")
    print(f'     "{log_basename}": {median_off:.1f},')


# ============================================================
# MAIN
# ============================================================

def run_smart_geosync():
    print("=" * 60)
    print("  GeoSync  ·  Auto-Calibrated  ·  Velocity-Corrected")
    print("=" * 60)

    backup_images()

    # ── 1. Read image EXIF timestamps (shared across all flights) ─
    print("\n[1/5] Reading image EXIF timestamps...")
    image_files = sorted([
        f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith('.jpg')
    ])
    print(f"  Found {len(image_files)} JPG files")

    images_data, errors = [], []
    for img in image_files:
        path = os.path.join(IMAGE_FOLDER, img)
        try:
            exif_dict = piexif.load(path)
            exif_ifd  = exif_dict.get("Exif", {})
            dt_bytes  = exif_ifd.get(piexif.ExifIFD.DateTimeOriginal)
            if not dt_bytes:
                errors.append(f"missing DateTimeOriginal: {img}"); continue
            dt = datetime.strptime(dt_bytes.decode().strip('\x00'), "%Y:%m:%d %H:%M:%S")
            subsec = 0.0
            ss = exif_ifd.get(piexif.ExifIFD.SubSecTimeOriginal)
            if ss:
                s = ss.decode().strip('\x00')
                if s.isdigit():
                    subsec = float("0." + s)
            images_data.append({'image_name': img, 'base_time': dt, 'subsec': subsec})
        except Exception as e:
            errors.append(f"EXIF error {img}: {e}")

    if errors:
        print(f"  ⚠  {len(errors)} images with issues (first 3):")
        for e in errors[:3]: print(f"     {e}")
    if not images_data:
        print("[ERROR] No valid images found. Exiting."); return

    df_images_base = pd.DataFrame(images_data).sort_values('base_time').reset_index(drop=True)

    has_subsec = (df_images_base['subsec'] > 0).any()
    if has_subsec:
        print(f"  ✅ SubSecTimeOriginal present")
    else:
        print(f"  ℹ  No SubSecTimeOriginal — distributing evenly within each second")
        df_images_base['count'] = df_images_base.groupby('base_time').cumcount()
        df_images_base['total'] = df_images_base.groupby('base_time')['image_name'].transform('count')
        df_images_base['subsec'] = df_images_base['count'] / df_images_base['total']

    tz_delta = CAMERA_TIMEZONE_OFFSET_HOURS * 3600
    df_images_base['base_time'] = (
        df_images_base['base_time'] + pd.to_timedelta(tz_delta, unit='s')
    )

    # ── 2. Process each flight log separately ─────────────────────
    print("\n[2/5] Processing each flight log separately...")
    all_flight_results  = []
    all_offsets_used    = {}
    assigned_images     = set()   # prevent double-assignment across flights

    for log_idx, log_file in enumerate(LOG_FILES):
        log_name = os.path.basename(log_file)
        print(f"\n  ── Flight {log_idx+1}: {log_name}")

        if not os.path.exists(log_file):
            print(f"     ✗ File not found — skipping."); continue

        df_log    = _load_single_log(log_file)
        gps_start = df_log['exact_time'].iloc[0]
        gps_end   = df_log['exact_time'].iloc[-1]
        print(f"     GPS window : {gps_start.strftime('%H:%M:%S')} → "
              f"{gps_end.strftime('%H:%M:%S')} UTC")
        print(f"     GPS points : {len(df_log):,}")

        isphoto_rows = df_log[df_log['isPhoto'] == 1][
            ['exact_time', 'latitude', 'longitude', 'speed(m/s)']
        ].reset_index(drop=True)
        print(f"     isPhoto    : {len(isphoto_rows)} events")

        # Per-file override or global default
        has_override  = log_name in CAMERA_CLOCK_OFFSET_PER_FILE
        manual_offset = CAMERA_CLOCK_OFFSET_PER_FILE.get(log_name,
                                                          CAMERA_CLOCK_OFFSET_SECONDS)
        if has_override:
            print(f"     offset     : {manual_offset:.1f} s  [MANUAL OVERRIDE — "
                  f"auto-calibration skipped]")

        # Find images for this flight window
        df_flight = _filter_images_for_log(
            df_images_base, gps_start, gps_end, manual_offset,
            assigned=assigned_images
        )
        print(f"     Images     : {len(df_flight)} in window")
        if len(df_flight) < 3:
            print(f"     ⚠  Too few images — skipping this flight."); continue

        # ── 3. Calibrate offset for this flight ───────────────────
        print(f"\n[3/5] Calibrating clock offset for flight {log_idx+1}...")
        if has_override:
            best_offset = manual_offset
        else:
            best_offset = auto_calibrate(df_flight, df_log, isphoto_rows, manual_offset)

        all_offsets_used[log_name] = best_offset

        # Apply final offset
        df_flight = _apply_offset(df_flight, best_offset)

        # ── 4. Interpolate GPS for this flight ────────────────────
        print(f"[4/5] Interpolating GPS for flight {log_idx+1}...")
        flight_result, gaps = interpolate_gps(df_flight, df_log)

        if gaps:
            for gs, ge in gaps:
                dur = (pd.Timestamp(ge) - pd.Timestamp(gs)).total_seconds()
                print(f"     Gap: {pd.Timestamp(gs).strftime('%H:%M:%S')} → "
                      f"{pd.Timestamp(ge).strftime('%H:%M:%S')} ({dur:.0f} s on ground)")

        assigned_images.update(flight_result['image_name'].tolist())
        all_flight_results.append(flight_result)
        print(f"     ✔ {len(flight_result)} images interpolated")

    # ── Combine all flights ───────────────────────────────────────
    if not all_flight_results:
        print("\n[ERROR] No GPS results from any flight. Exiting."); return

    final_images = (
        pd.concat(all_flight_results, ignore_index=True)
          .drop_duplicates(subset=['image_name'])
          .reset_index(drop=True)
    )

    # Merged log for filters (altitude ref, flycState)
    df_logs_all = pd.concat(
        [_load_single_log(f) for f in LOG_FILES if os.path.exists(f)],
        ignore_index=True
    ).sort_values('exact_time').reset_index(drop=True)

    # ── Mission-phase filter ──────────────────────────────────────
    if 'flycState' in final_images.columns:
        mission_mask = final_images['flycState'].str.strip() == 'Waypoint'
        non_mission  = (~mission_mask).sum()
        if non_mission:
            states = final_images.loc[~mission_mask, 'flycState'].value_counts().to_dict()
            print(f"\n  ✈  {non_mission} non-survey images removed "
                  f"(flycState: {states})")
        final_images = final_images[mission_mask].reset_index(drop=True)

    # ── Altitude cross-validation ─────────────────────────────────
    ground_elev     = df_logs_all['altitude_above_seaLevel(meters)'].iloc[0]
    mission_alt_min = ground_elev + 50.0
    airborne_mask   = final_images['altitude_above_seaLevel(meters)'] > mission_alt_min
    ground_excluded = (~airborne_mask).sum()
    if ground_excluded:
        print(f"  ⚠  {ground_excluded} photos excluded — drone on ground "
              f"(alt < {mission_alt_min:.0f} m ASL)")
    final_images = final_images[airborne_mask].reset_index(drop=True)

    total_orig = sum(len(r) for r in all_flight_results)
    skipped    = total_orig - len(final_images)
    print(f"\n  {len(final_images)} survey images,  {skipped} skipped total")

    # ── Velocity correction ───────────────────────────────────────
    dt_s = CAMERA_SHUTTER_LATENCY_MS / 1000.0

    def apply_correction(row):
        v_east  = row['xSpeed(m/s)']
        v_north = row['ySpeed(m/s)']
        lat     = row['latitude']
        dlat    = (v_north * dt_s) / 111_111
        dlon    = (v_east  * dt_s) / (111_111 * math.cos(math.radians(lat)))
        spd     = row['speed(m/s)']
        uncertainty = round(spd * 0.5, 2)
        return pd.Series({
            'lat_final':     lat + dlat,
            'lon_final':     row['longitude'] + dlon,
            'speed_mps':     round(spd, 2),
            'uncertainty_m': uncertainty,
            'flag': 'HIGH' if uncertainty > UNCERTAINTY_FLAG_M else 'OK',
        })

    corrections  = final_images.apply(apply_correction, axis=1)
    final_images = pd.concat([final_images, corrections], axis=1)

    n_high = (final_images['flag'] == 'HIGH').sum()
    if n_high:
        print(f"  ⚠  {n_high} photos flagged HIGH uncertainty")

    # ── 5. Inject EXIF ───────────────────────────────────────────
    print(f"\n[5/5] Injecting GPS EXIF into {len(final_images)} images...")
    success, failed = 0, 0
    for _, row in final_images.iterrows():
        path = os.path.join(IMAGE_FOLDER, row['image_name'])
        ok   = inject_gps_to_jpg(path, row['lat_final'], row['lon_final'],
                                  row['altitude_above_seaLevel(meters)'])
        if ok: success += 1
        else:  failed  += 1

    # ── Export CSV ────────────────────────────────────────────────
    csv_path = os.path.join(IMAGE_FOLDER, "gps_coordinates.csv")
    final_images[[
        'image_name', 'exact_time',
        'lat_final', 'lon_final',
        'altitude_above_seaLevel(meters)',
        'speed_mps', 'uncertainty_m', 'flag'
    ]].rename(columns={
        'lat_final': 'latitude',
        'lon_final': 'longitude',
        'altitude_above_seaLevel(meters)': 'altitude_m',
        'exact_time': 'datetime_utc',
    }).to_csv(csv_path, index=False)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  ✅ Success        : {success} images")
    if failed:          print(f"  ❌ Failed         : {failed} images")
    if ground_excluded: print(f"  🚫 Ground skip    : {ground_excluded}")
    if n_high:          print(f"  ⚠  High uncert.  : {n_high}")
    print(f"\n  Offsets used:")
    for lname, off in all_offsets_used.items():
        tag = " [MANUAL]" if lname in CAMERA_CLOCK_OFFSET_PER_FILE else " [auto]"
        print(f"    {lname}: {off:.2f} s{tag}")
    print(f"  Velocity correction : {CAMERA_SHUTTER_LATENCY_MS} ms latency")
    print(f"  CSV: {csv_path}")
    print("=" * 60)
    print("\n  ArcGIS: GeoTagged Photos to Points | GCS_WGS_1984")
    print("  Tip: colour-code points by 'flag' to see uncertainty")


if __name__ == "__main__":
    # ── Option A: Run full GPS injection pipeline ──────────────────
    # run_smart_geosync()

    # ── Option B: Calibrate offset from Metashape CSV ─────────────
    # Uncomment ONE block at a time, run, note the printed offset,
    # then set it in CAMERA_CLOCK_OFFSET_PER_FILE above.

    METASHAPE_CSV = r"/Users/snirtahasa/Library/Application Support/Claude/local-agent-mode-sessions/e408ffb8-c911-47df-aded-18f2a60fb283/0d13fc78-2d0e-476e-8498-2cd4cf5ef7df/local_5664cf64-bf56-4302-9cf9-305bb48c0faf/uploads/RGB_coordinates_information.csv"

    # Flight 1 calibration:
    calc_offset_from_metashape_csv(
        METASHAPE_CSV,
        LOG_FILES[0],   # March-7-2021-12-56-05-Flight-Airdata.csv
    )

    # Flight 2 calibration:
    calc_offset_from_metashape_csv(
        METASHAPE_CSV,
        LOG_FILES[1],   # March-7-2021-13-15-59-Flight-Airdata.csv
    )

    # ── After calibration: set offsets above and run Option A ──────
    # run_smart_geosync()
