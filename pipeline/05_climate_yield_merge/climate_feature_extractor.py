import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# מילון הממפה כל שנת מחקר לתאריך הטיסה המדויק שלה (YYYY-MM-DD)
# הנתונים יחושבו מה-1 בנובמבר של השנה הקודמת ועד לתאריך זה בדיוק.
FLIGHT_DATES = {
    2021: "2021-03-07",
    2022: "2022-03-02",
    2023: "2023-03-01",
    2024: "2024-02-29"
}

BASE_DIR = "/Users/snirtahasa/Thesis"
CLIMATE_DIR = os.path.join(BASE_DIR, "Climate_Data")
MASTER_EXCEL = os.path.join(BASE_DIR, "4band_mosaic", "Master_Trees_Time_Series_Plot1.xlsx")

COLUMN_TRANSLATION = {
    'תחנה': 'Station',
    'תאריך ושעה (שעון עולמי)': 'Datetime',   # UTC variant
    'תאריך ושעה (שעון קיץ)': 'Datetime',      # DST/summer-time variant
    'קרינה גלובלית (וואט/מ"ר)': 'Global_Radiation',
    'לחות יחסית (%)': 'Relative_Humidity',
    'טמפרטורה (C°)': 'Temperature',
    'טמפרטורת מקסימום (C°)': 'Max_Temperature',
    'טמפרטורת מינימום (C°)': 'Min_Temperature',
    'כמות גשם (מ"מ)': 'Rainfall',
    'מהירות רוח (מטר לשניה)': 'Wind_Speed'
}

# ==========================================
# CORE FUNCTIONS
# ==========================================
def calculate_dynamic_chilling_portions(hourly_temps):
    """Implements the Dynamic Model state machine (Fishman et al., 1987)."""
    A_0, A_1 = 139500, 2.567e18
    E_0, E_1 = 4153.5, 12888.8
    inter_stage_value = 0.0
    portions = 0.0
    
    for T_c in hourly_temps:
        T_k = T_c + 273.15
        k_0 = A_0 * np.exp(-E_0 / T_k)
        k_1 = A_1 * np.exp(-E_1 / T_k)
        
        if k_1 != 0:
            inter_stage_value = inter_stage_value * np.exp(-k_1) + (k_0 / k_1) * (1.0 - np.exp(-k_1))
        else:
            inter_stage_value += k_0
            
        if inter_stage_value >= 1.0:
            portions += 1.0
            inter_stage_value -= 1.0
            
    return portions

def process_climate_year(target_year, end_date_str, df_master):
    """
    מחשב פיצ'רים אגרו-אקלימיים מה-1 בנובמבר ועד לתאריך הטיסה הספציפי.
    """
    print(f"\n--- Processing Year: {target_year} | Cut-off Date: {end_date_str} ---")
    
    weather_file = os.path.join(CLIMATE_DIR, f"{target_year}.csv")
    if not os.path.exists(weather_file):
        print(f"Warning: Data for {target_year} not found. Skipping.")
        return df_master

    start_date_str = f"{target_year - 1}-11-01"

    # 1. Load & Filter (Strictly up to the flight date)
    df_raw = pd.read_csv(weather_file).rename(columns=COLUMN_TRANSLATION)
    if 'Datetime' not in df_raw.columns:
        raise ValueError(f"Datetime column not found. Actual columns: {df_raw.columns.tolist()}")
    df_raw['Datetime'] = pd.to_datetime(df_raw['Datetime'], dayfirst=True)
    
    # חיתוך הדאטה עד סוף יום הטיסה המדויק (23:59:59)
    end_date_filter = pd.to_datetime(end_date_str) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_filtered = df_raw[(df_raw['Datetime'] >= start_date_str) & (df_raw['Datetime'] <= end_date_filter)].copy()

    numeric_columns = ['Temperature', 'Max_Temperature', 'Min_Temperature', 'Relative_Humidity', 'Global_Radiation', 'Rainfall', 'Wind_Speed']
    for col in numeric_columns:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

    # 2. Resampling (10-min to Hourly)
    df_filtered.set_index('Datetime', inplace=True)
    agg_rules = {'Temperature': 'mean'}
    # Dedicated max/min columns may not exist in all CSV exports — fall back
    # to deriving daily max/min from hourly Temperature if they're absent.
    if 'Max_Temperature' in df_filtered.columns: agg_rules['Max_Temperature'] = 'max'
    if 'Min_Temperature' in df_filtered.columns: agg_rules['Min_Temperature'] = 'min'
    if 'Relative_Humidity' in df_filtered.columns: agg_rules['Relative_Humidity'] = 'mean'
    if 'Global_Radiation' in df_filtered.columns: agg_rules['Global_Radiation'] = 'mean'
    if 'Rainfall' in df_filtered.columns: agg_rules['Rainfall'] = 'sum'
    if 'Wind_Speed' in df_filtered.columns: agg_rules['Wind_Speed'] = 'mean'

    df_hourly = df_filtered.resample('1h').agg(agg_rules).dropna(subset=['Temperature']).reset_index()
    df_hourly['Date'] = df_hourly['Datetime'].dt.date
    df_hourly['Month'] = df_hourly['Datetime'].dt.month

    # 3. Calculate Features
    hourly_temps = df_hourly['Temperature'].to_numpy()
    chill_portions = calculate_dynamic_chilling_portions(hourly_temps)
    chill_hours_simple = np.sum(hourly_temps <= 7.0)
    
    df_jan_feb_hourly = df_hourly[df_hourly['Month'].isin([1, 2])]
    late_chill_portions = calculate_dynamic_chilling_portions(df_jan_feb_hourly['Temperature'].to_numpy()) if not df_jan_feb_hourly.empty else 0

    # GDD Calculation
    # Use dedicated Max/Min columns if available; otherwise derive from hourly Temperature.
    daily_agg = df_hourly.groupby('Date')
    if 'Max_Temperature' in df_hourly.columns and 'Min_Temperature' in df_hourly.columns:
        daily_stats = daily_agg.agg(
            Daily_Max=('Max_Temperature', 'max'),
            Daily_Min=('Min_Temperature', 'min'),
            Month=('Month', 'first')
        ).reset_index()
    else:
        daily_stats = daily_agg.agg(
            Daily_Max=('Temperature', 'max'),
            Daily_Min=('Temperature', 'min'),
            Month=('Month', 'first')
        ).reset_index()
    
    daily_stats['GDD_Base'] = ((daily_stats['Daily_Max'] + daily_stats['Daily_Min']) / 2) - 4.5
    daily_stats['GDD_Base'] = np.clip(daily_stats['GDD_Base'], 0, None)
    daily_stats['DTR'] = daily_stats['Daily_Max'] - daily_stats['Daily_Min']

    gdd_feb_only = daily_stats[daily_stats['Month'] == 2]['GDD_Base'].sum()
    gdd_jan_feb = daily_stats[daily_stats['Month'].isin([1, 2])]['GDD_Base'].sum()
    avg_dtr_feb = daily_stats[daily_stats['Month'] == 2]['DTR'].mean()

    df_feb_hourly = df_hourly[df_hourly['Month'] == 2]
    cumulative_radiation_feb = (df_feb_hourly['Global_Radiation'].sum() * 3600) / 1e6 if 'Global_Radiation' in df_feb_hourly.columns else 0
    
    is_dry = df_feb_hourly['Relative_Humidity'] < 35.0
    max_consecutive_dry_hours = is_dry.groupby((~is_dry).cumsum()).sum().max() if not is_dry.empty else 0

    transition_ratio = chill_portions / gdd_feb_only if gdd_feb_only > 0 else 0
    frost_hours = np.sum(hourly_temps <= 0.0)
    heat_stress_hours_feb = np.sum(df_feb_hourly['Temperature'] > 25.0) if not df_feb_hourly.empty else 0
    
    total_rain_feb = df_feb_hourly['Rainfall'].sum() if 'Rainfall' in df_feb_hourly.columns else 0
    high_wind_hours_feb = np.sum(df_feb_hourly['Wind_Speed'] > 5.0) if 'Wind_Speed' in df_feb_hourly.columns else 0

    print(f" -> Chill Portions: {chill_portions:.2f} | Late Chill: {late_chill_portions:.2f} | GDD Feb: {gdd_feb_only:.2f}")

    # 4. Data Injection
    df_master[f"Chill_Portions_{target_year}"] = round(chill_portions, 2)
    df_master[f"Late_Chill_{target_year}"] = round(late_chill_portions, 2)
    df_master[f"Chill_Hrs_{target_year}"] = int(chill_hours_simple)
    df_master[f"GDD_Feb_{target_year}"] = round(gdd_feb_only, 2)
    df_master[f"GDD_Jan_Feb_{target_year}"] = round(gdd_jan_feb, 2)
    df_master[f"Avg_DTR_Feb_{target_year}"] = round(avg_dtr_feb, 2)
    df_master[f"Rad_Feb_{target_year}"] = round(cumulative_radiation_feb, 2)
    df_master[f"Max_Dry_Hrs_{target_year}"] = int(max_consecutive_dry_hours)
    df_master[f"Trans_Ratio_{target_year}"] = round(transition_ratio, 3)
    df_master[f"Frost_Hrs_{target_year}"] = int(frost_hours)
    df_master[f"Heat_Stress_Feb_{target_year}"] = int(heat_stress_hours_feb)
    df_master[f"Rain_Feb_{target_year}"] = round(total_rain_feb, 1)
    df_master[f"High_Wind_Hrs_{target_year}"] = int(high_wind_hours_feb)

    return df_master

# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
if os.path.exists(MASTER_EXCEL):
    df_master_trees = pd.read_excel(MASTER_EXCEL)
    
    # לולאה על המילון: לוקחת את השנה ואת תאריך החיתוך שלה
    for target_year, flight_date in FLIGHT_DATES.items():
        df_master_trees = process_climate_year(target_year, flight_date, df_master_trees)
        
    df_master_trees.to_excel(MASTER_EXCEL, index=False)
    print("\n" + "=" * 70)
    print("SUCCESS: Unified Agro-Climatic Pipeline completed dynamically per flight date!")
    print("=" * 70)
else:
    print(f"CRITICAL ERROR: '{MASTER_EXCEL}' not found. Please verify the path.")