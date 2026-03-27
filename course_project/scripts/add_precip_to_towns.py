"""
add_precip_to_towns.py
----------------------
For each of Connecticut's 169 towns, this script:
  1. Computes the town centroid in a projected CRS (UTM 18N) then converts to
     WGS84 lat/lon for distance matching.
  2. Parses lat/lon from every CSV filename in the ERA5 aggregated_csvs directory
     (filenames like: lat_41_981_lon_-73_62795_time_series_weather.csv).
  3. Finds the nearest grid cell (by cosine-corrected angular distance).
  4. Reads that grid cell's CSV, parses dates robustly, filters to 2015-2023,
     and computes:
       - precip_days_annual : mean annual count of days with total_precipitation
                              >= 0.001 m (1 mm threshold), averaged over 9 years
       - precip_total_annual: mean annual total precipitation in mm,
                              averaged over 9 years
  5. Joins both fields onto the original towns GeoDataFrame and writes:
     ct_towns_enhanced_v2.gpkg (same folder as input).
  6. Prints a diagnostic table: town -> matched grid cell, for manual review.

Study period : 2015-01-01 through 2023-12-31 (9 complete calendar years)
Date format  : YYYY-MM-DD (ISO 8601) — confirmed from file inspection
Precip units : input CSVs are in meters; output precip_total_annual is in mm
Threshold    : 0.001 m (1 mm) per day to count as a precipitation day
"""

import os
import re
import math
import warnings
import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# PATHS - edit if needed
# ---------------------------------------------------------------------------
TOWNS_GPKG = (
    "/Users/jackbienvenuejr/Desktop/Desktop - jack-bienvenue/"
    "Spring2026Classes/STAT5605/STAT5605_AppliedStatII_Spring2026/"
    "course_project/data/ct_towns_enhanced.gpkg"
)

CSV_DIR = "/Volumes/JB_Fortress_L3/college_work/EEC/aggregated_csvs"

OUTPUT_GPKG = (
    "/Users/jackbienvenuejr/Desktop/Desktop - jack-bienvenue/"
    "Spring2026Classes/STAT5605/STAT5605_AppliedStatII_Spring2026/"
    "course_project/data/ct_towns_enhanced_v2.gpkg"
)

# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------
STUDY_START = pd.Timestamp("2015-01-01")
STUDY_END   = pd.Timestamp("2023-12-31")
STUDY_YEARS = 9
PRECIP_THRESHOLD_M = 0.001   # 1 mm expressed in meters
M_TO_MM = 1000.0

# Regex to parse lat and lon from filenames like:
#   lat_41_981_lon_-73_62795_time_series_weather.csv
FNAME_PATTERN = re.compile(
    r"^lat_(?P<lat>[0-9_]+)_lon_(?P<lon>-?[0-9_]+)_time_series_weather\.csv$"
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def parse_coord_from_filename(fname):
    """
    Parse the lat and lon embedded in a CSV filename.
    e.g. lat_41_981_lon_-73_62795_time_series_weather.csv -> (41.981, -73.62795)
    Convention: decimal point was encoded as underscore.
    We split on the FIRST underscore after the integer digits to recover the float.
    """
    m = FNAME_PATTERN.match(fname)
    if m is None:
        return None, None

    def underscore_to_float(s):
        negative = s.startswith("-")
        if negative:
            s = s[1:]
        parts = s.split("_", 1)
        value = float(parts[0]) if len(parts) == 1 else float(parts[0] + "." + parts[1])
        return -value if negative else value

    return underscore_to_float(m.group("lat")), underscore_to_float(m.group("lon"))


def angular_dist_sq(lat1, lon1, lat2, lon2):
    """
    Squared cosine-corrected angular distance - fast proxy for nearest neighbour
    within Connecticut's small spatial extent. No sqrt needed for argmin.
    """
    dlat = lat1 - lat2
    dlon = (lon1 - lon2) * math.cos(math.radians((lat1 + lat2) / 2.0))
    return dlat * dlat + dlon * dlon


def read_csv_robustly(path):
    """
    Read a weather CSV.
    - Tries comma first (confirmed format from file inspection), then tab.
    - Dates are YYYY-MM-DD ISO format (confirmed from file inspection).
    - Raises a descriptive error if parsing fails.
    """
    last_error = None
    for sep in (",", "\t"):
        try:
            df = pd.read_csv(path, sep=sep)
            if "date" not in df.columns or len(df.columns) < 3:
                continue
            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
            return df
        except Exception as e:
            last_error = e
            continue
    raise ValueError(
        f"Could not parse {path}\nLast error: {last_error}"
    )


# ---------------------------------------------------------------------------
# STEP 1 - Load towns; compute centroids in projected CRS, convert to WGS84
# ---------------------------------------------------------------------------
print("Loading ct_towns_enhanced.gpkg ...")
towns = gpd.read_file(TOWNS_GPKG)

# Compute centroids in UTM Zone 18N (EPSG:32618 - appropriate for Connecticut)
# to avoid the "geographic CRS centroid inaccuracy" warning.
# Then reproject to WGS84 for lat/lon matching against CSV filenames.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    centroids_utm   = towns.to_crs(epsg=32618).geometry.centroid
    centroids_wgs84 = centroids_utm.to_crs(epsg=4326)

town_lats = centroids_wgs84.y.values
town_lons = centroids_wgs84.x.values

print(f"  Loaded {len(towns)} towns.")
print(f"  Town GeoDataFrame columns: {list(towns.columns)}\n")

# ---------------------------------------------------------------------------
# STEP 2 - Scan CSV directory, parse grid cell coords from filenames
# ---------------------------------------------------------------------------
print(f"Scanning CSV directory: {CSV_DIR} ...")
all_fnames = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv")]
print(f"  Found {len(all_fnames)} CSV files.")

grid_index = []
skipped = 0
for fname in all_fnames:
    lat, lon = parse_coord_from_filename(fname)
    if lat is None:
        skipped += 1
        continue
    grid_index.append({"fname": fname, "lat": lat, "lon": lon})

print(f"  Parsed {len(grid_index)} filenames ({skipped} skipped due to unexpected naming).\n")

if not grid_index:
    raise RuntimeError(
        "No CSV filenames matched the expected pattern. "
        "Check FNAME_PATTERN against actual filenames in CSV_DIR."
    )

grid_lats = [g["lat"] for g in grid_index]
grid_lons = [g["lon"] for g in grid_index]

# ---------------------------------------------------------------------------
# STEP 3 - Identify the town name column for the diagnostic table
# ---------------------------------------------------------------------------
name_col = None
for candidate in ["TOWN", "town", "NAME", "name", "Town", "Name",
                   "TOWNNAME", "town_name", "TOWN_NAME", "MUNICIPALITY"]:
    if candidate in towns.columns:
        name_col = candidate
        break
if name_col is None:
    # Use first string column that is not geometry and not a pure numeric ID
    for col in towns.columns:
        if col == "geometry":
            continue
        if towns[col].dtype == object and not col.upper().startswith("GEOID"):
            name_col = col
            break

if name_col is None:
    print("  WARNING: Could not identify a town name column. Using row index.")
else:
    print(f"  Using '{name_col}' as the town name column.\n")

# ---------------------------------------------------------------------------
# STEP 4 - Match each town to its nearest grid cell; compute precip statistics
# ---------------------------------------------------------------------------
print("Matching towns to grid cells and computing precipitation statistics ...")
print(f"  (This will read {len(towns)} CSVs - may take a minute)\n")

results        = []
diagnostic_rows = []

for i in range(len(towns)):
    town_lat  = town_lats[i]
    town_lon  = town_lons[i]
    town_name = towns.iloc[i][name_col] if name_col else f"row_{i}"

    # --- Find nearest grid cell ---
    best_j    = min(range(len(grid_index)),
                    key=lambda j: angular_dist_sq(
                        town_lat, town_lon, grid_lats[j], grid_lons[j]))
    best_grid = grid_index[best_j]
    best_dist = angular_dist_sq(town_lat, town_lon, grid_lats[best_j], grid_lons[best_j])

    # --- Read and parse the matched CSV ---
    csv_path = os.path.join(CSV_DIR, best_grid["fname"])
    df = read_csv_robustly(csv_path)

    # --- Filter to study period 2015-2023 ---
    df = df[(df["date"] >= STUDY_START) & (df["date"] <= STUDY_END)].copy()

    # Warn if row count is unexpectedly low (>10 days short of expected ~3287)
    expected_days = 365 * STUDY_YEARS + 2   # leap years 2016, 2020
    if len(df) < expected_days - 10:
        print(f"  WARNING: '{town_name}' — {len(df)} rows in study period "
              f"(expected ~{expected_days}). Check CSV coverage.")

    # --- Per-year aggregation then average across years ---
    df["year"] = df["date"].dt.year
    annual = (
        df.groupby("year")["total_precipitation"]
        .agg(
            precip_days  = lambda x: int((x >= PRECIP_THRESHOLD_M).sum()),
            precip_total = lambda x: float(x.sum())
        )
        .reset_index()
    )

    precip_days_annual  = annual["precip_days"].mean()
    precip_total_annual = annual["precip_total"].mean() * M_TO_MM   # m -> mm

    results.append({
        "precip_days_annual":  round(precip_days_annual,  2),
        "precip_total_annual": round(precip_total_annual, 2),
    })

    diagnostic_rows.append({
        "Town":              town_name,
        "Grid lat":          best_grid["lat"],
        "Grid lon":          best_grid["lon"],
        "Dist (deg^2)":      round(best_dist, 8),
        "Precip days/yr":    round(precip_days_annual,  1),
        "Precip total (mm)": round(precip_total_annual, 1),
    })

print("  Done computing.\n")

# ---------------------------------------------------------------------------
# STEP 5 - Print diagnostic summary table
# ---------------------------------------------------------------------------
diag_df = pd.DataFrame(diagnostic_rows)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 120)
print("=" * 85)
print("DIAGNOSTIC SUMMARY: Town -> Matched Grid Cell")
print("=" * 85)
print(diag_df.to_string(index=False))
print("=" * 85)
print()

# ---------------------------------------------------------------------------
# STEP 6 - Attach new fields to original GeoDataFrame and write output
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
towns["precip_days_annual"]  = results_df["precip_days_annual"].values
towns["precip_total_annual"] = results_df["precip_total_annual"].values

print(f"Writing output GeoPackage to:\n  {OUTPUT_GPKG}\n")
towns.to_file(OUTPUT_GPKG, driver="GPKG")

print("Success! Output GeoPackage written with two new fields:")
print("  precip_days_annual  -- mean annual days with precip >= 1 mm (2015-2023)")
print("  precip_total_annual -- mean annual total precipitation in mm (2015-2023)")