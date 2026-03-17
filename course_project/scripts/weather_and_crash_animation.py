"""
ct_precip_crash_animation.py  [v4]
────────────────────────────
Changes from v3:
  - 15-minute sub-frames restored (INTERP_FRAMES = 3 → 4 frames per hour).
    Each real hourly grid is linearly blended with the next, giving smooth
    transitions without generating new data.
  - Slower frame rate (FPS = 3 by default → each real hour spans ~1.3 s).
  - Robust grid repair pass BEFORE rendering:
      Step 1 – Classify every hourly grid as PINK, WHITE, or OK.
               PINK  = grid max > PINK_THRESHOLD  (RBF blowup)
               WHITE = grid max < WHITE_THRESHOLD (all-dry / near-zero)
               OK    = anything in between
      Step 2 – Repair PINK grids:
               Look at the prior and next OK-or-WHITE grid.
               If next is WHITE and the grid TWO steps ahead is also WHITE
               (or all grids ahead are WHITE), replace pink with WHITE (zero).
               Otherwise replace with linear blend of prev-OK and next-OK.
      Step 3 – Repair suspicious WHITE grids that are adjacent to a PINK:
               A WHITE grid that immediately follows a PINK (before repair)
               and is sandwiched between the repaired PINK and a genuine
               OK-or-WHITE grid is left as-is (true dry period).
               A WHITE grid that sits between a genuine storm frame and a
               repaired-from-PINK frame is also left alone.
               Only WHITE grids that are isolated artefacts adjacent to PINK
               are scrutinised; all others are accepted as real dry hours.

Usage:
    python ct_precip_crash_animation.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import imageio

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

WEATHER_DIR   = "/Volumes/JB_Fortress_L3/college_work/EEC/merged_csvs/"
CRASH_CSV     = (
    "/Users/jackbienvenuejr/Desktop/Desktop - jack-bienvenue/Spring2026Classes/"
    "STAT5605/STAT5605_AppliedStatII_Spring2026/course_project/data/"
    "clean_crash_data/crash_data_weather.csv"
)
CT_TOWNS_GPKG = (
    "/Users/jackbienvenuejr/Desktop/Desktop - jack-bienvenue/Spring2026Classes/"
    "STAT5605/STAT5605_AppliedStatII_Spring2026/course_project/data/"
    "ct_towns_enhanced.gpkg"
)

ANIM_START = "2019-12-09 06:00"
ANIM_END   = "2019-12-14 05:00"

OUT_DIR   = "animation_output"
MP4_PATH  = os.path.join(OUT_DIR, "ct_storm_animation.mp4")
GIF_PATH  = os.path.join(OUT_DIR, "ct_storm_animation.gif")
FRAME_DIR = os.path.join(OUT_DIR, "frames")

# ── Playback ──────────────────────────────────────────────────────────────────
INTERP_FRAMES    = 3       # sub-frames between each real hourly grid (→ 15-min equiv.)
FPS              = 3       # output frames-per-second  (lower = slower)
                           # at FPS=3 and INTERP_FRAMES=3: each real hour ≈ 1.33 s
BLACK_END_FRAMES = 6

# ── Render quality ────────────────────────────────────────────────────────────
DPI      = 160
GRID_RES = 200

# ── Precipitation colour scale ────────────────────────────────────────────────
PRECIP_VMAX_MM = 2.5    # mm/hr at which colour saturates

# ── Frame classification thresholds (in mm/hr after conversion) ──────────────
# Any hourly grid whose mean-over-CT exceeds PINK_THRESHOLD is a blowup.
# Any hourly grid whose max-over-CT is below WHITE_THRESHOLD is all-dry.
PINK_THRESHOLD  = 8.0    # mm/hr mean  — well above any plausible CT storm avg
WHITE_THRESHOLD = 0.005  # mm/hr max   — essentially zero

# ── Interpolation guard ───────────────────────────────────────────────────────
M_TO_MM           = 1000.0
RBF_BLOWUP_FACTOR = 3.0
CLAMP_PERCENTILE  = 99

# ── Crash markers ─────────────────────────────────────────────────────────────
CRASH_COLOR        = "#C0392B"
CRASH_MARKER_SZ    = 18
CRASH_MARKER_ALPHA = 0.50
RING_COLOR         = "#FF6F00"
RING_SZ            = 90
RING_ALPHA         = 0.85
RING_LINEWIDTH     = 1.2

# ══════════════════════════════════════════════════════════════════════════════
# RADAR COLOUR MAP
# ══════════════════════════════════════════════════════════════════════════════
RADAR_COLORS = [
    (0.000, (1.00, 1.00, 1.00, 0.00)),
    (0.040, (0.56, 0.93, 0.56, 0.50)),
    (0.150, (0.13, 0.70, 0.13, 0.72)),
    (0.300, (0.60, 0.90, 0.00, 0.82)),
    (0.500, (1.00, 0.90, 0.00, 0.88)),
    (0.700, (1.00, 0.55, 0.00, 0.92)),
    (0.870, (0.90, 0.10, 0.10, 0.95)),
    (1.000, (0.75, 0.00, 0.75, 0.97)),
]
radar_cmap = LinearSegmentedColormap.from_list(
    "nws_radar", [(v, c) for v, c in RADAR_COLORS], N=512
)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CT BOUNDARY
# ══════════════════════════════════════════════════════════════════════════════
print("Loading CT boundary …")
try:
    ct_gdf = gpd.read_file(CT_TOWNS_GPKG)
except Exception:
    print("  gpkg not found — downloading Census boundary …")
    ct_gdf = gpd.read_file(
        "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_09_cousub_500k.zip"
    )

ct_gdf   = ct_gdf.to_crs(epsg=4326)
ct_union = ct_gdf.union_all()

CT_LON_MIN, CT_LAT_MIN, CT_LON_MAX, CT_LAT_MAX = ct_union.bounds
PAD     = 0.12
LON_MIN = CT_LON_MIN - PAD;  LON_MAX = CT_LON_MAX + PAD
LAT_MIN = CT_LAT_MIN - PAD;  LAT_MAX = CT_LAT_MAX + PAD

# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOAD WEATHER
# ══════════════════════════════════════════════════════════════════════════════
print("Loading weather CSVs …")

anim_start = pd.Timestamp(ANIM_START)
anim_end   = pd.Timestamp(ANIM_END)
hours      = pd.date_range(anim_start, anim_end, freq="h")

weather_files = glob.glob(os.path.join(WEATHER_DIR, "*.csv"))
records = []

for fp in tqdm(weather_files, desc="  Parsing stations", unit="file"):
    try:
        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip().str.lower()
        if not {"time", "tp", "latitude", "longitude"}.issubset(df.columns):
            continue

        df["time"] = pd.to_datetime(df["time"], infer_datetime_format=True)
        lat = df["latitude"].iloc[0]
        lon = df["longitude"].iloc[0]

        df = df[
            (df["time"] >= (anim_start - pd.Timedelta(hours=2))) &
            (df["time"] <= (anim_end   + pd.Timedelta(hours=1)))
        ].copy()
        if df.empty:
            continue

        df = df.sort_values("time")
        df["date"] = df["time"].dt.date

        def daily_delta(grp):
            grp = grp.sort_values("time").copy()
            grp["precip_hr_m"] = grp["tp"].diff().clip(lower=0)
            grp.loc[grp.index[0], "precip_hr_m"] = max(grp["tp"].iloc[0], 0)
            return grp

        df = df.groupby("date", group_keys=False).apply(daily_delta)
        df["precip_hr_mm"] = (df["precip_hr_m"] * M_TO_MM).clip(lower=0).fillna(0)

        for _, row in df.iterrows():
            records.append({
                "time":      row["time"],
                "lat":       lat,
                "lon":       lon,
                "precip_hr": row["precip_hr_mm"],
            })
    except Exception:
        pass

weather_long = pd.DataFrame(records)
weather_long = weather_long[
    (weather_long["time"] >= anim_start) &
    (weather_long["time"] <= anim_end)
]
print(f"  {weather_long['time'].nunique()} unique hours  |  "
      f"max station hourly: {weather_long['precip_hr'].max():.4f} mm/hr")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  LOAD CRASHES
# ══════════════════════════════════════════════════════════════════════════════
print("Loading crash data …")

crash_df = pd.read_csv(CRASH_CSV)
crash_df.columns = crash_df.columns.str.strip()
crash_df["crash_dt"] = (
    pd.to_datetime(crash_df["Date Of Crash"], infer_datetime_format=True)
    + pd.to_timedelta(crash_df["Hour of the Day"].astype(int), unit="h")
)
crash_window = crash_df[
    (crash_df["crash_dt"] >= anim_start) &
    (crash_df["crash_dt"] <= anim_end) &
    crash_df["Latitude"].notna() &
    crash_df["Longitude"].notna()
].copy()
print(f"  {len(crash_window):,} crashes in window")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  INTERPOLATION GRID & CT MASK
# ══════════════════════════════════════════════════════════════════════════════
lon_grid     = np.linspace(LON_MIN, LON_MAX, GRID_RES)
lat_grid     = np.linspace(LAT_MIN, LAT_MAX, GRID_RES)
LON_G, LAT_G = np.meshgrid(lon_grid, lat_grid)
grid_points  = np.column_stack([LAT_G.ravel(), LON_G.ravel()])

from shapely.vectorized import contains as shp_contains
ct_mask = shp_contains(
    ct_union, LON_G.ravel(), LAT_G.ravel()
).reshape(GRID_RES, GRID_RES)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  INTERPOLATION  (RBF + IDW fallback + clamp)
# ══════════════════════════════════════════════════════════════════════════════

def idw_interpolate(pts, vals, query, power=2.0):
    dists   = np.sqrt(((query[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    dists   = np.maximum(dists, 1e-10)
    weights = 1.0 / dists ** power
    return (weights * vals[None, :]).sum(axis=1) / weights.sum(axis=1)


def interpolate_precip(hour_df):
    pts  = hour_df[["lat", "lon"]].values
    vals = hour_df["precip_hr"].values.astype(float)

    if len(pts) < 4 or vals.max() < 1e-6:
        return np.zeros((GRID_RES, GRID_RES))

    station_max = vals.max()
    clamp_val   = np.percentile(vals, CLAMP_PERCENTILE)
    use_idw     = False

    try:
        rbf       = RBFInterpolator(pts, vals, kernel="thin_plate_spline",
                                    smoothing=0.5, degree=1)
        grid_vals = rbf(grid_points).reshape(GRID_RES, GRID_RES)
        grid_vals = np.clip(grid_vals, 0, None)
        if grid_vals.max() > RBF_BLOWUP_FACTOR * station_max:
            use_idw = True
    except Exception:
        use_idw = True

    if use_idw:
        grid_vals = idw_interpolate(pts, vals, grid_points).reshape(GRID_RES, GRID_RES)
        grid_vals = np.clip(grid_vals, 0, None)

    grid_vals = np.clip(grid_vals, 0, clamp_val)
    grid_vals = gaussian_filter(grid_vals, sigma=GRID_RES / 40)

    return grid_vals

# ══════════════════════════════════════════════════════════════════════════════
# 6.  COMPUTE RAW HOURLY GRIDS
# ══════════════════════════════════════════════════════════════════════════════
print("Interpolating raw hourly grids …")

raw_grids = {}
for ts, grp in tqdm(weather_long.groupby("time"), desc="  Interpolating", unit="hr"):
    raw_grids[ts] = interpolate_precip(grp)

for ts in hours:
    if ts not in raw_grids:
        raw_grids[ts] = np.zeros((GRID_RES, GRID_RES))

# ══════════════════════════════════════════════════════════════════════════════
# 7.  CLASSIFY & REPAIR GRIDS
#
#  Classification (applied only to CT-masked pixels):
#    PINK  = mean of CT pixels > PINK_THRESHOLD  → blowup, must repair
#    WHITE = max  of CT pixels < WHITE_THRESHOLD → all-dry, may be real or artefact
#    OK    = everything else                     → good data, keep as-is
#
#  Repair rules:
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │ Rule 1 (PINK → blend):                                              │
#  │   Find nearest OK frame before (prev_ok) and after (next_ok).      │
#  │   Replace PINK with linear blend: 0.5*prev_ok + 0.5*next_ok.       │
#  │   If no prev_ok exists, use next_ok alone.                         │
#  │   If no next_ok exists, use prev_ok alone.                         │
#  │                                                                     │
#  │ Rule 2 (PINK surrounded only by WHITE):                             │
#  │   If prev_ok and next_ok are both absent (all neighbours are WHITE  │
#  │   or PINK), replace PINK with zeros (WHITE).                        │
#  │                                                                     │
#  │ Rule 3 (WHITE adjacent to original PINK):                           │
#  │   A WHITE frame whose immediate neighbour was originally PINK is    │
#  │   inspected. If the neighbour on the OTHER side of the WHITE frame  │
#  │   is also WHITE (or absent), the WHITE frame is accepted as real.   │
#  │   This avoids wrongly zeroing out genuine dry periods that happen   │
#  │   to sit next to a repaired blowup.                                 │
#  └─────────────────────────────────────────────────────────────────────┘
# ══════════════════════════════════════════════════════════════════════════════
print("Classifying and repairing hourly grids …")

n_hours  = len(hours)
ts_list  = list(hours)

def ct_mean(grid):
    return grid[ct_mask].mean() if ct_mask.any() else 0.0

def ct_max(grid):
    return grid[ct_mask].max() if ct_mask.any() else 0.0

# --- Classify ---
labels = []   # 'PINK', 'WHITE', 'OK'
for ts in ts_list:
    g = raw_grids[ts]
    if ct_mean(g) > PINK_THRESHOLD:
        labels.append("PINK")
    elif ct_max(g) < WHITE_THRESHOLD:
        labels.append("WHITE")
    else:
        labels.append("OK")

original_labels = labels.copy()   # keep a record for Rule 3

pink_count = labels.count("PINK")
print(f"  Before repair: {pink_count} PINK, {labels.count('WHITE')} WHITE, "
      f"{labels.count('OK')} OK frames")

# --- Repair PINK frames (Rule 1 & 2) ---
repaired_grids = {ts: raw_grids[ts].copy() for ts in ts_list}

for i, ts in enumerate(ts_list):
    if labels[i] != "PINK":
        continue

    # Find nearest OK frame before this index
    prev_ok_idx = None
    for j in range(i - 1, -1, -1):
        if labels[j] == "OK":
            prev_ok_idx = j
            break

    # Find nearest OK frame after this index
    next_ok_idx = None
    for j in range(i + 1, n_hours):
        if labels[j] == "OK":
            next_ok_idx = j
            break

    if prev_ok_idx is None and next_ok_idx is None:
        # Surrounded entirely by WHITE/PINK → zero out (Rule 2)
        repaired_grids[ts] = np.zeros((GRID_RES, GRID_RES))
        labels[i] = "WHITE"
    elif prev_ok_idx is None:
        repaired_grids[ts] = repaired_grids[ts_list[next_ok_idx]].copy()
        labels[i] = "OK"
    elif next_ok_idx is None:
        repaired_grids[ts] = repaired_grids[ts_list[prev_ok_idx]].copy()
        labels[i] = "OK"
    else:
        # Blend prev and next OK grids
        prev_grid = repaired_grids[ts_list[prev_ok_idx]]
        next_grid = repaired_grids[ts_list[next_ok_idx]]
        # Weight by temporal proximity
        total_gap  = next_ok_idx - prev_ok_idx
        w_next     = (i - prev_ok_idx) / total_gap
        w_prev     = 1.0 - w_next
        repaired_grids[ts] = w_prev * prev_grid + w_next * next_grid
        labels[i] = "OK"

# --- Rule 3: inspect WHITE frames adjacent to original PINK ---
# These are already fine in repaired_grids (they retain their raw zeros).
# We just log them for transparency rather than modifying them.
suspicious_white = []
for i, ts in enumerate(ts_list):
    if labels[i] != "WHITE":
        continue
    # Was a neighbour originally PINK?
    left_orig  = original_labels[i - 1] if i > 0 else None
    right_orig = original_labels[i + 1] if i < n_hours - 1 else None
    if left_orig == "PINK" or right_orig == "PINK":
        # Check the far neighbour
        far_left  = original_labels[i - 2] if i > 1 else None
        far_right = original_labels[i + 2] if i < n_hours - 2 else None
        # Accept as real WHITE if the far neighbour is also WHITE (genuine dry gap)
        if (far_left in ("WHITE", None)) or (far_right in ("WHITE", None)):
            suspicious_white.append((i, ts, "accepted as genuine dry"))
        else:
            suspicious_white.append((i, ts, "kept as-is (surrounded by data)"))

repaired_count = sum(1 for o, n in zip(original_labels, labels) if o == "PINK" and n != "PINK")
print(f"  After repair : {labels.count('PINK')} PINK, {labels.count('WHITE')} WHITE, "
      f"{labels.count('OK')} OK frames")
print(f"  Repaired {repaired_count} PINK frames")
if suspicious_white:
    print(f"  {len(suspicious_white)} WHITE frames adjacent to original PINK (all accepted):")
    for idx, ts, note in suspicious_white:
        print(f"    [{idx:03d}] {ts}  → {note}")

# ══════════════════════════════════════════════════════════════════════════════
# 7b.  INTERPOLATE WHITE FRAMES SANDWICHED BETWEEN OK FRAMES
#
#  This pass runs AFTER all PINK repairs are complete so that any PINK→OK
#  conversions above are already baked into `labels` before we decide whether
#  a WHITE frame is truly surrounded by data on both sides.
#
#  A WHITE frame is filled if and only if:
#    - There is at least one OK frame somewhere before it  (prev_ok_idx), AND
#    - There is at least one OK frame somewhere after  it  (next_ok_idx).
#
#  The fill value is a temporally weighted blend of those two OK anchors,
#  using the same proximity weighting as the PINK repair above.
#
#  WHITE frames at the leading or trailing edge of the window (no OK on one
#  side) are left as zeros — they are genuine dry periods, not artefacts.
#  Labels are updated to "OK" for every frame that gets filled.
# ══════════════════════════════════════════════════════════════════════════════
print("Interpolating WHITE frames sandwiched between OK frames …")

white_filled = 0

for i, ts in enumerate(ts_list):
    if labels[i] != "WHITE":
        continue

    # Nearest OK frame before i
    prev_ok_idx = None
    for j in range(i - 1, -1, -1):
        if labels[j] == "OK":
            prev_ok_idx = j
            break

    # Nearest OK frame after i
    next_ok_idx = None
    for j in range(i + 1, n_hours):
        if labels[j] == "OK":
            next_ok_idx = j
            break

    # Only fill if OK anchors exist on BOTH sides
    if prev_ok_idx is None or next_ok_idx is None:
        continue

    prev_grid = repaired_grids[ts_list[prev_ok_idx]]
    next_grid = repaired_grids[ts_list[next_ok_idx]]
    total_gap = next_ok_idx - prev_ok_idx
    w_next    = (i - prev_ok_idx) / total_gap
    w_prev    = 1.0 - w_next

    repaired_grids[ts] = w_prev * prev_grid + w_next * next_grid
    labels[i] = "OK"
    white_filled += 1

print(f"  Filled {white_filled} WHITE frame(s) sandwiched between OK frames")
print(f"  Final labels : {labels.count('PINK')} PINK, {labels.count('WHITE')} WHITE, "
      f"{labels.count('OK')} OK")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  BUILD SUB-FRAME SEQUENCE  (15-min linear blends between hourly grids)
# ══════════════════════════════════════════════════════════════════════════════
# Each entry in sub_frames is (grid_2d, timestamp_for_crash_lookup, display_ts)
# - grid_2d          : the 2-D precipitation array to render
# - crash_hour_ts    : the real hourly timestamp used to accumulate crashes
# - display_ts       : interpolated timestamp shown on screen

sub_frames = []
n_sub = 1 + INTERP_FRAMES   # e.g. 4 sub-frames per hour at INTERP_FRAMES=3

for i, ts in enumerate(ts_list):
    ts_next  = ts_list[i + 1] if i + 1 < n_hours else ts
    g_curr   = repaired_grids[ts]
    g_next   = repaired_grids[ts_next]

    for sub in range(n_sub):
        alpha    = sub / n_sub
        g_frame  = (1 - alpha) * g_curr + alpha * g_next
        disp_ts  = ts + pd.Timedelta(minutes=15 * sub)

        sub_frames.append({
            "grid":         g_frame,
            "crash_hour":   ts,       # crashes accumulate at the real hour only
            "display_ts":   disp_ts,
            "is_real_hour": sub == 0, # True only for the real hourly frame
        })

print(f"\n  {len(sub_frames)} total sub-frames to render "
      f"({len(ts_list)} hours × {n_sub} sub-frames)")

# ══════════════════════════════════════════════════════════════════════════════
# 9.  RENDER FRAMES
# ══════════════════════════════════════════════════════════════════════════════
os.makedirs(FRAME_DIR, exist_ok=True)

norm   = mcolors.Normalize(vmin=0, vmax=PRECIP_VMAX_MM)
extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

print(f"Rendering {len(sub_frames)} frames …")

frame_paths   = []
crash_seen    = pd.DataFrame()
last_real_hour = None

for idx, sf in enumerate(tqdm(sub_frames, desc="  Frames", unit="frame")):
    ts        = sf["crash_hour"]
    disp_ts   = sf["display_ts"]
    grid_frame = sf["grid"]

    # Accumulate crashes once per real hour
    if sf["is_real_hour"] and ts != last_real_hour:
        new_crashes  = crash_window[crash_window["crash_dt"] == ts].copy()
        crash_seen   = pd.concat([crash_seen, new_crashes], ignore_index=True)
        last_real_hour = ts
        is_new_hour    = True
    else:
        is_new_hour = False

    # Track which crashes are "new this hour" across all sub-frames of that hour
    new_this_hour = crash_window[crash_window["crash_dt"] == ts]

    # ── Figure — fixed axes positions so colorbar never shifts ─────────────
    fig = plt.figure(figsize=(11, 7), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.02, 0.04, 0.88, 0.92])
    ax.set_facecolor("#F5F5F0")
    ax.set_aspect("equal")
    ax.set_xlim(CT_LON_MIN - 0.04, CT_LON_MAX + 0.04)
    ax.set_ylim(CT_LAT_MIN - 0.02, CT_LAT_MAX + 0.02)

    ct_gdf.plot(ax=ax, color="#ECEAE4", edgecolor="#BDBDBD",
                linewidth=0.35, zorder=1)

    masked = np.where(ct_mask, grid_frame, np.nan)
    ax.imshow(masked, extent=extent, origin="lower",
              cmap=radar_cmap, norm=norm,
              interpolation="bilinear", zorder=2)

    ct_gdf.boundary.plot(ax=ax, color="#9E9E9E", linewidth=0.3, zorder=3)

    # ── Crashes ──────────────────────────────────────────────────────────────
    if not crash_seen.empty:
        # All accumulated dots
        ax.scatter(
            crash_seen["Longitude"], crash_seen["Latitude"],
            c=CRASH_COLOR, s=CRASH_MARKER_SZ,
            edgecolors="none", alpha=CRASH_MARKER_ALPHA,
            zorder=5, marker="o",
        )
        # Rings for crashes that belong to the current real hour
        if not new_this_hour.empty:
            ax.scatter(
                new_this_hour["Longitude"], new_this_hour["Latitude"],
                s=RING_SZ, facecolors="none",
                edgecolors=RING_COLOR, linewidths=RING_LINEWIDTH,
                alpha=RING_ALPHA, zorder=6, marker="o",
            )

    # ── Colour bar — fixed axes position, never reflows ────────────────────
    cbar_ax = fig.add_axes([0.915, 0.15, 0.018, 0.65])
    sm = plt.cm.ScalarMappable(cmap=radar_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Precipitation (mm/hr)", fontsize=8, color="#333333")
    cbar.ax.yaxis.set_tick_params(color="#333333")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#333333", fontsize=7)

    # ── Timestamp ────────────────────────────────────────────────────────────
    ax.text(
        0.98, 0.97, disp_ts.strftime("%a %b %-d, %Y  %H:%M"),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color="#1A1A1A", fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        zorder=10,
    )

    ax.set_title("Connecticut — Precipitation & Traffic Crashes",
                 color="#1A1A1A", fontsize=11, fontweight="bold", pad=7)
    ax.set_axis_off()

    fp = os.path.join(FRAME_DIR, f"frame_{idx:05d}.png")
    fig.savefig(fp, dpi=DPI, facecolor="white")
    plt.close(fig)
    frame_paths.append(fp)

# ── Normalise all frames to a single canonical pixel size ────────────────────
# bbox_inches="tight" can produce slightly different pixel dimensions across
# frames (off by 1–2 px).  Read every frame, find the most common (h, w),
# and resize any outliers so numpy.stack doesn't raise a shape mismatch.
print("  Normalising frame sizes …")
from PIL import Image as PILImage
import collections

shapes = [imageio.imread(fp).shape[:2] for fp in frame_paths]
canonical_hw = collections.Counter(shapes).most_common(1)[0][0]
canonical_h, canonical_w = canonical_hw

resized = 0
for fp in frame_paths:
    img = imageio.imread(fp)
    if img.shape[:2] != canonical_hw:
        pil = PILImage.fromarray(img).resize(
            (canonical_w, canonical_h), PILImage.LANCZOS
        )
        imageio.imwrite(fp, np.array(pil))
        resized += 1

if resized:
    print(f"  Resized {resized} frame(s) to {canonical_w}×{canonical_h} px")
else:
    print(f"  All frames already {canonical_w}×{canonical_h} px — no resize needed")

# ── Black end frames ─────────────────────────────────────────────────────────
print(f"  Appending {BLACK_END_FRAMES} black end frames …")
black = np.zeros((canonical_h, canonical_w, 3), dtype=np.uint8)
for j in range(BLACK_END_FRAMES):
    bf = os.path.join(FRAME_DIR, f"frame_{len(sub_frames) + j:05d}.png")
    imageio.imwrite(bf, black)
    frame_paths.append(bf)

print(f"  {len(frame_paths)} total frames saved")

# ══════════════════════════════════════════════════════════════════════════════
# 10.  ASSEMBLE VIDEO
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nAssembling MP4 at {FPS} fps …")
try:
    writer = imageio.get_writer(
        MP4_PATH, fps=FPS, codec="libx264", quality=8, macro_block_size=1
    )
    for fp in tqdm(frame_paths, desc="  Encoding MP4", unit="frame"):
        writer.append_data(imageio.imread(fp))
    writer.close()
    print(f"  MP4 → {MP4_PATH}")
except Exception as e:
    print(f"  MP4 failed ({e})")

print(f"Assembling GIF at {FPS} fps …")
images = [imageio.imread(fp) for fp in tqdm(frame_paths, desc="  Reading frames")]
imageio.mimsave(GIF_PATH, images, fps=FPS, loop=0)
print(f"  GIF → {GIF_PATH}")

print("\n✓ Done!")