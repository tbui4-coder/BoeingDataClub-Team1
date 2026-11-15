"""
clean_data.py - Team 1 automated data cleaning (Deliverable 3, item 1)

What it does, for every flight in the FLIGHTS list below:
  1. Reads the raw altimeter, GPS tracker and ground station files.
  2. Checks for duplicates, wrong data types, mixed units, missing values,
     sensor spikes and formatting problems, and fixes what can be fixed.
  3. Puts all three files on ONE shared clock: seconds since liftoff.
  4. Writes the cleaned data to
       - cleaned/Team1_Cleaned_Data.xlsx: one tab per cleaned file ("F1 Altimeter", ...),
         a "Change Log" tab listing every change, and a "Data Quality" tab
       - cleaned/<flight>_altimeter.csv, _gps.csv, _ground_station.csv and events.csv
         (the other two scripts read these)

How to run (from this folder):
    pip install pandas numpy openpyxl
    python clean_data.py

Units in every cleaned file: feet, seconds, ft/s, degrees, volts, pascals, deg F.
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HERE = Path(__file__).parent

# openpyxl normally saves numbers with 16 significant digits, exposing tiny rounding errors.
# Those long digit runs can look like credit-card numbers to e-mail security filters,
# so save the shortest exact form instead.
import openpyxl.cell._writer as _xlsx_writer
_default_safe_string = _xlsx_writer.safe_string
_xlsx_writer.safe_string = lambda v: repr(v) if isinstance(v, float) else _default_safe_string(v)

# ===========================================================================
# FLIGHTS - edit this list to add a flight, then rerun all three scripts.
#   altimeter_type: "easymini" or "blueraven"
#   utc_offset_hours: -5 for Eastern Standard Time, -4 for Eastern Daylight Time
# ===========================================================================
FLIGHTS = [
    {"flight": "F1", "launch_date": "2026-02-14", "utc_offset_hours": -5,
     "altimeter_type": "easymini",
     "altimeter_file": "Altimeter_Data214.xlsx",
     "gps_file": "GPS_Tracker214.xlsx",
     "ground_station_file": "Ground_Station214.xlsx"},
    {"flight": "F2", "launch_date": "2026-03-01", "utc_offset_hours": -5,
     "altimeter_type": "blueraven",
     "altimeter_file": "Altimeter_Data_31.xlsx",
     "gps_file": "GPS_Tracker31.xlsx",
     "ground_station_file": "GPS_Ground_Station31.xlsx"},
]
RAW_FOLDER = "raw"                              # where the original files live
CLEANED_FOLDER = "cleaned"                      # where the scripts save their output
OUTPUT_WORKBOOK = "Team1_Cleaned_Data.xlsx"     # Excel file made inside CLEANED_FOLDER
CSV_FLOAT_FORMAT = "%.8g"   # at most 8 significant digits: plenty for this data, and avoids
                            # long digit runs that e-mail filters flag as card numbers


# ---------------------------------------------------------------------------
# Helper functions (also used by flight_metrics.py and plot_ground_tracks.py)
# ---------------------------------------------------------------------------
FT_PER_M = 3.28084                      # feet in one metre
PA_PER_ATM = 101325.0                   # pascals in one atmosphere
EARTH_RADIUS_FT = 6371000 * FT_PER_M    # mean Earth radius in feet


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------
def latlon_to_north_east_ft(lat, lon, lat0, lon0):
    """Convert latitude/longitude (degrees) into feet north and feet east of a
    reference point (lat0, lon0). Uses the flat-Earth approximation, which is
    accurate to well under a foot over a few thousand feet."""
    north = np.radians(lat - lat0) * EARTH_RADIUS_FT
    east = np.radians(lon - lon0) * EARTH_RADIUS_FT * np.cos(np.radians(lat0))
    return north, east


def distance_and_bearing(d_north, d_east):
    """Straight-line distance (ft) and compass bearing (degrees, 0 = north,
    90 = east) of a north/east offset."""
    dist = float(np.hypot(d_north, d_east))
    bearing = float(np.degrees(np.arctan2(d_east, d_north)) % 360)
    return dist, bearing


# ---------------------------------------------------------------------------
# Smoothing and outliers
# ---------------------------------------------------------------------------
def _timed(series, time_s):
    """Index a series by time so rolling windows are measured in seconds,
    not rows (the EasyMini logs 100 rows/s going up but 10 rows/s coming down)."""
    t = pd.to_timedelta(np.asarray(time_s, float) - float(np.min(time_s)), unit="s")
    return pd.Series(np.asarray(series, float), index=t)


def smooth(series, time_s, window_s=0.5):
    """Rolling median then rolling mean over `window_s` seconds.
    The median removes single-sample spikes, the mean removes jitter."""
    s = _timed(series, time_s)
    w = pd.Timedelta(seconds=window_s)
    s = s.rolling(w, center=True, min_periods=1).median()
    return s.rolling(w, center=True, min_periods=1).mean().to_numpy()


def flag_spikes(series, time_s, window_s=1.0, limit=40.0):
    """True where a sample sits more than `limit` units away from the rolling
    median of the surrounding `window_s` seconds (a likely sensor spike)."""
    s = _timed(series, time_s)
    med = s.rolling(pd.Timedelta(seconds=window_s), center=True, min_periods=1).median()
    return (s - med).abs().to_numpy() > limit


# ---------------------------------------------------------------------------
# Flight events
# ---------------------------------------------------------------------------
def choose_height_source(baro_apogee_ft, gps_apogee_ft, tolerance=0.05):
    """Use the barometer unless it disagrees with GPS by more than 5 %
    (the Blue Raven barometer read ~8 % high on flight 2)."""
    if gps_apogee_ft and abs(baro_apogee_ft - gps_apogee_ft) / gps_apogee_ft > tolerance:
        return "gps"
    return "barometer"


def detect_main_and_touchdown(time_s, height_ft, apogee_s,
                              main_rate_fts=40.0, still_rate_fts=3.0, hold_s=3.0):
    """Find when the main chute opened and when the rocket touched down.

    Main open  = first time after apogee that the descent rate drops below
                 `main_rate_fts` (ft/s) and stays below it for `hold_s` seconds.
    Touchdown  = first time after main open that height stops changing
                 (|rate| < `still_rate_fts`) for `hold_s` seconds.
    The height is first put on an even 0.5 s grid and smoothed so a noisy
    barometer doesn't trigger these early."""
    t = np.asarray(time_s, float)
    h = np.asarray(height_ft, float)
    ok = ~np.isnan(h)
    grid = np.arange(apogee_s, t[ok].max(), 0.5)
    hg = np.interp(grid, t[ok], h[ok])
    hg = pd.Series(hg).rolling(3, center=True, min_periods=1).median()
    hg = hg.rolling(3, center=True, min_periods=1).mean().to_numpy()
    rate = np.gradient(hg, grid)                      # ft/s, negative = falling
    hold = int(hold_s / 0.5)

    main_open = touchdown = None
    for i in range(len(grid) - hold):
        if grid[i] < apogee_s + 3:
            continue
        if main_open is None:
            if np.all(rate[i:i + hold] > -main_rate_fts):
                main_open = float(grid[i])
        elif np.all(np.abs(rate[i:i + hold]) < still_rate_fts):
            touchdown = float(grid[i])
            break

    # Refine touchdown: the first moment after main open that the rocket is
    # within 5 ft of its final resting height (the "still" test above starts
    # a little late because of smoothing).
    if touchdown is not None:
        ground = float(np.median(hg[grid >= touchdown]))
        near = (grid > main_open) & (np.abs(hg - ground) < 5)
        if near.any():
            touchdown = float(grid[near][0])
    return main_open, touchdown


def label_phase(time_s, events):
    """Label every row: pad, ascent, drogue, main, landed."""
    t = np.asarray(time_s, float)
    phase = np.full(len(t), "pad", dtype=object)
    phase[t >= 0] = "ascent"
    if events.get("apogee_s") is not None:
        phase[t >= events["apogee_s"]] = "drogue"
    if events.get("main_open_s") is not None:
        phase[t >= events["main_open_s"]] = "main"
    if events.get("touchdown_s") is not None:
        phase[t >= events["touchdown_s"]] = "landed"
    return phase



# ---------------------------------------------------------------------------
# Change log: every cleaning step records what it did here
# ---------------------------------------------------------------------------
class ChangeLog:
    def __init__(self):
        self.rows = []

    def add(self, flight, source, step, column, rows, detail):
        self.rows.append({"flight": flight, "file": source, "step": step,
                          "column": column, "rows_affected": rows, "detail": detail})

    def frame(self):
        return pd.DataFrame(self.rows)


LOG = ChangeLog()


def drop_duplicates(df, flight, source, subset=None, what="rows"):
    """Remove exact duplicate rows (or rows sharing the same timestamp)."""
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        LOG.add(flight, source, "Duplicates", subset[0] if subset else "all",
                removed, f"Removed {removed} duplicate {what} (kept the first copy).")
    return df


def drop_columns(df, cols, flight, source, reason):
    cols = [c for c in cols if c in df.columns]
    if cols:
        LOG.add(flight, source, "Dropped columns", ", ".join(cols[:6]) + (" ..." if len(cols) > 6 else ""),
                len(df), f"Dropped {len(cols)} column(s): {reason}")
    return df.drop(columns=cols)


# ---------------------------------------------------------------------------
# Altimeters
# ---------------------------------------------------------------------------
def clean_easymini(raw, flight, source):
    """Altus Metrum EasyMini: barometer only, metric units."""
    df = drop_duplicates(raw, flight, source)
    df = drop_columns(df, ["#version", "serial", "flight"], flight, source,
                      "same value in every row (file version, serial number, flight number).")
    LOG.add(flight, source, "Units", "altitude, height, speed, acceleration", len(df),
            "Converted metres to feet (x 3.28084) so all files use feet.")
    LOG.add(flight, source, "Units", "temperature", len(df), "Converted deg C to deg F to match the Blue Raven.")
    LOG.add(flight, source, "Dropped columns", "altitude", len(df),
            "Dropped altitude above sea level: computed from a standard atmosphere, puts the pad at 532 ft instead of ~645 ft. Use height_agl_ft.")

    out = pd.DataFrame({
        "time_s": df["time"],
        "height_agl_ft": df["height"] * FT_PER_M,
        "vertical_speed_fts": df["speed"] * FT_PER_M,
        "speed_source": "barometric estimate",
        "accel_fts2": df["acceleration"] * FT_PER_M,
        "pressure_pa": df["pressure"].astype(float),
        "temperature_f": df["temperature"] * 9 / 5 + 32,
        "battery_v": df["battery_voltage"],
        "drogue_v": df["drogue_voltage"],
        "main_v": df["main_voltage"],
        "altimeter_state": df["state_name"],
    })
    LOG.add(flight, source, "Note", "accel_fts2, vertical_speed_fts", len(out),
            "EasyMini has no accelerometer: these are filtered estimates from pressure. Treat boost-phase values as rough.")
    apogee_s = float(df.loc[df["state_name"] == "drogue", "time"].min())
    return out, apogee_s


def clean_blueraven(raw, flight, source):
    """Featherweight Blue Raven low-rate file: barometer + accelerometers + gyro."""
    df = drop_duplicates(raw, flight, source)
    t = df["Flight_Time_(s)"]

    # Event flags we keep (as True/False); everything else in the flag block is internal logic.
    keep_flags = {"Liftoff": "flag_liftoff", "Burnout_Coast": "flag_burnout", "Apogee": "flag_apogee",
                  "Apo_fired": "flag_drogue_fired", "Main_fired": "flag_main_fired",
                  "3rd_fired": "flag_ch3_fired", "4th_fired": "flag_ch4_fired",
                  "Tilt Exceeded 90deg": "flag_tilt_over_90"}
    logic_cols = [c for c in df.columns if c.split(".")[0] in
                  ("LT_AGL1", "GT_AGL2", "LT_FANG1", "GT_FANG2", "GT_FUTANG", "LT_TVAL1", "GT_TVAL2",
                   "LT_VEL1", "GT_VEL2", "LT_BVEL1", "GT_BURN", "Armed")]
    reserved = [c for c in df.columns if c.startswith("Reserved")]
    fer = [c for c in df.columns if "FER" in c]
    df = drop_columns(df, logic_cols, flight, source,
                      "per-channel deployment-logic flags (when a channel was allowed to fire); not needed for flight analysis.")
    df = drop_columns(df, reserved + ["Future_Angle_(deg)", "Sync"], flight, source,
                      "reserved columns and Future_Angle are 0 in every row; Sync only matters for merging the high-rate file.")
    df = drop_columns(df, fer + ["Normal_Ascent", "Accel_Vel_LE_0", "ECI_Vvel_le_0", "Press_Increasing"],
                      flight, source, "internal status registers and detection inputs (hex codes, undocumented bits).")
    df = drop_columns(df, ["Year", "Month", "Day", "Time"], flight, source,
                      "altimeter clock was 1 h 16 min off (synced from a phone). Replaced by time_local built from the GPS clock.")
    df = drop_columns(df, ["Baro_Altitude_ASL_(feet)"], flight, source,
                      "standard-atmosphere altitude puts the pad at ~99 ft above sea level; use height_agl_ft.")
    LOG.add(flight, source, "Units", "Baro_Press_(atm)", len(df), "Converted atmospheres to pascals (x 101325) to match the EasyMini.")

    apogee_s = float(t[df["Apogee"] == 1].min())
    after_apogee = (t >= apogee_s).to_numpy()

    inertial = {"Velocity_Up": "vertical_speed_fts", "Inertial_Altitude": "inertial_alt_ft",
                "Inertial_DR_Position": "inertial_downrange_ft", "Inertial_CR_position": "inertial_crossrange_ft",
                "Velocity_DR": "inertial_downrange_speed_fts", "Velocity_CR": "inertial_crossrange_speed_fts"}
    out = pd.DataFrame({"time_s": t, "height_agl_ft": df["Baro_Altitude_AGL_(feet)"].astype(float)})
    for src, dst in inertial.items():
        out[dst] = df[src].astype(float)
        out.loc[after_apogee, dst] = np.nan
    out["speed_source"] = np.where(after_apogee, "barometric (smoothed)", "inertial")
    LOG.add(flight, source, "Invalid values", ", ".join(inertial.values()), int(after_apogee.sum()),
            "Set inertial values to blank after apogee: they drift once the rocket tumbles (inertial altitude reaches -12,312 ft). "
            "They match GPS within 0.5 % up to apogee.")

    # Replace the blanked vertical speed after apogee with a smoothed barometric rate.
    h_s = smooth(out["height_agl_ft"], t.to_numpy(), window_s=1.0)
    baro_rate = np.gradient(h_s, t.to_numpy())
    out.loc[after_apogee, "vertical_speed_fts"] = baro_rate[after_apogee]

    out["accel_fts2"] = np.nan
    LOG.add(flight, source, "Missing values", "accel_fts2", len(out),
            "Left blank: the Blue Raven stores acceleration only in its separate 500 Hz high-rate file, which we don't have.")
    out["pressure_pa"] = df["Baro_Press_(atm)"] * PA_PER_ATM
    out["temperature_f"] = df["Temperature_(F)"]
    out["battery_v"] = df["Batt_Volts"]
    out["drogue_v"] = df["Apo_Volts"]
    out["main_v"] = df["Main_Volts"]
    out["ch3_v"] = df["3rd_Volts"]
    out["ch4_v"] = df["4th_Volts"]
    out["tilt_deg"] = df["Tilt_Angle_(deg)"]
    out["roll_deg"] = df["Roll_Angle_(deg)"]
    for src, dst in keep_flags.items():
        out[dst] = df[src].astype(bool)
    LOG.add(flight, source, "Data types", ", ".join(keep_flags.values()), len(out), "Converted 0/1 event flags to True/False.")
    return out, apogee_s


def finish_altimeter(out, flight, source):
    """Steps shared by both altimeter types."""
    t = out["time_s"].to_numpy()
    out["height_outlier"] = flag_spikes(out["height_agl_ft"], t, window_s=1.0, limit=40.0)
    n_out = int(out["height_outlier"].sum())
    LOG.add(flight, source, "Outliers", "height_agl_ft", n_out,
            f"Flagged {n_out} barometric spikes (> 40 ft from the 1 s rolling median) in height_outlier. Kept, not deleted.")
    out["height_agl_smooth_ft"] = smooth(out["height_agl_ft"].where(~out["height_outlier"]).interpolate(), t, 0.5)
    LOG.add(flight, source, "Added column", "height_agl_smooth_ft", len(out),
            "Added a smoothed height (spikes removed, 0.5 s rolling median + mean) for event detection and metrics.")
    if (np.diff(t) <= 0).any():
        out = out.sort_values("time_s").reset_index(drop=True)
        LOG.add(flight, source, "Ordering", "time_s", len(out), "Sorted rows by time.")
    return out


# ---------------------------------------------------------------------------
# GPS tracker
# ---------------------------------------------------------------------------
def clean_gps(raw, flight, source, cfg):
    df = drop_duplicates(raw, flight, source, subset=["UTCTIME"], what="rows with the same timestamp")

    # Fix the dates: time of day in UTCTIME is right, but its date and UNIXTIME are wrong.
    tod = pd.to_datetime(df["UTCTIME"].str.slice(12, 24), format="%H:%M:%S.%f")
    date = datetime.strptime(cfg["launch_date"], "%Y-%m-%d")
    time_utc = pd.Series([date + timedelta(hours=x.hour, minutes=x.minute, seconds=x.second,
                                           microseconds=x.microsecond) for x in tod])
    LOG.add(flight, source, "Formatting", "UTCTIME, UNIXTIME", len(df),
            f"UTCTIME has the right time of day but the wrong date ({df['UTCTIME'].iloc[0][:11]}); UNIXTIME is 5 h later still. "
            f"Rebuilt time_utc and time_local from the launch date {cfg['launch_date']} and the time of day.")

    lat0, lon0 = float(df["LAT"].iloc[0]), float(df["LON"].iloc[0])
    north, east = latlon_to_north_east_ft(df["LAT"], df["LON"], lat0, lon0)
    out = pd.DataFrame({
        "elapsed_s": df["UNIXTIME"] - df["UNIXTIME"].iloc[0],      # replaced by time_s after alignment
        "time_utc": time_utc,
        "lat_deg": df["LAT"], "lon_deg": df["LON"],
        "alt_msl_ft": df["ALT"], "alt_agl_ft": df["Altitude AGL"],
        "north_ft": north, "east_ft": east,
        "dist_from_pad_ft": np.hypot(north, east),
        "horiz_speed_fts": df["HORZV"].astype(float),
        "vert_speed_fts": df["VERTV"].astype(float),
        "heading_deg": df["HEAD"].astype(float),
        "sats_used": df["#SATS"], "fix_type": df["FIX"], "status_flags": df["FLAGS"],
        "sats_snr_gt40": df[">40"], "sats_snr_gt32": df[">32"], "sats_snr_gt24": df[">24"],
        "rssi_dbm": df["RSSI"], "battery_v": df["BATT"],
        "flag_launch": df["Launch detection"].astype(bool),
        "flag_apogee": df["Apogee detection"].astype(bool),
        "flag_landing": df["Landing detection"].astype(bool),
        "tracker_dist_ft": df["Distance (feet)"],
    })
    LOG.add(flight, source, "Added column", "north_ft, east_ft, dist_from_pad_ft", len(out),
            "Added position in feet north/east of the first fix (the pad), from LAT/LON.")

    still = out["horiz_speed_fts"] == 0
    out.loc[still, "heading_deg"] = np.nan
    LOG.add(flight, source, "Invalid values", "heading_deg", int(still.sum()),
            "Blanked heading when horizontal speed is 0: the direction of a stationary tracker is random.")

    gaps = out["elapsed_s"].diff()
    out["gap_before_s"] = gaps.round(2)
    out["gps_gap"] = gaps > 1.5
    for i in out.index[out["gps_gap"]]:
        LOG.add(flight, source, "Missing values", "all GPS columns", 1,
                f"No GPS fixes for {gaps[i]:.1f} s before elapsed {out['elapsed_s'][i]:.1f} s (marked in gps_gap). Not filled in.")
    neg = int((out["alt_agl_ft"] < -10).sum())
    if neg:
        LOG.add(flight, source, "Outliers", "alt_agl_ft", neg,
                f"{neg} rows below -10 ft AGL after landing (GPS altitude error). Kept; use the altimeter for height near the ground.")
    return out


def align_gps_to_altimeter(gps, alt, apogee_s):
    """Find the time shift that best lines up GPS height with altimeter height
    during the first 40 s of descent (allowing for a barometer scale error)."""
    t_alt = alt["time_s"].to_numpy()
    h_alt = alt["height_agl_smooth_ft"].to_numpy()
    best = (np.inf, 0.0)
    for shift in np.arange(0, 10, 0.05):
        tg = gps["elapsed_s"].to_numpy() - shift
        m = (tg > apogee_s) & (tg < apogee_s + 40)
        if m.sum() < 10:
            continue
        ha = np.interp(tg[m], t_alt, h_alt)
        hg = gps["alt_agl_ft"].to_numpy()[m]
        k = (ha @ hg) / (ha @ ha)                     # best scale factor
        err = np.mean((k * ha - hg) ** 2)
        if err < best[0]:
            best = (err, float(shift))
    return best[1], float(np.sqrt(best[0]))


# ---------------------------------------------------------------------------
# Ground station
# ---------------------------------------------------------------------------
def clean_ground_station(raw, flight, source, liftoff_local):
    df = drop_duplicates(raw, flight, source)
    found = [c for c in df.columns if c.startswith("Found")]
    df = drop_columns(df, found, flight, source, "recovery 'find' feature was never used (all 0, blank or a fixed -50).")

    time_local = pd.to_datetime(df["DATE"]).dt.normalize() + pd.to_timedelta(df["TIME"].astype(str))
    LOG.add(flight, source, "Formatting", "DATE, TIME", len(df), "Combined DATE and TIME into one time_local column.")

    out = pd.DataFrame({
        "time_local": time_local,
        "time_s": (time_local - liftoff_local).dt.total_seconds().round(3),
        "tracker_id": df["TRACKER"],
        "gs_lat_deg": df["GS Lat"], "gs_lon_deg": df["GS Lon"], "gs_alt_msl_ft": df["GS Alt asl"],
        "tracker_lat_deg": df["TRACKER Lat"], "tracker_lon_deg": df["TRACKER Lon"],
        "tracker_alt_msl_ft": df["TRACKER Alt asl"],
        "fix_type": df["FIX"],
        "status_flags": df["FLAGS"].astype(str).apply(lambda s: int(s, 16)),
        "horiz_speed_fts": df["HORZV"].astype(float), "vert_speed_fts": df["VERTV"].astype(float),
        "heading_deg": df["HEAD"].astype(float),
        "sats_total": df["#TOT"], "sats_snr_gt40": df[">40"], "sats_snr_gt32": df[">32"], "sats_snr_gt24": df[">24"],
        "rssi_down_dbm": df["RSSI Down"], "rssi_up_dbm": df["RSSI Up"],
        "horizontal_range_ft": df["Horizontal range (ft)"].astype(float),
        "alt_agl_ft": df["Alt AGL (ft)"].astype(float),
        "total_range_ft": df["Total range (ft)"].astype(float),
        "lora_max_range_kft": df["LoRa Max Range (kft)"], "lora_success_rate": df["LoRa Success Rate"],
        "tracker_batt_v": df["BATT"], "gs_batt_v": df["GS_batt_v"],
        "packets_down_sent": df["Down Sent"], "packets_down_received": df["Down Rcvd"],
        "packets_up_sent": df["Up Sent"], "packets_up_received": df["Up Rccvd"],
        "gs_asl_offset_ft": df["GS_ASL_offset"],
    })
    LOG.add(flight, source, "Data types", "status_flags", len(out), "Converted hex text (0x03) to whole numbers.")

    no_fix = (out["tracker_lat_deg"] == 0) | (out["fix_type"] == 0)
    cols = ["tracker_lat_deg", "tracker_lon_deg", "tracker_alt_msl_ft", "horizontal_range_ft",
            "alt_agl_ft", "total_range_ft", "heading_deg"]
    out.loc[no_fix, cols] = np.nan
    LOG.add(flight, source, "Invalid values", ", ".join(cols[:3]) + " ...", int(no_fix.sum()),
            "Blanked tracker position and ranges when the tracker had no GPS fix (logged as 0, which is a point off Africa).")
    still = out["horiz_speed_fts"] == 0
    out.loc[still, "heading_deg"] = np.nan
    LOG.add(flight, source, "Invalid values", "heading_deg", int(still.sum()),
            "Blanked heading when the tracker was not moving.")
    in_flight = out["time_s"].between(0, 200).sum()
    LOG.add(flight, source, "Coverage", "time_s", int(in_flight),
            f"Log runs from {out['time_s'].min():.0f} s to {out['time_s'].max():.0f} s relative to liftoff; "
            f"{in_flight} rows fall within the first 200 s of flight.")
    return out


# ---------------------------------------------------------------------------
# Data quality summary
# ---------------------------------------------------------------------------
def quality_rows(flight, name, raw, clean):
    missing = clean.isna().sum()
    missing = missing[missing > 0]
    return {"flight": flight, "file": name,
            "raw_rows": len(raw), "raw_columns": raw.shape[1],
            "clean_rows": len(clean), "clean_columns": clean.shape[1],
            "columns_with_blanks": ", ".join(f"{c} ({n})" for c, n in missing.items()) or "none"}


# ---------------------------------------------------------------------------
# Workbook output
# ---------------------------------------------------------------------------
def write_sheet(wb, title, df, after=None, widths=None):
    if title in wb.sheetnames:
        del wb[title]
    ws = wb.create_sheet(title)
    if after and after in wb.sheetnames:
        wb.move_sheet(ws, offset=wb.sheetnames.index(after) + 1 - wb.sheetnames.index(title))
    hdr_font = Font(name="Arial", bold=True, color="FFFFFF")
    hdr_fill = PatternFill("solid", fgColor="26408F")
    body = Font(name="Arial")
    ws.append(list(df.columns))
    for c in ws[1]:
        c.font, c.fill = hdr_font, hdr_fill
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for row in df.itertuples(index=False):
        vals = []
        for v in row:
            if isinstance(v, (np.floating, float)):
                v = None if np.isnan(v) else float(f"{float(v):.8g}")   # at most 8 significant digits
            elif isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.bool_):
                v = bool(v)
            elif isinstance(v, pd.Timestamp):
                # Written as text: Excel stores real dates as long decimal numbers, which
                # e-mail filters can mistake for credit-card numbers.
                v = v.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            vals.append(v)
        ws.append(vals)
    for i, col in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(i)].width = (widths or {}).get(col, max(12, min(26, len(col) + 3)))
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = body
    ws.freeze_panes = "B2"
    return ws


def main():
    raw_dir = HERE / RAW_FOLDER
    clean_dir = HERE / CLEANED_FOLDER
    clean_dir.mkdir(exist_ok=True)
    wb_path = clean_dir / OUTPUT_WORKBOOK

    quality, tabs, events_all = [], [], {}
    for fc in FLIGHTS:
        fl = fc["flight"]
        print(f"Cleaning {fl} ...")

        # Altimeter
        raw_alt = pd.read_excel(raw_dir / fc["altimeter_file"])
        if fc["altimeter_type"] == "easymini":
            alt, apogee_s = clean_easymini(raw_alt, fl, fc["altimeter_file"])
        else:
            alt, apogee_s = clean_blueraven(raw_alt, fl, fc["altimeter_file"])
        alt = finish_altimeter(alt, fl, fc["altimeter_file"])

        # GPS, then line it up with the altimeter clock
        raw_gps = pd.read_excel(raw_dir / fc["gps_file"])
        gps = clean_gps(raw_gps, fl, fc["gps_file"], fc)
        shift, rmse = align_gps_to_altimeter(gps, alt, apogee_s)
        gps.insert(0, "time_s", (gps["elapsed_s"] - shift).round(3))
        gps = gps.drop(columns="elapsed_s")
        utc_offset = timedelta(hours=fc["utc_offset_hours"])
        liftoff_utc = gps["time_utc"].iloc[0] + timedelta(seconds=shift)
        liftoff_local = liftoff_utc + utc_offset
        gps.insert(2, "time_local", gps["time_utc"] + utc_offset)
        LOG.add(fl, fc["gps_file"], "Time alignment", "time_s", len(gps),
                f"Shifted GPS time by {shift:.2f} s so time_s = 0 at altimeter liftoff "
                f"(GPS and altimeter heights then agree to {rmse:.0f} ft RMS during descent). Liftoff: {liftoff_local:%Y-%m-%d %H:%M:%S} local.")

        # Events (needed for the phase column) and which height to trust
        baro_apogee = float(alt["height_agl_smooth_ft"].max())
        gps_apogee = float(gps["alt_agl_ft"].max())
        source = choose_height_source(baro_apogee, gps_apogee)
        if source == "barometer":
            main_open, touchdown = detect_main_and_touchdown(alt["time_s"], alt["height_agl_smooth_ft"], apogee_s)
        else:
            main_open, touchdown = detect_main_and_touchdown(gps["time_s"], gps["alt_agl_ft"], apogee_s)
        events = {"apogee_s": apogee_s, "main_open_s": main_open, "touchdown_s": touchdown,
                  "height_source": source, "gps_time_shift_s": shift,
                  "liftoff_local": f"{liftoff_local:%Y-%m-%d %H:%M:%S}"}
        events_all[fl] = events
        LOG.add(fl, "all", "Added column", "phase", len(alt) + len(gps),
                f"Labelled each row pad / ascent / drogue / main / landed. Apogee {apogee_s:.2f} s (altimeter flag), "
                f"main open {main_open:.1f} s, touchdown {touchdown:.1f} s (detected from {source} height).")
        alt.insert(1, "phase", label_phase(alt["time_s"], events))
        gps.insert(1, "phase", label_phase(gps["time_s"], events))
        alt.insert(0, "flight", fl)
        alt.insert(2, "time_local", [liftoff_local + timedelta(seconds=float(s)) for s in alt["time_s"]])
        gps.insert(0, "flight", fl)

        # Ground station
        raw_gs = pd.read_excel(raw_dir / fc["ground_station_file"])
        gs = clean_ground_station(raw_gs, fl, fc["ground_station_file"], liftoff_local)
        gs.insert(0, "flight", fl)
        gs.insert(2, "phase", label_phase(gs["time_s"], events))

        # Save
        for name, raw, df in (("altimeter", raw_alt, alt), ("gps", raw_gps, gps), ("ground_station", raw_gs, gs)):
            df.to_csv(clean_dir / f"{fl}_{name}.csv", index=False, float_format=CSV_FLOAT_FORMAT)
            quality.append(quality_rows(fl, f"{fl}_{name}", raw, df))
        tabs += [(f"{fl} Altimeter", alt), (f"{fl} GPS", gps), (f"{fl} Ground Station", gs)]

    events_table = pd.DataFrame.from_dict(events_all, orient="index").rename_axis("flight").reset_index()
    events_table.to_csv(clean_dir / "events.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    # Excel file for people: Change Log and Data Quality first, then one tab per cleaned file
    wb = Workbook()
    wb.remove(wb.active)
    write_sheet(wb, "Change Log", LOG.frame(), widths={"detail": 100, "column": 34, "file": 26, "step": 16})
    write_sheet(wb, "Data Quality", pd.DataFrame(quality), widths={"columns_with_blanks": 90, "file": 22})
    for ws_name in ("Change Log", "Data Quality"):
        for row in wb[ws_name].iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
    for title, df in tabs:
        write_sheet(wb, title, df)
    wb.save(wb_path)
    print(f"Done. {len(LOG.rows)} changes logged. Saved {CLEANED_FOLDER}/{OUTPUT_WORKBOOK} and the CSV files.")
    for fl, ev in events_all.items():
        print(f"  {fl}: apogee {ev['apogee_s']:.2f} s, main open {ev['main_open_s']:.1f} s, "
              f"touchdown {ev['touchdown_s']:.1f} s, height source {ev['height_source']}, GPS shift {ev['gps_time_shift_s']:.2f} s")


if __name__ == "__main__":
    main()
