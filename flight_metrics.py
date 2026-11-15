"""
flight_metrics.py - Team 1 flight metrics (Deliverable 3, item 2)

Reads the cleaned files made by clean_data.py and calculates, for every flight:
maximum altitude, flight duration, ascent and descent time, descent rates,
horizontal displacement at apogee, landing coordinates, landing distance and
direction, drift in each phase and the wind that caused it.

Nothing is typed in by hand: event times (apogee, main chute opening,
touchdown) come from cleaned/events.csv, which clean_data.py detects
automatically from the data.

How to run (after clean_data.py):
    python flight_metrics.py

Output:
    - a table printed in the terminal
    - cleaned/metrics.csv
    - a "Metrics" tab in cleaned/Team1_Cleaned_Data.xlsx
"""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from clean_data import (CLEANED_FOLDER, CSV_FLOAT_FORMAT, FLIGHTS, OUTPUT_WORKBOOK, distance_and_bearing,
                        latlon_to_north_east_ft, write_sheet)

HERE = Path(__file__).parent
FTS_TO_MPH = 0.681818


def compass(bearing):
    """Turn a bearing in degrees into a compass word like 'NE'."""
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int((bearing + 11.25) // 22.5) % 16]


def position_at(gps, t):
    """GPS north/east (ft from pad) at time t, by linear interpolation."""
    return (float(np.interp(t, gps["time_s"], gps["north_ft"])),
            float(np.interp(t, gps["time_s"], gps["east_ft"])))


def flight_metrics(alt, gps, ev):
    """Return a list of (metric, value, units, how) for one flight."""
    t_apo, t_main, t_td = ev["apogee_s"], ev["main_open_s"], ev["touchdown_s"]
    src = ev["height_source"]

    # Height series we trust for this flight (barometer unless it disagrees with GPS by > 5 %)
    if src == "barometer":
        t_h, h = alt["time_s"].to_numpy(), alt["height_agl_smooth_ft"].to_numpy()
    else:
        t_h, h = gps["time_s"].to_numpy(), gps["alt_agl_ft"].to_numpy()
    height_at = lambda t: float(np.interp(t, t_h, h))
    ground = float(np.median(h[t_h >= t_td])) if (t_h >= t_td).any() else 0.0
    apogee_ft = float(np.max(h[(t_h > 0) & (t_h < t_main)]))
    h_main = height_at(t_main)

    m = []
    add = lambda name, value, units, how: m.append((name, value, units, how))

    add("Liftoff (local time)", ev["liftoff_local"], "", "GPS clock at altimeter liftoff")
    add("Height source used", src, "", "barometer unless it disagrees with GPS apogee by more than 5 %")

    # --- Altitude ---------------------------------------------------------
    add("Maximum altitude (apogee)", apogee_ft, "ft AGL", "max of the trusted height (see Height source used)")
    add("Apogee, barometric (smoothed)", float(alt["height_agl_smooth_ft"].max()), "ft AGL", "max of height_agl_smooth_ft")
    add("Apogee, GPS", float(gps["alt_agl_ft"].max()), "ft AGL", "max of GPS alt_agl_ft")
    if "inertial_alt_ft" in alt and alt["inertial_alt_ft"].notna().any():
        add("Apogee, inertial", float(alt["inertial_alt_ft"].max()), "ft", "max of Blue Raven inertial_alt_ft")

    # --- Times ------------------------------------------------------------
    add("Apogee time", t_apo, "s", "altimeter's own apogee detection")
    add("Main chute opening time", t_main, "s", "descent rate drops below 40 ft/s and stays there 3 s")
    add("Main chute opening height", h_main, "ft AGL", "trusted height at main opening")
    add("Touchdown time", t_td, "s", "first time within 5 ft of final resting height")
    add("Flight duration", t_td, "s", "touchdown - liftoff")
    add("Ascent duration", t_apo, "s", "apogee - liftoff")
    add("Descent duration", t_td - t_apo, "s", "touchdown - apogee")
    add("Time under drogue", t_main - t_apo, "s", "main opening - apogee")
    add("Time under main", t_td - t_main, "s", "touchdown - main opening")

    # --- Speeds -----------------------------------------------------------
    drogue_rate = (apogee_ft - h_main) / (t_main - t_apo)
    main_rate = (h_main - ground) / (t_td - t_main)
    add("Average descent rate, drogue", drogue_rate, "ft/s", "(apogee - main height) / time under drogue")
    add("Average descent rate, main", main_rate, "ft/s", "(main height - ground) / time under main")
    add("Average descent rate, overall", (apogee_ft - ground) / (t_td - t_apo), "ft/s", "(apogee - ground) / descent duration")
    up = alt[(alt["time_s"] > 0) & (alt["time_s"] < t_apo)]
    add("Peak upward speed", float(up["vertical_speed_fts"].max()), "ft/s", "max vertical_speed_fts before apogee")
    add("Peak speed source", up["speed_source"].iloc[0], "", "inertial (measured) or barometric estimate")
    if "tilt_deg" in alt:
        add("Tilt from vertical at apogee", float(np.interp(t_apo, alt["time_s"], alt["tilt_deg"])), "deg", "Blue Raven tilt_deg")

    # --- Positions --------------------------------------------------------
    pad = gps.iloc[0]
    landed = gps[gps["time_s"] >= t_td + 2]
    landed = landed if len(landed) else gps.tail(3)
    land_lat, land_lon = float(landed["lat_deg"].median()), float(landed["lon_deg"].median())
    add("Pad latitude", float(pad["lat_deg"]), "deg", "first GPS fix")
    add("Pad longitude", float(pad["lon_deg"]), "deg", "first GPS fix")
    add("Landing latitude", land_lat, "deg", "median of GPS fixes after touchdown")
    add("Landing longitude", land_lon, "deg", "median of GPS fixes after touchdown")
    n, e = latlon_to_north_east_ft(land_lat, land_lon, pad["lat_deg"], pad["lon_deg"])
    dist, brg = distance_and_bearing(n, e)
    add("Landing displacement", dist, "ft", "straight line, pad to landing (Method 1)")
    add("Landing direction", brg, "deg", "bearing from north: 0 = N, 90 = E, 180 = S, 270 = W")
    add("Landing direction (compass)", compass(brg), "", "16-point compass")
    add("Landing displacement, north", float(n), "ft", "negative = south")
    add("Landing displacement, east", float(e), "ft", "negative = west")

    na, ea = position_at(gps, t_apo)
    d_apo, b_apo = distance_and_bearing(na, ea)
    add("Horizontal displacement at apogee", d_apo, "ft", "GPS position at apogee vs pad")
    add("Direction at apogee", b_apo, "deg", "bearing from north")
    if "inertial_downrange_ft" in alt and alt["inertial_downrange_ft"].notna().any():
        dr = float(np.interp(t_apo, alt["time_s"], alt["inertial_downrange_ft"].ffill()))
        cr = float(np.interp(t_apo, alt["time_s"], alt["inertial_crossrange_ft"].ffill()))
        add("Horizontal displacement at apogee, inertial", float(np.hypot(dr, cr)), "ft", "Blue Raven inertial position at apogee")

    flown = gps[(gps["time_s"] >= 0) & (gps["time_s"] <= t_td)]
    add("Ground track length", float(np.hypot(flown["north_ft"].diff(), flown["east_ft"].diff()).sum()),
        "ft", "sum of distances between GPS fixes, liftoff to touchdown")

    # --- Drift in each phase, and the wind it implies --------------------------
    for phase, t0, t1 in (("ascent", 0.0, t_apo), ("drogue", t_apo, t_main), ("main", t_main, t_td)):
        n0, e0 = position_at(gps, t0)
        n1, e1 = position_at(gps, t1)
        d, b = distance_and_bearing(n1 - n0, e1 - e0)
        add(f"Drift during {phase}", d, "ft", "GPS movement between the start and end of the phase")
        add(f"Drift direction during {phase}", b, "deg", "bearing from north")
        if phase != "ascent":
            wind = d / (t1 - t0)
            add(f"Wind estimate under {phase}", wind * FTS_TO_MPH, "mph", "drift / time under that parachute, blowing toward the drift direction")
    return m


def main():
    clean_dir = HERE / CLEANED_FOLDER
    events = pd.read_csv(clean_dir / "events.csv").set_index("flight").to_dict("index")

    # One column per flight; a metric that only exists for one flight (e.g. tilt) is blank for the other.
    order, info, values = [], {}, {}
    for fc in FLIGHTS:
        fl = fc["flight"]
        alt = pd.read_csv(clean_dir / f"{fl}_altimeter.csv")
        gps = pd.read_csv(clean_dir / f"{fl}_gps.csv")
        col = f"{fl} ({fc['launch_date']})"
        values[col] = {}
        for name, value, units, how in flight_metrics(alt, gps, events[fl]):
            if name not in info:
                order.append(name)
                info[name] = (units, how)
            values[col][name] = value
    table = pd.DataFrame({"metric": order})
    for col, vals in values.items():
        table[col] = [vals.get(name, "") for name in order]
    table["units"] = [info[name][0] for name in order]
    table["how"] = [info[name][1] for name in order]

    # Round for display: 5 decimals for latitude/longitude, 1 decimal for everything else
    def tidy(value, metric):
        if not isinstance(value, (float, np.floating)):
            return value
        return round(float(value), 5 if metric.endswith(("latitude", "longitude")) else 1)

    for c in table.columns[1:-2]:
        table[c] = [tidy(v, metric) for v, metric in zip(table[c], table["metric"])]

    table.to_csv(clean_dir / "metrics.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(table.drop(columns="how").to_string(index=False))

    wb_path = clean_dir / OUTPUT_WORKBOOK
    wb = load_workbook(wb_path)
    write_sheet(wb, "Metrics", table, after="Data Quality", widths={"metric": 40, "units": 10, "how": 80})
    wb.save(wb_path)
    print(f"\nSaved {CLEANED_FOLDER}/metrics.csv and the 'Metrics' tab in {CLEANED_FOLDER}/{OUTPUT_WORKBOOK}")


if __name__ == "__main__":
    main()
