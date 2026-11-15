# Team 1 flight data pipeline

Python scripts that turn the raw rocket files into clean data, flight metrics, charts and machine learning models.

| File or folder | What it is |
| --- | --- |
| `raw/` | Put the original data files here before rerunning the scripts (left out of the zip because e-mail filters flag them). |
| `clean_data.py` | Step 1. Cleans every raw file. The list of flights is at the top of this file. |
| `flight_metrics.py` | Step 2. Calculates max altitude, flight and descent time, descent rates, displacement, landing coordinates and phase drift. |
| `plot_ground_tracks.py` | Step 3. Draws the ground-track map of every flight, coloured by phase. |
| `models.py` | Step 4. Trains linear regression, polynomial regression and a decision tree to predict the descent, tests each on the flight it wasn't trained on, and charts the results. |
| `cleaned/` | Output. `Team1_Cleaned_Data.xlsx` (cleaned data, Change Log, Data Quality, Metrics), plus CSV copies the scripts read. |
| `figures/` | Output. `ground_tracks.png`, `model_predictions.png`, `model_errors.png`. |

## Run it

```
pip install pandas numpy openpyxl matplotlib scikit-learn
python clean_data.py
python flight_metrics.py
python plot_ground_tracks.py
python models.py
```

Run them in this order. Each one can be rerun at any time; it replaces its own output.

## Add a new flight

1. Put the three raw files in `raw/`.
2. Open `clean_data.py`, find the `FLIGHTS` list near the top, copy one flight entry and change the name, date, altimeter type (`easymini` or `blueraven`) and file names. Use `utc_offset_hours: -4` for flights during daylight saving time.
3. Run all four scripts again.

## What the cleaning does

- Removes duplicate rows (67 in flight 1's altimeter, 1 in flight 2's GPS).
- Converts everything to feet, seconds, ft/s, pascals and deg F.
- Puts all three files on one clock: `time_s` = seconds since liftoff. The GPS is lined up with the altimeter automatically, and the wrong dates in the tracker file are corrected.
- Adds a `phase` column (pad, ascent, drogue, main, landed) with the same definition on every file.
- Blanks values that are known to be wrong instead of deleting rows: Blue Raven inertial data after apogee, tracker positions logged as 0 with no GPS fix, and headings while the tracker is standing still.
- Flags (but keeps) barometric spikes in `height_outlier`, and GPS dropouts in `gps_gap`.
- Drops columns that are constant, unused or internal (for example the 56 Blue Raven logic flags).
- Writes every change to the **Change Log** tab.

## Column naming

Lower case with the unit at the end: `height_agl_ft`, `vertical_speed_fts`, `time_s`, `tilt_deg`, `pressure_pa`. True/False columns start with `flag_`.
