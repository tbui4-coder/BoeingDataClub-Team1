"""
plot_ground_tracks.py - Team 1 visualization (Deliverable 3, item 4)

Draws the GPS ground track of every flight on one top-down map, coloured by
flight phase (ascent, under drogue, under main), with a dashed line from the
pad to each landing spot. This is the "where the drift came from" chart.

How to run (after clean_data.py):
    pip install matplotlib
    python plot_ground_tracks.py

Output: figures/ground_tracks.png (high resolution, for slides and docs)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from clean_data import CLEANED_FOLDER, FLIGHTS, distance_and_bearing

HERE = Path(__file__).parent

# Colours: one per phase, the same on every flight
PHASE_STYLE = {
    "ascent": {"color": "#2a78d6", "label": "Ascent"},
    "drogue": {"color": "#eb6834", "label": "Under drogue"},
    "main":   {"color": "#a8a79f", "label": "Under main"},
}
INK, QUIET, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def compass(bearing):
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int((bearing + 11.25) // 22.5) % 16]


def plot_flight(ax, gps, events, label):
    """Draw one flight's track, apogee dot, landing dot and net-offset line."""
    flown = gps[(gps["time_s"] >= 0) & (gps["time_s"] <= events["touchdown_s"] + 1)]

    # Track, one colour per phase. Each phase starts where the previous one ended,
    # so the line has no gaps.
    for phase, style in PHASE_STYLE.items():
        part = flown[flown["phase"].isin([phase])]
        if part.empty:
            continue
        prev = flown[flown["time_s"] < part["time_s"].min()].tail(1)
        part = pd.concat([prev, part])
        ax.plot(part["east_ft"], part["north_ft"], color=style["color"], lw=2.4,
                solid_capstyle="round", zorder=3)

    # Landing point = median of GPS fixes after touchdown
    landed = gps[gps["time_s"] >= events["touchdown_s"] + 2]
    landed = landed if len(landed) else gps.tail(3)
    e_land, n_land = landed["east_ft"].median(), landed["north_ft"].median()
    dist, brg = distance_and_bearing(n_land, e_land)

    # Apogee point
    e_apo = np.interp(events["apogee_s"], gps["time_s"], gps["east_ft"])
    n_apo = np.interp(events["apogee_s"], gps["time_s"], gps["north_ft"])

    ax.plot([0, e_land], [0, n_land], ls=(0, (4, 4)), color=QUIET, lw=1.1, zorder=2)
    ax.scatter([e_apo], [n_apo], s=36, color=QUIET, edgecolor="white", lw=1.5, zorder=4)
    ax.scatter([e_land], [n_land], s=60, color=INK, edgecolor="white", lw=1.5, zorder=5)
    return {"label": label, "e_land": e_land, "n_land": n_land, "dist": dist, "brg": brg,
            "e_apo": e_apo, "n_apo": n_apo}


def main():
    clean_dir = HERE / CLEANED_FOLDER
    events = pd.read_csv(clean_dir / "events.csv").set_index("flight").to_dict("index")
    out_dir = HERE / "figures"
    out_dir.mkdir(exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(9, 8.5))

    results = []
    for fc in FLIGHTS:
        fl = fc["flight"]
        gps = pd.read_csv(clean_dir / f"{fl}_gps.csv")
        date = pd.Timestamp(fc["launch_date"]).strftime("%b %-d")
        results.append(plot_flight(ax, gps, events[fl], f"Flight {fl[1:]}, {date}"))

    # Labels next to each landing spot and apogee; placed on the side away from the track
    for r in results:
        # Label on the outside of the landing spot, away from the tracks
        dx = -14
        ha = "right"
        ax.annotate(f"{r['label']}\n{r['dist']:,.0f} ft toward {r['brg']:.0f}° ({compass(r['brg'])})",
                    (r["e_land"], r["n_land"]), xytext=(dx, 0 if r["e_land"] < 0 else 16), textcoords="offset points",
                    ha=ha, va="center", fontsize=10, color=INK, fontweight="bold", linespacing=1.4)
        ax.annotate("apogee", (r["e_apo"], r["n_apo"]), xytext=(8, -12), textcoords="offset points",
                    fontsize=9, color=QUIET)
    ax.scatter([0], [0], s=50, color=QUIET, edgecolor="white", lw=1.5, zorder=5)
    ax.annotate("Pad", (0, 0), xytext=(8, -14), textcoords="offset points", fontsize=9, color=INK)

    # Axes: equal scale so distances look right, light grid, no box
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(colors=QUIET, length=0)
    ax.set_xlabel("Feet east of pad (negative = west)", color=QUIET)
    ax.set_ylabel("Feet north of pad", color=QUIET)
    xs = [r["e_land"] for r in results] + [0]
    left_room = 1150 if min(xs) < 0 else 500        # room for a label west of a western landing
    ax.set_xlim(min(xs) - left_room, max(xs) + 400)

    # Title states the finding; legend explains the colours
    biggest = max(results, key=lambda r: np.hypot(r["e_apo"], r["n_apo"]))
    up = np.hypot(biggest["e_apo"], biggest["n_apo"])
    fig.text(0.06, 0.965, f"{biggest['label'].split(',')[0]} went {up:,.0f} ft sideways before its chutes opened",
             ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.06, 0.928, "GPS ground tracks, feet from each flight's pad, north up, equal scale",
             ha="left", va="top", fontsize=10, color=QUIET)
    handles = [Line2D([], [], color=s["color"], lw=3, label=s["label"]) for s in PHASE_STYLE.values()]
    handles.append(Line2D([], [], color=QUIET, lw=1.1, ls=(0, (4, 4)), label="Net offset, pad to landing"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False,
              fontsize=9, handlelength=1.8, columnspacing=1.2, borderaxespad=0.3)

    fig.text(0.06, 0.012, "Source: GPS tracker files, cleaned by clean_data.py. Phases timed from each altimeter.",
             fontsize=8, color=QUIET)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.09)
    path = out_dir / "ground_tracks.png"
    fig.savefig(path, dpi=200, facecolor="white")
    print(f"Saved {path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
