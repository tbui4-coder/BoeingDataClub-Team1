"""
models.py - Team 1 machine learning models (Deliverable 4)

Trains three models to predict how the rocket comes down, and tests each one
fairly: train on one flight, test on the other, then swap.

    Linear regression       a straight line
    Polynomial regression   a curve that can bend (degree 2 = quadratic)
    Decision tree           "if this, then that" steps

What the models predict
    Input  (feature): seconds since apogee
    Output (target):  how far the rocket has fallen since apogee, in feet
    Height above ground = that flight's apogee - predicted fall.

    Lining both flights up at apogee means a model trained on one flight can be
    used on the other even though they peaked at different heights.

Why not predict landing displacement directly?
    We only have two landings, and a model can't learn from two examples.
    Predicting the descent (hundreds of points per flight) is the first step:
    descent time multiplied by wind gives drift, which sets landing displacement.

How to run (after clean_data.py):
    pip install scikit-learn matplotlib
    python models.py

Output:
    cleaned/model_results.csv       error of every model, both test directions
    cleaned/model_predictions.csv   actual vs predicted height, every 0.5 s
    figures/model_predictions.png   predicted vs actual descent
    figures/model_errors.png        average error of each model
    "Model Results" tab in cleaned/Team1_Cleaned_Data.xlsx
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from clean_data import CLEANED_FOLDER, CSV_FLOAT_FORMAT, FLIGHTS, OUTPUT_WORKBOOK, write_sheet

HERE = Path(__file__).parent
STEP_S = 0.5            # put every flight on the same 0.5 s time grid

# ---------------------------------------------------------------------------
# The models. To try another one, add a line here: every model is trained,
# tested, saved and plotted the same way.
# ---------------------------------------------------------------------------
MODELS = {
    "Linear regression": LinearRegression(),
    "Polynomial regression (degree 2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "Decision tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=0),
}
MODEL_COLORS = {"Linear regression": "#2a78d6",
                "Polynomial regression (degree 2)": "#eb6834",
                "Decision tree": "#1baf7a"}
INK, QUIET, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


# ---------------------------------------------------------------------------
# 1. Build one tidy table of descent data per flight
# ---------------------------------------------------------------------------
def descent_table(flight, events, clean_dir):
    """Height during descent on an even 0.5 s grid, from the height source the
    cleaning step trusted for this flight (barometer, or GPS if the barometer
    disagreed with GPS by more than 5 %)."""
    ev = events[flight]
    if ev["height_source"] == "barometer":
        df = pd.read_csv(clean_dir / f"{flight}_altimeter.csv")
        t, h = df["time_s"].to_numpy(), df["height_agl_smooth_ft"].to_numpy()
    else:
        df = pd.read_csv(clean_dir / f"{flight}_gps.csv")
        t, h = df["time_s"].to_numpy(), df["alt_agl_ft"].to_numpy()

    grid = np.arange(ev["apogee_s"], ev["touchdown_s"] + 1e-9, STEP_S)
    height = np.interp(grid, t, h)
    apogee_ft = float(height[0])
    ground_ft = float(np.median(h[t >= ev["touchdown_s"]]))
    return pd.DataFrame({
        "flight": flight,
        "time_since_apogee_s": grid - ev["apogee_s"],
        "height_agl_ft": height - ground_ft,             # 0 ft = where it landed
        "fall_since_apogee_ft": apogee_ft - height,
        "apogee_ft": apogee_ft - ground_ft,
        "phase": np.where(grid < ev["main_open_s"], "drogue", "main"),
    })


# ---------------------------------------------------------------------------
# 2. Train on one flight, test on the other (leave-one-flight-out)
# ---------------------------------------------------------------------------
def evaluate(data):
    """Return (results table, predictions table)."""
    results, predictions = [], []
    flights = data["flight"].unique()
    for test in flights:
        train_df = data[data["flight"] != test]
        test_df = data[data["flight"] == test]
        X_train = train_df[["time_since_apogee_s"]].to_numpy()
        y_train = train_df["fall_since_apogee_ft"].to_numpy()
        X_test = test_df[["time_since_apogee_s"]].to_numpy()
        apogee = test_df["apogee_ft"].iloc[0]
        actual = test_df["height_agl_ft"].to_numpy()

        pred_table = test_df[["flight", "time_since_apogee_s", "phase", "height_agl_ft"]].copy()
        pred_table.insert(0, "trained_on", ", ".join(f for f in flights if f != test))
        for name, model in MODELS.items():
            model.fit(X_train, y_train)
            predicted = apogee - model.predict(X_test)
            err = predicted - actual
            pred_table[name] = predicted
            results.append({
                "model": name,
                "trained_on": ", ".join(f for f in flights if f != test),
                "tested_on": test,
                "rmse_ft": float(np.sqrt(mean_squared_error(actual, predicted))),
                "mae_ft": float(mean_absolute_error(actual, predicted)),
                "max_error_ft": float(np.abs(err).max()),
                "r2": float(r2_score(actual, predicted)),
                "mae_drogue_ft": float(np.abs(err[test_df["phase"] == "drogue"]).mean()),
                "mae_main_ft": float(np.abs(err[test_df["phase"] == "main"]).mean()),
            })
        predictions.append(pred_table)
    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Charts
# ---------------------------------------------------------------------------
def style_axes(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(colors=QUIET, length=0)


def plot_predictions(pred, events, path):
    tests = list(pred["flight"].unique())
    fig, axes = plt.subplots(1, len(tests), figsize=(12, 5.6), sharey=True)
    for ax, test in zip(np.atleast_1d(axes), tests):
        p = pred[pred["flight"] == test]
        t = p["time_since_apogee_s"]
        ax.plot(t, p["height_agl_ft"], color=INK, lw=3.2, label="Actual", zorder=3)
        for name in MODELS:
            ax.plot(t, p[name], color=MODEL_COLORS[name], lw=1.8, label=name, zorder=4)
        main_t = events[test]["main_open_s"] - events[test]["apogee_s"]
        main_h = float(np.interp(main_t, t, p["height_agl_ft"]))
        ax.annotate("main chute opens", (main_t, main_h), xytext=(18, 46), textcoords="offset points",
                    fontsize=9, color=QUIET, arrowprops=dict(arrowstyle="-", color=QUIET, lw=0.8))
        style_axes(ax)
        ax.set_title(f"Tested on {test}, trained on {p['trained_on'].iloc[0]}", loc="left",
                     fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel("Seconds since apogee", color=QUIET)
        ax.set_ylim(bottom=-200)
    np.atleast_1d(axes)[0].set_ylabel("Height above ground (ft)", color=QUIET)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.885), ncol=4, frameon=False, fontsize=9.5)
    fig.text(0.06, 0.965, "Every model misses where the main chute opens on the other flight",
             fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.918, "Predicted vs actual height during descent, each model trained on one flight and tested on the other",
             fontsize=10, color=QUIET, va="top")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.12, wspace=0.08)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_errors(results, path):
    names = list(MODELS)
    tests = list(results["tested_on"].unique())
    shades = {tests[0]: 1.0, tests[-1]: 0.55}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bar_h = 0.36
    for j, test in enumerate(tests):
        for i, name in enumerate(names):
            r = results[(results["model"] == name) & (results["tested_on"] == test)].iloc[0]
            y = i + (j - (len(tests) - 1) / 2) * (bar_h + 0.04)
            ax.barh(y, r["mae_ft"], height=bar_h, color=MODEL_COLORS[name], alpha=shades.get(test, 1.0))
            ax.text(r["mae_ft"] + 8, y, f"{r['mae_ft']:,.0f} ft  (tested on {test})", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=INK)
    ax.invert_yaxis()
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average error in height (ft), lower is better", color=QUIET)
    ax.set_xlim(0, results["mae_ft"].max() * 1.35)
    best = results.groupby("model")["mae_ft"].mean().idxmin()
    fig.text(0.02, 0.96, f"{best} has the smallest average error", fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.89, "Mean absolute error when predicting the flight the model was not trained on",
             fontsize=10, color=QUIET, va="top")
    fig.subplots_adjust(left=0.27, right=0.97, top=0.8, bottom=0.15)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    clean_dir = HERE / CLEANED_FOLDER
    fig_dir = HERE / "figures"
    fig_dir.mkdir(exist_ok=True)
    events = pd.read_csv(clean_dir / "events.csv").set_index("flight").to_dict("index")

    data = pd.concat([descent_table(fc["flight"], events, clean_dir) for fc in FLIGHTS], ignore_index=True)
    print("Descent rows per flight:", data.groupby("flight").size().to_dict())

    results, pred = evaluate(data)
    results.to_csv(clean_dir / "model_results.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    pred.to_csv(clean_dir / "model_predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    with pd.option_context("display.width", 200, "display.float_format", "{:,.1f}".format):
        print(results.to_string(index=False))

    plot_predictions(pred, events, fig_dir / "model_predictions.png")
    plot_errors(results, fig_dir / "model_errors.png")

    wb_path = clean_dir / OUTPUT_WORKBOOK
    wb = load_workbook(wb_path)
    write_sheet(wb, "Model Results", results.round(2), after="Metrics", widths={"model": 34})
    wb.save(wb_path)
    print(f"\nSaved model results, predictions and two charts (figures/), and the 'Model Results' tab.")


if __name__ == "__main__":
    main()
