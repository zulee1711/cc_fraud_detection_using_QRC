"""Plot Arm A Tanh, RFF, and QELM results found under results/arma.

Reads completed metrics.json/config.json run pairs recursively.
Does not run experiments.

Usage from the repository root:
    python results/arma/plot.py
    python results/arma/plot.py --split validation
    python results/arma/plot.py --split test
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent
PLOTS_DIR = RESULTS_DIR / "plots"

# Consistent typography across the figure.
FONT_SIZE = 8
TITLE_SIZE = 9

METRICS = (
    ("average_precision", "All fraud"),
    ("average_precision_amount", "Amount fraud"),
    #  ("average_precision_terminal", "Terminal"),
    ("average_precision_customer", "Customer fraud"),
)

METHODS = (
    ("classical", "Tanh random map"),
    ("rff", "Random Fourier features (RFF)"),
    ("qrc", "QELM"),
)

COLORS = {
    "classical": "#e07a32",
    "rff": "#2a9d8f",
    "qrc": "#3478b9",
}


def get_method(args):
    """Identify Tanh, RFF, or QELM from saved run configuration."""
    command = args.get("command")

    if command == "qrc":
        return "qrc"

    if command == "rff":
        return "rff"

    if command == "classical":
        map_method = args.get(
            "random_map_method",
            args.get("classical_model", "tanh"),
        )

        if map_method == "rff":
            return "rff"
        if map_method in ("tanh", "classical"):
            return "classical"

    return None


def load_runs():
    """Load complete result pairs; skip incomplete or unrelated folders."""
    runs = []

    for metrics_path in sorted(RESULTS_DIR.rglob("metrics.json")):
        try:
            config_path = metrics_path.with_name("config.json")

            if not config_path.is_file():
                print(f"Skipping {metrics_path.parent}: config.json is missing")
                continue

            metrics = json.loads(metrics_path.read_text())
            config = json.loads(config_path.read_text())
            args = config.get("args", {})

            method = get_method(args)
            if method is None:
                continue

            seed = args.get("seed")
            if seed is None:
                print(f"Skipping {metrics_path.parent}: run seed is missing")
                continue

            if not all(split in metrics for split in ("validation", "test")):
                print(
                    f"Skipping {metrics_path.parent}: "
                    "validation or test metrics are incomplete"
                )
                continue

            runs.append(
                {
                    "command": method,
                    "seed": int(seed),
                    "metrics": metrics,
                    "path": metrics_path.parent,
                }
            )

        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"Skipping unreadable run {metrics_path.parent}: {exc}")

    return runs


def plot_split(runs, split):
    """Plot mean AP, standard deviation, and individual seeds."""
    available_metrics = [
        (key, label)
        for key, label in METRICS
        if any(key in run["metrics"].get(split, {}) for run in runs)
    ]

    if not available_metrics:
        print(f"No AP metrics found for the {split} split; no plot created.")
        return

    methods = [
        (command, label)
        for command, label in METHODS
        if any(run["command"] == command for run in runs)
    ]

    ncols = 2
    nrows = int(np.ceil(len(available_metrics) / ncols))

    fig = plt.figure(figsize=(7, 2 * nrows))
    gs = fig.add_gridspec(2, 4)

    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 1:3]),
    ]

    method_y = {command: i for i, (command, _) in enumerate(methods)}

    for ax, (metric, title) in zip(axes, available_metrics):

        for command, _ in methods:
            values = [
                (
                    run["seed"],
                    run["metrics"][split][metric],
                )
                for run in runs
                if run["command"] == command
                and metric in run["metrics"].get(split, {})
                and isinstance(
                    run["metrics"][split][metric],
                    (int, float),
                )
                and np.isfinite(run["metrics"][split][metric])
            ]

            if not values:
                continue

            scores = np.asarray(
                [score for _, score in values],
                dtype=float,
            )

            mean = float(scores.mean())
            std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
            y = method_y[command]

            # Individual seed scores, vertically jittered to reduce overlap.
            jitter = (
                np.linspace(-0.10, 0.10, len(scores))
                if len(scores) > 1
                else np.array([0.0])
            )

            ax.scatter(
                scores,
                y + jitter,
                color=COLORS[command],
                s=22,
                alpha=0.65,
                edgecolors="none",
                zorder=3,
            )

            # Mean and ±1 standard deviation.
            ax.errorbar(
                mean,
                y,
                xerr=std,
                fmt="D",
                markersize=5,
                color=COLORS[command],
                ecolor="#333333",
                elinewidth=1.2,
                capsize=3,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=4,
            )

        ax.set_title(title, fontsize=TITLE_SIZE, pad=6)
        ax.set_xlim(0, 1)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()

        ax.grid(axis="x", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(
            axis="x",
            labelsize=FONT_SIZE,
            length=3,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # Hide unused panels when fewer than four metrics are available.
    for ax in axes[len(available_metrics) :]:
        ax.set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[command],
            marker="D",
            linestyle="-",
            markersize=5,
            linewidth=1.2,
            label=label,
        )
        for command, label in methods
    ]

    fig.suptitle(
        " ",
        fontsize=TITLE_SIZE,
        y=0.97,
    )

    # Reserve space below the plots for the x-axis label and legend.
    # Place the label above the legend with a clear gap between them.
    fig.supxlabel(
        "Average precision",
        fontsize=FONT_SIZE,
        y=0.155,
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=8,
    )

    # Use manual margins to avoid tight_layout moving the label or legend.
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.90,
        bottom=0.25,
        hspace=0.40,
        wspace=0.25,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output = PLOTS_DIR / f"ap_{split}_vs_feature_map.pdf"

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("both", "validation", "test"),
        default="both",
        help="which evaluation split(s) to plot",
    )

    args = parser.parse_args()
    runs = load_runs()

    if not runs:
        raise SystemExit(
            f"No complete Tanh, RFF, or QELM runs found under {RESULTS_DIR}"
        )

    for command, label in METHODS:
        seeds = sorted({run["seed"] for run in runs if run["command"] == command})
        if seeds:
            print(f"{label}: loaded seeds {seeds}")

    splits = ("validation", "test") if args.split == "both" else (args.split,)

    for split in splits:
        plot_split(runs, split)


if __name__ == "__main__":
    main()
