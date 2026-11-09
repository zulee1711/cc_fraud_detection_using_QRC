from pathlib import Path
import re
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent / "max_delay"
OUTPUT_DIR = RESULTS_DIR / "plots"

OUTPUT_FILE = OUTPUT_DIR / "normalized_ipc_vs_max_delay.pdf"
LAST_ROW_OUTPUT_FILE = OUTPUT_DIR / "normalized_ipc_vs_max_delay_window20.pdf"
TOTAL_CAPACITY_OUTPUT_FILE = OUTPUT_DIR / "normalized_total_capacity_vs_max_delay.pdf"

WINDOW_LENGTHS = (5, 10, 15, 20)

HAMILTONIANS = (
    "random",
    "tfim",
    "tfim-longitudinal",
    "tfim-native",
    "native-exchange",
)

BACKENDS = ("shots", "estimator-exact")

METRICS = (
    ("normalized_degree_1", "Degree 1"),
    ("normalized_degree_2", "Degree 2"),
    ("normalized_degree_3", "Degree 3"),
    ("normalized_total_capacity", "Total capacity"),
)

TOTAL_CAPACITY_METRIC = "normalized_total_capacity"

COLORS = {
    "random": "#3478b9",
    "tfim": "#e07a32",
    "tfim-longitudinal": "#2a9d8f",
    "tfim-native": "#a05ab5",
    "native-exchange": "#c44e52",
}

MARKERS = {
    "random": "o",
    "tfim": "s",
    "tfim-longitudinal": "^",
    "tfim-native": "D",
    "native-exchange": "P",
}

LINESTYLES = {
    "shots": "-",
    "estimator-exact": "--",
}


# ---------------------------------------------------------------------
# Parse filenames and results
# ---------------------------------------------------------------------

FILENAME_PATTERN = re.compile(
    r"(?P<backend>shots|estimator-exact)"
    r"_precision(?P<precision>[\d.]+)"
    r"_depth(?P<depth>\d+)"
    r"_window(?P<window_length>\d+)"
    r"_maxdelay(?P<max_delay>\d+)"
    r"_seed(?P<seed>\d+)\.txt$"
)


def parse_log(path):
    """Extract experiment parameters and normalized IPC metrics."""
    match = FILENAME_PATTERN.search(path.name)

    if not match:
        return None

    params = match.groupdict()
    params["window_length"] = int(params["window_length"])
    params["max_delay"] = int(params["max_delay"])
    params["seed"] = int(params["seed"])

    hamiltonian = path.parent.name

    if hamiltonian not in HAMILTONIANS:
        return None

    text = path.read_text(errors="replace")

    ipc_match = re.search(
        r"Information Processing Capacity \(IPC\)\s*(\{.*?\})",
        text,
        flags=re.DOTALL,
    )

    if not ipc_match:
        print(f"Warning: no IPC result found in {path}")
        return None

    try:
        ipc = json.loads(ipc_match.group(1))
    except json.JSONDecodeError:
        print(f"Warning: could not parse IPC JSON in {path}")
        return None

    metrics = {}

    for metric, _ in METRICS:
        value = ipc.get(metric)

        if value is not None:
            metrics[metric] = float(value)

    return {
        **params,
        "hamiltonian": hamiltonian,
        "metrics": metrics,
    }


def load_results():
    """Load all matching log files."""
    records = []

    for path in RESULTS_DIR.rglob("*.txt"):
        record = parse_log(path)

        if record is not None:
            records.append(record)

    if not records:
        raise RuntimeError(f"No matching IPC logs found under {RESULTS_DIR}")

    print(f"Loaded {len(records)} log files.")
    return records


# ---------------------------------------------------------------------
# Aggregate results across seeds
# ---------------------------------------------------------------------


def aggregate_results(records):
    """Calculate means and standard deviations across seeds."""
    grouped = {}

    for record in records:
        key = (
            record["hamiltonian"],
            record["backend"],
            record["window_length"],
            record["max_delay"],
        )

        grouped.setdefault(
            key,
            {metric: [] for metric, _ in METRICS},
        )

        for metric, value in record["metrics"].items():
            grouped[key][metric].append(value)

    aggregated = {}

    for key, metric_values in grouped.items():
        aggregated[key] = {}

        for metric, values in metric_values.items():
            if not values:
                continue

            values = np.asarray(values, dtype=float)

            aggregated[key][metric] = {
                "mean": float(np.mean(values)),
                "std": (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
                "n": len(values),
            }

    return aggregated


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------


def plot_metric(
    ax,
    aggregated,
    window_length,
    metric,
    metric_title,
    show_ylabel=True,
):
    """Plot a metric against maximum delay for one window length."""
    values_for_axis = []

    for hamiltonian in HAMILTONIANS:
        for backend in BACKENDS:
            points = []

            for (
                h,
                b,
                window,
                max_delay,
            ), values in aggregated.items():

                if (
                    h == hamiltonian
                    and b == backend
                    and window == window_length
                    and metric in values
                ):
                    mean = values[metric]["mean"]
                    std = values[metric]["std"]

                    points.append((max_delay, mean, std))
                    values_for_axis.extend([mean - std, mean + std])

            if not points:
                continue

            points.sort(key=lambda point: point[0])

            x = np.array([point[0] for point in points])
            y = np.array([point[1] for point in points])
            yerr = np.array([point[2] for point in points])

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=COLORS[hamiltonian],
                marker=MARKERS[hamiltonian],
                linestyle=LINESTYLES[backend],
                markersize=4,
                linewidth=1.2,
                elinewidth=0.8,
                capsize=2,
                alpha=0.85,
            )

    ax.set_title(metric_title, fontsize=11, pad=10)
    ax.set_xlabel("Maximum delay")

    if show_ylabel:
        ax.set_ylabel("Normalized IPC")

    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=8)

    valid_delays = sorted(
        {
            max_delay
            for (_, _, window, max_delay) in aggregated
            if window == window_length
        }
    )

    if valid_delays:
        ax.set_xticks(valid_delays)

    # Independently scale the y-axis of this subplot.
    if values_for_axis:
        ymin = min(values_for_axis)
        ymax = max(values_for_axis)
        span = ymax - ymin

        padding = 0.05 * span if span > 0 else max(abs(ymax) * 0.05, 0.01)

        ax.set_ylim(ymin - padding, ymax + padding)


def add_legends(fig, reservoir_y=0.015, backend_y=0.015):
    """Add reservoir and backend legends."""
    reservoir_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[hamiltonian],
            marker=MARKERS[hamiltonian],
            linestyle="None",
            markersize=5,
            label=hamiltonian,
        )
        for hamiltonian in HAMILTONIANS
    ]

    backend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=LINESTYLES[backend],
            linewidth=1.5,
            label=backend,
        )
        for backend in BACKENDS
    ]

    fig.legend(
        handles=reservoir_handles,
        title="Reservoir",
        loc="lower center",
        bbox_to_anchor=(0.36, reservoir_y),
        ncol=len(HAMILTONIANS),
        frameon=False,
        fontsize=10,
    )

    fig.legend(
        handles=backend_handles,
        title="Backend type",
        loc="lower center",
        bbox_to_anchor=(0.82, backend_y),
        ncol=1,
        frameon=False,
        fontsize=10,
    )


# ---------------------------------------------------------------------
# Full 4x4 plot
# ---------------------------------------------------------------------


def plot_results(aggregated):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=len(WINDOW_LENGTHS),
        ncols=len(METRICS),
        figsize=(4 * len(METRICS), 4 * len(WINDOW_LENGTHS)),
        sharex="col",
        sharey=False,
        squeeze=False,
    )

    for row, window_length in enumerate(WINDOW_LENGTHS):
        for col, (metric, metric_title) in enumerate(METRICS):
            ax = axes[row, col]

            plot_metric(
                ax=ax,
                aggregated=aggregated,
                window_length=window_length,
                metric=metric,
                metric_title=metric_title,
                show_ylabel=(col == 0),
            )

            if row > 0:
                ax.set_title("")

            if col == 0:
                ax.set_ylabel(
                    f"Window length = {window_length}\nNormalized IPC",
                    labelpad=8,
                )

            if row < len(WINDOW_LENGTHS) - 1:
                ax.set_xlabel("")

    add_legends(fig)

    fig.suptitle(
        "",
        fontsize=14,
        y=0.99,
    )

    fig.tight_layout(rect=[0.04, 0.12, 1, 0.95])
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")

    print(f"Saved full plot to {OUTPUT_FILE}")
    plt.close(fig)


# ---------------------------------------------------------------------
# Separate 2x2 plot for window length 20
# ---------------------------------------------------------------------


def plot_last_row(aggregated):
    """Save the four metrics for window length 20 as a 2x2 plot."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    window_length = WINDOW_LENGTHS[-1]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 7),
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    for idx, (metric, metric_title) in enumerate(METRICS):
        ax = axes[idx // 2, idx % 2]

        plot_metric(
            ax=ax,
            aggregated=aggregated,
            window_length=window_length,
            metric=metric,
            metric_title=metric_title,
            show_ylabel=True,
        )

    add_legends(fig, reservoir_y=0.015, backend_y=0.015)

    fig.suptitle(
        f"",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0.14, 1, 0.93])
    fig.savefig(LAST_ROW_OUTPUT_FILE, bbox_inches="tight")

    print(f"Saved window-20 plot to {LAST_ROW_OUTPUT_FILE}")
    plt.close(fig)


# ---------------------------------------------------------------------
# Separate 2x2 plot for total capacity across all window lengths
# ---------------------------------------------------------------------


def plot_total_capacity(aggregated):
    """Save total capacity for all window lengths in a 2x2 plot."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 7),
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    for idx, window_length in enumerate(WINDOW_LENGTHS):
        ax = axes[idx // 2, idx % 2]

        plot_metric(
            ax=ax,
            aggregated=aggregated,
            window_length=window_length,
            metric=TOTAL_CAPACITY_METRIC,
            metric_title=f"Window length = {window_length}",
            show_ylabel=True,
        )

    add_legends(fig, reservoir_y=0.015, backend_y=0.015)

    fig.suptitle(
        "",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0.14, 1, 0.93])
    fig.savefig(TOTAL_CAPACITY_OUTPUT_FILE, bbox_inches="tight")

    print(f"Saved total-capacity plot to {TOTAL_CAPACITY_OUTPUT_FILE}")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    records = load_results()
    aggregated = aggregate_results(records)

    plot_results(aggregated)
    plot_last_row(aggregated)
    plot_total_capacity(aggregated)


if __name__ == "__main__":
    main()
