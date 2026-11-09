from pathlib import Path
import json
import re
import warnings
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# Configuration
# ================================================================

ROOT = Path("results/benchmarks/ipc/observables_copy")
PLOT_DIR = Path("results/benchmarks/ipc/plots")

HAMILTONIANS = [
    "random",
    "tfim",
    "tfim-longitudinal",
    "tfim-native",
    "native-exchange",
]

OBSERVABLES = ["Z", "Z+ZZ", "XYZ"]
PRECISIONS = ["0.01", "0.05", "0.1"]

SERIES = [
    ("estimator-exact", None, "Exact estimator"),
    ("shots", "0.01", "Shots (precision 0.01)"),
    ("shots", "0.05", "Shots (precision 0.05)"),
    ("shots", "0.1", "Shots (precision 0.1)"),
]

FIGURE_NAME = "ipc_observables_normalized.pdf"
DEGREE_FIGURE_NAME = "ipc_observables_normalized_degrees.pdf"
ABSOLUTE_FIGURE_NAME = "ipc_observables_absolute_degrees.pdf"

DPI = 200
SHOW_PLOTS = True
SHOW_STD = True


# ================================================================
# Filename parsing
# ================================================================


def parse_filename(path):
    name = path.stem
    name = re.sub(r"_copy\d+$", "", name)

    match = re.match(
        r"^(estimator-exact|shots|spinpulse)_(.+)$",
        name,
        re.IGNORECASE,
    )

    if not match:
        return None

    backend = match.group(1).lower()
    rest = match.group(2)

    seed_match = re.search(r"_seed(\d+)", rest, re.IGNORECASE)
    seed = int(seed_match.group(1)) if seed_match else None

    if seed_match:
        rest = rest[: seed_match.start()]

    precision_match = re.search(
        r"_precision(.+)$",
        rest,
        re.IGNORECASE,
    )

    if precision_match:
        precision_text = precision_match.group(1)
        rest = rest[: precision_match.start()]
        precisions = re.findall(r"\d+(?:\.\d+)?", precision_text)
    else:
        precisions = []

    hamiltonian = None
    observable = None

    for candidate in sorted(HAMILTONIANS, key=len, reverse=True):
        prefix = candidate + "_"

        if rest.startswith(prefix):
            hamiltonian = candidate
            observable = rest[len(prefix) :]
            break

    if hamiltonian is None:
        return None

    observable = observable.strip("_- ")

    aliases = {
        "Z": "Z",
        "Z_ZZ": "Z+ZZ",
        "Z+ZZ": "Z+ZZ",
        "XYZ": "XYZ",
    }

    observable = aliases.get(observable, observable)

    return {
        "backend": backend,
        "hamiltonian": hamiltonian,
        "observable": observable,
        "precisions": precisions,
        "seed": seed,
    }


# ================================================================
# Read IPC values
# ================================================================


def read_log_values(path, config):
    """
    Extract normalized and absolute IPC values from IPC JSON blocks.

    Each metrics dictionary contains the available values:
        normalized_total
        normalized_degree_1, normalized_degree_2, normalized_degree_3
        absolute_total
        absolute_degree_1, absolute_degree_2, absolute_degree_3
    """

    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return []

    blocks = re.findall(
        r"Information Processing Capacity\s*\(IPC\)\s*(\{.*?\})",
        text,
        flags=re.DOTALL,
    )

    if not blocks:
        warnings.warn(f"No IPC JSON block found in {path}")
        return []

    parsed = []

    for block in blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue

        metrics = {}

        # Total capacity
        for source_key, target_key in [
            ("normalized_total_capacity", "normalized_total"),
            ("total_capacity", "absolute_total"),
        ]:
            value = data.get(source_key)

            if value is not None:
                metrics[target_key] = float(value)

        # Individual degree capacities
        for degree in range(1, 4):
            for prefix, output_prefix in [
                ("normalized_", "normalized_"),
                ("", "absolute_"),
            ]:
                source_key = f"{prefix}degree_{degree}"
                target_key = f"{output_prefix}degree_{degree}"

                value = data.get(source_key)

                if value is not None:
                    metrics[target_key] = float(value)

        if metrics:
            parsed.append(metrics)

    if not parsed:
        warnings.warn(f"No usable IPC values found in {path}")
        return []

    if config["backend"] == "estimator-exact":
        return [(None, parsed[-1])]

    precisions = config["precisions"]

    if len(precisions) == 1 and len(parsed) == 1:
        return [(precisions[0], parsed[0])]

    if len(precisions) > 1 and len(parsed) == len(precisions):
        warnings.warn(f"{path.name}: matching IPC values to precisions by order.")
        return list(zip(precisions, parsed))

    if len(precisions) == 1:
        return [(precisions[0], parsed[-1])]

    warnings.warn(
        f"Cannot match {len(parsed)} IPC block(s) "
        f"to precisions {precisions} in {path.name}; skipping."
    )

    return []


# ================================================================
# Load and aggregate results
# ================================================================


def load_observable_results(root=ROOT):
    """
    Returns four dictionaries:

    normalized_total_results:
        (observable, hamiltonian, backend, precision) -> values

    normalized_degree_results:
        (degree, observable, hamiltonian, backend, precision) -> values

    absolute_total_results:
        (observable, hamiltonian, backend, precision) -> values

    absolute_degree_results:
        (degree, observable, hamiltonian, backend, precision) -> values
    """

    if not root.exists():
        raise FileNotFoundError(f"Log directory not found: {root}")

    normalized_total_results = defaultdict(list)
    normalized_degree_results = defaultdict(list)

    absolute_total_results = defaultdict(list)
    absolute_degree_results = defaultdict(list)

    files = sorted(
        path
        for hamiltonian in HAMILTONIANS
        for path in (root / hamiltonian).glob("*.txt")
    )

    loaded_normalized_total = 0
    loaded_normalized_degrees = 0
    loaded_absolute_total = 0
    loaded_absolute_degrees = 0
    skipped = 0

    print(f"Found {len(files)} log files in {root}\n")

    for path in files:
        config = parse_filename(path)

        if config is None:
            print(f"SKIP: Unrecognized filename: {path}")
            skipped += 1
            continue

        if config["observable"] not in OBSERVABLES:
            continue

        if config["hamiltonian"] not in HAMILTONIANS:
            continue

        if config["hamiltonian"] != path.parent.name:
            print(f"SKIP: Hamiltonian folder does not match filename: {path}")
            skipped += 1
            continue

        # SpinPulse is intentionally excluded.
        if config["backend"] not in {"shots", "estimator-exact"}:
            continue

        values = read_log_values(path, config)

        if not values:
            skipped += 1
            continue

        for precision, metrics in values:

            if config["backend"] == "shots":
                if precision not in PRECISIONS:
                    print(f"SKIP: Unsupported precision {precision}: " f"{path.name}")
                    skipped += 1
                    continue
            else:
                precision = None

            # ----------------------------------------------------
            # Normalized total capacity
            # ----------------------------------------------------

            if "normalized_total" in metrics:
                key = (
                    config["observable"],
                    config["hamiltonian"],
                    config["backend"],
                    precision,
                )

                normalized_total_results[key].append(metrics["normalized_total"])
                loaded_normalized_total += 1

            # ----------------------------------------------------
            # Normalized individual degrees
            # ----------------------------------------------------

            for degree in range(1, 4):
                metric = f"normalized_degree_{degree}"

                if metric not in metrics:
                    continue

                key = (
                    degree,
                    config["observable"],
                    config["hamiltonian"],
                    config["backend"],
                    precision,
                )

                normalized_degree_results[key].append(metrics[metric])
                loaded_normalized_degrees += 1

            # ----------------------------------------------------
            # Absolute total capacity
            # ----------------------------------------------------

            if "absolute_total" in metrics:
                key = (
                    config["observable"],
                    config["hamiltonian"],
                    config["backend"],
                    precision,
                )

                absolute_total_results[key].append(metrics["absolute_total"])
                loaded_absolute_total += 1

            # ----------------------------------------------------
            # Absolute individual degrees
            # ----------------------------------------------------

            for degree in range(1, 4):
                metric = f"absolute_degree_{degree}"

                if metric not in metrics:
                    continue

                key = (
                    degree,
                    config["observable"],
                    config["hamiltonian"],
                    config["backend"],
                    precision,
                )

                absolute_degree_results[key].append(metrics[metric])
                loaded_absolute_degrees += 1

    print(f"\nLoaded normalized total IPC values: {loaded_normalized_total}")
    print(f"Loaded normalized degree values: {loaded_normalized_degrees}")
    print(f"Loaded absolute total IPC values: {loaded_absolute_total}")
    print(f"Loaded absolute degree values: {loaded_absolute_degrees}")
    print(f"Skipped files/values: {skipped}")

    return (
        normalized_total_results,
        normalized_degree_results,
        absolute_total_results,
        absolute_degree_results,
    )


# ================================================================
# Plotting helper
# ================================================================


def plot_metric(
    ax,
    observable,
    metric,
    results,
    normalized=True,
    show_std=True,
):
    """
    Plot the four series for one observable and one IPC metric.

    metric:
        "total", "degree_1", "degree_2", or "degree_3"

    normalized:
        True  -> normalized IPC, y-axis fixed to [0, 1]
        False -> absolute IPC, y-axis automatically scaled
    """

    n_series = len(SERIES)
    bar_width = 0.18
    group_spacing = 1.0
    group_centres = np.arange(len(HAMILTONIANS)) * group_spacing

    for series_index, (backend, precision, label) in enumerate(SERIES):
        offset = (series_index - (n_series - 1) / 2) * bar_width

        means = []
        errors = []

        for hamiltonian in HAMILTONIANS:

            if metric == "total":
                key = (
                    observable,
                    hamiltonian,
                    backend,
                    precision,
                )
            else:
                degree = int(metric.split("_")[1])

                key = (
                    degree,
                    observable,
                    hamiltonian,
                    backend,
                    precision,
                )

            values = results.get(key, [])

            if values:
                means.append(float(np.mean(values)))

                errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            else:
                means.append(np.nan)
                errors.append(0.0)

        if backend == "estimator-exact":
            ax.bar(
                group_centres + offset,
                means,
                width=bar_width,
                yerr=errors if show_std else None,
                capsize=3,
                label=label,
                facecolor="none",
                edgecolor="C4",
                linestyle="--",
                linewidth=1.8,
                hatch="///",
            )

        else:
            ax.bar(
                group_centres + offset,
                means,
                width=bar_width,
                yerr=errors if show_std else None,
                capsize=3,
                label=label,
            )

    ax.set_xticks(group_centres)
    ax.set_xticklabels(
        HAMILTONIANS,
        rotation=35,
        ha="right",
    )

    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    if normalized:
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))


# ================================================================
# Figure 1: Normalized total IPC
# ================================================================


def plot_total_capacity(
    total_results,
    output_dir=PLOT_DIR,
    show_std=SHOW_STD,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 8),
        sharey=False,
    )

    for ax, observable in zip(axes, OBSERVABLES):
        plot_metric(
            ax,
            observable,
            "total",
            total_results,
            normalized=True,
            show_std=show_std,
        )

        ax.set_title(f"Observables: {observable}")

    fig.supylabel("Normalized total IPC", x=0.005)
    fig.supxlabel("Hamiltonian", y=0.25)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.19),
        ncol=4,
        frameon=False,
        fontsize=13,
    )

    fig.suptitle("", y=0.98)
    fig.tight_layout(rect=[0.02, 0.23, 0.96, 0.92])

    output_path = output_dir / FIGURE_NAME
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")

    print(f"\nSaved plot to: {output_path}")

    return fig


# ================================================================
# Generic 3x4 degree-wise figure
#
# Rows:
#     Z
#     Z+ZZ
#     XYZ
#
# Columns:
#     Total capacity
#     Degree 1
#     Degree 2
#     Degree 3
# ================================================================


def plot_degree_grid(
    total_results,
    degree_results,
    output_path,
    normalized=True,
    show_std=SHOW_STD,
):
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(27, 19),
        sharey=False,
    )

    if normalized:
        metrics = [
            ("total", "Normalized total IPC"),
            ("degree_1", "Normalized degree 1"),
            ("degree_2", "Normalized degree 2"),
            ("degree_3", "Normalized degree 3"),
        ]
        ylabel_prefix = "Normalized IPC"
    else:
        metrics = [
            ("total", "Total IPC"),
            ("degree_1", "Degree 1"),
            ("degree_2", "Degree 2"),
            ("degree_3", "Degree 3"),
        ]
        ylabel_prefix = "IPC"

    for row, observable in enumerate(OBSERVABLES):

        for col, (metric, title) in enumerate(metrics):
            ax = axes[row, col]

            if metric == "total":
                data = total_results
            else:
                data = degree_results

            plot_metric(
                ax,
                observable,
                metric,
                data,
                normalized=normalized,
                show_std=show_std,
            )

            ax.set_title(title, fontsize=14, pad=12)
            ax.tick_params(axis="both", labelsize=9)

            if col == 0:
                ax.set_ylabel(
                    f"{observable}\n{ylabel_prefix}",
                    labelpad=10,
                )

            if row == 2:
                ax.set_xlabel("Hamiltonian", labelpad=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=False,
        fontsize=13,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.96,
        bottom=0.13,
        wspace=0.1,
        hspace=0.3,
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    print(f"Saved plot to: {output_path}")

    return fig


# ================================================================
# Figure 2: Normalized total capacity + individual degrees
# ================================================================


def plot_normalized_degree_capacities(
    total_results,
    degree_results,
    output_dir=PLOT_DIR,
    show_std=SHOW_STD,
):
    output_path = output_dir / DEGREE_FIGURE_NAME

    return plot_degree_grid(
        total_results=total_results,
        degree_results=degree_results,
        output_path=output_path,
        normalized=True,
        show_std=show_std,
    )


# ================================================================
# Figure 3: Absolute total capacity + individual degrees
# ================================================================


def plot_absolute_degree_capacities(
    total_results,
    degree_results,
    output_dir=PLOT_DIR,
    show_std=SHOW_STD,
):
    output_path = output_dir / ABSOLUTE_FIGURE_NAME

    return plot_degree_grid(
        total_results=total_results,
        degree_results=degree_results,
        output_path=output_path,
        normalized=False,
        show_std=show_std,
    )


# ================================================================
# Missing configurations
# ================================================================


def print_missing_configurations(total_results):
    print("\nMissing configurations:")
    missing_count = 0

    for observable in OBSERVABLES:
        for hamiltonian in HAMILTONIANS:
            for backend, precision, label in SERIES:
                key = (
                    observable,
                    hamiltonian,
                    backend,
                    precision,
                )

                if not total_results.get(key):
                    missing_count += 1
                    print(f"  {observable:4s} | " f"{hamiltonian:20s} | {label}")

    if missing_count == 0:
        print("  None")


# ================================================================
# Main plotting function
# ================================================================


def plot_observables(
    root=ROOT,
    output_dir=PLOT_DIR,
    show=SHOW_PLOTS,
    show_std=SHOW_STD,
):
    (
        normalized_total_results,
        normalized_degree_results,
        absolute_total_results,
        absolute_degree_results,
    ) = load_observable_results(root)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Normalized total IPC (1x3).
    fig1 = plot_total_capacity(
        normalized_total_results,
        output_dir=output_dir,
        show_std=show_std,
    )

    # Figure 2: Normalized total and degree-wise IPC (3x4).
    fig2 = plot_normalized_degree_capacities(
        normalized_total_results,
        normalized_degree_results,
        output_dir=output_dir,
        show_std=show_std,
    )

    # Figure 3: Absolute total and degree-wise IPC (3x4).
    fig3 = plot_absolute_degree_capacities(
        absolute_total_results,
        absolute_degree_results,
        output_dir=output_dir,
        show_std=show_std,
    )

    print_missing_configurations(normalized_total_results)

    print("\nPDF output check:")
    for filename in [
        FIGURE_NAME,
        DEGREE_FIGURE_NAME,
        ABSOLUTE_FIGURE_NAME,
    ]:
        path = output_dir / filename

        if path.exists():
            print(f"  OK: {path} ({path.stat().st_size:,} bytes)")
        else:
            print(f"  ERROR: File was not created: {path}")

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

    return (
        normalized_total_results,
        normalized_degree_results,
        absolute_total_results,
        absolute_degree_results,
    )


if __name__ == "__main__":
    plot_observables()
