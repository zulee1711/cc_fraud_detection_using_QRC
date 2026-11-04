from pathlib import Path
import json
import re
import warnings
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

OBSERVABLES = ["Z", "Z+ZZ"]

FIGURE_NAME = "ipc_observables_spinpulse_normalized.pdf"
DEGREE_FIGURE_NAME = "ipc_observables_spinpulse_degree_wise.pdf"

DPI = 200
SHOW_PLOTS = True
SHOW_STD = True

SHOTS_PRECISIONS = ["0.05", "0.01"]
SPINPULSE_PRECISIONS = ["0.05", "0.01"]

# Error bar styling for the bar means.
ERROR_COLOR = "black"
ERROR_LINEWIDTH = 1.8
ERROR_CAPSIZE = 5
ERROR_CAPTHICK = 1.8

# Offset shot dots slightly to the right of their corresponding bars.
SHOT_DOT_OFFSET = 0.12
SHOT_DOT_SIZE = 45

METRICS = [
    ("normalized_total_capacity", "Total IPC"),
    ("normalized_degree_1", "Degree 1"),
    ("normalized_degree_2", "Degree 2"),
    ("normalized_degree_3", "Degree 3"),
]


# ================================================================
# Filename parsing
# ================================================================


def parse_filename(path):
    name = re.sub(r"_copy\d+$", "", path.stem)

    match = re.match(
        r"^(estimator-exact|spinpulse|shots)_(.+)$",
        name,
        re.IGNORECASE,
    )

    if not match:
        return None

    backend = match.group(1).lower()
    rest = match.group(2)

    precision_match = re.search(
        r"_precision_?([^_]+)",
        rest,
        re.IGNORECASE,
    )

    precision = precision_match.group(1) if precision_match else None

    if precision_match:
        rest = rest[: precision_match.start()] + rest[precision_match.end() :]

    seed_match = re.search(r"_seed(\d+)", rest, re.IGNORECASE)
    seed = int(seed_match.group(1)) if seed_match else None

    if seed_match:
        rest = rest[: seed_match.start()] + rest[seed_match.end() :]

    hamiltonian = None
    remainder = None

    for candidate in sorted(HAMILTONIANS, key=len, reverse=True):
        prefix = candidate + "_"

        if rest.startswith(prefix):
            hamiltonian = candidate
            remainder = rest[len(prefix) :]
            break

    if hamiltonian is None:
        return None

    observable = None
    suffix = ""

    for candidate in sorted(OBSERVABLES, key=len, reverse=True):
        if remainder == candidate or remainder.startswith(candidate + "_"):
            observable = candidate
            suffix = remainder[len(candidate) :].strip("_- ")
            break

    if observable is None:
        return None

    if precision is not None:
        try:
            precision = f"{float(precision):.2f}"
        except ValueError:
            return None

    if backend == "spinpulse" and precision is None:
        precision = "0.05"

    if backend == "estimator-exact":
        variant = "Exact estimator"

    elif backend == "shots":
        variant = None

    else:
        config_label = suffix.replace("_", ", ")

        if config_label:
            variant = f"SpinPulse ({config_label}, precision {precision})"
        else:
            variant = f"SpinPulse (precision {precision})"

    return {
        "backend": backend,
        "hamiltonian": hamiltonian,
        "observable": observable,
        "variant": variant,
        "seed": seed,
        "precision": precision,
    }


# ================================================================
# Read IPC metrics
# ================================================================


def read_ipc_metrics(path):
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None

    blocks = re.findall(
        r"Information Processing Capacity \(IPC\)\s*(\{.*?\})",
        text,
        flags=re.DOTALL,
    )

    if not blocks:
        warnings.warn(f"No IPC JSON block found in {path}")
        return None

    for block in reversed(blocks):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue

        metrics = {}

        for metric, _ in METRICS:
            value = data.get(metric)
            metrics[metric] = float(value) if value is not None else None

        if metrics["normalized_total_capacity"] is not None:
            return metrics

    warnings.warn(f"No usable IPC metrics found in {path}")
    return None


# ================================================================
# Load and aggregate results
# ================================================================


def load_results(root=ROOT):
    if not root.exists():
        raise FileNotFoundError(f"Log directory not found: {root}")

    results = defaultdict(list)
    shots_results = defaultdict(list)

    files = sorted(root.rglob("*.txt"))

    loaded = 0
    skipped = 0

    print(f"Found {len(files)} log files in {root}\n")

    for path in files:
        config = parse_filename(path)

        if config is None:
            continue

        backend = config["backend"]

        if backend not in {"spinpulse", "estimator-exact", "shots"}:
            continue

        if config["observable"] not in OBSERVABLES:
            continue

        if backend == "shots":
            if config["precision"] not in SHOTS_PRECISIONS:
                continue

        if backend == "spinpulse":
            if config["precision"] not in SPINPULSE_PRECISIONS:
                continue

        metrics = read_ipc_metrics(path)

        if metrics is None:
            skipped += 1
            continue

        if backend == "shots":
            key = (
                config["observable"],
                config["hamiltonian"],
                config["precision"],
            )
            shots_results[key].append(metrics)

        else:
            key = (
                config["observable"],
                config["hamiltonian"],
                config["variant"],
            )
            results[key].append(metrics)

        loaded += 1

    print(f"Loaded IPC results: {loaded}")
    print(f"Skipped logs without usable values: {skipped}")

    variants = sorted({key[2] for key in results if key[2] != "Exact estimator"})

    print("\nSpinPulse variants found:")
    for variant in variants:
        print(f"  {variant}")

    return results, shots_results, variants


# ================================================================
# Statistics
# ================================================================


def get_metric_statistics(values, metric):
    metric_values = [item[metric] for item in values if item.get(metric) is not None]

    if not metric_values:
        return np.nan, 0.0

    mean = float(np.mean(metric_values))

    std = float(np.std(metric_values, ddof=1)) if len(metric_values) > 1 else 0.0

    return mean, std


def get_variant_precision(variant):
    if "precision 0.01" in variant:
        return "0.01"
    return "0.05"


# ================================================================
# Series and legend
# ================================================================


def get_series(spinpulse_variants):
    return [
        ("Exact estimator", "exact"),
        *[(variant, "spinpulse") for variant in spinpulse_variants],
    ]


def add_legend(
    fig,
    axes,
    spinpulse_variants,
    variant_colors,
    bbox_to_anchor=(0.5, 0.01),
    ncol=3,
):
    handles, labels = axes[0].get_legend_handles_labels()

    for precision in SHOTS_PRECISIONS:
        matching_variant = next(
            (
                variant
                for variant in spinpulse_variants
                if f"precision {precision}" in variant
            ),
            None,
        )

        if matching_variant is None:
            continue

        color = variant_colors.get(matching_variant)

        if color is None:
            continue

        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=7,
                label=f"Shots (precision {precision})",
            )
        )

        labels.append(f"Shots (precision {precision})")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=min(ncol, len(labels)),
        frameon=False,
        fontsize=11,
    )


# ================================================================
# Plot one subplot
# ================================================================


def plot_subplot(
    ax,
    observable,
    metric,
    results,
    shots_results,
    series,
    variant_colors,
    show_std=True,
):
    n_series = len(series)
    bar_width = min(0.8 / n_series, 0.24)

    group_centres = np.arange(len(HAMILTONIANS))

    for series_index, (variant, backend) in enumerate(series):
        offset = (series_index - (n_series - 1) / 2) * bar_width

        means = []
        errors = []

        for hamiltonian in HAMILTONIANS:
            key = (observable, hamiltonian, variant)
            values = results.get(key, [])

            mean, std = get_metric_statistics(values, metric)

            means.append(mean)
            errors.append(std)

        positions = group_centres + offset

        if backend == "exact":
            ax.bar(
                positions,
                means,
                width=bar_width,
                yerr=errors if show_std else None,
                error_kw={
                    "ecolor": ERROR_COLOR,
                    "elinewidth": ERROR_LINEWIDTH,
                    "capsize": ERROR_CAPSIZE,
                    "capthick": ERROR_CAPTHICK,
                    "zorder": 4,
                },
                label=variant,
                facecolor="none",
                edgecolor="C0",
                linestyle="--",
                linewidth=1.8,
                hatch="///",
                zorder=2,
            )

        else:
            bars = ax.bar(
                positions,
                means,
                width=bar_width,
                yerr=errors if show_std else None,
                error_kw={
                    "ecolor": ERROR_COLOR,
                    "elinewidth": ERROR_LINEWIDTH,
                    "capsize": ERROR_CAPSIZE,
                    "capthick": ERROR_CAPTHICK,
                    "zorder": 4,
                },
                label=variant,
                zorder=2,
            )

            if bars.patches:
                variant_colors[variant] = bars.patches[0].get_facecolor()

            precision = get_variant_precision(variant)

            for i, hamiltonian in enumerate(HAMILTONIANS):
                shots_key = (
                    observable,
                    hamiltonian,
                    precision,
                )

                shots_values = shots_results.get(shots_key, [])

                shots_mean, _ = get_metric_statistics(
                    shots_values,
                    metric,
                )

                if not np.isnan(shots_mean) and not np.isnan(means[i]):
                    # Place the shot marker slightly to the right
                    # of the bar so it cannot be mistaken for the
                    # bar's mean or error bar.
                    ax.scatter(
                        positions[i] + SHOT_DOT_OFFSET,
                        shots_mean,
                        color=bars.patches[i].get_facecolor(),
                        edgecolor="black",
                        linewidth=0.8,
                        s=SHOT_DOT_SIZE,
                        zorder=5,
                    )

    ax.set_xticks(group_centres)
    ax.set_xticklabels(
        HAMILTONIANS,
        rotation=35,
        ha="right",
    )

    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


# ================================================================
# Figure 1: Original total normalized IPC (1x2)
# ================================================================


def create_total_figure(
    results,
    shots_results,
    spinpulse_variants,
    output_dir,
):
    series = get_series(spinpulse_variants)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 8),
        sharey=False,
    )

    variant_colors = {}

    for ax, observable in zip(axes, OBSERVABLES):
        plot_subplot(
            ax=ax,
            observable=observable,
            metric="normalized_total_capacity",
            results=results,
            shots_results=shots_results,
            series=series,
            variant_colors=variant_colors,
            show_std=SHOW_STD,
        )

        ax.set_title(f"Observables: {observable}")

    fig.supylabel("Normalized total IPC", x=0.05, y=0.6)
    fig.supxlabel("Hamiltonian", y=0.29)

    add_legend(
        fig,
        axes,
        spinpulse_variants,
        variant_colors,
        bbox_to_anchor=(0.5, 0.2),
        ncol=3,
    )

    fig.tight_layout(rect=[0.08, 0.27, 0.98, 0.92])

    output_path = output_dir / FIGURE_NAME

    fig.savefig(
        output_path,
        format="pdf",
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    print(f"Saved total IPC figure: {output_path}")

    return fig


# ================================================================
# Figure 2: Degree-wise normalized IPC (2x4)
# ================================================================


def create_degree_figure(
    results,
    shots_results,
    spinpulse_variants,
    output_dir,
):
    series = get_series(spinpulse_variants)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(25, 12),
        sharey=False,
    )

    variant_colors = {}

    for row, observable in enumerate(OBSERVABLES):
        for col, (metric, metric_title) in enumerate(METRICS):
            ax = axes[row, col]

            plot_subplot(
                ax=ax,
                observable=observable,
                metric=metric,
                results=results,
                shots_results=shots_results,
                series=series,
                variant_colors=variant_colors,
                show_std=SHOW_STD,
            )

            ax.set_title(
                f"{observable}: {metric_title}",
                fontsize=14,
            )

            if col == 0:
                ax.set_ylabel("Normalized IPC")

            if row == 1:
                ax.set_xlabel("Hamiltonian")

    add_legend(
        fig,
        axes.flatten(),
        spinpulse_variants,
        variant_colors,
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
    )

    fig.tight_layout(
        rect=[0.02, 0.10, 0.99, 0.97],
        w_pad=1.5,
        h_pad=2.0,
    )

    output_path = output_dir / DEGREE_FIGURE_NAME

    fig.savefig(
        output_path,
        format="pdf",
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    print(f"Saved degree-wise IPC figure: {output_path}")

    return fig


# ================================================================
# Main
# ================================================================


def plot_observables(
    root=ROOT,
    output_dir=PLOT_DIR,
    show=SHOW_PLOTS,
):
    results, shots_results, spinpulse_variants = load_results(root)

    if not spinpulse_variants:
        raise RuntimeError(
            "No SpinPulse results were found. Check the filenames "
            "and confirm that SpinPulse logs are in the input folder."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_fig = create_total_figure(
        results,
        shots_results,
        spinpulse_variants,
        output_dir,
    )

    degree_fig = create_degree_figure(
        results,
        shots_results,
        spinpulse_variants,
        output_dir,
    )

    total_path = output_dir / FIGURE_NAME
    degree_path = output_dir / DEGREE_FIGURE_NAME

    print("\nPDF output check:")
    for path in [total_path, degree_path]:
        if path.exists():
            print(f"  OK: {path} " f"({path.stat().st_size:,} bytes)")
        else:
            print(f"  ERROR: File was not created: {path}")

    print("\nMissing configurations:")

    missing_count = 0
    series = get_series(spinpulse_variants)

    for observable in OBSERVABLES:
        for hamiltonian in HAMILTONIANS:

            for variant, _ in series:
                key = (observable, hamiltonian, variant)

                if not results.get(key):
                    missing_count += 1
                    print(f"  {observable:4s} | " f"{hamiltonian:20s} | {variant}")

            for precision in SHOTS_PRECISIONS:
                shots_key = (
                    observable,
                    hamiltonian,
                    precision,
                )

                if not shots_results.get(shots_key):
                    missing_count += 1
                    print(
                        f"  {observable:4s} | "
                        f"{hamiltonian:20s} | "
                        f"Shots precision {precision}"
                    )

    if missing_count == 0:
        print("  None")

    if show:
        plt.show()

    plt.close(total_fig)
    plt.close(degree_fig)

    return results


if __name__ == "__main__":
    plot_observables()
