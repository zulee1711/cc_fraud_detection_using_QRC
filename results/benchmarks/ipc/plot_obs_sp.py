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

OBSERVABLES = ["Z", "Z+ZZ"]

FIGURE_NAME = "ipc_observables_spinpulse_normalized.png"
DPI = 200
SHOW_PLOTS = True
SHOW_STD = True


# ================================================================
# Filename parsing
# ================================================================


def parse_filename(path):
    """
    Recognizes filenames such as:

        estimator-exact_random_Z_seed1.txt
        spinpulse_random_Z_seed1.txt
        spinpulse_random_Z_noise_pink_t2_50000_seed1.txt
        spinpulse_tfim_Z+ZZ_noise_pink_t2_50000_seed2.txt

    Any suffix after the observable is used to distinguish
    SpinPulse configurations.
    """
    name = re.sub(r"_copy\d+$", "", path.stem)

    match = re.match(
        r"^(estimator-exact|spinpulse)_(.+)$",
        name,
        re.IGNORECASE,
    )
    if not match:
        return None

    backend = match.group(1).lower()
    rest = match.group(2)

    # Extract seed.
    seed_match = re.search(r"_seed(\d+)", rest, re.IGNORECASE)
    seed = int(seed_match.group(1)) if seed_match else None

    if seed_match:
        rest = rest[: seed_match.start()]

    # Remove precision suffix if present.
    rest = re.sub(r"_precision.+$", "", rest, flags=re.IGNORECASE)

    # Identify Hamiltonian, longest name first.
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

    # Identify observable and preserve any trailing configuration.
    observable = None
    suffix = ""

    for candidate in sorted(OBSERVABLES, key=len, reverse=True):
        if remainder == candidate or remainder.startswith(candidate + "_"):
            observable = candidate
            suffix = remainder[len(candidate) :].strip("_- ")
            break

    if observable is None:
        return None

    if backend == "estimator-exact":
        variant = "Exact estimator"
    elif suffix:
        variant = "SpinPulse (" + suffix.replace("_", ", ") + ")"
    else:
        variant = "SpinPulse"

    return {
        "backend": backend,
        "hamiltonian": hamiltonian,
        "observable": observable,
        "variant": variant,
        "seed": seed,
    }


# ================================================================
# Read normalized IPC values
# ================================================================


def read_normalized_capacity(path):
    """Read the last normalized_total_capacity value from a log."""
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

    values = []

    for block in blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue

        value = data.get("normalized_total_capacity")
        if value is not None:
            values.append(float(value))

    if not values:
        warnings.warn(f"No normalized_total_capacity found in {path}")
        return None

    return values[-1]


# ================================================================
# Load and aggregate results
# ================================================================


def load_results(root=ROOT):
    """
    Returns:
        results[(observable, hamiltonian, variant)]
            = list of normalized total IPC values across seeds.
    """
    if not root.exists():
        raise FileNotFoundError(f"Log directory not found: {root}")

    results = defaultdict(list)
    files = sorted(root.rglob("*.txt"))

    loaded = 0
    skipped = 0

    print(f"Found {len(files)} log files in {root}\n")

    for path in files:
        config = parse_filename(path)

        if config is None:
            continue

        if config["backend"] not in {"spinpulse", "estimator-exact"}:
            continue

        value = read_normalized_capacity(path)

        if value is None:
            skipped += 1
            continue

        key = (
            config["observable"],
            config["hamiltonian"],
            config["variant"],
        )

        results[key].append(value)
        loaded += 1

    print(f"Loaded normalized IPC values: {loaded}")
    print(f"Skipped logs without usable values: {skipped}")

    variants = sorted({key[2] for key in results if key[2] != "Exact estimator"})

    print("\nSpinPulse variants found:")
    for variant in variants:
        print(f"  {variant}")

    return results, variants


# ================================================================
# Plotting
# ================================================================


def plot_observables(
    root=ROOT,
    output_dir=PLOT_DIR,
    show=True,
    show_std=SHOW_STD,
):
    results, spinpulse_variants = load_results(root)

    if not spinpulse_variants:
        raise RuntimeError(
            "No SpinPulse results were found. Check the filenames "
            "and confirm that SpinPulse logs are in the input folder."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series = [
        ("Exact estimator", "exact"),
        *[(variant, "spinpulse") for variant in spinpulse_variants],
    ]

    n_series = len(series)
    bar_width = min(0.8 / n_series, 0.24)

    group_centres = np.arange(len(HAMILTONIANS))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 8),
        sharey=False,
    )

    for ax, observable in zip(axes, OBSERVABLES):
        for series_index, (variant, backend) in enumerate(series):
            offset = (series_index - (n_series - 1) / 2) * bar_width

            means = []
            errors = []

            for hamiltonian in HAMILTONIANS:
                key = (observable, hamiltonian, variant)
                values = results.get(key, [])

                if values:
                    means.append(float(np.mean(values)))
                    errors.append(
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    )
                else:
                    means.append(np.nan)
                    errors.append(0.0)

            positions = group_centres + offset

            if backend == "exact":
                # Dashed outline and hatched fill for the baseline.
                ax.bar(
                    positions,
                    means,
                    width=bar_width,
                    yerr=errors if show_std else None,
                    capsize=3,
                    label=variant,
                    facecolor="none",
                    edgecolor="C0",
                    linestyle="--",
                    linewidth=1.8,
                    hatch="///",
                )
            else:
                ax.bar(
                    positions,
                    means,
                    width=bar_width,
                    yerr=errors if show_std else None,
                    capsize=3,
                    label=variant,
                )

        ax.set_title(f"Observables: {observable}")
        ax.set_xticks(group_centres)
        ax.set_xticklabels(
            HAMILTONIANS,
            rotation=35,
            ha="right",
        )
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    # One shared y-axis label on the right.
    fig.supylabel("Normalized total IPC", x=0.05, y=0.6)

    # One shared Hamiltonian label.
    fig.supxlabel("Hamiltonian", y=0.29)

    # Shared legend at the bottom.
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.2),
        ncol=min(3, len(labels)),
        frameon=False,
        fontsize=10,
    )

    fig.suptitle(
        "",
        y=0.98,
    )

    fig.tight_layout(rect=[0.08, 0.27, 0.98, 0.92])

    output_path = output_dir / FIGURE_NAME
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    print(f"\nSaved plot to: {output_path}")

    print("\nMissing configurations:")
    missing_count = 0

    for observable in OBSERVABLES:
        for hamiltonian in HAMILTONIANS:
            for variant, _ in series:
                key = (observable, hamiltonian, variant)

                if not results.get(key):
                    missing_count += 1
                    print(f"  {observable:4s} | " f"{hamiltonian:20s} | {variant}")

    if missing_count == 0:
        print("  None")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return results


if __name__ == "__main__":
    plot_observables()


# from pathlib import Path
# import json
# import re
# import warnings
# from collections import defaultdict

# import numpy as np
# import matplotlib.pyplot as plt

# # ================================================================
# # Configuration
# # ================================================================

# ROOT = Path("results/benchmarks/ipc/observables_copy")
# PLOT_DIR = Path("results/benchmarks/ipc/plots")

# HAMILTONIANS = [
#     "random",
#     "tfim",
#     "tfim-longitudinal",
#     "tfim-native",
#     "native-exchange",
# ]

# OBSERVABLES = ["Z", "Z+ZZ", "XYZ"]

# FIGURE_NAME = "ipc_observables_spinpulse_normalized.png"
# DPI = 200
# SHOW_PLOTS = True
# SHOW_STD = True


# # ================================================================
# # Filename parsing
# # ================================================================


# def parse_filename(path):
#     """
#     Recognizes filenames such as:

#         estimator-exact_random_Z_seed1.txt
#         spinpulse_random_Z_seed1.txt
#         spinpulse_random_Z_noise_pink_t2_50000_seed1.txt
#         spinpulse_tfim_Z+ZZ_noise_pink_t2_50000_seed2.txt

#     Any suffix after the observable is used to distinguish
#     SpinPulse configurations.
#     """
#     name = re.sub(r"_copy\d+$", "", path.stem)

#     match = re.match(
#         r"^(estimator-exact|spinpulse)_(.+)$",
#         name,
#         re.IGNORECASE,
#     )
#     if not match:
#         return None

#     backend = match.group(1).lower()
#     rest = match.group(2)

#     # Extract seed.
#     seed_match = re.search(r"_seed(\d+)", rest, re.IGNORECASE)
#     seed = int(seed_match.group(1)) if seed_match else None

#     if seed_match:
#         rest = rest[: seed_match.start()]

#     # Remove precision suffix if present.
#     rest = re.sub(r"_precision.+$", "", rest, flags=re.IGNORECASE)

#     # Identify Hamiltonian, longest name first.
#     hamiltonian = None
#     remainder = None

#     for candidate in sorted(HAMILTONIANS, key=len, reverse=True):
#         prefix = candidate + "_"
#         if rest.startswith(prefix):
#             hamiltonian = candidate
#             remainder = rest[len(prefix) :]
#             break

#     if hamiltonian is None:
#         return None

#     # Identify observable and preserve any trailing configuration.
#     observable = None
#     suffix = ""

#     for candidate in sorted(OBSERVABLES, key=len, reverse=True):
#         if remainder == candidate or remainder.startswith(candidate + "_"):
#             observable = candidate
#             suffix = remainder[len(candidate) :].strip("_- ")
#             break

#     if observable is None:
#         return None

#     if backend == "estimator-exact":
#         variant = "Exact estimator"
#     elif suffix:
#         # Convert suffix tokens to a readable label.
#         variant = "SpinPulse (" + suffix.replace("_", ", ") + ")"
#     else:
#         variant = "SpinPulse"

#     return {
#         "backend": backend,
#         "hamiltonian": hamiltonian,
#         "observable": observable,
#         "variant": variant,
#         "seed": seed,
#     }


# # ================================================================
# # Read normalized IPC values
# # ================================================================


# def read_normalized_capacity(path):
#     """Read the last normalized_total_capacity value from a log."""
#     try:
#         text = path.read_text(errors="replace")
#     except OSError as exc:
#         warnings.warn(f"Could not read {path}: {exc}")
#         return None

#     blocks = re.findall(
#         r"Information Processing Capacity \(IPC\)\s*(\{.*?\})",
#         text,
#         flags=re.DOTALL,
#     )

#     if not blocks:
#         warnings.warn(f"No IPC JSON block found in {path}")
#         return None

#     values = []

#     for block in blocks:
#         try:
#             data = json.loads(block)
#         except json.JSONDecodeError:
#             continue

#         value = data.get("normalized_total_capacity")
#         if value is not None:
#             values.append(float(value))

#     if not values:
#         warnings.warn(f"No normalized_total_capacity found in {path}")
#         return None

#     return values[-1]


# # ================================================================
# # Load and aggregate results
# # ================================================================


# def load_results(root=ROOT):
#     """
#     Returns:
#         results[(observable, hamiltonian, variant)]
#             = list of normalized total IPC values across seeds.
#     """
#     if not root.exists():
#         raise FileNotFoundError(f"Log directory not found: {root}")

#     results = defaultdict(list)
#     files = sorted(root.rglob("*.txt"))

#     loaded = 0
#     skipped = 0

#     print(f"Found {len(files)} log files in {root}\n")

#     for path in files:
#         config = parse_filename(path)

#         if config is None:
#             continue

#         if config["backend"] not in {"spinpulse", "estimator-exact"}:
#             continue

#         value = read_normalized_capacity(path)

#         if value is None:
#             skipped += 1
#             continue

#         key = (
#             config["observable"],
#             config["hamiltonian"],
#             config["variant"],
#         )

#         results[key].append(value)
#         loaded += 1

#     print(f"Loaded normalized IPC values: {loaded}")
#     print(f"Skipped logs without usable values: {skipped}")

#     variants = sorted({key[2] for key in results if key[2] != "Exact estimator"})

#     print("\nSpinPulse variants found:")
#     for variant in variants:
#         print(f"  {variant}")

#     return results, variants


# # ================================================================
# # Plotting
# # ================================================================


# def plot_observables(
#     root=ROOT,
#     output_dir=PLOT_DIR,
#     show=True,
#     show_std=SHOW_STD,
# ):
#     results, spinpulse_variants = load_results(root)

#     if not spinpulse_variants:
#         raise RuntimeError(
#             "No SpinPulse results were found. Check the filenames "
#             "and confirm that SpinPulse logs are in the input folder."
#         )

#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     series = [
#         ("Exact estimator", "exact"),
#         *[(variant, "spinpulse") for variant in spinpulse_variants],
#     ]

#     n_series = len(series)
#     bar_width = min(0.8 / n_series, 0.24)

#     group_centres = np.arange(len(HAMILTONIANS))

#     fig, axes = plt.subplots(
#         1,
#         3,
#         figsize=(20, 8),
#         sharey=False,
#     )

#     for ax, observable in zip(axes, OBSERVABLES):
#         for series_index, (variant, backend) in enumerate(series):
#             offset = (series_index - (n_series - 1) / 2) * bar_width

#             means = []
#             errors = []

#             for hamiltonian in HAMILTONIANS:
#                 key = (observable, hamiltonian, variant)
#                 values = results.get(key, [])

#                 if values:
#                     means.append(float(np.mean(values)))
#                     errors.append(
#                         float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
#                     )
#                 else:
#                     means.append(np.nan)
#                     errors.append(0.0)

#             positions = group_centres + offset

#             if backend == "exact":
#                 # Dashed outline and hatched fill for the baseline.
#                 ax.bar(
#                     positions,
#                     means,
#                     width=bar_width,
#                     yerr=errors if show_std else None,
#                     capsize=3,
#                     label=variant,
#                     facecolor="none",
#                     edgecolor="C0",
#                     linestyle="--",
#                     linewidth=1.8,
#                     hatch="///",
#                 )
#             else:
#                 ax.bar(
#                     positions,
#                     means,
#                     width=bar_width,
#                     yerr=errors if show_std else None,
#                     capsize=3,
#                     label=variant,
#                 )

#         ax.set_title(f"Observables: {observable}")
#         ax.set_xticks(group_centres)
#         ax.set_xticklabels(
#             HAMILTONIANS,
#             rotation=35,
#             ha="right",
#         )
#         ax.grid(axis="y", alpha=0.3)
#         ax.set_axisbelow(True)

#     # One shared y-axis label on the right.
#     fig.supylabel("Normalized total IPC", x=0.99)

#     # One shared Hamiltonian label.
#     fig.supxlabel("Hamiltonian", y=0.13)

#     # Shared legend at the bottom.
#     handles, labels = axes[0].get_legend_handles_labels()

#     fig.legend(
#         handles,
#         labels,
#         loc="lower center",
#         bbox_to_anchor=(0.5, 0.015),
#         ncol=min(3, len(labels)),
#         frameon=False,
#         fontsize=10,
#     )

#     fig.suptitle(
#         "Normalized IPC: SpinPulse vs Exact Estimator",
#         y=0.98,
#     )

#     fig.tight_layout(rect=[0.02, 0.27, 0.96, 0.92])

#     output_path = output_dir / FIGURE_NAME
#     fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
#     print(f"\nSaved plot to: {output_path}")

#     print("\nMissing configurations:")
#     missing_count = 0

#     for observable in OBSERVABLES:
#         for hamiltonian in HAMILTONIANS:
#             for variant, _ in series:
#                 key = (observable, hamiltonian, variant)

#                 if not results.get(key):
#                     missing_count += 1
#                     print(f"  {observable:4s} | " f"{hamiltonian:20s} | {variant}")

#     if missing_count == 0:
#         print("  None")

#     if show:
#         plt.show()
#     else:
#         plt.close(fig)

#     return results


# if __name__ == "__main__":
#     plot_observables()
