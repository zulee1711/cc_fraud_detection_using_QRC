"""Figures for E6 (experiments/spinpulse_noise_models.py): how the SpinPulse noise types affect the QRC.

Reads the `compare` output and, at one two-qubit fidelity, draws:

noise_types_memory     memory capacity per delay, exact expectation values and finite shots
noise_types_summary    total memory capacity and feature signal per model

Each figure is saved as PNG and PDF, with the plotted numbers in a CSV of the same name, and a
summary table (mean and standard error over seeds) is printed for the report.

    python -m experiments.plot_spinpulse_noise_models --target 0.99 --precision 0.05
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.spinpulse_noise_models import OUT_DIR

TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"  # same palette as results/observable_sets
# model -> (label, colour, line style)
MODELS = {
    "ideal": ("ideal circuit", MUTED, "--"),
    "depolarizing": ("depolarizing", "#eda100", "-"),
    "sp-pink": ("SpinPulse pink (1/f)", "#eb6834", "-"),
    "sp-quasistatic": ("SpinPulse quasistatic", "#2a78d6", "-"),
    "sp-white": ("SpinPulse white", "#1baf7a", "-"),
}


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def save(fig, out_dir, name, table):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    table.to_csv(out_dir / f"{name}.csv", index=False)


def mean_sem(df, value, keys):
    g = df.groupby(keys)[value]
    out = g.agg(["mean", "count"]).reset_index()
    out["sem"] = g.std(ddof=1).fillna(0.0).values / np.sqrt(out["count"])
    return out


def select(df, target):
    """Rows of the ideal circuit and of every noisy model at the given 2q fidelity."""
    noisy = df.model != "ideal"
    keep = ~noisy | np.isclose(pd.to_numeric(df.target_f2, errors="coerce"), target)
    models = [m for m in MODELS if m in set(df.model)]
    return df[keep & df.model.isin(models)], models


def memory_figure(df, models, precisions, target, out_dir):
    t = mean_sem(df, "capacity", ["model", "precision", "delay"])
    fig, axes = plt.subplots(1, len(precisions), figsize=(5.2 * len(precisions), 3.6), sharey=True, squeeze=False)
    for ax, precision in zip(axes[0], precisions):
        style(ax)
        for m in models:
            label, colour, ls = MODELS[m]
            d = t[(t.model == m) & (t.precision == precision)].sort_values("delay")
            ax.fill_between(d.delay, d["mean"] - d["sem"], d["mean"] + d["sem"], color=colour, alpha=0.15, lw=0)
            ax.plot(d.delay, d["mean"], color=colour, ls=ls, lw=2, marker="o", ms=3.5, label=label)
        shots = "exact expectation values" if precision == 0 else f"{int(np.ceil(1 / precision**2))} shots per observable"
        ax.set_title(shots, color=TEXT, fontsize=10.5)
        ax.set_xlabel("delay d (transactions)", color=MUTED)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0][0].set_ylabel("memory capacity $C_d$", color=TEXT)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9.5)
    # fig.suptitle(f"Memory per delay at two-qubit fidelity {target:.1%}", color=TEXT, y=1.2, fontsize=11.5)
    save(fig, out_dir, "noise_types_memory", t)


def summary_figure(df, models, precisions, target, out_dir):
    total = df.groupby(["model", "precision", "seed"], as_index=False).agg(capacity=("capacity", "sum"),
                                                                             signal=("signal", "first"))
    mc = mean_sem(total, "capacity", ["model", "precision"])
    sig = mean_sem(total[total.precision == 0], "signal", ["model"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    style(ax1)
    style(ax2)
    x = np.arange(len(models))
    width = 0.8 / len(precisions)
    for j, precision in enumerate(precisions):
        d = mc[mc.precision == precision].set_index("model").reindex(models)
        alpha = 1.0 if precision == 0 else 0.5
        name = "exact" if precision == 0 else f"{int(np.ceil(1 / precision**2))} shots"
        ax1.bar(x + (j - (len(precisions) - 1) / 2) * width, d["mean"], width, yerr=d["sem"],
                color=[MODELS[m][1] for m in models], alpha=alpha, edgecolor=MUTED, lw=0.6,
                error_kw=dict(ecolor=MUTED, lw=1), label=name)
    ax1.set_xticks(x, [MODELS[m][0].replace("SpinPulse ", "SP ") for m in models], rotation=20, ha="right")
    ax1.set_ylabel("total memory capacity", color=TEXT)
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_title("memory (solid: exact, light: shots)", color=TEXT, fontsize=10.5)

    d = sig.set_index("model").reindex(models)
    ax2.bar(x, d["mean"], 0.6, yerr=d["sem"], color=[MODELS[m][1] for m in models], edgecolor=MUTED, lw=0.6,
            error_kw=dict(ecolor=MUTED, lw=1))
    ax2.set_xticks(x, [MODELS[m][0].replace("SpinPulse ", "SP ") for m in models], rotation=20, ha="right")
    ax2.set_ylabel("feature spread (signal)", color=TEXT)
    ax2.set_title("signal, exact", color=TEXT, fontsize=10.5)
    # fig.suptitle(f"Noise types at two-qubit fidelity {target:.1%}", color=TEXT, y=1.03, fontsize=11.5)
    fig.tight_layout()
    table = mc.merge(sig.rename(columns={"mean": "signal", "sem": "signal_sem", "count": "n"}), on="model", how="left")
    save(fig, out_dir, "noise_types_summary", table)
    return mc, sig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", type=Path, default=OUT_DIR / "compare.csv", help="output of `compare`")
    p.add_argument("--target", type=float, default=0.99, help="two-qubit fidelity to plot")
    p.add_argument("--precision", type=float, default=0.05, help="shot precision of the shots panels")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR / "figures")
    args = p.parse_args(argv)

    df, models = select(pd.read_csv(args.compare), args.target)
    precisions = [q for q in (0.0, args.precision) if np.isclose(df.precision, q).any()]
    if df.empty or not precisions:
        raise SystemExit(f"no rows at target {args.target} in {args.compare}")
    df = df.assign(precision=df.precision.map(lambda q: min(precisions, key=lambda r: abs(r - q))))

    memory_figure(df, models, precisions, args.target, args.out_dir)
    mc, sig = summary_figure(df, models, precisions, args.target, args.out_dir)

    print(f"two-qubit fidelity {args.target}: total memory capacity (mean +- s.e. over seeds) and signal")
    for m in models:
        cells = [f"{r['mean']:.2f} +- {r['sem']:.2f}" for _, r in mc[mc.model == m].sort_values("precision").iterrows()]
        s = sig[sig.model == m]
        print(f"  {MODELS[m][0]:<24} MC {' | '.join(cells):<28} signal {s['mean'].iloc[0]:.4f}")
    print(f"figures in {args.out_dir}")
    return args.out_dir


if __name__ == "__main__":
    main()