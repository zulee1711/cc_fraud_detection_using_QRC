"""Figures for the arm B experiments (test split by default, seeds 42-44).

- sweep1: reset QRC vs logistic regression on the raw window, L in {1, 2, 3, 5, 8, 10}.
- noreset_l5 / noreset_l8_l10: same QRC without input reset; L = 5 also at gaussian precision 0.01 / 0.05.
- lstm: LSTM (hidden 32) on the same windows as sweep1.

Figures:
- dissipation.png: customer AP and amount AP vs L, reset vs no reset vs logistic.
- sweep1_categories.png: AP per fraud category vs L, reset QRC vs logistic and LSTM
  (terminal AP is at chance, ~0.005, omitted).
- noise_l5.png: customer AP at L = 5 (no reset) vs readout precision.

Validation is only used to pick the decision threshold, which AP ignores, so both splits are held out for AP;
test is the reported one. `--split validation` writes the same figures with a `_validation` suffix.

Usage: .venv/bin/python results/armb/plots/plot.py [--split test|validation]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results/armb")
OUT = RESULTS / "plots"

RESET = "#2a78d6"
NO_RESET = "#eb6834"
LSTM = "#1baf7a"
LOGISTIC = "#52514e"
TEXT = "#0b0b0b"
MUTED = "#8a8985"
GRID = "#e6e5e0"

SPLIT = "test"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": TEXT,
    "ytick.labelcolor": TEXT,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def load():
    sweep = pd.read_csv(RESULTS / "sweep1" / "summary.csv")
    noreset = pd.concat([pd.read_csv(RESULTS / d / "summary.csv") for d in ("noreset_l5", "noreset_l8_l10")])
    logistic = sweep[sweep["run"].str.endswith("_baseline")].set_index("window_length")
    reset = sweep[sweep["run"].str.endswith("_qrc")]
    exact = noreset[noreset["backend"] == "estimator-exact"]
    noisy = noreset[noreset["window_length"] == 5]
    lstm = pd.read_csv(RESULTS / "lstm" / "summary.csv")
    return logistic, reset, exact, noisy, lstm


def suffix():
    return "" if SPLIT == "test" else f"_{SPLIT}"


def plot_qrc(ax, df, metric, color, label):
    """Mean line with one dot per seed."""
    stats = df.groupby("window_length")[metric].agg(["mean", "std"])
    ax.plot(stats.index, stats["mean"], color=color, lw=2, marker="o", ms=6, label=label, zorder=3)
    ax.scatter(df["window_length"], df[metric], color=color, s=14, alpha=0.45, lw=0, zorder=2)
    return stats


def dissipation(logistic, reset, exact):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    lengths = [5, 8, 10]
    reset_l = reset[reset["window_length"].isin(lengths)]
    panels = [
        (f"{SPLIT}_average_precision_customer", "Customer AP"),
        (f"{SPLIT}_average_precision_amount", "Amount AP"),
    ]
    for ax, (metric, title) in zip(axes, panels):
        ax.plot(lengths, logistic.loc[lengths, metric], color=LOGISTIC, lw=1.5, ls="--", marker="s", ms=5,
                label="logistic (raw window)", zorder=2)
        r = plot_qrc(ax, reset_l, metric, RESET, "QRC, input reset")
        n = plot_qrc(ax, exact, metric, NO_RESET, "QRC, no reset (unitary)")
        for stats, color in ((r, RESET), (n, NO_RESET)):
            ax.annotate(f"{stats['mean'].iloc[-1]:.2f}", (10, stats["mean"].iloc[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", color=TEXT, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=TEXT)
        ax.set_xticks(lengths)
        ax.set_xlim(4.5, 11)
        ax.set_xlabel("window length L")
    axes[0].set_ylabel(f"{SPLIT} AP")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.02, -0.01), ncol=3)
    title = fig.suptitle("Without in-window dissipation the reservoir loses the current transaction and stops improving with L",
                 x=0.02, y=1.06, ha="left", fontsize=11, color=TEXT)
    note = fig.text(0.02, -0.13, "1 input + 3 memory qubits, depth 4, XYZ+XYZ-2body, exact. Dots: seeds 42-44; line: mean.",
             fontsize=8, color=MUTED)
    fig.savefig(OUT / f"dissipation{suffix()}.png")
    # PDF for the report: title and setup note go in the LaTeX caption instead.
    title.remove()
    note.remove()
    fig.savefig(OUT / f"dissipation{suffix()}.pdf")
    plt.close(fig)


def sweep1_categories(logistic, reset, lstm):
    panels = [
        (f"{SPLIT}_average_precision_customer", "Customer compromise"),
        (f"{SPLIT}_average_precision_amount", "Amount"),
        (f"{SPLIT}_average_precision", "All frauds"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    lengths = sorted(reset["window_length"].unique())
    for ax, (metric, title) in zip(axes, panels):
        ax.plot(lengths, logistic.loc[lengths, metric], color=LOGISTIC, lw=1.5, ls="--", marker="s", ms=5,
                label="logistic (raw window)")
        plot_qrc(ax, reset, metric, RESET, "QRC, input reset")
        plot_qrc(ax, lstm, metric, LSTM, "LSTM (hidden 32)")
        ax.set_title(title, loc="left", fontsize=10, color=TEXT)
        ax.set_xticks(lengths)
        ax.set_xlabel("window length L")
    axes[0].set_ylabel(f"{SPLIT} AP")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.02, -0.01), ncol=3)
    fig.suptitle("Arm B sweep 1: reset QRC vs logistic regression and LSTM, AP per fraud category",
                 x=0.02, y=1.06, ha="left", fontsize=11, color=TEXT)
    fig.savefig(OUT / f"sweep1_categories{suffix()}.png")
    plt.close(fig)


def noise_l5(logistic, reset, noisy):
    metric = f"{SPLIT}_average_precision_customer"
    levels = [("exact", noisy["backend"] == "estimator-exact"),
              ("0.01", noisy["backend_precision"] == 0.01),
              ("0.05", noisy["backend_precision"] == 0.05)]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x = np.arange(len(levels))
    means = [noisy.loc[mask, metric].mean() for _, mask in levels]
    for i, (_, mask) in enumerate(levels):
        ax.scatter(np.full(mask.sum(), i), noisy.loc[mask, metric], color=NO_RESET, s=18, alpha=0.5, lw=0)
    ax.plot(x, means, color=NO_RESET, lw=2, marker="o", ms=7, label="QRC L=5, no reset")
    for xi, m in zip(x, means):
        ax.annotate(f"{m:.3f}", (xi, m), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9)
    refs = [(logistic.loc[5, metric], "logistic L=5"), (logistic.loc[1, metric], "logistic L=1 (no memory)")]
    for y, label in refs:
        ax.axhline(y, color=LOGISTIC, lw=1, ls="--")
        ax.text(len(levels) - 0.6, y, label, va="bottom", ha="right", fontsize=8, color=LOGISTIC)
    ax.set_xticks(x, [name for name, _ in levels])
    ax.set_xlim(-0.4, len(levels) - 0.5)
    ax.set_xlabel("readout precision (gaussian std)")
    ax.set_ylabel(f"{SPLIT} customer AP")
    ax.legend(loc="upper right")
    ax.set_title("Readout noise erases the memory edge at L=5", loc="left", fontsize=11, color=TEXT)
    fig.savefig(OUT / f"noise_l5{suffix()}.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    SPLIT = parser.parse_args().split
    OUT.mkdir(exist_ok=True)
    logistic, reset, exact, noisy, lstm = load()
    dissipation(logistic, reset, exact)
    sweep1_categories(logistic, reset, lstm)
    noise_l5(logistic, reset, noisy)
