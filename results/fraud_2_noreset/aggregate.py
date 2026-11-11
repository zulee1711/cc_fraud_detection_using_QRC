"""Aggregate results/fraud_2_noreset/runs (no input reset) with the fraud_2 records (input reset, logistic, ESN-45,
LSTM): runs.csv, tables.md, noreset_ap_vs_L.{pdf,png}. Works on a partial sweep.

Usage: .venv/bin/python results/fraud_2_noreset/aggregate.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
FRAUD_2 = HERE.parent / "fraud_2" / "runs"
METRICS = {"average_precision_customer": "customer AP", "average_precision_amount": "amount AP",
           "average_precision": "overall AP", "card_precision_top_k": "card precision@100"}
READOUT = "XYZ-strings"
QRC = {"tfim": "TFIM", "random": "random circuit"}
STYLE = {"tfim": "#2a78d6", "random": "#1baf7a"}  # fraud_2's categorical slots for these reservoirs
CLASSICAL = {"logistic": ("logistic, raw window", "--", "None"), "esn": ("ESN, 45 units", "-", "s"),
             "lstm": ("LSTM", ":", "v")}
TEXT, MUTED, GRID, INK = "#0b0b0b", "#8a8985", "#e4e3df", "#52514e"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "savefig.dpi": 200,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "figure.facecolor": "white"})


def load():
    rows = []
    for path in [*sorted(HERE.glob("runs/*.json")), *sorted(FRAUD_2.glob("*.json"))]:
        for r in json.loads(path.read_text()):
            model = {"baseline": "logistic"}.get(r["model"], r["model"])
            if model == "qrc":
                if r["reservoir"] not in QRC or r["observables"] != READOUT:
                    continue
                model = r["reservoir"]
            elif model == "esn" and r.get("n_units") != 45:
                continue
            for split in ("validation", "test"):
                rows.append(dict(model=model, reset=r.get("reset_inputs", True) if model in QRC else None,
                                 L=r["L"], seed=r["seed"],
                                 budget="exact" if r.get("budget") is None else str(r["budget"]), split=split,
                                 **{k: r["metrics"][split][k] for k in METRICS}))
    return pd.DataFrame(rows)


def qrc(df, model, reset, budget="exact", split="test"):
    return df[(df["model"] == model) & (df["reset"] == reset) & (df["budget"] == budget) & (df["split"] == split)]


def plot(df):
    panels = ("average_precision_customer", "average_precision_amount")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.22})
    test = df[df["split"] == "test"]
    for ax, metric in zip(axes, panels):
        for model, (label, ls, marker) in CLASSICAL.items():
            mean = test[test["model"] == model].groupby("L")[metric].mean()
            if not mean.empty:
                ax.plot(mean.index, mean.values, color=INK if model != "lstm" else TEXT, ls=ls, lw=1.4,
                        marker=marker, ms=5, mfc="white", mew=1.2, zorder=2, label=label)
        for model, color in STYLE.items():
            for reset in (True, False):
                sub = qrc(df, model, reset)
                if sub.empty:
                    continue
                mean = sub.groupby("L")[metric].mean()
                ax.scatter(sub["L"], sub[metric], color=color, s=8, alpha=0.25, lw=0, zorder=3)
                ax.plot(mean.index, mean.values, color=color, ls="-" if reset else "--", lw=2, marker="o",
                        ms=6, mfc=color if reset else "white", mew=1.6, zorder=4,
                        label=f"{QRC[model]}, {'input reset' if reset else 'no reset'}")
        ax.set_xlabel("window length L")
        ax.set_ylabel(f"test {METRICS[metric]}")
        ax.set_xticks(sorted(df["L"].unique()))
    axes[0].set_title("Customer-compromise fraud (needs memory)", loc="left", color=TEXT, fontsize=10)
    axes[1].set_title("Amount fraud (current transaction only)", loc="left", color=TEXT, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(f"Sliding-window QRC with and without input reset ({READOUT} readout, exact; mean of 10 seeds)",
                 x=0.125, ha="left", color=TEXT, fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"noreset_ap_vs_L.{ext}")


def fmt(s):
    return f"{s.mean():.3f} ± {s.std():.3f}" if len(s) else "–"


def tables(df):
    out = ["# fraud_2_noreset tables", "",
           f"Test metrics, mean ± std over seeds 42–51, {READOUT} readout. Δ = no reset − reset, paired by seed "
           "(t-test p).", ""]
    test = df[df["split"] == "test"]
    classical = {m: test[test["model"] == m].groupby("L") for m in CLASSICAL}
    for metric in ("average_precision_customer", "average_precision_amount"):
        for budget in ("exact", "100000", "10000"):
            out += [f"## {METRICS[metric]}, {budget if budget == 'exact' else f'{int(budget):.0e} shots'}", "",
                    "| L | " + " | ".join(f"{QRC[m]} reset | {QRC[m]} no reset | Δ (p)" for m in QRC)
                    + " | logistic | ESN-45 | LSTM |",
                    "|" + "---|" * (1 + 3 * len(QRC) + 3)]
            for L in sorted(df["L"].unique()):
                cells = []
                for m in QRC:
                    a = qrc(df, m, True, budget).query("L == @L").set_index("seed")[metric]
                    b = qrc(df, m, False, budget).query("L == @L").set_index("seed")[metric]
                    both = a.index.intersection(b.index)
                    if len(both) > 1:
                        d = b[both] - a[both]
                        delta = f"{d.mean():+.3f} ({stats.ttest_rel(b[both], a[both]).pvalue:.1g})"
                    else:
                        delta = "–"
                    cells += [fmt(a), fmt(b), delta]
                for m in CLASSICAL:
                    g = classical[m]
                    cells.append(f"{g.get_group(L)[metric].mean():.3f}" if L in g.groups else "–")
                out.append(f"| {L} | " + " | ".join(cells) + " |")
            out.append("")
    (HERE / "tables.md").write_text("\n".join(out))


if __name__ == "__main__":
    df = load()
    df.to_csv(HERE / "runs.csv", index=False)
    tables(df)
    plot(df)
    print(f"{len(df)} rows; wrote runs.csv, tables.md, noreset_ap_vs_L.{{pdf,png}}")
