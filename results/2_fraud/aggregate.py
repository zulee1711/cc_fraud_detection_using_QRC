"""Aggregate results/2_fraud/runs/*.json: runs.csv, tables.md, fraud_ap_vs_L.{pdf,png} and
fraud_shots.png. Works on a partial sweep:
missing cells are left out.

Usage: PYTHONPATH=. .venv/bin/python results/2_fraud/aggregate.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
HERE = Path("results/2_fraud")
METRICS = {"average_precision_customer": "customer AP", "average_precision_amount": "amount AP",
           "average_precision_terminal": "terminal AP", "average_precision": "overall AP",
           "card_precision_top_k": "card precision@100", "roc_auc": "ROC AUC"}
READOUTS = {"Z+ZZ": 7, "XYZ-strings": 45}  # observable set -> width (= equal-width ESN units)
QRC = {"tfim": "TFIM", "native-exchange": "native exchange", "random": "random circuit"}
CLASSICAL = {"esn": "ESN, equal width", "logistic": "logistic, raw window", "lstm": "LSTM"}
# QRC: categorical slots 1-3 (validated all-pairs); classical: neutral ink, identity from line style + marker
STYLE = {"tfim": ("#2a78d6", "-", "o"), "native-exchange": ("#eb6834", "-", "D"), "random": ("#1baf7a", "-", "^"),
         "esn": ("#52514e", "-", "s"), "logistic": ("#52514e", "--", "None"), "lstm": ("#0b0b0b", ":", "v")}
TEXT, MUTED, GRID = "#0b0b0b", "#8a8985", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "savefig.dpi": 200, "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "figure.facecolor": "white"})


def load():
    rows = []
    for path in sorted((HERE / "runs").glob("*.json")):
        for r in json.loads(path.read_text()):
            model = {"baseline": "logistic"}.get(r["model"], r["model"])
            for split in ("validation", "test"):
                rows.append(dict(model=r.get("reservoir", model) if model == "qrc" else model, L=r["L"],
                                 seed=r["seed"], n_units=r.get("n_units"), observables=r.get("observables"),
                                 budget="exact" if r.get("budget") is None else str(r["budget"]), split=split,
                                 **{k: r["metrics"][split][k] for k in METRICS}))
    return pd.DataFrame(rows)


def select(df, model, readout, split="test", budget="exact"):
    """Rows of one model in the comparison for `readout`: the QRC read out with it, the ESN of its width."""
    sub = df[(df["model"] == model) & (df["split"] == split)]
    if model in QRC:
        return sub[(sub["observables"] == readout) & (sub["budget"] == budget)]
    if model == "esn":
        return sub[sub["n_units"] == READOUTS[readout]]
    return sub


def plot(df):
    panels = ("average_precision_customer", "average_precision_amount", "average_precision_terminal")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), gridspec_kw={"hspace": 0.32, "wspace": 0.22, "width_ratios": (1.3, 1, 1)})
    Ls = sorted(df["L"].unique())
    for row, readout in enumerate(READOUTS):
        for ax, metric in zip(axes[row], panels):
            ends = []
            for model in (*QRC, *CLASSICAL):
                sub = select(df, model, readout)
                if sub.empty:
                    continue
                color, ls, marker = STYLE[model]
                mean = sub.groupby("L")[metric].mean()
                if model in QRC or model == "esn":
                    ax.scatter(sub["L"], sub[metric], color=color, s=8, alpha=0.25, lw=0, zorder=2)
                ax.plot(mean.index, mean.values, color=color, ls=ls, lw=2 if model in QRC else 1.6, marker=marker,
                        ms=5, mfc="white" if model == "esn" else color, mew=1.4, zorder=3,
                        label=QRC.get(model) or CLASSICAL[model] + (f" ({READOUTS[readout]} units)"
                                                                    if model == "esn" else ""))
                ends.append((mean.values[-1], model))
            if metric == "average_precision_customer":  # direct labels at the line ends, spread to avoid overlaps
                ends.sort()
                placed = []
                for y, model in ends:
                    y = max(y, placed[-1] + 0.055) if placed else y
                    placed.append(y)
                    name = QRC.get(model) or {"esn": f"ESN-{READOUTS[readout]}", "logistic": "logistic",
                                              "lstm": "LSTM"}[model]
                    ax.text(Ls[-1] + 1, y, name, fontsize=7.5, color=TEXT, va="center")
                ax.set_xlim(-0.5, Ls[-1] + 11)
            amount = {"Z+ZZ": (0.85, 1.0), "XYZ-strings": (0.94, 1.0)}[readout]  # per row: XYZ-strings spans .95-1
            ax.set_ylim(*{"average_precision_amount": amount, "average_precision_terminal": (0, 0.05)}
                        .get(metric, (0, 1)))
            ax.set_xticks(Ls)
            if metric != "average_precision_customer":
                ax.set_xlim(-0.5, Ls[-1] + 1.5)
            if row == 0:
                ax.set_title(METRICS[metric], loc="left", fontsize=10, color=TEXT, fontweight="bold")
            if row == 1:
                ax.set_xlabel("window length L")
        axes[row, 0].set_ylabel(f"QRC readout {readout} ({READOUTS[readout]} obs.)\naverage precision (test)")
    handles = {}
    for ax in axes[:, 0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    order = [*QRC.values(), CLASSICAL["logistic"], CLASSICAL["lstm"],
             *(f"{CLASSICAL['esn']} ({n} units)" for n in READOUTS.values())]
    order = [k for k in order if k in handles]
    fig.legend([handles[k] for k in order], order, loc="upper left", bbox_to_anchor=(0.06, 0.035), ncol=3,
               frameon=False, title="QRC (colour)  ·  classical baselines (grey)", title_fontsize=8,
               alignment="left")
    for ext in ("pdf", "png"):  # pdf for the report (caption in LaTeX), png for quick viewing
        fig.savefig(HERE / f"fraud_ap_vs_L.{ext}")
    plt.close(fig)


def plot_shots(df, metric="average_precision_customer", readout="XYZ-strings"):
    """Customer AP vs L per reservoir for the XYZ-strings readout, one line per shot budget (ordinal blue ramp,
    darker = more precise), against logistic regression and the equal-width ESN (noise-free)."""
    budgets = {"exact": ("exact", "#104281"), "100000": ("10⁵ shots / window", "#2a78d6"),
               "10000": ("10⁴ shots / window", "#86b6ef")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, gridspec_kw={"wspace": 0.12})
    Ls = sorted(df["L"].unique())
    for ax, model in zip(axes, QRC):
        for b, (label, color) in budgets.items():
            sub = select(df, model, readout, budget=b)
            mean = sub.groupby("L")[metric].mean()
            ax.scatter(sub["L"], sub[metric], color=color, s=8, alpha=0.35, lw=0, zorder=2)
            ax.plot(mean.index, mean.values, color=color, lw=2, marker="o", ms=4.5, zorder=3, label=label)
        for ref, label in (("logistic", "logistic, raw window"),
                           ("esn", f"ESN, equal width ({READOUTS[readout]} units)")):
            mean = select(df, ref, readout).groupby("L")[metric].mean()
            ax.plot(mean.index, mean.values, color=STYLE[ref][0], ls=STYLE[ref][1], lw=1.4, marker=STYLE[ref][2],
                    ms=4.5, mfc="white", mew=1.2, zorder=2.5, label=label)
        ax.set_xticks(Ls)
        ax.set_ylim(0, 0.6)
        ax.set_title(QRC[model], loc="left", fontsize=10, color=TEXT, fontweight="bold")
        ax.set_xlabel("window length L")
    axes[0].set_ylabel(f"customer AP (test)\nQRC readout {readout} ({READOUTS[readout]} obs.)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.06, 0.0), ncol=3,
               frameon=False)
    fig.text(0.06, -0.14, "Shot model: per measurement setting, multinomial sampling of the exact outcome "
             "distribution of every window; the budget is split evenly over the 3 settings (all-X, all-Y, all-Z). "
             "Train, validation and test\nfeatures are all sampled. Dots: seeds 42-51; lines: mean over seeds. Grey: "
             "logistic regression on the raw window and the untuned equal-width ESN, noise-free. data/sc3_c1000_x6, "
             "feature set 1, input reset.", fontsize=7.5, color=MUTED, va="top")
    fig.savefig(HERE / "fraud_shots.png")
    plt.close(fig)


def mean_std(g):
    return g.mean().round(3).astype(str) + " ± " + g.std().round(3).fillna(0).astype(str)


def tables(df):
    out = ["# Section 7.2 fraud sweep: mean ± std over seeds 42-51 (test unless stated)\n"]
    for split in ("test", "validation"):
        for readout in READOUTS:
            out.append(f"## {split}, QRC readout {readout} (exact), ESN-{READOUTS[readout]}\n")
            for metric in METRICS:
                frame = {m: mean_std(select(df, m, readout, split).groupby("L")[metric])
                         for m in (*QRC, *CLASSICAL) if not select(df, m, readout, split).empty}
                out.append(f"### {METRICS[metric]}\n\n```\n{pd.DataFrame(frame).to_string()}\n```\n")
    out.append("## Customer AP under shot noise (test)\n")
    sub = df[df["model"].isin(list(QRC)) & (df["split"] == "test")]
    t = mean_std(sub.groupby(["observables", "model", "budget", "L"])["average_precision_customer"]).unstack("L")
    out.append(f"```\n{t.to_string()}\n```\n")
    out.append("## Tests on customer AP (test, exact)\n")
    out.append("QRC - logistic: one-sample t-test of the QRC seeds against the deterministic logistic value. "
               "QRC - ESN (equal width) and QRC - LSTM: Welch t-test. Two-sided p-values.\n")
    m = "average_precision_customer"
    for readout in READOUTS:
        rows = []
        for model in QRC:
            for L in sorted(df["L"].unique()):
                q = select(df, model, readout).query("L == @L")[m]
                if len(q) < 2:
                    continue
                row = dict(qrc=model, L=L, n=len(q), mean=round(q.mean(), 3))
                lg = select(df, "logistic", readout).query("L == @L")[m]
                if len(lg):
                    row["- logistic"] = round(q.mean() - lg.iloc[0], 3)
                    row["p logistic"] = stats.ttest_1samp(q, lg.iloc[0]).pvalue
                for ref in ("esn", "lstm"):
                    r = select(df, ref, readout).query("L == @L")[m]
                    if len(r) > 1:
                        row[f"- {ref}"] = round(q.mean() - r.mean(), 3)
                        row[f"p {ref}"] = stats.ttest_ind(q, r, equal_var=False).pvalue
                rows.append(row)
        if rows:
            t = pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2g}")
            out.append(f"### {readout} (ESN-{READOUTS[readout]})\n\n```\n{t}\n```\n")
    (HERE / "tables.md").write_text("\n".join(out))


if __name__ == "__main__":
    df = load()
    df.to_csv(HERE / "runs.csv", index=False)
    print(f"{len(df) // 2} records")
    tables(df)
    plot(df)
    plot_shots(df)
