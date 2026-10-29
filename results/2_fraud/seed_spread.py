"""How much the reservoir seed matters in the section 7.2 fraud sweep. Reads results/2_fraud/runs/*.json (via
aggregate.load) and writes:

  seed_spread_lines.png      test customer AP vs L, one line per seed, against logistic regression
  seed_spread_selection.png  validation vs test customer AP per seed (is the spread predictable?) and the test
                             AP of the seed picked on validation vs the seed mean
  seed_spread_n_seeds.png    how often a study with n seeds gets the sign of the 10-seed gap right
  seed_spread_shots.png      seed std vs L per shot budget
  seed_spread_ne_esn.{pdf,png}  native exchange (XYZ-strings) and ESN-45, one line per seed, superposed
                             (report figure, caption in LaTeX)
  seed_spread.md             the numbers behind the figures

The data split and logistic regression are fixed; the seed draws the reservoir (QRC: random circuit / couplings,
ESN: weights; LSTM: init) and, under shots, the sampling.

Usage: PYTHONPATH=. .venv/bin/python results/2_fraud/seed_spread.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from aggregate import GRID, HERE, MUTED, QRC, READOUTS, STYLE, TEXT, load, select  # noqa: E402  (also sets rcParams)

M = "average_precision_customer"
SEEDED = (*QRC, "esn")
NAME = {**QRC, "esn": "ESN"}
CAPTION = ("data/sc3_c1000_x6, feature set 1, 1+3 qubits, input reset; seeds 42-51. The data split and logistic "
           "regression are fixed; the seed draws the reservoir.")


def name(model, readout):
    return f"ESN-{READOUTS[readout]}" if model == "esn" else NAME[model]


def per_seed(df, model, readout, split="test", budget="exact"):
    """(L x seed) table of customer AP."""
    return select(df, model, readout, split, budget).pivot_table(index="L", columns="seed", values=M)


def logistic(df, split="test"):
    return select(df, "logistic", None, split).set_index("L")[M].sort_index()


def caption(fig, text, y):
    fig.text(0.06, y, text, fontsize=7.5, color=MUTED, va="top")


def plot_lines(df):
    """Every seed as its own line: the band between seeds is as wide as the gaps the comparison rests on."""
    lg = logistic(df)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), sharex=True, sharey="row",
                             gridspec_kw={"hspace": 0.3, "wspace": 0.1})
    for row, readout in enumerate(READOUTS):
        for ax, model in zip(axes[row], SEEDED):
            t = per_seed(df, model, readout)
            color = STYLE[model][0]
            for s in t.columns:
                ax.plot(t.index, t[s], color=color, lw=0.9, alpha=0.45, zorder=2)
            ax.plot(t.index, t.mean(axis=1), color=color, lw=2.2, marker="o", ms=4, zorder=3, label="seed mean")
            ax.plot(lg.index, lg.values, color=TEXT, ls="--", lw=1.3, zorder=2.5, label="logistic, raw window")
            above = (t.gt(lg.reindex(t.index), axis=0)).sum(axis=1)
            for L, k in above.items():  # seeds above logistic, per L, along the bottom
                if L > 1:
                    ax.text(L, 0.015, f"{k}", fontsize=7, color=MUTED, ha="center")
            ax.set_xticks(t.index)
            ax.set_ylim(0, 0.62)
            ax.set_title(name(model, readout), loc="left", fontsize=10, color=TEXT, fontweight="bold")
            if row == 1:
                ax.set_xlabel("window length L")
        axes[row, 0].set_ylabel(f"QRC readout {readout} ({READOUTS[readout]} obs.)\ncustomer AP (test, exact)")
    axes[0, 0].plot([], [], color=STYLE["tfim"][0], lw=0.9, alpha=0.45, label="one seed")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.06, 0.035), ncol=3,
               frameon=False)
    caption(fig, "Thin lines: one seed each; bold: mean over the 10 seeds. Small numbers: seeds (of 10) above "
            "logistic regression at that L.\n" + CAPTION, -0.01)
    fig.savefig(HERE / "seed_spread_lines.png")
    plt.close(fig)


def selection(df, readout):
    """Per model and L: test AP of the seed with the best / worst validation AP, the seed mean, Spearman rho."""
    rows = []
    for model in SEEDED:
        val, test = per_seed(df, model, readout, "validation"), per_seed(df, model, readout)
        for L in test.index:
            v, t = val.loc[L], test.loc[L]
            rows.append(dict(model=model, L=L, mean=t.mean(), std=t.std(), best_val=t[v.idxmax()],
                             worst_val=t[v.idxmin()], best_test=t.max(), worst_test=t.min(),
                             rho=v.corr(t, method="spearman") if v.std() > 0 else np.nan))
    return pd.DataFrame(rows)


def plot_selection(df, L_scatter=15):
    """Top: validation vs test per seed at one L, the spread is a property of the drawn reservoir, so
    validation sees it. Bottom: picking the seed on validation vs averaging over seeds, relative to logistic."""
    lg = logistic(df)
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.22, height_ratios=(1, 1.05))
    for col, model in enumerate(SEEDED):
        ax = fig.add_subplot(gs[0, col])
        for readout, mfc in zip(READOUTS, ("white", None)):
            val = per_seed(df, model, readout, "validation").loc[L_scatter]
            test = per_seed(df, model, readout).loc[L_scatter]
            rho = val.corr(test, method="spearman")
            ax.scatter(val, test, s=34, color=STYLE[model][0], facecolors=mfc or STYLE[model][0], lw=1.3,
                       edgecolors="white" if mfc is None else STYLE[model][0], zorder=3,
                       label=f"{readout if model != 'esn' else name(model, readout)}  ρ = {rho:.2f}")
        ax.set_title(f"{NAME[model]}, L = {L_scatter}", loc="left", fontsize=10, color=TEXT, fontweight="bold")
        ax.set_xlabel("customer AP, validation")
        ax.legend(loc="upper left", frameon=False, fontsize=7.5, handletextpad=0.2)
        if col == 0:
            ax.set_ylabel("customer AP, test")
    for col, readout in enumerate(READOUTS):
        ax = fig.add_subplot(gs[1, 2 * col:2 * col + 2])
        sel = selection(df, readout)
        for model in SEEDED:
            s = sel[sel["model"] == model].set_index("L")
            gap = lambda c: s[c] - lg.reindex(s.index)  # noqa: E731
            color = STYLE[model][0]
            ax.fill_between(s.index, gap("worst_test"), gap("best_test"), color=color, alpha=0.12, lw=0, zorder=1)
            ax.plot(s.index, gap("mean"), color=color, lw=1.4, ls=":", zorder=2)
            ax.plot(s.index, gap("best_val"), color=color, lw=2, marker="o", ms=4, zorder=3,
                    label=name(model, readout))
        ax.axhline(0, color=TEXT, lw=1, ls="--", zorder=1.5)
        ax.set_xticks(sorted(df["L"].unique()))
        ax.set_xlabel("window length L")
        ax.set_title(f"QRC readout {readout} ({READOUTS[readout]} obs.)", loc="left", fontsize=10, color=TEXT,
                     fontweight="bold")
        ax.legend(loc="lower left", frameon=False, fontsize=7.5, ncol=2)
        if col == 0:
            ax.set_ylabel("customer AP (test) − logistic")
    caption(fig, "Top: one dot per seed; ρ = Spearman rank correlation between validation and test over the 10 "
            "seeds (open: Z+ZZ / ESN-7, filled: XYZ-strings / ESN-45).\nBottom: solid = test AP of the seed with "
            "the best validation AP, dotted = mean over seeds, band = best to worst seed on test; all minus "
            "logistic regression (dashed zero).\n" + CAPTION, 0.035)
    fig.savefig(HERE / "seed_spread_selection.png")
    plt.close(fig)


def sign_agreement(a, b, n, draws, rng):
    """P(mean of n seeds of a − ref has the sign of the full gap). b: scalar (logistic) or seeds of a reference
    model, also subsampled to n seeds."""
    full = a.mean() - (b if np.isscalar(b) else b.mean())
    ga = np.array([rng.choice(a, n, replace=False).mean() for _ in range(draws)])
    gb = b if np.isscalar(b) else np.array([rng.choice(b, n, replace=False).mean() for _ in range(draws)])
    return np.mean(np.sign(ga - gb) == np.sign(full)), full


def plot_n_seeds(df, readout="XYZ-strings", Ls=(10, 15, 20, 30), draws=4000):
    """A study that runs n seeds instead of 10: how often does it get the direction of the 10-seed gap right?"""
    rng = np.random.default_rng(0)
    lg = logistic(df)
    esn = per_seed(df, "esn", readout)
    ramp = dict(zip(Ls, ("#c7dcf6", "#86b6ef", "#2a78d6", "#104281")))  # ordinal: darker = longer window
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, gridspec_kw={"wspace": 0.1})
    panels = [("tfim", "logistic"), ("tfim", "esn"), ("native-exchange", "logistic"), ("native-exchange", "esn")]
    ns = range(1, 10)
    for ax, (model, ref) in zip(axes, panels):
        q = per_seed(df, model, readout)
        for L in Ls:
            out = [sign_agreement(q.loc[L].values, lg[L] if ref == "logistic" else esn.loc[L].values, n, draws, rng)
                   for n in ns]
            ax.plot(list(ns), [p for p, _ in out], color=ramp[L], lw=2, marker="o", ms=4,
                    label=f"L = {L}  (gap {out[0][1]:+.3f})")
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.set_ylim(0.3, 1.02)
        ax.set_xticks(list(ns))
        ax.set_xlabel("seeds run")
        refname = "logistic" if ref == "logistic" else name("esn", readout)
        ax.set_title(f"{NAME[model]} vs {refname}", loc="left", fontsize=10, color=TEXT, fontweight="bold")
        ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    axes[0].set_ylabel("P(sign of the gap matches\nthe 10-seed result)")
    caption(fig, f"QRC readout {readout}, exact, test customer AP. For each n, {draws} random subsets of n of the "
            "10 seeds (ESN subsampled independently); gap in the legend = 10-seed mean difference. 0.5 = coin "
            "flip.\n" + CAPTION, -0.06)
    fig.savefig(HERE / "seed_spread_n_seeds.png")
    plt.close(fig)


def plot_shots(df):
    """Seed std vs L per budget, next to the std of the equal-width ESN and the TFIM - logistic gap."""
    budgets = {"exact": ("exact", "#104281"), "100000": ("10⁵ shots / window", "#2a78d6"),
               "10000": ("10⁴ shots / window", "#86b6ef")}
    lg = logistic(df)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.3, "wspace": 0.1})
    for row, readout in enumerate(READOUTS):
        esn = per_seed(df, "esn", readout).std(axis=1)
        for ax, model in zip(axes[row], QRC):
            for b, (label, color) in budgets.items():
                sd = per_seed(df, model, readout, budget=b).std(axis=1)
                ax.plot(sd.index, sd.values, color=color, lw=2, marker="o", ms=4, label=f"seed std, {label}")
            ax.plot(esn.index, esn.values, color=STYLE["esn"][0], lw=1.4, marker="s", ms=4, mfc="white",
                    label="seed std, equal-width ESN")
            gap = (per_seed(df, model, readout).mean(axis=1) - lg).abs()
            ax.plot(gap.index, gap.values, color=TEXT, lw=1.3, ls="--", label="|exact seed mean − logistic|")
            ax.set_xticks(sorted(df["L"].unique()))
            ax.set_title(f"{QRC[model]}", loc="left", fontsize=10, color=TEXT, fontweight="bold")
            if row == 1:
                ax.set_xlabel("window length L")
        axes[row, 0].set_ylabel(f"QRC readout {readout} ({READOUTS[readout]} obs.)\ncustomer AP (test)")
    for ax in axes.flat:
        ax.set_ylim(0, 0.4)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.06, 0.035), ncol=3,
               frameon=False)
    caption(fig, "Std over seeds 42-51 of test customer AP; under shots the seed also draws the sampling. Dashed: "
            "size of the exact gap to logistic regression, for scale.\n" + CAPTION, -0.035)
    fig.savefig(HERE / "seed_spread_shots.png")
    plt.close(fig)


def plot_ne_vs_esn(df, readout="XYZ-strings"):
    """Report figure: per-seed lines of native exchange (XYZ-strings readout) and the equal-width ESN on one axis.
    No caption (it lives in LaTeX)."""
    lg = logistic(df)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ends = []
    for model in ("esn", "native-exchange"):
        t = per_seed(df, model, readout)
        color = STYLE[model][0]
        for s in t.columns:
            ax.plot(t.index, t[s], color=color, lw=0.9, alpha=0.4, zorder=2)
        mean = t.mean(axis=1)
        ax.plot(mean.index, mean.values, color=color, lw=2.2, marker=STYLE[model][2], ms=4.5, zorder=3,
                label=f"{name(model, readout)}: seed mean")
        ends.append((mean.iloc[-1], f"{name(model, readout)}\nstd {t.loc[t.index[-1]].std():.3f}"))
    ax.plot(lg.index, lg.values, color=TEXT, ls="--", lw=1.2, zorder=2.5, label="logistic, raw window")
    ends.append((lg.iloc[-1], "logistic"))
    ends.sort()
    placed = []
    for y, text in ends:  # direct labels at the line ends, spread to avoid overlaps
        y = max(y, placed[-1] + 0.06) if placed else y
        placed.append(y)
        ax.text(lg.index[-1] + 1, y, text, fontsize=7.5, color=TEXT, va="center")
    ax.plot([], [], color=MUTED, lw=0.9, label="one seed (thin)")
    ax.set_xticks(lg.index)
    ax.set_xlim(-0.5, lg.index[-1] + 7)
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("window length L")
    ax.set_ylabel("customer AP (test, exact)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    for ext in ("pdf", "png"):  # pdf for the report (caption in LaTeX), png for quick viewing
        fig.savefig(HERE / f"seed_spread_ne_esn.{ext}")
    plt.close(fig)

def table(df):
    lg = logistic(df)
    out = ["# Seed spread, test customer AP (exact), seeds 42-51\n",
           "best_val: test AP of the seed with the highest validation AP. rho: Spearman(validation, test) over "
           "seeds. above_lg: seeds above logistic.\n"]
    for readout in READOUTS:
        sel = selection(df, readout)
        sel["logistic"] = sel["L"].map(lg)
        sel["above_lg"] = [int((per_seed(df, m, readout).loc[L] > lg[L]).sum()) for m, L in zip(sel["model"], sel["L"])]
        sel["model"] = [name(m, readout) for m in sel["model"]]
        out.append(f"## {readout}\n\n```\n{sel.to_string(index=False, float_format=lambda v: f'{v:.3f}')}\n```\n")
    (HERE / "seed_spread.md").write_text("\n".join(out))


if __name__ == "__main__":
    df = load()
    table(df)
    plot_lines(df)
    plot_selection(df)
    plot_n_seeds(df)
    plot_shots(df)
    plot_ne_vs_esn(df)
