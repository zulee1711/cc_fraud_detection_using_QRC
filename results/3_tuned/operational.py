"""Operational metrics of the tuned models (results/3_tuned) at the report's operating threshold, with logistic regression
and the LSTM of results/2_fraud as references (L = 15).

The threshold is the one of the report (Evaluation Methods) and of experiments.cli: for each fitted model and seed, the
validation threshold that minimises false positives subject to validation recall >= r_min (qrc.decision, Linh's
select_thresholds_min_fp; r_min = 0.80 by default), applied unchanged to the test scores. At r_min = 0.80 the script
checks that it reproduces the threshold and test metrics stored by the runs. Metrics at that threshold are those of
ClassicalReadout.evaluate (precision, recall, FPR, F1, MCC, balanced accuracy), plus the recall of each fraud category,
alerts per day and Card Precision@K (qrc.decision.card_precision_top_ks, handbook definition). A sweep over r_min shows
the trade-off around the operating point.

Writes operational.md, operational_table.tex, operational.csv (per model, seed and r_min) and operational.{pdf,png}.
Usage: PYTHONPATH=. .venv/bin/python results/3_tuned/operational.py
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

from qrc.decision import card_precision_top_ks, select_thresholds_min_fp

matplotlib.use("Agg")
_spec = importlib.util.spec_from_file_location("tuned", "results/3_tuned/tuned.py")
tuned = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tuned)

HERE = Path("results/3_tuned")
FRAUD = Path("results/2_fraud")
R_MIN = 0.80  # the report's and the CLI's default
R_SWEEP = np.round(np.arange(0.50, 0.951, 0.05), 2)
TOP_KS = (10, 25, 50, 100)
CATEGORIES = ("customer", "amount", "terminal")
MODELS = {  # label -> (colour, line style, marker)
    "TFIM tuned": ("#2a78d6", "-", "o"),
    "ESN-45 tuned": ("#52514e", "-", "s"),
    "logistic (raw window)": ("#52514e", "--", "None"),
    "LSTM": ("#0b0b0b", ":", "v"),
}
INK, MUTED, GRID = "#0b0b0b", "#8a8985", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "figure.facecolor": "white"})


def runs():
    """{label: [(seed, stored test metrics, scores {validation, test})]}."""
    out = {"TFIM tuned": [], "ESN-45 tuned": []}
    for label, model in (("TFIM tuned", "tfim"), ("ESN-45 tuned", "esn")):
        for s in tuned.EVAL_SEEDS:
            record = next(r for r in json.loads((HERE / "runs" / f"{model}_tuned_s{s}.json").read_text())
                          if r["budget"] is None)
            z = np.load(HERE / "runs" / "scores" / f"{model}_tuned_s{s}_exact.npz")
            out[label].append((s, record["metrics"], {k: z[f"scores_{k}"] for k in ("validation", "test")}))
    for label, name in (("logistic (raw window)", "baseline_L15"), ("LSTM", "lstm_L15_s{}")):
        out[label] = []
        for s in ([None] if "{}" not in name else tuned.EVAL_SEEDS):
            record = json.loads((FRAUD / "runs" / f"{name.format(s)}.json").read_text())[0]
            z = np.load(Path(record["run_dir"]) / "scores.npz")
            out[label].append((s, record["metrics"], {k: z[f"scores_{k}"] for k in ("validation", "test")}))
    return out


def at_threshold(y, scores, category, timestamps, threshold):
    """ClassicalReadout.evaluate's operating-point metrics (prediction = score >= threshold), per-category recall
    and alerts per day."""
    pred = (scores >= threshold).astype(int)
    days = pd.to_datetime(timestamps).normalize().nunique()
    m = dict(precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
             false_positive_rate=float(np.mean(pred[y == 0] == 1)), f1_score=f1_score(y, pred, zero_division=0),
             matthews_corr_coef=matthews_corrcoef(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
             alerts_per_day=pred.sum() / days, false_alerts_per_day=(pred & (y == 0)).sum() / days)
    for c in CATEGORIES:
        m[f"recall_{c}"] = float(pred[category == c].mean())
    return m


def evaluate_all(data):
    rows = []
    for label, entries in runs().items():
        for seed, stored, scores in entries:
            thresholds = select_thresholds_min_fp(data["y_validation"], scores["validation"], min_recalls=R_SWEEP)
            cp = card_precision_top_ks(data["y_test"], scores["test"], data["groups_test"], data["timestamps_test"],
                                       top_ks=TOP_KS)
            for r_min, t in thresholds.items():
                m = at_threshold(data["y_test"], scores["test"], data["category_test"], data["timestamps_test"], t)
                rows.append(dict(model=label, seed=seed, r_min=r_min, threshold=t, **m,
                                 **{f"cp@{k}": v for k, v in cp.items()}))
                if np.isclose(r_min, R_MIN):  # same threshold and metrics as the stored run
                    assert np.isclose(t, stored["validation"]["threshold_used"]), (label, seed)
                    for k in ("precision", "recall", "false_positive_rate", "f1_score", "matthews_corr_coef",
                              "balanced_accuracy"):
                        assert np.isclose(m[k], stored["test"][k], atol=1e-9), (label, seed, k)
                    assert np.isclose(cp[100], stored["test"]["card_precision_top_k"]), (label, seed)
    return pd.DataFrame(rows)


COLUMNS = {"precision": "precision", "recall": "recall", "false_positive_rate": "FPR", "f1_score": "F1",
           "matthews_corr_coef": "MCC", "balanced_accuracy": "bal. acc.", "recall_customer": "recall customer",
           "recall_amount": "recall amount", "recall_terminal": "recall terminal",
           "alerts_per_day": "alerts / day", "false_alerts_per_day": "false alerts / day",
           **{f"cp@{k}": f"CP@{k}" for k in TOP_KS}}


def tables(df):
    at = df[np.isclose(df["r_min"], R_MIN)]
    g = at.groupby("model", sort=False)[list(COLUMNS)]
    mean, std = g.mean().loc[list(MODELS)], g.std().loc[list(MODELS)]
    fmt = {k: (".1%" if k == "false_positive_rate" else ".0f" if "per_day" in k else ".3f") for k in COLUMNS}
    md = ["| model | " + " | ".join(COLUMNS.values()) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    for model in MODELS:
        cells = [f"{mean.loc[model, k]:{fmt[k]}}" + (f" ± {std.loc[model, k]:{fmt[k]}}"
                                                     if not np.isnan(std.loc[model, k]) and fmt[k] == ".3f" else "")
                 for k in COLUMNS]
        md.append(f"| {model} | " + " | ".join(cells) + " |")
    out = [f"# Operational metrics at the operating threshold (r_min = {R_MIN:.2f}), test split, L = 15\n",
           "Threshold per model and seed: min validation FP s.t. validation recall >= r_min (qrc.decision), applied "
           "unchanged to test. Mean ± std over seeds 42-51 (logistic regression is deterministic).\n", "\n".join(md),
           "\n## Sweep over r_min (mean over seeds)\n", "```",
           df.groupby(["model", "r_min"], sort=False)[["recall", "precision", "recall_customer",
                                                       "false_alerts_per_day"]].mean().round(3).to_string(), "```"]
    (HERE / "operational.md").write_text("\n".join(out) + "\n")

    keys = ("precision", "recall", "false_positive_rate", "f1_score", "matthews_corr_coef", "balanced_accuracy",
            "recall_customer", "alerts_per_day", "cp@100")
    tex = ["\\begin{tabular}{l" + "c" * len(keys) + "}", "\\toprule",
           " & ".join(["Model", "Precision", "Recall", "FPR", "F1", "MCC", "Bal.\\ acc.", "Recall (customer)",
                       "Alerts/day", "CP@100"]) + " \\\\", "\\midrule"]
    for model in MODELS:
        cells = []
        for k in keys:
            v = mean.loc[model, k]
            cells.append(f"{100 * v:.1f}\\%" if k == "false_positive_rate" else f"{v:.0f}" if k == "alerts_per_day"
                         else f"{v:.3f}")
        tex.append(" & ".join([model] + cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (HERE / "operational_table.tex").write_text("\n".join(tex) + "\n")


def figure(df):
    """False alerts per day on test against r_min (the threshold rule's only parameter), mean over seeds."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for model, (color, ls, marker) in MODELS.items():
        m = df[df["model"] == model].groupby("r_min")["false_alerts_per_day"].mean()
        ax.plot(m.index, m.values, color=color, ls=ls, lw=2 if "tuned" in model else 1.5, marker=marker, ms=4,
                mfc="white" if model == "ESN-45 tuned" else color, label=model.replace(" (raw window)", ""))
    ax.axvline(R_MIN, color=MUTED, lw=1, zorder=0)
    ax.set_yscale("log")
    ax.text(R_MIN, ax.get_ylim()[1], " operating point", color=INK, fontsize=8, va="top")
    ax.set_xlabel("minimum validation recall $r_{\\min}$")
    ax.set_ylabel("false alerts per day (test)")
    ax.set_xticks(R_SWEEP[::2])
    ax.legend(loc="lower right", frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"operational.{ext}")
    plt.close(fig)


def main():
    data = tuned.load_data()
    df = evaluate_all(data)
    df.to_csv(HERE / "operational.csv", index=False)
    tables(df)
    figure(df)
    print((HERE / "operational.md").read_text().split("## Sweep")[0])


if __name__ == "__main__":
    main()
