"""Operational metrics of the tuned models (results/3_tuned) at two threshold rules, with logistic regression and the LSTM
of results/2_fraud as references (L = 15).

Both thresholds are chosen per fitted model and seed on its own validation scores and applied unchanged to the test
scores (report, Evaluation Methods); predictions are score >= threshold.
    min-recall rule  the report's and experiments.cli's rule: minimise validation false positives subject to
                     validation recall >= r_min (qrc.decision.select_thresholds_min_fp, Linh's; r_min = 0.80).
                     At r_min = 0.80 the script checks that it reproduces the threshold and test metrics stored by
                     the runs.
    FPR rule         maximise validation recall subject to validation FPR <= f_max, ties to fewer false positives
                     (qrc.decision.select_thresholds_max_recall; f_max = 1 %).
Metrics at the threshold are those of ClassicalReadout.evaluate (precision, recall, FPR, F1, MCC, balanced accuracy),
plus the recall of each fraud category, alerts per day and Card Precision@K (qrc.decision.card_precision_top_ks,
handbook definition; threshold-free). Sweeps over r_min and f_max show the trade-off around each operating point.

Writes operational.md, operational_table_<rule>.tex, operational.csv (per model, seed, rule and level) and
operational_<rule>.{pdf,png}.
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

from qrc.decision import card_precision_top_ks, select_thresholds_max_recall, select_thresholds_min_fp

matplotlib.use("Agg")
_spec = importlib.util.spec_from_file_location("tuned", "results/3_tuned/tuned.py")
tuned = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tuned)

HERE = Path("results/3_tuned")
FRAUD = Path("results/2_fraud")
R_MIN = 0.80  # the report's and the CLI's default
R_SWEEP = np.round(np.arange(0.50, 0.951, 0.05), 2)
F_MAX = 0.01
F_SWEEP = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
RULES = {"min_recall": ("min-recall rule", f"r_min = {R_MIN:.2f}", R_MIN),
         "max_fpr": ("FPR rule", f"f_max = {F_MAX:.0%}", F_MAX)}
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
            yv, sv = data["y_validation"], scores["validation"]
            cp = card_precision_top_ks(data["y_test"], scores["test"], data["groups_test"], data["timestamps_test"],
                                       top_ks=TOP_KS)
            for rule, thresholds in (("min_recall", select_thresholds_min_fp(yv, sv, min_recalls=R_SWEEP)),
                                     ("max_fpr", select_thresholds_max_recall(yv, sv, max_fprs=F_SWEEP))):
                for level, t in thresholds.items():
                    m = at_threshold(data["y_test"], scores["test"], data["category_test"], data["timestamps_test"], t)
                    val = at_threshold(yv, sv, data["category_validation"], data["timestamps_validation"], t)
                    rows.append(dict(model=label, seed=seed, rule=rule, level=level, threshold=t,
                                     val_recall=val["recall"], val_fpr=val["false_positive_rate"], **m,
                                     **{f"cp@{k}": v for k, v in cp.items()}))
                    if rule == "min_recall" and np.isclose(level, R_MIN):  # = the stored run
                        assert np.isclose(t, stored["validation"]["threshold_used"]), (label, seed)
                        for k in ("precision", "recall", "false_positive_rate", "f1_score", "matthews_corr_coef",
                                  "balanced_accuracy"):
                            assert np.isclose(m[k], stored["test"][k], atol=1e-9), (label, seed, k)
                        assert np.isclose(cp[100], stored["test"]["card_precision_top_k"]), (label, seed)
                    if rule == "max_fpr":
                        assert val["false_positive_rate"] <= level + 1e-12, (label, seed, level)
    return pd.DataFrame(rows)


COLUMNS = {"precision": "precision", "recall": "recall", "false_positive_rate": "FPR", "f1_score": "F1",
           "matthews_corr_coef": "MCC", "balanced_accuracy": "bal. acc.", "recall_customer": "recall customer",
           "recall_amount": "recall amount", "recall_terminal": "recall terminal",
           "alerts_per_day": "alerts / day", "false_alerts_per_day": "false alerts / day",
           "val_recall": "val recall", "val_fpr": "val FPR", **{f"cp@{k}": f"CP@{k}" for k in TOP_KS}}


def operating_point(df, rule):
    level = RULES[rule][2]
    at = df[(df["rule"] == rule) & np.isclose(df["level"], level)]
    g = at.groupby("model", sort=False)[list(COLUMNS)]
    return g.mean().loc[list(MODELS)], g.std().loc[list(MODELS)]


def tables(df):
    fmt = {k: (".2%" if k in ("false_positive_rate", "val_fpr") else ".0f" if "per_day" in k else ".3f")
           for k in COLUMNS}
    out = ["# Operational metrics, test split, L = 15\n",
           "Thresholds chosen per model and seed on validation, applied unchanged to test. Mean ± std over seeds "
           "42-51 (logistic regression is deterministic). CP@K does not depend on the threshold.\n"]
    for rule, (name, setting, _) in RULES.items():
        mean, std = operating_point(df, rule)
        md = ["| model | " + " | ".join(COLUMNS.values()) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
        for model in MODELS:
            cells = [f"{mean.loc[model, k]:{fmt[k]}}" + (f" ± {std.loc[model, k]:{fmt[k]}}"
                                                         if not np.isnan(std.loc[model, k]) and fmt[k] == ".3f"
                                                         else "") for k in COLUMNS]
            md.append(f"| {model} | " + " | ".join(cells) + " |")
        out += [f"## {name} ({setting})\n", "\n".join(md), ""]

        keys = ("precision", "recall", "false_positive_rate", "f1_score", "matthews_corr_coef", "balanced_accuracy",
                "recall_customer", "alerts_per_day")
        tex = ["\\begin{tabular}{l" + "c" * len(keys) + "}", "\\toprule",
               " & ".join(["Model", "Precision", "Recall", "FPR", "F1", "MCC", "Bal.\\ acc.", "Recall (customer)",
                           "Alerts/day"]) + " \\\\", "\\midrule"]
        for model in MODELS:
            cells = [f"{100 * mean.loc[model, k]:.{1 if rule == 'min_recall' else 2}f}\\%"
                     if k == "false_positive_rate" else f"{mean.loc[model, k]:.0f}" if k == "alerts_per_day"
                     else f"{mean.loc[model, k]:.3f}" for k in keys]
            tex.append(" & ".join([model] + cells) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
        (HERE / f"operational_table_{rule}.tex").write_text("\n".join(tex) + "\n")
    for rule, name in (("min_recall", "r_min"), ("max_fpr", "f_max")):
        sweep = df[df["rule"] == rule].groupby(["model", "level"], sort=False)[
            ["recall", "precision", "recall_customer", "false_alerts_per_day"]].mean().round(3)
        out += [f"## Sweep over {name} (mean over seeds)\n", "```", sweep.to_string(), "```", ""]
    (HERE / "operational.md").write_text("\n".join(out))


def figure(df):
    """Two single-panel figures (test split, mean over seeds): operational_min_recall.{pdf,png}, false alerts per day
    against r_min (min-recall rule); operational_max_fpr.{pdf,png}, customer-fraud recall against f_max (FPR rule)."""
    panels = {"min_recall": ("false_alerts_per_day", R_MIN, "minimum validation recall $r_{\\min}$",
                             "false alerts per day (test)"),
              "max_fpr": ("recall_customer", F_MAX, "maximum validation false-positive rate $f_{\\max}$",
                          "customer-fraud recall (test)")}
    for rule, (metric, level, xlabel, ylabel) in panels.items():
        fig, ax = plt.subplots(figsize=(6, 3.8))
        scale = 100 if rule == "max_fpr" else 1  # FPR budgets in %
        for model, (color, ls, marker) in MODELS.items():
            m = df[(df["model"] == model) & (df["rule"] == rule)].groupby("level")[metric].mean()
            ax.plot(scale * m.index, m.values, color=color, ls=ls, lw=2 if "tuned" in model else 1.5, marker=marker,
                    ms=4, mfc="white" if model == "ESN-45 tuned" else color, label=model.replace(" (raw window)", ""))
        ax.axvline(scale * level, color=MUTED, lw=1, zorder=0)
        if rule == "min_recall":
            ax.set_yscale("log")
            ax.set_xticks(R_SWEEP[::2])
        else:
            ax.set_xscale("log")
            ax.set_xticks([100 * f for f in F_SWEEP], [f"{100 * f:g}%" for f in F_SWEEP])
            ax.minorticks_off()
            ax.set_ylim(0, 1)
        ax.legend(loc="lower right", frameon=False)
        ax.text(scale * level, ax.get_ylim()[1], " operating point", color=INK, fontsize=8, va="top")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for ext in ("pdf", "png"):
            fig.savefig(HERE / f"operational_{rule}.{ext}")
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
