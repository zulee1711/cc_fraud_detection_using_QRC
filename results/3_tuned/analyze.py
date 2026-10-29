"""Tables, paired tests and figure for results/3_tuned (tuned.py): writes report.md and
tuning.{pdf,png}. References (untuned TFIM / ESN-45, logistic regression, LSTM at L = 15) come from results/2_fraud.

Usage: PYTHONPATH=. .venv/bin/python results/3_tuned/analyze.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy import stats

matplotlib.use("Agg")
optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = Path("results/3_tuned")
FRAUD = Path("results/2_fraud")
L = 15
SEEDS = tuple(range(42, 52))
TUNE_SEEDS = (42, 43, 44)
HELD_OUT = tuple(s for s in SEEDS if s not in TUNE_SEEDS)
METRICS = {"average_precision_customer": "customer AP", "average_precision_amount": "amount AP",
           "average_precision": "overall AP", "card_precision_top_k": "CP@100"}
BOUNDS = {"tfim": dict(h=(0.01, 10.0), time=(0.1, 20.0), encoding_scaling=(0.1, 2 * np.pi), alpha=(1e-4, 1e3)),
          "esn": dict(spectral_radius=(0.05, 2.5), leak_rate=(0.01, 1.0), input_scaling=(0.01, 30.0),
                      alpha=(1e-4, 1e3))}
LOG = {"spectral_radius": False}  # every other dimension is searched log-uniformly
BLUE_MID, GREY, INK, MUTED, GRID = ("#2a78d6", "#52514e", "#0b0b0b",
                                    "#8a8985", "#e4e3df")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "figure.facecolor": "white"})


def _record(path, observables="XYZ-strings", budget=None):
    records = json.loads(Path(path).read_text())
    return next(r for r in records if r.get("observables") in (None, observables) and r.get("budget") == budget)


def _scores_2fraud(kind, seed, budget="exact"):
    """Validation/test scores of a results/2_fraud run at L = 15, with labels and categories."""
    if kind == "tfim":
        z = np.load(FRAUD / "runs" / "scores" / f"qrc_tfim_L{L}_s{seed}_XYZ-strings_{budget}.npz")
        return dict(z)
    name = {"esn": f"esn45_L{L}_s{seed}", "logistic": f"baseline_L{L}", "lstm": f"lstm_L{L}_s{seed}"}[kind]
    run_dir = json.loads((FRAUD / "runs" / f"{name}.json").read_text())[0]["run_dir"]
    return dict(np.load(Path(run_dir) / "scores.npz"))


def load():
    """{label: {seed: (metrics, scores)}}; scores = dict of scores/y/category per split."""
    labels = _scores_2fraud("tfim", 42)  # y and category are the same for every model
    labels = {k: v for k, v in labels.items() if not k.startswith("scores")}
    out = {}
    for label, model, budget in (("TFIM tuned", "tfim", None), ("TFIM tuned, 10⁵ shots", "tfim", 100_000),
                                 ("TFIM tuned, 10⁴ shots", "tfim", 10_000), ("ESN-45 tuned", "esn", None)):
        runs = {}
        for s in SEEDS:
            path = HERE / "runs" / f"{model}_tuned_s{s}.json"
            if path.exists():
                z = np.load(HERE / "runs" / "scores" / f"{model}_tuned_s{s}_{budget or 'exact'}.npz")
                runs[s] = (_record(path, budget=budget)["metrics"], {**labels, **dict(z)})
        out[label] = runs
    for label, kind, budget, pattern in (
            ("TFIM untuned", "tfim", "exact", "qrc_tfim_L15_s{}.json"),
            ("TFIM untuned, 10⁵ shots", "tfim", "100000", "qrc_tfim_L15_s{}.json"),
            ("ESN-45 untuned", "esn", None, "esn45_L15_s{}.json"), ("LSTM", "lstm", None, "lstm_L15_s{}.json")):
        b = None if budget in (None, "exact") else int(budget)
        out[label] = {s: (_record(FRAUD / "runs" / pattern.format(s), budget=b)["metrics"],
                          _scores_2fraud(kind, s, budget or "exact")) for s in SEEDS}
    out["logistic (raw window)"] = {None: (_record(FRAUD / "runs" / "baseline_L15.json")["metrics"],
                                           _scores_2fraud("logistic", None))}
    return out


def mean_std(values):
    v = np.asarray(values, float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def tuning_table():
    rows = ["| model | trials | untuned val AP | best val AP | best trial | best setting | at a bound |",
            "|---|---|---|---|---|---|---|"]
    histories = {}
    for model, label in (("tfim", "TFIM"), ("esn", "ESN-45")):
        db = HERE / f"study_{model}.db"
        if not db.exists():
            continue
        s = optuna.load_study(study_name=model, storage=f"sqlite:///{db}")
        done = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]
        histories[label] = np.maximum.accumulate([t.value for t in done])
        edge = []
        for k, v in s.best_params.items():
            lo, hi = BOUNDS[model][k]
            f = ((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if LOG.get(k, True)
                 else (v - lo) / (hi - lo))
            if f < 0.03 or f > 0.97:
                edge.append(k)
        setting = ", ".join(f"{k} {v:.3g}" for k, v in s.best_params.items())
        rows.append(f"| {label} | {len(done)} | {done[0].value:.4f} | {s.best_value:.4f} | {s.best_trial.number} | "
                    f"{setting} | {', '.join(edge) or 'none'} |")
    return "\n".join(rows), histories


def ap_table(runs, seeds):
    rows = ["| model | " + " | ".join(METRICS.values()) + " | val customer AP |", "|---" * (len(METRICS) + 2) + "|"]
    for label, by_seed in runs.items():
        vals = [by_seed[s] for s in (by_seed if None in by_seed else seeds) if s in by_seed]
        if vals:
            rows.append(f"| {label} | " + " | ".join(mean_std([m["test"][k] for m, _ in vals]) for k in METRICS)
                        + f" | {mean_std([m['validation']['average_precision_customer'] for m, _ in vals])} |")
    return "\n".join(rows)


def paired(runs, seeds):
    rows = ["| a − b | mean diff | bootstrap 95% CI | paired t p | Wilcoxon p | a wins |", "|---|---|---|---|---|---|"]
    rng = np.random.default_rng(0)
    for a, b in (("TFIM tuned", "ESN-45 tuned"), ("TFIM tuned, 10⁵ shots", "ESN-45 tuned"),
                 ("TFIM tuned", "TFIM untuned"), ("ESN-45 tuned", "ESN-45 untuned"),
                 ("TFIM untuned", "ESN-45 untuned")):
        common = [s for s in seeds if s in runs[a] and s in runs[b]]
        if len(common) < 3:
            continue
        d = np.array([runs[a][s][0]["test"]["average_precision_customer"]
                      - runs[b][s][0]["test"]["average_precision_customer"] for s in common])
        boot = rng.choice(d, (10_000, len(d))).mean(1)
        rows.append(f"| {a} − {b} | {d.mean():+.3f} | [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]"
                    f" | {stats.ttest_rel(d, np.zeros_like(d)).pvalue if d.std() else float('nan'):.2g} | "
                    f"{stats.wilcoxon(d).pvalue:.2g} | {(d > 0).sum()}/{len(d)} |")
    for a in ("TFIM tuned", "ESN-45 tuned"):
        common = [s for s in seeds if s in runs[a]]
        if len(common) < 3:
            continue
        lg = runs["logistic (raw window)"][None][0]["test"]["average_precision_customer"]
        d = np.array([runs[a][s][0]["test"]["average_precision_customer"] for s in common]) - lg
        rows.append(f"| {a} − logistic | {d.mean():+.3f} | | {stats.ttest_1samp(d, 0).pvalue:.2g} (1-sample) | | "
                    f"{(d > 0).sum()}/{len(d)} |")
    return "\n".join(rows)


def figure(runs, histories):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": (1, 1.25), "wspace": 0.3})
    for label, color in (("TFIM", BLUE_MID), ("ESN-45", GREY)):
        if label in histories:
            h = histories[label]
            ax1.plot(np.arange(len(h)), h, color=color, lw=2, drawstyle="steps-post")
            ax1.text(len(h) - 1 + 1.5, h[-1], f"{label} {h[-1]:.3f}", color=INK, fontsize=8, va="center")
    ax1.set_xlabel("trial")
    ax1.set_ylabel("best validation customer AP so far")
    ax1.set_xlim(0, 118)
    ax1.set_title("Tuning", loc="left", fontsize=10, color=INK, fontweight="bold")
    order = [("TFIM untuned", BLUE_MID, "white"), ("TFIM tuned", BLUE_MID, BLUE_MID),
             ("ESN-45 untuned", GREY, "white"),
             ("ESN-45 tuned", GREY, GREY)]
    for i, (label, color, face) in enumerate(order):
        vals = [m["test"]["average_precision_customer"] for m, _ in runs[label].values()]
        if not vals:
            continue
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax2.scatter(np.full(len(vals), i) + jitter, vals, s=16, color=color, facecolor=face, lw=1, zorder=3)
        ax2.plot([i - 0.25, i + 0.25], [np.mean(vals)] * 2, color=INK, lw=1.5, zorder=4)
    for label, ls in (("logistic (raw window)", "--"), ("LSTM", ":")):
        v = np.mean([m["test"]["average_precision_customer"] for m, _ in runs[label].values()])
        ax2.axhline(v, color=GREY, ls=ls, lw=1.2, zorder=1)
        ax2.text(len(order) + 0.25, v + 0.004, label.split(" (")[0], fontsize=8, color=INK, va="bottom", ha="right")
    ax2.set_xticks(range(len(order)), [o[0].replace(" ", "\n", 1).replace(", ", "\n") for o in order], fontsize=8)
    ax2.set_xlim(-0.5, len(order) + 0.3)
    ax2.set_ylabel("customer AP (test)")
    ax2.set_title("Test", loc="left", fontsize=10, color=INK, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"tuning.{ext}")
    plt.close(fig)


def main():
    runs = load()
    table, histories = tuning_table()
    out = ["# Equal-budget tuning at L = 15: TFIM (2 Trotter steps, XYZ-strings) vs ESN-45\n",
           "Protocol in tuned.py. References (untuned, logistic, LSTM) from results/2_fraud at L = 15.\n",
           "## Tuning (validation customer AP, mean over seeds 42-44)\n", table, "",
           "## Test, mean ± std over seeds 42-51\n", ap_table(runs, SEEDS), "",
           "## Test, held-out seeds 45-51\n", ap_table(runs, HELD_OUT), "",
           "## Paired differences in test customer AP, seeds 42-51\n", paired(runs, SEEDS), "",
           "## Paired differences in test customer AP, held-out seeds 45-51\n", paired(runs, HELD_OUT), "",
           "Operational metrics at the report's operating threshold: operational.py -> operational.md.\n"]
    (HERE / "report.md").write_text("\n".join(out))
    figure(runs, histories)
    print((HERE / "report.md").read_text())


if __name__ == "__main__":
    main()
