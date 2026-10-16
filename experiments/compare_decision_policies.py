"""
Compare fraud decision policies on the same model scores.

It trains/runs each requested model once, then compares:
  1. the fixed 0.5 decision threshold;
  2. validation-selected thresholds that minimize false positives subject to different minimum-recall constraints;
  3. Card Precision@K for different daily review capacities K.

The test split is NEVER used to select a threshold. Thresholds are selected on validation and then frozen for test evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from experiments import cli
from experiments.lstm_baseline import run_lstm_baseline
from qrc.decision import card_precision_top_k, select_threshold_min_fp
from qrc.readout import ClassicalReadout


EVAL_SPLITS = ("validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--models",
        nargs="+",
        choices=("baseline", "qrc", "lstm"),
        default=["baseline"],
        help="models to compare; baseline is the fastest starting point",
    )

    # Shared data options.
    parser.add_argument("--data", choices=("simulate", "load"), default="simulate")
    parser.add_argument("--size", choices=tuple(cli.SIZES), default="medium")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=cli.PROJECT_ROOT / "data")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--feature-set", type=int, choices=tuple(cli.FEATURE_SETS), default=7)
    parser.add_argument("--window-length", type=int, default=3)

    # Decision-policy sweep.
    parser.add_argument(
        "--min-recalls",
        nargs="+",
        type=float,
        default=[0.60, 0.70, 0.80, 0.90],
        help="validation recall constraints for the min-FP threshold policy",
    )
    parser.add_argument(
        "--top-ks",
        nargs="+",
        type=int,
        default=[10, 25, 50, 100],
        help="daily card/customer review capacities for Card Precision@K",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.5,
        help="legacy fixed threshold included as a reference",
    )

    # Classical readout.
    parser.add_argument("--alpha", type=float, default=1.0)

    # QRC options.
    parser.add_argument("--backend", choices=cli.BACKEND_NAMES, default="estimator-exact")
    parser.add_argument("--precision", type=float, default=0.05)
    parser.add_argument("--n-input-qubits", type=int, default=None)
    parser.add_argument("--encoding-layers", type=int, default=1)
    parser.add_argument("--encoding-scaling", type=float, default=float(np.pi))
    parser.add_argument("--n-mem-qubits", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--entangler", choices=("cx", "cry"), default="cx")
    parser.add_argument("--reset-inputs", action="store_true")
    parser.add_argument("--observables", default="Z")
    parser.add_argument("--seed", type=int, default=42)

    # LSTM options.
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cli.PROJECT_ROOT / "results" / "decision_policy_comparison",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.fixed_threshold <= 1.0:
        raise ValueError("--fixed-threshold must be between 0 and 1")
    if not args.min_recalls:
        raise ValueError("provide at least one --min-recalls value")
    if any(not 0.0 < value <= 1.0 for value in args.min_recalls):
        raise ValueError("every --min-recalls value must be in (0, 1]")
    if not args.top_ks or any(value < 1 for value in args.top_ks):
        raise ValueError("every --top-ks value must be >= 1")


def _common_cli_argv(args: argparse.Namespace, command: str) -> list[str]:
    argv = [
        command,
        "--data",
        args.data,
        "--size",
        args.size,
        "--data-seed",
        str(args.data_seed),
        "--data-dir",
        str(args.data_dir),
        "--feature-set",
        str(args.feature_set),
        "--window-length",
        str(args.window_length),
    ]
    if args.start_date is not None:
        argv += ["--start-date", args.start_date]
    if args.end_date is not None:
        argv += ["--end-date", args.end_date]
    return argv


def prepare_shared_data(args: argparse.Namespace) -> dict:
    # prepare_data only uses the common/data arguments, so using the baseline
    # parser gives us a correctly shaped Namespace without duplicating project
    # defaults here.
    parsed = cli.build_parser().parse_args(_common_cli_argv(args, "baseline"))
    data = cli.prepare_data(parsed)
    cli.print_data_summary(data)
    return data


def _qrc_args(args: argparse.Namespace) -> argparse.Namespace:
    argv = _common_cli_argv(args, "qrc") + [
        "--alpha",
        str(args.alpha),
        "--backend",
        args.backend,
        "--precision",
        str(args.precision),
        "--encoding-layers",
        str(args.encoding_layers),
        "--encoding-scaling",
        str(args.encoding_scaling),
        "--n-mem-qubits",
        str(args.n_mem_qubits),
        "--depth",
        str(args.depth),
        "--entangler",
        args.entangler,
        "--observables",
        args.observables,
        "--seed",
        str(args.seed),
    ]
    if args.n_input_qubits is not None:
        argv += ["--n-input-qubits", str(args.n_input_qubits)]
    if args.reset_inputs:
        argv += ["--reset-inputs"]
    return cli.build_parser().parse_args(argv)


def classical_scores(features: dict, data: dict, alpha: float) -> dict[str, np.ndarray]:
    readout = ClassicalReadout(alpha=alpha)
    readout.fit(features["train"], data["y_train"])
    return {
        split: np.asarray(readout.predict(features[split]), dtype=float)
        for split in EVAL_SPLITS
    }


def score_model(model: str, data: dict, args: argparse.Namespace) -> dict[str, np.ndarray]:
    if model == "baseline":
        return classical_scores(cli.baseline_features(data), data, args.alpha)

    if model == "qrc":
        features, _ = cli.qrc_features(data, _qrc_args(args))
        return classical_scores(features, data, args.alpha)

    # Train LSTM only once. min_recall affects only threshold selection after
    # training, so any valid value can be used here; thresholds are recomputed
    # below from the returned validation scores.
    splits = {
        split: (data[f"X_{split}"], data[f"y_{split}"])
        for split in ("train", "validation", "test")
    }
    result = run_lstm_baseline(
        splits,
        seed=args.seed,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        min_recall=args.min_recalls[0],
    )
    return {
        split: np.asarray(result["scores"][split], dtype=float)
        for split in EVAL_SPLITS
    }


def classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y_pred = (scores >= threshold).astype(int)

    negative = y_true == 0
    false_positive_rate = (
        float(np.mean(y_pred[negative] == 1)) if np.any(negative) else 0.0
    )
    false_positives = int(np.sum((y_pred == 1) & (y_true == 0)))
    true_positives = int(np.sum((y_pred == 1) & (y_true == 1)))

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": false_positive_rate,
        "false_positives": false_positives,
        "true_positives": true_positives,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "matthews_corr_coef": float(matthews_corrcoef(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def threshold_rows(
    model: str,
    data: dict,
    scores: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[dict]:
    rows: list[dict] = []

    policies: list[tuple[str, float | None, float]] = [
        ("fixed_0.5", None, args.fixed_threshold)
    ]
    for min_recall in sorted(set(args.min_recalls)):
        threshold = select_threshold_min_fp(
            data["y_validation"],
            scores["validation"],
            min_recall=min_recall,
        )
        policies.append(("min_fp_with_min_recall", min_recall, threshold))

    for policy, min_recall, threshold in policies:
        for split in EVAL_SPLITS:
            row = {
                "model": model,
                "policy": policy,
                "min_recall_constraint": min_recall,
                "split": split,
                **classification_metrics(
                    data[f"y_{split}"], scores[split], threshold
                ),
            }
            rows.append(row)
    return rows


def topk_rows(
    model: str,
    data: dict,
    scores: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[dict]:
    rows: list[dict] = []
    for top_k in sorted(set(args.top_ks)):
        for split in EVAL_SPLITS:
            value = card_precision_top_k(
                data[f"y_{split}"],
                scores[split],
                data[f"groups_{split}"],
                data[f"timestamps_{split}"],
                top_k=top_k,
            )
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "top_k": top_k,
                    "card_precision_top_k": float(value),
                }
            )
    return rows


def write_summary(
    threshold_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "Decision-policy comparison",
        "==========================",
        "",
        "Threshold policy (TEST split)",
        "-----------------------------",
    ]

    test_threshold = threshold_df[threshold_df["split"] == "test"].copy()
    cols = [
        "model",
        "policy",
        "min_recall_constraint",
        "threshold",
        "precision",
        "recall",
        "false_positive_rate",
        "false_positives",
        "f1_score",
        "pr_auc",
    ]
    lines.append(test_threshold[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    lines += ["", "Card Precision@K (TEST split)", "--------------------------------"]
    test_topk = topk_df[topk_df["split"] == "test"].copy()
    lines.append(test_topk.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    lines += [
        "",
        "Interpretation",
        "--------------",
        "* fixed_0.5 is the legacy reference.",
        "* min_fp_with_min_recall selects its threshold on validation only and freezes it for test.",
        "* Card Precision@K is a ranking/capacity metric and does not choose the probability threshold.",
        "* Compare lower test false-positive rate against the resulting test recall to understand the customer-protection trade-off.",
        "* Compare Card Precision@K at the K that best matches investigator capacity.",
    ]
    (output_dir / "comparison_summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_shared_data(args)
    all_threshold_rows: list[dict] = []
    all_topk_rows: list[dict] = []

    for model in args.models:
        print(f"\n=== {model.upper()} ===", flush=True)
        scores = score_model(model, data, args)
        all_threshold_rows.extend(threshold_rows(model, data, scores, args))
        all_topk_rows.extend(topk_rows(model, data, scores, args))

    threshold_df = pd.DataFrame(all_threshold_rows)
    topk_df = pd.DataFrame(all_topk_rows)

    threshold_path = args.output_dir / "threshold_comparison.csv"
    topk_path = args.output_dir / "topk_comparison.csv"
    threshold_df.to_csv(threshold_path, index=False)
    topk_df.to_csv(topk_path, index=False)
    write_summary(threshold_df, topk_df, args.output_dir)

    print("\nThreshold comparison (test):")
    print(
        threshold_df[threshold_df["split"] == "test"][
            [
                "model",
                "policy",
                "min_recall_constraint",
                "threshold",
                "precision",
                "recall",
                "false_positive_rate",
                "false_positives",
                "f1_score",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nCard Precision@K (test):")
    print(
        topk_df[topk_df["split"] == "test"].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    print(f"\nSaved comparison results to: {args.output_dir}")
    print(f"  {threshold_path.name}")
    print(f"  {topk_path.name}")
    print("  comparison_summary.txt")


if __name__ == "__main__":
    main()