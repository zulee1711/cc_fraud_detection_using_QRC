"""
Check for the advantage of Arm B in realistic hardware conditions.

Arm B's reset QRC beats logistic regression on customer-compromise fraud (customer AP) at
L = 5 to 8, but Gaussian readout noise already erases that edge (results/armb/README.md).
This script adds the other half of the hardware: pulse-level gate noise from SpinPulse.

For each hardware level and seed it computes the reservoir features of every split once
(cached in --cache-dir), then scores them at each readout precision: Gaussian noise of
width `precision` added to every feature, fresh for every split, as the `gaussian` backend
does. The readout, the decision threshold and the per-category AP are the CLI's own
(`fit_and_evaluate`, `add_derived_metrics`). Logistic regression on the raw window is
scored on the same windows as the reference.

The reservoir and data are given as an ordinary qrc command after the options, e.g. Arm B's
sweep1 setup at L = 5:

    python -m experiments.spinpulse_armb --levels ideal sp-ideal 1 2 5 --precisions 0 0.01 0.05 \
        qrc --data load --data-dir data/sc3_c1000_x6 --feature-set 1 --window-length 5 \
        --n-mem-qubits 3 --depth 4 --reset-inputs --observables XYZ+XYZ-2body

Levels (as in experiments.spinpulse_memory): ideal (exact ideal circuit), sp-ideal (realistic
chip without noise), or a coherence factor (1 = today's chip: T2* = 10 us, TJ* = 5 us).
Use --limit to time a configuration on a subset of windows before a full run.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import experiments.cli as cli
from experiments.spinpulse_memory import make_backend
from qrc.backends.spinpulse import NOISE_TYPES, SpinPulseBackend

FIELDS = [
    "label", "model", "level", "t2_us", "tj_us", "seed", "window_length", "reservoir",
    "precision", "split", "average_precision", "average_precision_amount",
    "average_precision_terminal", "average_precision_customer", "card_precision_top_k",
    "noise_stderr", "window_us", "quantum_seconds",
]
READOUT_SEED_OFFSET = 30_000


def subsample(data, limit, seed):
    """At most `limit` windows per split, drawn at random (same for every level and seed)."""
    rng = np.random.default_rng(seed)
    for split in cli.SPLITS:
        n = len(data[f"y_{split}"])
        if n <= limit:
            continue
        keep = np.sort(rng.choice(n, limit, replace=False))
        for key in [k for k in data if k.endswith(f"_{split}")]:
            data[key] = np.asarray(data[key])[keep]
    return data


def reservoir_features(level, seed, args, qrc_args, data):
    """Features of every split for one level and seed (loaded from the cache when present)."""
    cache = args.cache_dir / f"{args.label}_{level}_seed{seed}.npz"
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        return {s: stored[s] for s in cli.SPLITS}, dict(stored["info"].item())

    qrc_args.seed = seed
    protocol, reservoir_info = cli.build_protocol(qrc_args)  # encoder, reservoir and observables
    case = argparse.Namespace(**vars(args), reset=qrc_args.reset_inputs, precision=0.0)
    backend, t2_us, tj_us = make_backend(level, case, seed)
    protocol.backend = backend

    start = time.perf_counter()
    features, stderr = {}, []
    for split in cli.SPLITS:
        features[split] = protocol.run(np.asarray(data[f"X_{split}"]))
        if getattr(backend, "last_stderr", None) is not None:
            stderr.append(float(np.mean(backend.last_stderr)))
    window_us = ""
    if isinstance(backend, SpinPulseBackend):
        steps = qrc_args.window_length * backend._timeline.step_duration
        window_us = round(steps * backend.config.time_step_ns / 1000, 3)
    info = dict(
        t2_us=t2_us or "", tj_us=tj_us or "", window_us=window_us,
        noise_stderr=float(np.mean(stderr)) if stderr else "",
        quantum_seconds=round(time.perf_counter() - start, 1),
        reservoir=reservoir_info["hamiltonian"],
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, info=np.array(info, dtype=object), **features)
    return features, info


def score(features, data, qrc_args, precision, seed):
    """Per-split metrics after adding Gaussian readout noise of width `precision`."""
    if precision > 0:
        rng = np.random.default_rng([seed + READOUT_SEED_OFFSET, round(precision * 1e6)])
        features = {s: f + rng.normal(0.0, precision, f.shape) for s, f in features.items()}
    metrics, scores = cli.fit_and_evaluate(features, data, qrc_args.alpha, qrc_args.min_recall)
    cli.add_derived_metrics(metrics, scores, data, qrc_args.top_k)
    return metrics


def rows_for(metrics, **common):
    return [
        {**common, "split": split, **{k: metrics[split].get(k, "") for k in FIELDS if k.startswith(("average", "card"))}}
        for split in cli.EVAL_SPLITS
    ]


def done_runs(path, label):
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {(r["model"], r["level"], r["seed"], r["precision"]) for r in csv.DictReader(f) if r["label"] == label}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", nargs="+", default=["ideal", "sp-ideal", "1", "2", "5"],
                   help="hardware levels: ideal, sp-ideal, or coherence factors (1 = today's chip)")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--precisions", nargs="+", type=float, default=[0.0, 0.01, 0.05],
                   help="Gaussian readout noise widths added to the features (0 = exact readout)")
    p.add_argument("--noise", choices=[n for n in NOISE_TYPES if n != "none"], default="pink")
    p.add_argument("--qubit-noise", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coupling-noise", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--noise-samples", type=int, default=10, help="noise realizations per window")
    p.add_argument("--time-step-ns", type=float, default=10.0)
    p.add_argument("--limit", type=int, default=None, help="use at most this many windows per split")
    p.add_argument("--label", default="main", help="name of this sweep in the output and cache files")
    p.add_argument("--out", type=Path, default=cli.PROJECT_ROOT / "results" / "spinpulse_armb" / "armb.csv")
    p.add_argument("--cache-dir", type=Path, default=cli.PROJECT_ROOT / "results" / "spinpulse_armb" / "features")
    return p


def main(argv=None):
    args, qrc_argv = build_parser().parse_known_args(argv)
    qrc_args = cli.build_parser().parse_args(qrc_argv)
    if qrc_args.command != "qrc":
        raise SystemExit("give a `qrc ...` command after the options")
    qrc_args.backend = "numpy"  # only used to build the protocol; the level picks the real backend

    data = cli.prepare_data(qrc_args)
    if args.limit:
        data = subsample(data, args.limit, seed=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = done_runs(args.out, args.label)
    write_header = not args.out.exists()

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        # reference: logistic regression on the raw window, same windows and readout noise model
        raw = cli.baseline_features(data)
        for precision in args.precisions:
            if ("logistic", "", "", str(precision)) not in done:
                metrics = score(raw, data, qrc_args, precision, seed=0)
                writer.writerows(rows_for(
                    metrics, label=args.label, model="logistic", level="", seed="",
                    window_length=qrc_args.window_length, precision=precision,
                ))
        f.flush()

        for level in args.levels:
            for seed in args.seeds:
                if all(("qrc", level, str(seed), str(p)) in done for p in args.precisions):
                    continue
                features, info = reservoir_features(level, seed, args, qrc_args, data)
                for precision in args.precisions:
                    if ("qrc", level, str(seed), str(precision)) in done:
                        continue
                    metrics = score(features, data, qrc_args, precision, seed)
                    writer.writerows(rows_for(
                        metrics, label=args.label, model="qrc", level=level, seed=seed,
                        window_length=qrc_args.window_length, precision=precision, **info,
                    ))
                    test = metrics["test"]
                    print(f"[{args.label}] {level:>8} seed {seed} precision {precision:g}: customer AP "
                          f"{metrics['validation'].get('average_precision_customer', float('nan')):.3f} / "
                          f"{test.get('average_precision_customer', float('nan')):.3f}  "
                          f"({info['quantum_seconds']} s quantum)", flush=True)
                f.flush()
    return args.out


if __name__ == "__main__":
    main()