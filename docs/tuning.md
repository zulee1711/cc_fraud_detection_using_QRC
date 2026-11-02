# Hyperparameter search (Optuna)

`experiments/tune.py` searches the reservoir configuration with [Optuna](https://optuna.org)
(multi-objective TPE sampler). Each trial is a full `qrc` run. It **maximises** the
**validation** score (average precision by default; `--metric` takes any reported metric, e.g.
`average_precision_customer` or `card_precision_top_k`) and **minimises** the costs below at the
same time. The result is a **Pareto front**: the trials that no other trial beats on every
objective at once. The test split is never used for selection.

## Quick examples

```bash
# Simplest: accuracy against circuit depth, on the small dataset
python -m experiments.tune --study-name depth --objectives depth --size small --n-trials 30

# The full multi-objective search, with sampled expectation values
python -m experiments.tune --n-trials 100 --size medium --backend gaussian

# Accuracy against hardware size only, averaged over three reservoir seeds
python -m experiments.tune --study-name size --objectives qubits depth --seeds 1 2 3 --n-trials 50
```

## Objectives

| Cost (`--objectives`) | What it measures                                                                                                                  |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `qubits`              | Input + memory qubits                                                                                                             |
| `depth`               | Depth of the circuit for one window (L × encoding + reservoir), transpiled to `rz`/`sx`/`x`/`cz` on a linear chain of qubits     |
| `bases`               | Measurement bases: observables that can be read from the same shots count once (`Z` and `Z+ZZ`: 1, `XYZ`: 3)                     |
| `shots`               | Shots per basis, ≈ 1/precision². Searched over ≈10³ … 10⁵. Needs `--backend gaussian`, `shots` or `spinpulse`                    |
| `wall_time`           | Seconds to simulate the reservoir over all splits (machine-dependent)                                                            |

By default all costs are objectives, except `shots` with an exact backend. Pass a subset with
`--objectives qubits depth`, or `--objectives` alone for a single-objective search.

## Search space

Defined in `suggest_params` in `experiments/tune.py`; the ranges are constants at the top of the file.
The default column is the value `experiments.cli qrc` uses when the option is not given, i.e. without tuning:

| Parameter                       | Values searched                                                               | Default (`cli qrc`)                    |
| ------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------- |
| encoding                        | `angle`, `dense`, `reupload` (`--encodings`)                                  | `auto` (angle with one qubit per feature) |
| input qubits                    | dense: ⌈F/2⌉ … F−1; angle / reupload: F (F = number of features)              | F                                      |
| memory qubits                   | 2 … 5, with input + memory ≤ `--max-qubits` (default 12)                      | 2                                      |
| encoding layers                 | 2 … 3 (reupload only)                                                          | 1                                      |
| encoding scaling                | π/4 … π (above π, two different values can give the same state)               | π                                      |
| reservoir                       | `random` or a Hamiltonian (`--hamiltonians`)                                  | `random`                               |
| depth, entangler                | 1 … 6, `cx` / `cry`; random circuit only                                      | 2, `cx`                                |
| topology, evolution time        | `--topologies`, 0.1 … 10 (log); Hamiltonians only                             | `linear`, 1.0                          |
| Trotter steps                   | 2 … 3; Hamiltonians only (see `--trotter-steps` in the [CLI reference](cli_reference.md#reservoir-and-backend-qrc-only)) | 2 |
| observables                     | `Z`, `Z+ZZ`, `XYZ`, `XYZ+XYZ-2body`                                           | `Z`                                    |
| readout `alpha`                 | 1e-1 … 1e5 (log)                                                              | 1.0                                    |
| precision                       | 0.03, 0.01, 0.003 (≈10³, 10⁴, 10⁵ shots); only with the `shots` objective     | 0.05 (≈400 shots)                      |

Parts of each reservoir are random but fixed: the gate angles of the random circuit and the
couplings of the Hamiltonians. The seed picks that random draw. `--seeds 1 2 3` runs every trial
with three draws and scores it by their average, so a configuration can't win through one lucky
draw (each trial takes 3× longer). Without `--seeds`, every trial uses `--seed` (default `42`).

## Options

Tuner options: `--n-trials`, `--timeout`, `--study-name`, `--output-dir` (default `results/tune`),
`--storage`, `--metric`, `--objectives`, `--seeds`, `--max-qubits`, `--encodings`, `--hamiltonians`,
`--topologies`, `--sampler-seed` and `--n-jobs`. Every other option goes to `qrc` unchanged and is fixed for the
whole study (data, feature set, window length, backend, ...). Passing an option the search sets
(e.g. `--n-mem-qubits`) is an error.

`--n-jobs N` runs N trials at once, in threads that share the prepared data (so memory grows little).
It speeds up the `numpy` backend and `gaussian` / `shots` with `--simulator numpy` (about 3.6× with 6 threads on a
6-core laptop); the Aer backends (`estimator-exact`, `gaussian`, `shots`) gain nothing from it. Use at
most one thread per physical core.

## Outputs

A study is saved in `results/tune/<study-name>/`:

- `study.db`: the Optuna study. Running the same command again **resumes** it. Resuming with
  different fixed options is refused, so that all trials in a study stay comparable.
- `runs/`: every trial as an ordinary `qrc` run, with its own `summary.csv`
- `trials.csv`: one row per trial, with its parameters, objective values, costs (also two-qubit
  gate count and depth, and number of observables) and validation and test metrics
- `pareto.json`: the Pareto front, each trial with the `experiments.cli qrc` command that reproduces it

At the end, the tuner prints the Pareto front as a table.
To browse a study interactively: `pip install optuna-dashboard`, then
`optuna-dashboard sqlite:///results/tune/<study-name>/study.db`.
