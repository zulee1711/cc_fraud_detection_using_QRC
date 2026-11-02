# Command-line reference

Everything about `experiments/cli.py` and `experiments/sweep.py`. For a first run, see the
[README](../README.md). `python -m experiments.cli <command> --help` lists every option.

## Contents

1. [Commands](#1-commands)
2. [Metrics](#2-metrics)
3. [Options](#3-options)
4. [Backends](#4-backends)
5. [Encodings](#5-encodings)
6. [Run outputs](#6-run-outputs)
7. [Reproducibility](#7-reproducibility)
8. [Sweeps](#8-sweeps)
9. [Other scripts](#9-other-scripts)

---

## 1. Commands

```bash
python -m experiments.cli <command> [options]
```

| Command    | What it does                                                                                             |
| ---------- | -------------------------------------------------------------------------------------------------------- |
| `prepare`  | Data preparation only. Prints the split shapes and fraud counts.                                         |
| `baseline` | Classical control: the logistic readout fitted directly on the raw (flattened) windows.                  |
| `lstm`     | Classical control: an LSTM trained on the windows; validation loss picks the best epoch (needs PyTorch). |
| `esn`      | Classical reservoir control: a fixed random echo state network on each window, then the same logistic readout as `qrc`. |
| `qrc`      | Runs the quantum reservoir on each window, then fits the same logistic readout on its outputs.           |

All commands run the same data preparation:

```
data (simulated or loaded) → feature engineering → DataProcessor → per-customer windows
```

Models are fitted on **train only**. Validation is used for two things only: the decision threshold
(the one with the fewest false positives that still catches at least `--min-recall` of the frauds;
it is then frozen and reused on test) and, for `lstm`, early stopping on validation loss (the
hyperparameter search, [docs/tuning.md](tuning.md), also selects on validation).

## 2. Metrics

Reported on **validation and test**:

- threshold-free: average precision (AP) and ROC-AUC;
- at the selected threshold: F1, precision, recall, false positive rate, Matthews correlation
  coefficient (MCC) and balanced accuracy;
- **Card Precision@K** (`card_precision_top_k`): every day, the `--top-k` customers with the highest score are
  reviewed, and the metric is the mean daily fraction of them that are fraudulent (as in the
  Fraud Detection Handbook; a detected card is dropped on later days);
- **AP per fraud category** (`average_precision_{amount,terminal,customer}`), when the data carries the
  simulator's fraud scenarios: each category ranked against the genuine transactions. `amount` frauds give themselves
  away by their amount alone, `terminal` frauds need the terminal's history and `customer` frauds need
  the customer's history (see `qrc/evaluation.py`).

## 3. Options

### Data (every command)

| Option                        | Default    | Meaning                                                                                                                                              |
| ----------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--data simulate\|load`       | `simulate` | Simulate a dataset, or load the split `.pkl` files                                                                                                   |
| `--size small\|medium\|large` | `medium`   | Simulation preset: 25 / 250 / 5000 customers over 365 days (~14k / ~170k / ~3.5M transactions), all with the shared dataset's ~0.9% frauds; `large` is the shared dataset |
| `--data-seed`                 | `0`        | Simulation seed                                                                                                                                      |
| `--save-data DIR`             | –          | Also write the simulated splits to `DIR` (reload with `--data load --data-dir DIR`)                                                                  |
| `--data-dir DIR`              | `data/`    | Where `--data load` reads `transactions_{train,validation,test}.pkl`                                                                                 |
| `--start-date`, `--end-date`  | –          | Restrict the loaded period (`YYYY-MM-DD`)                                                                                                            |
| `--feature-set 1\|7\|10\|15\|20` | `7`     | `1`: the raw log amount only. `7`, `10`, `15`: selected by the feature analysis, each extends the previous ([list](data_pipeline.md#4-feature-analysis-and-the-feature-sets)). `20`: the log amount and four customer amount features |
| `--window-length`             | `3`        | Number of transactions per customer window                                                                                                           |

The presets scale the shared dataset (5000 customers) down while keeping its statistics: twice as many
terminals as customers, the same number of terminals per customer and customers per terminal, and
compromise rates proportional to the population, so the fraud rate and scenario mix are the same at
every size. `small` (~130 frauds) is for smoke and end-to-end tests: its splits hold only a few
compromised customers, so per-category metrics are noisy (validation `average_precision_customer` is
often `NaN`). `medium` (~1.6k frauds) is the smallest size with meaningful per-category metrics. `large`
with `--data-seed 0` regenerates the shared dataset (about 7 minutes); `--data load` is faster.

### Readout, evaluation and output (`baseline`, `lstm`, `esn`, `qrc`)

`--alpha` is not used by `lstm`.

| Option         | Default        | Meaning                                                                                       |
| -------------- | -------------- | --------------------------------------------------------------------------------------------- |
| `--alpha`      | `1.0`          | Readout regularisation (`C = 1/alpha`)                                                        |
| `--min-recall` | `0.8`          | Minimum validation recall when picking the decision threshold (fewest false positives above it) |
| `--top-k`      | `100`          | Customers reviewed per day for Card Precision@K                                               |
| `--output-dir` | `results/runs` | Where runs are saved                                                                          |

### LSTM (`lstm` only)

| Option            | Default | Meaning                                                    |
| ----------------- | ------- | ---------------------------------------------------------- |
| `--hidden-size`   | `32`    | LSTM hidden units                                          |
| `--epochs`        | `50`    | Maximum training epochs                                    |
| `--batch-size`    | `256`   | Training batch size                                        |
| `--learning-rate` | `1e-3`  | Adam learning rate                                         |
| `--patience`      | `8`     | Epochs without validation-loss improvement before stopping |
| `--seed`          | `42`    | Training seed                                              |

### Echo state network (`esn` only)

Each window starts from a zero state (like the reset-window QRC) and the state after its last
transaction is the readout input.

| Option              | Default | Meaning                                                             |
| ------------------- | ------- | ------------------------------------------------------------------- |
| `--n-units`         | `50`    | Reservoir neurons                                                   |
| `--spectral-radius` | `0.9`   | Largest absolute eigenvalue of the recurrent weights (memory decay) |
| `--leak-rate`       | `1.0`   | State update rate in (0, 1]; smaller is slower, with longer memory  |
| `--input-scaling`   | `1.0`   | Input weights are uniform in `[-s, s]`                              |
| `--bias-scaling`    | `0.2`   | Neuron biases are uniform in `[-s, s]`                              |
| `--density`         | `1.0`   | Fraction of non-zero recurrent weights                              |
| `--seed`            | `42`    | Reservoir seed                                                      |

### Reservoir and backend (`qrc` only)

| Option               | Default            | Meaning                                                                                                                   |
| -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `--backend`          | `estimator-exact`  | `estimator-exact`, `gaussian`, `shots`, `numpy` or `spinpulse` ([below](#4-backends))                                     |
| `--precision`        | `0.05`             | Target standard error per expectation value (`gaussian` / `shots` / `spinpulse`; `0` is exact, `spinpulse` only)          |
| `--simulator`        | `aer`              | Simulator behind `gaussian` and `shots`: `aer` or `numpy` (much faster up to ~8 qubits)                                   |
| `--n-input-qubits`   | number of features | Input qubits. With fewer qubits than features, dense angle encoding is used ([below](#5-encodings))                       |
| `--n-mem-qubits`     | `2`                | Memory qubits                                                                                                             |
| `--hamiltonian`      | `random`           | `random`, `tfim`, `tfim-longitudinal`, `tfim-native` or `native-exchange`                                                 |
| `--topology`         | `linear`           | Hamiltonian connectivity: `linear`, `ring` or `fully-connected`; ignored for `--hamiltonian random`                       |
| `--evolution-time`   | `1.0`              | Hamiltonian evolution time between encodings; ignored for `--hamiltonian random`                                          |
| `--field`            | `1.0`              | Multiplies the Hamiltonian's Z field term, not its couplings; ignored for `--hamiltonian random`                          |
| `--trotter-steps`    | `2`                | Trotter steps per evolution; ignored for `--hamiltonian random`. With one step, a `tfim` reservoir with input reset gives the same features for an input `u` and its mirror `1 − u`, hence the default of 2 |
| `--depth`            | `2`                | Reservoir circuit depth                                                                                                   |
| `--entangler`        | `cx`               | `cx` or `cry`                                                                                                             |
| `--encoding`         | `auto`             | `angle`, `dense` or `reupload`. `auto` picks one from the qubit count and `--encoding-layers` ([below](#5-encodings))     |
| `--encoding-layers`  | `1`                | Times each transaction is encoded (data re-uploading). `1` is plain angle encoding                                        |
| `--encoding-scaling` | `π`                | Rotation angle per unit of (scaled) feature, in each encoding layer                                                       |
| `--reset-inputs`     | off                | Reset the input qubits before each transaction is encoded (density-matrix simulation, slower)                             |
| `--observables`      | `Z`                | E.g. `Z`, `Z+ZZ`, `XX-all`, `XYZ`, `XYZ-strings` (see `build_observables` in `qrc/observables.py`)                                       |
| `--seed`             | `42`               | Reservoir and sampling seed                                                                                               |

The reservoir is either a random circuit (the default) or a Trotterized Hamiltonian model. The
Hamiltonian and topology are recorded in the run configuration and summary.

The SpinPulse backend has its own options (`--hardware`, `--noise`, `--t2`, `--dd`, ...); see
`python -m experiments.cli qrc --help`.

## 4. Backends

- `estimator-exact` (default): exact, noiseless simulation with the Qiskit Aer Estimator. Windows are sent to Aer in batches.
- `numpy`: exact, noiseless batched simulation in plain NumPy. Same results as `estimator-exact`, but much faster
  for small registers (up to ~8 qubits).
- `gaussian`: exact expectation values plus Gaussian noise of width `--precision`. This is a cheap stand-in for shot noise.
  With `--simulator numpy` the exact values come from the `numpy` backend instead of Aer. The noise is drawn the same
  way, so the features are the same up to machine precision, 4–6× faster on 4–5 qubits.
- `shots`: real shot sampling. The observables are grouped into measurement bases (e.g. 3 for `XYZ` and
  `XYZ-strings`), every basis gets `ceil(1/precision²)` shots, and all observables of a basis are estimated from the
  same outcomes. With `--simulator numpy` the outcomes are drawn from the exact distribution of each window's state,
  which is much faster than Aer for small registers. For a total budget of N shots per window, pass
  `--precision` = √(number of bases / N), e.g. `0.0054773` for 10⁵ shots with `XYZ-strings`.
- `spinpulse`: pulse-level noisy simulation of a spin-qubit chip with SpinPulse (qubit-frequency noise, dynamical
  decoupling, pulse shapes). `--precision 0` gives exact expectation values under the noise.

## 5. Encodings

- **Angle encoding** (`--encoding angle`): one feature per input qubit, as a `RY` rotation of `scaling × feature`.
- **Dense angle encoding** (`--encoding dense`): used automatically when `--n-input-qubits` is smaller than the number of
  features (with `--encoding-layers 1`). Each qubit carries up to two features: one as a `RY`
  rotation, one as a `RZ` rotation. So the number of features can be at most twice `--n-input-qubits`.
- **Re-uploading** (`--encoding reupload`, or `--encoding-layers` > 1): the transaction is angle-encoded several times, with
  fixed random `RZ` rotations in between. Each feature then enters with frequencies up to
  `layers × scaling`.

## 6. Run outputs

Each `baseline`, `lstm`, `esn` and `qrc` run creates a directory `results/runs/<timestamp>_<command>/` containing:

- `config.json`: every argument, the data source, the features used and the split sizes
- `metrics.json`: validation and test metrics, plus the reservoir runtime for `esn` and `qrc` or the training time for `lstm`
- `scores.npz`: labels and predicted scores per split, for plotting PR/ROC curves later

Each run also appends one row to **`results/runs/summary.csv`**, so you can compare runs side
by side. A row records:

- the run: its name, the git commit (with `-dirty` if there were uncommitted changes) and the command;
- the data: its source, feature set, window length, and the size and fraud count (per category, when known) of each split;
- for `qrc`, the reservoir: backend, precision, encoding (type, layers and scaling), qubit counts, depth,
  entangler, input reset, observables, seed, Hamiltonian, topology, evolution time and Trotter steps, plus the reservoir runtime;
- for `esn`, the reservoir: units, spectral radius, leak rate, input and bias scaling, density, seed and runtime;
- for `lstm`, the training seed and training time;
- the readout: `alpha` (empty for `lstm`), `--min-recall`, the selected decision threshold, `--top-k`, and every
  [metric](#2-metrics) on validation and test.

## 7. Reproducibility

A simulated run is fully determined by `--size`, `--data-seed` and `--seed`, and all three are
recorded in `config.json`. To share the exact data with someone else, save it once and load it
from then on:

```bash
python -m experiments.cli prepare --size medium --save-data data/sim_medium
python -m experiments.cli qrc --data load --data-dir data/sim_medium
```

Loading saved data gives the same results as simulating it again.

## 8. Sweeps

`experiments/sweep.py` runs a list of `experiments.cli` commands one after the other and prepares each
dataset only once. The jobs file holds one command per line, without the `python -m experiments.cli`
prefix (blank lines and `#` comments are skipped):

```
# jobs.txt
baseline --data load --feature-set 7 --window-length 1
qrc --data load --feature-set 7 --window-length 1 --seed 42
qrc --data load --feature-set 7 --window-length 1 --seed 43
```

```bash
python -m experiments.sweep jobs.txt
```

Jobs with the same data settings (source, feature set, window length) share the prepared data. A failing
job is reported and the sweep moves on. Results go to each job's `--output-dir` and `summary.csv` as usual.

## 9. Other scripts

The CLI is the recommended entry point. These scripts are still useful for exploration. Run them as
modules from the repository root, e.g. `python -m experiments.run_benchmarks`.

| Script                                    | Purpose                                                                        |
| ----------------------------------------- | ------------------------------------------------------------------------------ |
| `experiments/compare_decision_policies.py` | Fixed 0.5 threshold vs. validation-selected thresholds (several minimum recalls) vs. Card Precision@K (several K), on the same model scores |
| `experiments/run_data.py`           | Data → features → analysis plots → model input, from the daily split files     |
| `experiments/dataset_analysis.py`   | Exploration and feature analysis of a simulated dataset                        |
| `experiments/run_benchmarks.py`     | QRC on synthetic tasks with known answers (memory, parity, NARMA10, IPC)       |

For IPC, `--ipc-reservoir-depth` sets the random-circuit depth or Hamiltonian Trotter steps,
depending on the selected reservoir.
