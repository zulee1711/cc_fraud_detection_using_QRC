# Credit-card fraud detection with Quantum Reservoir Computing

This repository explores **Quantum Reservoir Computing (QRC)** for detecting fraudulent
credit-card transactions. Each transaction is turned into engineered features, and the
transactions are grouped into short per-customer windows. A quantum reservoir processes
each window, and a classical logistic readout is trained on the measured observables.

Everything you need to run an experiment is in one command-line tool, `experiments/cli.py`.

---

## Quickstart

Requires **Python 3.11+** (`python --version`).

```bash
# 1. Clone, including the thelab/ submodule
git clone --recurse-submodules https://github.com/zulee1711/cc_fraud_detection_using_QRC.git
cd cc_fraud_detection_using_QRC

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install the project (editable, with all dependencies)
pip install -e .

# 4. Run your first experiment
python -m experiments.cli baseline --size small
python -m experiments.cli qrc      --size small
```

No dataset download is needed for this: by default the data is simulated on the fly.
On a laptop, the `small` preset takes a couple of seconds for `baseline` and under 10 seconds for `qrc`.

> `pip install -e .` installs the `qrc` package in editable mode, so changes you make under
> `qrc/` take effect immediately without reinstalling.

The optional LSTM baseline also needs PyTorch:

```bash
pip install -e ".[classical]"
```

---

## Running experiments

Every experiment is one command:

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
hyperparameter search, [below](#hyperparameter-search-optuna), also selects on validation).

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

Run `python -m experiments.cli <command> --help` to list every option.

### Examples

```bash
# Classical baseline on a mid-sized simulated dataset
python -m experiments.cli baseline --size medium

# LSTM baseline on a small-sized simulated dataset
python -m experiments.cli lstm --size small --epochs 10

# Echo state network (classical reservoir) baseline
python -m experiments.cli esn --size small --n-units 50 --spectral-radius 0.9

# QRC with the exact Qiskit Estimator
python -m experiments.cli qrc --size medium --backend estimator-exact

# The same exact simulation in plain NumPy, much faster for small registers (up to ~8 qubits)
python -m experiments.cli qrc --size medium --backend numpy

# QRC with shot noise (~1/precision² shots per expectation value)
python -m experiments.cli qrc --size medium --backend shots --precision 0.05

# Pulse-level noisy simulation with SpinPulse (exact expectation values, 1/f qubit-frequency noise)
python -m experiments.cli qrc --size small --backend spinpulse --noise pink --t2 50000 --precision 0

# Decision threshold that keeps at least 90% validation recall; Card Precision@50 per day
python -m experiments.cli baseline --size medium --min-recall 0.9 --top-k 50

# A bigger reservoir and more features
python -m experiments.cli qrc --feature-set 10 --n-mem-qubits 3 --depth 3 --observables Z+ZZ

# The shared dataset from SharePoint (see "Data" below)
python -m experiments.cli qrc --data load --backend gaussian --precision 0.05

# Run with specific topology and hamiltonian
python -m experiments.cli qrc --size small --n-input-qubits 4 --n-mem-qubits 3 --hamiltonian native-exchange --topology fully-connected
python -m experiments.cli qrc --size small --n-input-qubits 6 --n-mem-qubits 3 --hamiltonian tfim --topology ring
python -m experiments.cli qrc --size medium --hamiltonian tfim --topology ring
```

### Options

**Data**, available for every command:

| Option                        | Default    | Meaning                                                                                                                                              |
| ----------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--data simulate\|load`       | `simulate` | Simulate a dataset, or load the split `.pkl` files                                                                                                   |
| `--size small\|medium\|large` | `medium`   | Simulation preset: 25 / 250 / 5000 customers over 365 days (~14k / ~170k / ~3.5M transactions), all with the shared dataset's ~0.9% frauds; `large` is the shared dataset |
| `--data-seed`                 | `0`        | Simulation seed                                                                                                                                      |
| `--save-data DIR`             | –          | Also write the simulated splits to `DIR` (reload with `--data load --data-dir DIR`)                                                                  |
| `--data-dir DIR`              | `data/`    | Where `--data load` reads `transactions_{train,validation,test}.pkl`                                                                                 |
| `--start-date`, `--end-date`  | –          | Restrict the loaded period (`YYYY-MM-DD`)                                                                                                            |
| `--feature-set 1\|7\|10\|15`   | `7`        | `1`: the raw log amount only. `7`, `10`, `15`: selected by the feature analysis, each extends the previous ([list](docs/data_pipeline.md#4-feature-analysis-and-the-feature-sets)) |
| `--window-length`             | `3`        | Number of transactions per customer window                                                                                                           |

The presets scale the shared dataset (5000 customers) down while keeping its statistics: twice as many
terminals as customers, the same number of terminals per customer and customers per terminal, and
compromise rates proportional to the population, so the fraud rate and scenario mix are the same at
every size. `small` (~130 frauds) is for smoke and end-to-end tests: its splits hold only a few
compromised customers, so per-category metrics are noisy (validation `average_precision_customer` is
often `NaN`). `medium` (~1.6k frauds) is the smallest size with meaningful per-category metrics. `large`
with `--data-seed 0` regenerates the shared dataset (about 7 minutes); `--data load` is faster.

**Readout, evaluation and output**, for `baseline`, `lstm`, `esn` and `qrc` (`--alpha` is not used by `lstm`):

| Option         | Default        | Meaning                                                                                       |
| -------------- | -------------- | --------------------------------------------------------------------------------------------- |
| `--alpha`      | `1.0`          | Readout regularisation (`C = 1/alpha`)                                                        |
| `--min-recall` | `0.8`          | Minimum validation recall when picking the decision threshold (fewest false positives above it) |
| `--top-k`      | `100`          | Customers reviewed per day for Card Precision@K                                               |
| `--output-dir` | `results/runs` | Where runs are saved                                                                          |

**LSTM**, for `lstm` only:

| Option            | Default | Meaning                                                    |
| ----------------- | ------- | ---------------------------------------------------------- |
| `--hidden-size`   | `32`    | LSTM hidden units                                          |
| `--epochs`        | `50`    | Maximum training epochs                                    |
| `--batch-size`    | `256`   | Training batch size                                        |
| `--learning-rate` | `1e-3`  | Adam learning rate                                         |
| `--patience`      | `8`     | Epochs without validation-loss improvement before stopping |
| `--seed`          | `42`    | Training seed                                              |

**Echo state network**, for `esn` only. Each window starts from a zero state (like the reset-window QRC) and the
state after its last transaction is the readout input:

| Option              | Default | Meaning                                                             |
| ------------------- | ------- | ------------------------------------------------------------------- |
| `--n-units`         | `50`    | Reservoir neurons                                                   |
| `--spectral-radius` | `0.9`   | Largest absolute eigenvalue of the recurrent weights (memory decay) |
| `--leak-rate`       | `1.0`   | State update rate in (0, 1]; smaller is slower, with longer memory  |
| `--input-scaling`   | `1.0`   | Input weights are uniform in `[-s, s]`                              |
| `--bias-scaling`    | `0.2`   | Neuron biases are uniform in `[-s, s]`                              |
| `--density`         | `1.0`   | Fraction of non-zero recurrent weights                              |
| `--seed`            | `42`    | Reservoir seed                                                      |

**Reservoir and backend**, for `qrc` only:

| Option               | Default            | Meaning                                                                                                                   |
| -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `--backend`          | `estimator-exact`  | `estimator-exact`, `gaussian`, `shots`, `numpy` or `spinpulse` (see below)                                                |
| `--precision`        | `0.05`             | Target standard error per expectation value (`gaussian` / `shots` / `spinpulse`; `0` is exact, `spinpulse` only)          |
| `--simulator`        | `aer`              | What computes the exact values `gaussian` adds noise to: `aer` or `numpy` (same features, much faster)                    |
| `--n-input-qubits`   | number of features | Input qubits. With fewer qubits than features, dense angle encoding is used (see below)                                   |
| `--n-mem-qubits`     | `2`                | Memory qubits                                                                                                             |
| `--hamiltonian`      | `random`           | `random`, `tfim`, `tfim-longitudinal`, `tfim-native` or `native-exchange`                                                 |
| `--topology`         | `linear`           | Hamiltonian connectivity: `linear`, `ring` or `fully-connected`; ignored for `--hamiltonian random`                       |
| `--evolution-time`   | `1.0`              | Hamiltonian evolution time between encodings; ignored for `--hamiltonian random`                                          |
| `--trotter-steps`    | `2`                | Trotter steps per evolution; ignored for `--hamiltonian random`. One step makes reset `tfim` blind to u → 1−u             |
| `--depth`            | `2`                | Reservoir circuit depth                                                                                                   |
| `--entangler`        | `cx`               | `cx` or `cry`                                                                                                             |
| `--encoding`         | `auto`             | `angle`, `dense` or `reupload`. `auto` picks one from the qubit count and `--encoding-layers` (see below)                 |
| `--encoding-layers`  | `1`                | Times each transaction is encoded (data re-uploading). `1` is plain angle encoding                                        |
| `--encoding-scaling` | `π`                | Rotation angle per unit of (scaled) feature, in each encoding layer                                                       |
| `--reset-inputs`     | off                | Reset the input qubits before each transaction is encoded (density-matrix simulation, slower)                             |
| `--observables`      | `Z`                | E.g. `Z`, `Z+ZZ`, `XX-all`, `XYZ` (see `build_observables` in `qrc/observables.py`)                                       |
| `--seed`             | `42`               | Reservoir and sampling seed                                                                                               |

The backends:

- `estimator-exact` (default): exact, noiseless simulation with the Qiskit Aer Estimator. Windows are sent to Aer in batches.
- `gaussian`: exact expectation values plus Gaussian noise of width `--precision`. This is a cheap stand-in for shot noise.
  With `--simulator numpy` the exact values come from the `numpy` backend instead of Aer. The noise is drawn the same
  way, so the features are the same up to machine precision, 4–6× faster on 4–5 qubits.
- `shots`: real shot sampling with about `1/precision²` shots.
- `numpy`: exact, noiseless batched simulation in plain NumPy. Same results as `estimator-exact`, but much faster
  for small registers (up to ~8 qubits).
- `spinpulse`: pulse-level noisy simulation with SpinPulse. 

The reservoir can be a random circuit (the default) or a Trotterized Hamiltonian model.
The Hamiltonian choice and topology are included in the run configuration and summary.

The encodings:

- **Angle encoding** (`--encoding angle`): one feature per input qubit, as a `RY` rotation of `scaling × feature`.
- **Dense angle encoding** (`--encoding dense`): used automatically when `--n-input-qubits` is smaller than the number of
  features (with `--encoding-layers 1`). Each qubit carries up to two features: one as a `RY`
  rotation, one as a `RZ` rotation. So the number of features can be at most twice `--n-input-qubits`.
- **Re-uploading** (`--encoding reupload`, or `--encoding-layers` > 1): the transaction is angle-encoded several times, with
  fixed random `RZ` rotations in between. Each feature then enters with frequencies up to
  `layers × scaling`.

### Results

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
  [metric](#running-experiments) on validation and test.

### Reproducibility

A simulated run is fully determined by `--size`, `--data-seed` and `--seed`, and all three are
recorded in `config.json`. To share the exact data with someone else, save it once and load it
from then on:

```bash
python -m experiments.cli prepare --size medium --save-data data/sim_medium
python -m experiments.cli qrc --data load --data-dir data/sim_medium
```

Loading saved data gives the same results as simulating it again.

### Running many configurations (sweeps)

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

### Hyperparameter search (Optuna)

`experiments/tune.py` searches the reservoir configuration with [Optuna](https://optuna.org)
(multi-objective TPE sampler). Each trial is a full `qrc` run. It **maximises** the
**validation** score (average precision by default; `--metric` takes any reported metric, e.g.
`average_precision_customer` or `card_precision_top_k`) and **minimises** the costs below at the
same time. The result is a **Pareto front**: the trials that no other trial beats on every
objective at once. The test split is never used for selection.

| Cost (`--objectives`) | What it measures                                                                                                                  |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `qubits`              | Input + memory qubits                                                                                                             |
| `depth`               | Depth of the circuit for one window (L × encoding + reservoir), transpiled to `rz`/`sx`/`x`/`cz` on a linear chain of qubits     |
| `bases`               | Measurement bases: observables that can be read from the same shots count once (`Z` and `Z+ZZ`: 1, `XYZ`: 3)                     |
| `shots`               | Shots per basis, ≈ 1/precision². Searched over ≈10³ … 10⁵. Needs `--backend gaussian`, `shots` or `spinpulse`               |
| `wall_time`           | Seconds to simulate the reservoir over all splits (machine-dependent)                                                            |

By default all costs are objectives, except `shots` with an exact backend. Pass a subset with
`--objectives qubits depth`, or `--objectives` alone for a single-objective search.

```bash
# Simplest: accuracy against circuit depth, on the small dataset
python -m experiments.tune --study-name depth --objectives depth --size small --n-trials 30

# The full multi-objective search, with sampled expectation values
python -m experiments.tune --n-trials 100 --size medium --backend gaussian

# Accuracy against hardware size only, averaged over three reservoir seeds
python -m experiments.tune --study-name size --objectives qubits depth --seeds 1 2 3 --n-trials 50
```

The search space (`suggest_params` in `experiments/tune.py`; ranges are constants at the top of the file).
The default is the value `experiments.cli qrc` uses when the option is not given, i.e. without tuning:

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
| Trotter steps                   | 2 … 3; Hamiltonians only (1 step makes reset `tfim` blind to u → 1−u)          | 2                                      |
| observables                     | `Z`, `Z+ZZ`, `XYZ`, `XYZ+XYZ-2body`                                           | `Z`                                    |
| readout `alpha`                 | 1e-1 … 1e5 (log)                                                              | 1.0                                    |
| precision                       | 0.03, 0.01, 0.003 (≈10³, 10⁴, 10⁵ shots); only with the `shots` objective     | 0.05 (≈400 shots)                      |

Parts of each reservoir are random but fixed: the gate angles of the random circuit and the
couplings of the Hamiltonians. The seed picks that random draw. `--seeds 1 2 3` runs every trial
with three draws and scores it by their average, so a configuration can't win through one lucky
draw (each trial takes 3× longer). Without `--seeds`, every trial uses `--seed` (default `42`).

With the `shots` objective, the precision is searched too.

Tuner options: `--n-trials`, `--timeout`, `--study-name`, `--output-dir` (default `results/tune`),
`--storage`, `--metric`, `--objectives`, `--seeds`, `--max-qubits`, `--encodings`, `--hamiltonians`,
`--topologies`, `--sampler-seed` and `--n-jobs`. Every other option goes to `qrc` unchanged and is fixed for the
whole study (data, feature set, window length, backend, ...).

`--n-jobs N` runs N trials at once, in threads that share the prepared data (so memory grows little).
It speeds up the `numpy` backend and `gaussian` with `--simulator numpy` (about 3.6× with 6 threads on a
6-core laptop); the Aer backends (`estimator-exact`, `gaussian`, `shots`) gain nothing from it. Use at most one thread per physical core. Passing an option the search sets
(e.g. `--n-mem-qubits`) is an error.

A study is saved in `results/tune/<study-name>/`:

- `study.db`: the Optuna study. Running the same command again **resumes** it. Resuming with
  different fixed options is refused, so that all trials in a study stay comparable.
- `runs/`: every trial as an ordinary `qrc` run, with its own `summary.csv`
- `trials.csv`: one row per trial, with its parameters, objective values, costs (also two-qubit
  gate count and depth, and number of observables) and validation and test metrics
- `pareto.json`: the Pareto front, each trial with the `experiments.cli qrc` command that reproduces it

At the end, the tuner prints the Pareto front as a table.
To browse a study interactively: `pip install optuna-dashboard`, then
`optuna-dashboard sqlite:///results/tune/qrc/study.db`.

---

## Data

The team's shared dataset is on [SharePoint](https://tud365.sharepoint.com/:f:/r/sites/stud-JIPQuobly/Gedeelde%20documenten/Datasets/v1_simulated_dataset?d=w85e103b01b5d4ec189c07807a959dabf&csf=1&web=1&e=d6ftcI).

To use it with `--data load`, download the three split files into `data/`:

```
data/
├── transactions_train.pkl
├── transactions_validation.pkl
└── transactions_test.pkl
```

These files are produced by `qrc/datasets/split_dataset.py`. That script also writes daily files
under `data/train/`, `data/validation/` and `data/test/`, which `experiments/run_data.py` reads.
`data/` and `raw_data/` (unsplit simulator output) are both git-ignored.

For how transactions become model input (feature engineering, the full feature catalog,
NaN handling, preprocessing and windowing), see **[docs/data_pipeline.md](docs/data_pipeline.md)**.

---

## Other scripts

The CLI is the recommended entry point. These scripts are still useful for exploration:

| Script                                    | Purpose                                                                        |
| ----------------------------------------- | ------------------------------------------------------------------------------ |
| `experiments/compare_decision_policies.py` | Fixed 0.5 threshold vs. validation-selected thresholds (several minimum recalls) vs. Card Precision@K (several K), on the same model scores |
| `experiments/run_data.py`           | Data → features → analysis plots → model input, from the daily split files     |
| `experiments/dataset_analysis.py`   | Exploration and feature analysis of a simulated dataset                        |
| `experiments/run_benchmarks.py`     | QRC on synthetic tasks with known answers (memory, parity, NARMA10, IPC)       |

Run them as modules from the repository root, e.g. `python -m experiments.run_benchmarks`.
For IPC, `--ipc-reservoir-depth` sets the random-circuit depth or Hamiltonian
Trotter steps, depending on the selected reservoir.

## Using the package in your own code

After `pip install -e .`, `qrc` can be imported from anywhere:

```python
from qrc.encodings import AngleEncoding
from qrc.reservoirs import RandomCircuitReservoir
from qrc.observables import build_observables
from qrc.backends.estimator import EstimatorBackend
from qrc.protocol import QRCProtocol

encoder = AngleEncoding(num_qubits=4)
reservoir = RandomCircuitReservoir(num_input_qubits=4, num_mem_qubits=2, depth=2, rng=42)
observables = build_observables("Z+ZZ", reservoir.num_qubits)
protocol = QRCProtocol(encoder, reservoir, observables, EstimatorBackend())

features = protocol.run(X)   # X: (n_windows, window_length, n_features)
```

`qrc_features` in `experiments/cli.py` shows the complete wiring.

## Tests

```bash
pytest
```

CI runs the same suite on every push (`.github/workflows/tests.yml`).

---

## Project structure

```
cc_fraud_detection_using_QRC/
├── qrc/                     # the project package
│   ├── datasets/            # transaction simulation, train/validation/test split, file I/O
│   ├── features/            # fraud feature engineering
│   ├── analysis/            # exploration, feature scoring and selection, plots
│   ├── benchmarks/          # synthetic tasks: memory, parity, NARMA10
│   ├── backends/            # Qiskit Aer Estimator, NumPy (exact, small registers), SpinPulse (pulse-level noise)
│   ├── baselines/           # classical baselines: LSTM (model and training), echo state network
│   ├── processing.py        # final model input: feature selection, imputation, scaling
│   ├── sequences.py         # 2D features → (n, window_length, features) per-customer windows
│   ├── encodings.py         # input encodings (AngleEncoding, ...)
│   ├── hamiltonians.py      # reservoir Hamiltonians
│   ├── trotter.py           # Trotterised time evolution
│   ├── reservoirs.py        # reservoir circuits
│   ├── observables.py       # measured observables
│   ├── protocol.py          # encode → evolve → measure loop
│   ├── readout.py           # classical readout layers
│   ├── decision.py          # decision threshold selection, Card Precision@K
│   ├── evaluation.py        # fraud categories and per-category average precision
│   └── logger.py
├── experiments/             # experiment scripts (not packaged)
│   ├── cli.py               # ← the experiment runner
│   ├── sweep.py             # many cli.py commands, data prepared once
│   └── tune.py              # Optuna hyperparameter search over cli.py's qrc
├── docs/                    # data pipeline, feature catalog, experiment plan (experiments.md)
├── tests/                   # pytest suite
├── thelab/                  # reference QRC implementation (git submodule)
├── data/                    # train/validation/test splits (git-ignored)
├── raw_data/                # unsplit simulator output (git-ignored)
└── results/                 # run outputs and figures (git-ignored)
```
