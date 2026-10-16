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
On a laptop, the `small` preset takes a few seconds for `baseline` and about 10–15 seconds for `qrc`.

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
| `qrc`      | Runs the quantum reservoir on each window, then fits the same logistic readout on its outputs.           |

All commands run the same data preparation:

```
data (simulated or loaded) → feature engineering → DataProcessor → per-customer windows
```

Models are fitted on **train only**. Average precision (AP), ROC-AUC, F1, precision, recall, Matthews correlation
coefficient (MCC) and balanced accuracy are reported on **validation and test**. Nothing is tuned on
validation, except that the `lstm` command uses validation loss for early stopping.

Run `python -m experiments.cli <command> --help` to list every option.

### Examples

```bash
# Classical baseline on a mid-sized simulated dataset
python -m experiments.cli baseline --size medium

# LSTM baseline on a small-sized simulated dataset
python -m experiments.cli lstm --size small --epochs 10

# QRC with the exact Qiskit Estimator
python -m experiments.cli qrc --size medium --backend estimator-exact

# QRC with shot noise (~1/precision² shots per expectation value)
python -m experiments.cli qrc --size medium --backend shots --precision 0.05

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
| `--size small\|medium\|large` | `medium`   | Simulation preset: 10 / 50 / 100 customers over 365 days                                                                                             |
| `--data-seed`                 | `0`        | Simulation seed                                                                                                                                      |
| `--save-data DIR`             | –          | Also write the simulated splits to `DIR` (reload with `--data load --data-dir DIR`)                                                                  |
| `--data-dir DIR`              | `data/`    | Where `--data load` reads `transactions_{train,validation,test}.pkl`                                                                                 |
| `--start-date`, `--end-date`  | –          | Restrict the loaded period (`YYYY-MM-DD`)                                                                                                            |
| `--feature-set 1\|7\|10\|15`   | `7`        | `1`: the raw log amount only. `7`, `10`, `15`: selected by the feature analysis, each extends the previous ([list](docs/data_pipeline.md#4-feature-analysis-and-the-feature-sets)) |
| `--window-length`             | `3`        | Number of transactions per customer window                                                                                                           |

**Readout and output**, for `baseline` and `qrc` (`--output-dir` also applies to `lstm`):

| Option         | Default        | Meaning                                |
| -------------- | -------------- | -------------------------------------- |
| `--alpha`      | `1.0`          | Readout regularisation (`C = 1/alpha`) |
| `--output-dir` | `results/runs` | Where runs are saved                   |

**LSTM**, for `lstm` only:

| Option            | Default | Meaning                                                    |
| ----------------- | ------- | ---------------------------------------------------------- |
| `--hidden-size`   | `32`    | LSTM hidden units                                          |
| `--epochs`        | `50`    | Maximum training epochs                                    |
| `--batch-size`    | `256`   | Training batch size                                        |
| `--learning-rate` | `1e-3`  | Adam learning rate                                         |
| `--patience`      | `8`     | Epochs without validation-loss improvement before stopping |
| `--seed`          | `42`    | Training seed                                              |

**Reservoir and backend**, for `qrc` only:

| Option               | Default            | Meaning                                                                                                                   |
| -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `--backend`          | `estimator-exact`  | `estimator-exact`, `gaussian` or `shots` (see below)                                                                      |
| `--precision`        | `0.05`             | Target standard error per expectation value (`gaussian` / `shots`)                                                        |
| `--n-input-qubits`   | number of features | Input qubits. With fewer qubits than features, dense angle encoding is used (see below)                                   |
| `--n-mem-qubits`     | `2`                | Memory qubits                                                                                                             |
| `--hamiltonian`      | `random`           | `random`, `tfim`, `tfim-longitudinal`, `tfim-native` or `native-exchange`                                                 |
| `--topology`         | `linear`           | Hamiltonian connectivity: `linear`, `ring` or `fully-connected`; ignored for `--hamiltonian random`                       |
| `--depth`            | `2`                | Reservoir circuit depth                                                                                                   |
| `--entangler`        | `cx`               | `cx` or `cry`                                                                                                             |
| `--encoding-layers`  | `1`                | Times each transaction is encoded (data re-uploading). `1` is plain angle encoding                                        |
| `--encoding-scaling` | `π`                | Rotation angle per unit of (scaled) feature, in each encoding layer                                                       |
| `--reset-inputs`     | off                | Reset the input qubits before each transaction is encoded (density-matrix simulation, slower)                             |
| `--observables`      | `Z`                | E.g. `Z`, `Z+ZZ`, `XX-all`, `XYZ` (see `build_observables` in `qrc/observables.py`)                                       |
| `--seed`             | `42`               | Reservoir and sampling seed                                                                                               |

The backends:

- `estimator-exact` (default): exact, noiseless simulation with the Qiskit Aer Estimator. Windows are sent to Aer in batches.
- `gaussian`: exact expectation values plus Gaussian noise of width `--precision`. This is a cheap stand-in for shot noise.
- `shots`: real shot sampling with about `1/precision²` shots.

The reservoir can be a random circuit (the default) or a Trotterized Hamiltonian model.
The Hamiltonian choice and topology are included in the run configuration and summary.

The encodings:

- **Angle encoding** (default): one feature per input qubit, as a `RY` rotation of `scaling × feature`.
- **Dense angle encoding**: used automatically when `--n-input-qubits` is smaller than the number of
  features (with `--encoding-layers 1`). Each qubit carries up to two features: one as a `RY`
  rotation, one as a `RZ` rotation. So the number of features can be at most twice `--n-input-qubits`.
- **Re-uploading** (`--encoding-layers` > 1): the transaction is angle-encoded several times, with
  fixed random `RZ` rotations in between. Each feature then enters with frequencies up to
  `layers × scaling`.

### Results

Each `baseline`, `lstm` and `qrc` run creates a directory `results/runs/<timestamp>_<command>/` containing:

- `config.json`: every argument, the data source, the features used and the split sizes
- `metrics.json`: validation and test metrics, plus the reservoir runtime for `qrc` or the training time for `lstm`
- `scores.npz`: labels and predicted scores per split, for plotting PR/ROC curves later

Each run also appends one row to **`results/runs/summary.csv`**, so you can compare runs side
by side. A row records:

- the run: its name, the git commit (with `-dirty` if there were uncommitted changes) and the command;
- the data: its source, feature set, window length, and the size and fraud count of each split;
- for `qrc`, the reservoir: backend, precision, encoding layers and scaling, qubit counts, depth,
  entangler, input reset, observables and seed, plus the reservoir runtime;
- for `lstm`, the training seed and training time;
- the readout: `alpha` (empty for `lstm`), the decision threshold, and every metric (average precision, ROC-AUC,
  F1, precision, recall, MCC, balanced accuracy) on validation and test.

`results/` is git-ignored.

### Reproducibility

A simulated run is fully determined by `--size`, `--data-seed` and `--seed`, and all three are
recorded in `config.json`. To share the exact data with someone else, save it once and load it
from then on:

```bash
python -m experiments.cli prepare --size medium --save-data data/sim_medium
python -m experiments.cli qrc --data load --data-dir data/sim_medium
```

Loading saved data gives the same results as simulating it again.

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

| Script                              | Purpose                                                                        |
| ----------------------------------- | ------------------------------------------------------------------------------ |
| `experiments/run_data.py`           | Data → features → analysis plots → model input, from the daily split files     |
| `experiments/dataset_analysis.py`   | Exploration and feature analysis of a simulated dataset                        |
| `experiments/pipeline.py`           | The original single-file end-to-end QRC run (configuration is set in the file) |
| `experiments/pipeline_estimator.py` | Same as above, comparing the Estimator backends (`--precision`)                |
| `experiments/run_benchmarks.py`     | QRC on synthetic tasks with known answers (memory, parity, NARMA10)            |

Run them as modules from the repository root, e.g. `python -m experiments.run_benchmarks`.

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
│   ├── backends/            # Qiskit Aer Estimator, spin-pulse
│   ├── baselines/           # classical baseline models (LSTM)
│   ├── processing.py        # final model input: feature selection, imputation, scaling
│   ├── sequences.py         # 2D features → (n, window_length, features) per-customer windows
│   ├── encodings.py         # input encodings (AngleEncoding, ...)
│   ├── hamiltonians.py      # reservoir Hamiltonians
│   ├── trotter.py           # Trotterised time evolution
│   ├── reservoirs.py        # reservoir circuits
│   ├── observables.py       # measured observables
│   ├── protocol.py          # encode → evolve → measure loop
│   ├── readout.py           # classical readout layers
│   └── logger.py
├── experiments/             # experiment scripts (not packaged)
│   └── cli.py               # ← the experiment runner
├── docs/                    # data pipeline and feature catalog
├── tests/                   # pytest suite
├── thelab/                  # reference QRC implementation (git submodule)
├── data/                    # train/validation/test splits (git-ignored)
├── raw_data/                # unsplit simulator output (git-ignored)
└── results/                 # run outputs and figures (git-ignored)
```
