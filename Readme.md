## Clone repository
```bash
git clone --recurse-submodules https://github.com/zulee1711/cc_fraud_detection_using_QRC.git
```

**Note:** `thelab/` is a git submodule (a reference QRC implementation the company coaches suggested us).
`--recurse-submodules` populates it on clone. If you have already cloned without it, or
if `thelab/` looks empty, run:
```bash
git submodule update --init
```

## Download datasets
Datasets are available at:
[Credit Risk Datasets](https://tud365.sharepoint.com/:f:/r/sites/stud-JIPQuobly/Gedeelde%20documenten/Datasets?d=w89895bce21bd4788ad9c63fe4314ead9&csf=1&web=1&e=KTtJkA).

Download the datasets and place them in the `raw_data` folder of the cloned repository.

## Setup
### Check Python version:
`python --version`

Check if the version is 3.11 or higher. If not, install the latest version of Python from the official website: https://www.python.org/downloads/

### Create a virtual environment and activate it:
On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install the project:
```bash
pip install -e .
```
This installs the `qrc` package in editable mode (your edits to `qrc/` take
effect immediately, no reinstall needed) along with its dependencies.
```

## Project structure

```
cc_fraud_detection_using_QRC/
├── qrc/                    # the project package — the import root
│   │                       #   datasets → features → analysis → processing →
│   │                       #   quantum core → readout  (imports flow one way;
│   │                       #   the quantum core imports none of the layers above)
│   ├── datasets/           # transaction simulation, splitting, file I/O
│   │   ├── create_dataset.py
│   │   ├── split_dataset.py
│   │   └── dataIO.py
│   ├── features/           # fraud feature engineering
│   │   ├── feature_engineer.py
│   │   └── help_functions.py
│   ├── analysis/           # exploration, feature scoring and selection
│   │   ├── core.py
│   │   ├── plots.py
│   │   ├── feature_analysis.py
│   │   └── feature_selection.py
│   ├── processing.py       # final model input: selection, imputation, scaling
│   ├── sequences.py        # (to come) 2D features → (M, L, N) windows
│   ├── encodings.py        # input encodings (AngleEncoding, ...)
│   ├── hamiltonians.py     # reservoir Hamiltonians
│   ├── trotter.py          # Trotterised time evolution
│   ├── reservoirs.py       # reservoir definitions
│   ├── observables.py      # measured observables
│   ├── protocol.py         # encode → evolve → measure loop
│   ├── readout.py          # classical readout layer
│   ├── logger.py
│   └── backends/           # execution backends
│       ├── base.py
│       ├── sampled.py
│       ├── spinpulse.py
│       └── statevector.py
├── experiments/            # experiment wiring — scripts, not a package
│   ├── run_data.py         # data → features → analysis → model input
│   ├── pipeline.py         # one end-to-end QRC run
│   └── cli.py              # (to come) launcher
├── examples/               # spin-pulse usage examples
├── notebook/               # scratch notebooks
├── thelab/                 # reference implementation (git submodule)
├── raw_data/               # source datasets (git-ignored)
└── data/                   # daily train/validation/test splits (git-ignored)
    ├── train/
    ├── validation/
    └── test/
```

### Where the dataset goes

`qrc/datasets/` is code. The dataset *files* live in `data/`, which is git-ignored.

`data/` should contain the daily `.pkl` splits under `train/`, `validation/` and `test/`, produced by `split_dataset.py` and read by `experiments/run_data.py`.
They can be found on [the Sharepoint](https://tud365.sharepoint.com/sites/stud-JIPQuobly/Gedeelde%20documenten/Forms/AllItems.aspx?d=w85e103b01b5d4ec189c07807a959dabf&csf=1&web=1&e=Gnh3Ry&CID=b41e9a08%2D4ee4%2D4a04%2D8b29%2Dd93caea790eb&FolderCTID=0x0120005EE06ABC7353574FA007EB3176FE5935&id=%2Fsites%2Fstud%2DJIPQuobly%2FGedeelde%20documenten%2FDatasets%2Fsimulated%5Fdataset).

### Importing project code

After `pip install -e .` the `qrc` package is importable from anywhere — you no
longer need to run Python from the repository root or set `PYTHONPATH`:

```python
from qrc.encodings import AngleEncoding
from qrc.hamiltonians import build_disordered_tfim
```

## Run examples:
You can run the Jupyter Notebook or the Python script to see the basic usage of the application.

To run the Python script, use the following commands:
On Windows:
```bash
cd examples
python BasicUsage.py
```

On macOS/Linux:
```bash
cd examples
python3 BasicUsage.py
```