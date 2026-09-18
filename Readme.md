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

### Install dependencies:
```bash
pip install -r requirements.txt
```

## Project structure

```
cc_fraud_detection_using_QRC/
├── qrc/                  # QRC library — the import root for project code
│   ├── encodings.py      # input encodings (AngleEncoding, ...)
│   ├── hamiltonians.py   # reservoir Hamiltonians
│   ├── trotter.py        # Trotterised time evolution
│   ├── reservoirs.py     # reservoir definitions
│   ├── readout.py        # measurement / readout layer
│   ├── protocol.py       # end-to-end QRC protocol
│   └── backends/         # execution backends
│       ├── base.py
│       ├── sampled.py
│       ├── spinpulse.py
│       └── statevector.py
├── examples/             # spin-pulse usage examples
├── notebook/             # scratch notebooks
├── thelab/               # reference implementation (git submodule)
└── raw_data/             # datasets you download (git-ignored)
```

### Importing project code

Run Python from the repository root and import through the `qrc` package:

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