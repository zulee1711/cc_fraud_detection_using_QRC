# Arm A: Amount Features

Arm A compares classical random feature maps with a quantum extreme learning
machine (QELM) using the same transaction-level amount features. With a window
length of 1, the experiment tests whether the QELM is competitive when only the
current transaction is available.

## Shared setup

- **Data:** `data/sc3_c1000_x6`
- **Feature set:** 20, with
  `CUSTOMER_AMOUNT_DEVIATION_30D`, `CUSTOMER_AMOUNT_RATIO_90D`,
  `CUSTOMER_AMOUNT_SHIFT_7D_30D`, `CUSTOMER_ACTIVITY_AMOUNT_30D`, and
  `TX_AMOUNT_LOG`
- **Window length:** 1
- **Metric:** average precision overall and by fraud category

## Experiment folders

| Folder | Experiment |
|---|---|
| `tanh/` | Classical tanh random feature map |
| `rff/` | Classical random Fourier feature map |
| `qelm/` | QELM with 5 input qubits, no memory qubits, depth 4, exact backend |
| `noise/` | QELM with the SpinPulse backend and pink noise; the active jobs vary `t2` |
| `plots/` | Validation and test comparison figures |

The classical map width is 5, matching the QELM's five Z observables. Each
method folder contains its run outputs and summary; the root `summary.csv`
collects the combined results. Run directories contain per-run configuration
and metrics, while score arrays are regenerated locally and are not tracked.

## Run and plot

Run the jobs from the repository root:

```bash
python -m experiments.sweep results/arma/jobs.txt
```

Generate the validation and test comparison plots with:

```bash
python results/arma/plot.py
```

Figures are written to `results/arma/plots/`.
