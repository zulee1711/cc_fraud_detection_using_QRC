# Data pipeline and feature catalog

This document describes how raw transactions become model input. It covers data loading,
feature engineering, feature analysis, preprocessing and per-customer windowing, and lists
every engineered feature. For running experiments, see the [main README](../Readme.md).

## Contents

1. [Overview](#1-overview)
2. [Entry points](#2-entry-points)
3. [Feature engineering](#3-feature-engineering)
4. [Feature analysis and the feature sets](#4-feature-analysis-and-the-feature-sets)
5. [DataProcessor](#5-dataprocessor)
6. [Per-customer windowing](#6-per-customer-windowing)
7. [Feature catalog](#7-feature-catalog)
8. [NaN handling](#8-nan-handling)
9. [Identifiers and leakage](#9-identifiers-and-leakage)
10. [Module map](#10-module-map)

---

## 1. Overview

```
simulate (create_dataset.py) or load (dataIO.py)
      ↓
split train / validation / test by date (split_dataset.py)
      ↓
concatenate into full_data, drop TX_FRAUD_SCENARIO
      ↓
feature engineering on full_data, split back by date (FeatureEngineer)
      ↓
feature analysis on train only (FeatureAnalyzer)  ← offline; produces the feature sets
      ↓
select features, impute, scale (DataProcessor)
      ↓
per-customer windows of length L (window_by_customer_id)
      ↓
QRC reservoir / baseline → logistic readout
```

## 2. Entry points

| Script                              | What it runs                                                                                          |
|-------------------------------------|-------------------------------------------------------------------------------------------------------|
| `experiments/cli.py`                | The whole pipeline above, without feature analysis. Uses the fixed feature sets of [section 4](#4-feature-analysis-and-the-feature-sets) |
| `experiments/run_data.py`           | Data → feature engineering → feature analysis and plots → model input, from the daily split files in `data/` |
| `experiments/dataset_analysis.py`   | Exploration and feature analysis of a freshly simulated dataset                                        |
| `experiments/pipeline.py`           | The original single-file end-to-end run (see [below](#pipelinepy))                                     |

Every entry point merges the splits before feature engineering:

```python
full_data = pd.concat([train, validation, test], ignore_index=True)
full_data = full_data.drop(columns=["TX_FRAUD_SCENARIO"])   # avoid leakage
```

### `pipeline.py`

`experiments/pipeline.py` predates the CLI and is configured by editing the file itself:

- The dataset is simulated from the `SIMULATION` dictionary at the bottom of the file
  (10 customers, 100 terminals, 365 days). For a quicker test, reduce it, e.g.
  `n_customers=5, n_terminals=20, nb_days=10`.
- It skips feature analysis and uses a hand-picked feature set, `manual_feature_sets[6]`:
  `TERMINAL_RISK_7D`, `TERMINAL_RISK_CHANGE_7D_30D`, `TERMINAL_RISK_CHANGE_7D_180D`,
  `CUSTOMER_AMOUNT_RATIO_180D`, `CUSTOMER_AMOUNT_RATIO_90D`, `CUSTOMER_AMOUNT_DEVIATION_30D`.

The CLI covers the same ground with `--size` and `--feature-set`, so prefer it for new experiments.

## 3. Feature engineering

**Source:** `qrc/features/feature_engineer.py` (orchestration) and `qrc/features/help_functions.py`
(the feature implementations).

```python
from qrc.features import FeatureEngineer

feature_engineer = FeatureEngineer(windows=(1, 7, 30, 90, 180), save=False)
train_features, validation_features, test_features = feature_engineer.run(
    full_data, train, validation, test,
)
```

> 💡 **Feature engineering runs on `full_data`, not on each split separately.**
> Historical features can need transactions from the previous split. For example, the first
> validation transaction may depend on the customer's last days of training. `run()` computes
> the features on `full_data`, then splits the result back along the original date boundaries
> (`split_features_by_dates`).

The steps run in this order:

```
clean_amount
    ↓
add_time_features
    ↓
customer_spending_behaviour_rolling
    ↓
terminal_count_risk_rolling
    ↓
add_behavior              (use_behavior=True)
    ↓
add_combination_features  (use_combinations=True)
    ↓
add_deviation_features    (use_deviations=True)
```

The default historical windows are `(1, 7, 30, 90, 180)` days. With these windows, the
pipeline produces about 80 columns.

### Rule for historical features

```
CUSTOMER_ID
    ↓
same customer's transactions
    ↓
TX_DATETIME < current transaction
    ↓
historical window
    ↓
engineered customer feature
```

A transaction is never part of its own history. The rolling windows use `closed="left"`.

`CUSTOMER_ID` is only used internally as a grouping key. `FeatureEngineer.run()` has no
`customer_id` argument. To inspect one customer's history by hand:

```python
customer_transactions = full_data[full_data["CUSTOMER_ID"] == customer_id].sort_values("TX_DATETIME")

history_30d = full_data[
    (full_data["CUSTOMER_ID"] == customer_id)
    & (full_data["TX_DATETIME"] < transaction_time)
    & (full_data["TX_DATETIME"] >= transaction_time - pd.Timedelta(days=30))
].sort_values("TX_DATETIME")
```

## 4. Feature analysis and the feature sets

**Source:** `qrc/analysis/feature_analysis.py`, `qrc/analysis/feature_selection.py`, `qrc/analysis/core.py`.

```python
experiments = analyzer.run(train_features)
```

Feature analysis uses **training data only**. To check that the results are stable, it is
repeated over several sample seeds (which rows are sampled, default `99`) and several selection
seeds (randomness inside Information Gain and the Random Forest, default `[42, 99, 712, 2026]`).
It uses these methods:

- Pearson correlation with the target
- Information Gain (mutual information)
- Random Forest importance
- Random Forest ranking stability across selection seeds
- PCA

From these results we chose three nested feature sets. They are defined in `FEATURE_SETS` in
`experiments/cli.py` and selected with `--feature-set`:

| Set | Adds |
|-----|------|
| **7**  | `TERMINAL_RISK_7D`, `TERMINAL_RISK_CHANGE_7D_30D`, `TERMINAL_RISK_CHANGE_7D_90D`, `CUSTOMER_AMOUNT_RATIO_90D`, `CUSTOMER_AMOUNT_DEVIATION_30D`, `TX_AMOUNT_LOG`, `TX_DURING_NIGHT` |
| **10** | set 7 + `TERMINAL_RISK_30D`, `TERMINAL_RISK_CHANGE_1D_90D`, `CUSTOMER_NB_TX_30D` |
| **15** | set 10 + `TERMINAL_RISK_1D`, `CUSTOMER_AMOUNT_SHIFT_7D_30D`, `CUSTOMER_ACTIVITY_AMOUNT_30D`, `CUSTOMER_TIME_GAP_SECONDS`, `TX_DURING_WEEKEND` |

## 5. DataProcessor

**Source:** `qrc/processing.py`

`DataProcessor` turns the selected engineered columns into numeric model arrays. Feature
engineering produces meaningful transaction, customer and terminal variables. `DataProcessor`
only prepares them for the model, so the two stages can be used independently.

```python
from qrc.processing import DataProcessor

processor = DataProcessor(feature_sets=FEATURE_SETS)
result = processor.process(train_features, validation_features, test_features, feature_set=7)
```

The processing order:

```
selected features
      ↓
replace +inf / -inf with NaN
      ↓
fill NaN with the TRAIN medians
      ↓
fit MinMaxScaler on TRAIN
      ↓
transform train / validation / test
```

The returned dictionary contains:

```python
{
    "X_train": ..., "X_validation": ..., "X_test": ...,                  # (n, features)
    "y_train": ..., "y_validation": ..., "y_test": ...,                  # TX_FRAUD
    "groups_train": ..., "groups_validation": ..., "groups_test": ...,   # CUSTOMER_ID, used for windowing
    "features": [...],
}
```

## 6. Per-customer windowing

**Source:** `qrc/sequences.py`

The reservoir needs a time axis, so `window_by_customer_id(result, L)` converts each
`X_{split}` from shape `(n, features)` to `(n, L, features)`. Each window holds the customer's
`L` most recent transactions up to and including the one being classified, ordered oldest first.
A window can reach back into an earlier split, so validation windows can include training
transactions. `y_{split}` is unchanged: row `i` still describes the same transaction. The CLI
sets `L` with `--window-length` (default 3).

---

## 7. Feature catalog

### Original columns

| Variable | Description | Example |
|---|---|---|
| `TRANSACTION_ID` | Unique identifier of the transaction | 1234567 |
| `TX_DATETIME` | Date and time of the transaction | 2025-07-06 09:53:41 |
| `CUSTOMER_ID` | Customer making the transaction | 123 |
| `TERMINAL_ID` | Payment terminal where the transaction took place | 456 |
| `TX_AMOUNT` | Transaction amount (`-1` = missing) | 75.40 |
| `TX_TIME_SECONDS` | Seconds since the start of the dataset | 6432000 |
| `TX_TIME_DAYS` | Days since the start of the dataset | 74 |
| `TX_FRAUD` | **Target**: 0 = legitimate, 1 = fraud | 0 / 1 |
| `TX_FRAUD_SCENARIO` | Simulated fraud scenario that generated the transaction. **Never use in a model** | 1, 2, 3 |

### Transaction amount — `clean_amount`

| Feature | Definition | NaN handling | Meaning |
|---|---|---|---|
| `TX_AMOUNT_MISSING` | `1` if `TX_AMOUNT == -1`, else `0` | Never NaN | Missing-amount indicator |
| `TX_AMOUNT_CLEAN` | `TX_AMOUNT` with `-1` replaced by NaN | `-1 → NaN` | Amount with missing values made explicit |
| `TX_AMOUNT_LOG` | `log1p(max(TX_AMOUNT_CLEAN, 0))` | NaN propagates | Amounts are right-skewed; the log compresses large values |

### Time — `add_time_features`

| Feature | Definition | Meaning |
|---|---|---|
| `TX_HOUR` | Hour of `TX_DATETIME` | Hour of the day |
| `TX_DAY_OF_WEEK` | `0 = Mon, …, 6 = Sun` | Day of the week |
| `TX_MONTH` | Month of `TX_DATETIME` | Month |
| `TX_DURING_WEEKEND` | `1` on Saturday/Sunday, else `0` | Weekend transaction |
| `TX_DURING_NIGHT` | `1` if hour `≤ 6`, else `0` | Transaction between 00:00 and 06:59 |
| `TX_TIME_SIN`, `TX_TIME_COS` | `sin`/`cos(2π · seconds_since_midnight / 86400)` | Position on the 24-hour clock, cyclic |

### Customer rolling — `customer_spending_behaviour_rolling`

For every `w` in `(1, 7, 30, 90, 180)`, grouped by `CUSTOMER_ID` over the previous `w` days,
excluding the current transaction:

| Feature | Definition | NaN handling | Meaning |
|---|---|---|---|
| `CUSTOMER_NB_TX_{w}D` | Number of previous transactions | No history → `0` | How many transactions did this customer make in the last `w` days? |
| `CUSTOMER_AVG_AMOUNT_{w}D` | Mean previous `TX_AMOUNT_CLEAN` | No history → `0` | The customer's typical amount over the last `w` days |

### Terminal rolling — `terminal_count_risk_rolling`

Fraud labels only become known after an investigation, so terminal features use a
**7-day delay**. For every `w`, the window is `[t − (7 + w) days, t − 7 days)`.

| Feature | Definition | NaN handling | Meaning |
|---|---|---|---|
| `TERMINAL_NB_TX_{w}D` | Number of terminal transactions in the delayed window | No history → `0` | Terminal activity |
| `TERMINAL_RISK_{w}D` | `frauds / transactions` in the delayed window | No transactions → `0` | Share of fraudulent transactions at this terminal |

### Customer behaviour — `add_behavior`

| Feature | Definition | NaN handling | Meaning |
|---|---|---|---|
| `CUSTOMER_TIME_GAP_SECONDS` | Seconds since the customer's previous transaction | First transaction → `-1` | A very short gap can indicate a burst of transactions |

`add_behavior` also computes `CUSTOMER_AMOUNT_RATIO_30D`, but `add_combination_features`
overwrites it (see below and [section 8](#8-nan-handling)).

### Combinations — `add_combination_features`

For every pair `short < long` of windows:

| Feature | Definition | NaN handling | Meaning |
|---|---|---|---|
| `CUSTOMER_TX_RATE_{short}D_{long}D` | `CUSTOMER_NB_TX_{short}D / CUSTOMER_NB_TX_{long}D` | `±inf`, `0/0` → NaN | Recent vs. long-term transaction activity |
| `CUSTOMER_AMOUNT_SHIFT_{short}D_{long}D` | `CUSTOMER_AVG_AMOUNT_{short}D / CUSTOMER_AVG_AMOUNT_{long}D` | `±inf`, `0/0` → NaN | Recent vs. long-term spending level |
| `TERMINAL_RISK_CHANGE_{short}D_{long}D` | `TERMINAL_RISK_{short}D − TERMINAL_RISK_{long}D` (a difference) | Never NaN | Has the terminal recently become riskier than usual? |

For every window `w`:

| Feature | Definition | NaN handling | Meaning |
|---|---|---|---|
| `CUSTOMER_AMOUNT_RATIO_{w}D` | `TX_AMOUNT_CLEAN / CUSTOMER_AVG_AMOUNT_{w}D` | `±inf`, `0/0` → NaN | Current amount relative to the customer's average |
| `CUSTOMER_ACTIVITY_AMOUNT_{w}D` | `CUSTOMER_NB_TX_{w}D × TX_AMOUNT_CLEAN` | NaN propagates | Transaction frequency × current amount |

And once:

| Feature | Definition | NaN handling | Meaning |
|---|---|---|---|
| `AMOUNT_PER_TIME_GAP` | `TX_AMOUNT_CLEAN / CUSTOMER_TIME_GAP_SECONDS` | First transaction or zero gap → NaN | Money moved relative to how quickly the customer transacts |

### Deviations — `add_deviation_features`

| Feature | Definition | NaN handling | Meaning |
|---|---|---|---|
| `CUSTOMER_AMOUNT_DEVIATION_30D` | `TX_AMOUNT_CLEAN − CUSTOMER_AVG_AMOUNT_30D` | NaN propagates | How far the amount is from the customer's 30-day normal |
| `CUSTOMER_AMOUNT_MULTIPLIER_30D` | `TX_AMOUNT_CLEAN / CUSTOMER_AVG_AMOUNT_30D` | `±inf` → NaN | Same values as `CUSTOMER_AMOUNT_RATIO_30D` |

---

## 8. NaN handling

NaN values are handled in two stages.

**Stage A — feature engineering.** Some undefined histories are deliberately set to a value:

```
no customer history   → CUSTOMER_NB_TX_*D = 0, CUSTOMER_AVG_AMOUNT_*D = 0
no terminal history   → TERMINAL_NB_TX_*D = 0, TERMINAL_RISK_*D = 0
first transaction     → CUSTOMER_TIME_GAP_SECONDS = -1
```

Other operations deliberately produce NaN, mainly divisions by zero (`amount / 0 → NaN`).
Feature engineering therefore does not remove every NaN.

**Stage B — `DataProcessor`.** Before the data reaches a model, any remaining `±inf` becomes
NaN, and every NaN is filled with the **train** median. The same medians are used for validation
and test, and the scaler is fitted on train only.

> **Note on `CUSTOMER_AMOUNT_RATIO_30D`.** `add_behavior` defines it with `0` when the average
> is `≤ 0` and `-99` for `±inf`. `add_combination_features` runs later and recomputes every
> `CUSTOMER_AMOUNT_RATIO_{w}D`, including 30D, with NaN for these cases. The NaN version is
> the one the model sees. This makes it identical to `CUSTOMER_AMOUNT_MULTIPLIER_30D`.

## 9. Identifiers and leakage

| Column | Purpose | Model feature? |
|---|---|---|
| `TRANSACTION_ID` | Identifies a transaction | No. Excluded from feature selection |
| `CUSTOMER_ID` | Grouping key for customer history and windowing | No. Excluded from PCA |
| `TERMINAL_ID` | Grouping key for terminal history | No. Excluded from PCA |
| `TX_FRAUD_SCENARIO` | How the simulator generated a fraud | **Never.** It leaks the label. Dropped before feature engineering and excluded from selection |

## 10. Module map

| Module | Role |
|---|---|
| `qrc/datasets/create_dataset.py` | Transaction and fraud simulation |
| `qrc/datasets/split_dataset.py` | Date-based train/validation/test split; `split_features_by_dates` |
| `qrc/datasets/dataIO.py` | Loading and saving (`load_splits`, `load_split_files`, `dumping`) |
| `qrc/features/feature_engineer.py` | `FeatureEngineer`: feature engineering orchestration |
| `qrc/features/help_functions.py` | Feature implementations |
| `qrc/analysis/core.py` | Dataset overview, daily stats, correlation helpers |
| `qrc/analysis/feature_analysis.py` | Feature-analysis experiments across seeds, stability report |
| `qrc/analysis/feature_selection.py` | Pearson / Information Gain / Random Forest / PCA |
| `qrc/analysis/plots.py` | Exploration and feature-analysis plots |
| `qrc/processing.py` | `DataProcessor`: model preprocessing |
| `qrc/sequences.py` | `window_by_customer_id`: per-customer windows |
| `experiments/run_data.py` | End-to-end data orchestration with analysis |
