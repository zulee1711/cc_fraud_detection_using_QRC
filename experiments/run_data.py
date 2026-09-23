#%%
from pathlib import Path

import pandas as pd

import qrc
from qrc.logger import get_logger
from qrc.datasets import load_splits, split_dataset
from qrc.datasets.create_dataset import generate_dataset, add_frauds
from qrc.analysis import overview, get_daily_stats, FeatureAnalyzer
from qrc.analysis.plots import (
    plot_amount_time_distributions,
    plot_fraud_and_transactions_stats,
    plot_fraud_rate_over_time,
)
from qrc.features import FeatureEngineer
from qrc.processing import DataProcessor

PROJECT_ROOT = Path(qrc.__file__).resolve().parents[1]

logger = get_logger(__name__)

DATA = PROJECT_ROOT / "data"
RESULT = PROJECT_ROOT / "results"
PNG = RESULT / "figures"
PNG.mkdir(parents=True, exist_ok=True)

PLOTS = True
SAVES = True

#%% For debugging
DEBUG = False
if DEBUG:
    (PLOTS, SAVES) = (False, False)

#%%
WINDOWS = (1, 7, 30, 90, 180)

#%%
EXCLUDED_FEATURES = [
    "TRANSACTION_ID",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",  # removed to avoid data leakage
]

#%%
NUMERIC_COLUMNS = [
    'TRANSACTION_ID',
    'TX_AMOUNT',
    'TX_TIME_SECONDS',
    'TX_TIME_DAYS',
    'TX_FRAUD',
]

#%%
GENERATE = True

SIMULATION = dict(
    n_customers=100,
    n_terminals=1000,
    nb_days=365,  # must cover the largest feature window
    start_date="2025-01-01",
    r=5,
    default_random_state=0,
)

#%%
if GENERATE:
    customer_profiles, terminal_profiles, transactions = generate_dataset(**SIMULATION)
    transactions = add_frauds(customer_profiles, terminal_profiles, transactions)
    train, validation, test = split_dataset(transactions, train_ratio=0.70, validation_ratio=0.15)
else:
    splits = load_splits(DATA)
    train, validation, test = (splits['train'], splits['validation'], splits['test'])

full_data = pd.concat([train, validation, test], ignore_index=True)
full_data.drop(columns=['TX_FRAUD_SCENARIO'], inplace=True)

#%% Data Exploration
for df in [(train, "train"), (validation, "validation"), (test, "test")]:
    overview(df[0], name=f"{df[1]}", output_path=RESULT / f"{df[1]}_overview.txt")

#%%
daily_stats = get_daily_stats(full_data)

if PLOTS:
    plot_amount_time_distributions(
        full_data,
        output_path=PNG / "amount_time_distributions.png",
    )

    plot_fraud_and_transactions_stats(
        daily_stats,
        output_path=PNG / "fraud_and_transactions_stats.png",
    )

    plot_fraud_rate_over_time(
        daily_stats,
        output_path=PNG / "fraud_rate_over_time.png",
    )

#%% Feature Engineering
feature_engineer = FeatureEngineer(
    windows=(1, 7, 30, 90, 180),
    output_dir=RESULT,
    save=SAVES,
)

train_features, validation_features, test_features = (
    feature_engineer.run(
        full_data,
        train,
        validation,
        test,
    )
)

#%%
for df, name in [(train_features, "train"), (validation_features, "validation"), (test_features, "test")]:
    overview(
        df,
        name=f"{name}_features",
        output_path=RESULT / f"{name}_features_overview.txt",
    )

#%% Feature Analysis
analyzer = FeatureAnalyzer(
    result_dir=RESULT,
    plot_dir=PNG,
    save=SAVES,
    plots=PLOTS,
)

experiments = analyzer.run(
    train_features
)

#%% Feature Selection
feature_sets = {
    5: experiments[99].candidate_feature_sets[5],
    10: experiments[99].candidate_feature_sets[10],
    15: experiments[99].candidate_feature_sets[15],
}

manual_feature_sets = {
    6: [
        'TERMINAL_RISK_7D',
        'TERMINAL_RISK_CHANGE_7D_30D',
        'TERMINAL_RISK_CHANGE_7D_180D',
        'CUSTOMER_AMOUNT_RATIO_180D',
        # 'TERMINAL_RISK_CHANGE_7D_90D',
        # 'TERMINAL_RISK_30D',
        'CUSTOMER_AMOUNT_RATIO_90D',
        'CUSTOMER_AMOUNT_DEVIATION_30D',
        # 'CUSTOMER_AMOUNT_MULTIPLIER_30D',
        # 'CUSTOMER_AMOUNT_RATIO_30D',
    ],
}

#%% Data Processing
processor = DataProcessor(
    feature_sets=manual_feature_sets,  # feature_sets,
)

input_data = processor.process(
    train_features,
    validation_features,
    test_features,
    feature_set=6,
)

#%%
X_train = input_data['X_train']
X_validation = input_data['X_validation']
X_test = input_data['X_test']
y_train = input_data['y_train']
y_validation = input_data['y_validation']
y_test = input_data['y_test']
input_features = input_data['features']