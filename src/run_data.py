#%%
import sys

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import *
from data_processing import *
from feature_engineering import FeatureEngineer
from feature_analysis import FeatureAnalyzer
from data_processing import DataProcessor

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

#%% Data Processing
processor = DataProcessor(
    feature_sets=feature_sets,
)

qrc_5 = processor.process(
    train_features,
    validation_features,
    test_features,
    feature_set=5,
)