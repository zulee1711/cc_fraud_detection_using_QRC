#%%
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from io import StringIO

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

#%%
WINDOWS = (1, 7, 30, 90, 180)

#%%
NUMERIC_COLUMNS = [
    'TRANSACTION_ID',
    'TX_AMOUNT',
    'TX_TIME_SECONDS',
    'TX_TIME_DAYS',
    'TX_FRAUD',
    'TX_FRAUD_SCENARIO',
]

EXCLUDED_FEATURES = [
    "TRANSACTION_ID",
    "TX_FRAUD_SCENARIO",
]

#%%
def overview(df, name='data', output_path=None, print_report=True):
    """
    Dataset inspection
    """
    records = []

    def add(section, metric, value=None, count=None, rate=None):
        records.append({
            'section': section,
            'metric': metric,
            'value': value,
            'count': count,
            'rate': rate,
        })

    rows = len(df)
    add('overview', 'rows', rows)
    add('overview', 'columns', len(df.columns))
    if "TX_DATETIME" in df.columns:
        add('overview', 'date_min', df.TX_DATETIME.min())
        add('overview', 'date_max', df.TX_DATETIME.max())

    for column, dtype in df.dtypes.items():
        add('dtypes', column, str(dtype))

    missing = df.isna().sum()
    for column, count in missing[missing.gt(0)].items():
        add(
            'missing',
            column,
            count=int(count),
            rate=count / rows if rows else 0,
        )

    for column in NUMERIC_COLUMNS:

        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce")

        count = int(values.eq(-1).sum())

        if count:
            add(
                "sentinel_-1",
                column,
                count=count,
                rate=count / rows if rows else 0,
            )

    if 'TX_FRAUD' in df.columns:
        fraud = pd.to_numeric(df["TX_FRAUD"], errors="coerce")
        add('fraud', 'rate', value=fraud.mean())

        for value, count in fraud.value_counts().items():
            add('fraud_counts', str(value), count=int(count))

    report = pd.DataFrame(records)

    text = StringIO()
    text.write(f'\n{name.upper()}\n')
    text.write(f'Rows: {rows:,} | Columns: {len(df.columns)}\n')
    text.write(
        f'Date: {df.TX_DATETIME.min()} -> {df.TX_DATETIME.max()}\n'
    )

    text.write('\nDtypes:\n')
    text.write(df.dtypes.to_string())

    text.write('\n\nMissing:\n')
    if missing.gt(0).any():
        text.write(missing[missing.gt(0)].to_string())
    else:
        text.write('None')

    text.write('\n\n-1 sentinel counts:\n')
    sentinel_rows = report[report['section'].eq('sentinel_-1')]
    if sentinel_rows.empty:
        text.write('  None')
    else:
        for row in sentinel_rows.itertuples():
            text.write(
                f'  {row.metric}: {row.count:,} '
                f'({row.rate:.4%})'
            )

    if 'TX_FRAUD' in df.columns:
        text.write(f'\n\nFraud rate: {df.TX_FRAUD.mean():.6%}\n')
        text.write(df.TX_FRAUD.value_counts().to_string())

    report_text = text.getvalue()

    if print_report:
        print(report_text)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() == '.csv':
            report.to_csv(output_path, index=False)
        else:
            output_path.write_text(report_text, encoding='utf-8')

    return report

#%%
def get_daily_stats(transactions_df):

    grouped = transactions_df.groupby(
        "TX_TIME_DAYS"
    )

    nb_tx_per_day = (
        grouped["TRANSACTION_ID"]
        .count()
    )

    nb_fraud_per_day = (
        grouped["TX_FRAUD"]
        .sum()
    )

    nb_fraud_customer_per_day = (
        transactions_df[
            transactions_df["TX_FRAUD"] == 1
        ]
        .groupby("TX_TIME_DAYS")["CUSTOMER_ID"]
        .nunique()
    )

    fraud_rate_per_day = (
        nb_fraud_per_day
        / nb_tx_per_day
    )

    return pd.DataFrame({
        "nb_tx_per_day": nb_tx_per_day,
        "nb_fraud_per_day": nb_fraud_per_day,
        "nb_fraud_customer_per_day":
            nb_fraud_customer_per_day,
        "fraud_rate_per_day":
            fraud_rate_per_day,
    })

#%%
def get_numeric_features(
    df,
    target=None,
    exclude=None,
):
    exclude = set(exclude or [])

    if target is not None:
        exclude.add(target)

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    return [
        column
        for column in numeric_columns
        if column not in exclude
    ]

#%%
def remove_constant_features(df):
    """
    Remove numeric features with zero variance.
    """
    numeric_df = df.select_dtypes(
        include=np.number
    )

    constant_columns = (
        numeric_df.nunique(dropna=False) <= 1
    )

    return numeric_df.loc[
        :,
        ~constant_columns,
    ]

#%%
def correlation_matrix(df, method='pearson', exclude=None,):
    """
    Compute correlation matrix for numeric columns
    """
    columns = get_numeric_features(
        df,
        exclude=exclude,
    )

    numeric_df = (
        df[columns]
        .replace([np.inf, -np.inf], np.nan)
    )
    numeric_df = numeric_df.loc[
        :,
        numeric_df.nunique(dropna=False) > 1,
    ]
    corr_matrix = numeric_df.corr(method=method)
    return corr_matrix

#%%
def pearson_correlation(df, target, exclude=None,):
    """
    Compute Pearson correlation between numeric columns and a target column
    """
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' is not in the DataFrame."
        )

    if exclude is None:
        exclude = EXCLUDED_FEATURES

    columns = get_numeric_features(
        df,
        target=target,
        exclude=exclude,
    )

    target_values = pd.to_numeric(
        df[target],
        errors="coerce",
    )

    correlations = {}

    for column in columns:
        feature_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        valid = (
                feature_values.notna()
                & target_values.notna()
                & np.isfinite(feature_values)
                & np.isfinite(target_values)
        )

        correlations[column] = (
            feature_values[valid]
            .corr(target_values[valid])
            if valid.sum() >= 2
            else np.nan
        )

    result = pd.Series(
        correlations,
        name="pearson",
    )

    return result.sort_values(
        key=lambda x: x.abs(),
        ascending=False,
    )

#%%
def sample_for_feature_analysis(
    df,
    sample_size=100000,
    random_state=99,
):
    """
    Create a representative sample for computationally
    expensive feature analysis.
    """

    if len(df) <= sample_size:
        return df.copy()

    return df.sample(
        n=sample_size,
        random_state=random_state,
    )