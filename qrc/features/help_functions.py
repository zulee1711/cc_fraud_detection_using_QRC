#%%
from pathlib import Path
import joblib
import re
import numpy as np
import pandas as pd

#%%
#%%
def clean_amount(df):
    amount = pd.to_numeric(
        df['TX_AMOUNT'],
        errors='coerce',
    )

    # -1 is the dataset's missing-value sentinel.
    df['TX_AMOUNT_MISSING'] = (
            amount == -1
    ).astype('int8')

    df['TX_AMOUNT_CLEAN'] = amount.mask(
        amount == -1,
        np.nan,
    )

    # Explicit pandas numeric dtype prevents object-dtype ufunc errors.
    df['TX_AMOUNT_CLEAN'] = pd.to_numeric(
        df['TX_AMOUNT_CLEAN'],
        errors='coerce',
    ).astype('float64')

    df['TX_AMOUNT_LOG'] = np.log1p(
        df['TX_AMOUNT_CLEAN'].clip(lower=0)
    )

    return df

#%%
def add_time_features(df):
    dt = pd.to_datetime(df.TX_DATETIME)
    df['TX_HOUR'] = dt.dt.hour
    df['TX_DAY_OF_WEEK'] = dt.dt.dayofweek
    df['TX_MONTH'] = dt.dt.month
    # df['TX_WEEK_OF_YEAR'] = dt.dt.isocalendar().week.astype(int)

    # Weekend transaction: Saturday (5) and Sunday (6)
    df['TX_DURING_WEEKEND'] = (dt.dt.dayofweek >= 5).astype('int8')

    # Night time transaction: 0:00 - 6:00
    df['TX_DURING_NIGHT'] = (dt.dt.hour <= 6).astype('int8')

    # Encode time of day as sin/cos to capture cyclical nature
    sec = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
    df['TX_TIME_SIN'] = np.sin(2 * np.pi * sec / 86400)
    df['TX_TIME_COS'] = np.cos(2 * np.pi * sec / 86400)
    return df

#%%
def customer_spending_behaviour_rolling(df, windows):
    out = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME']).copy()
    out['_COUNT'] = 1
    g = out.groupby('CUSTOMER_ID', sort=False)

    for w in windows:
        out[f'CUSTOMER_NB_TX_{w}D'] = (
            g.rolling(
                f'{w}D',
                on='TX_DATETIME',
                closed='left',
                min_periods=1,
            )['_COUNT']
            .sum()
            .reset_index(drop=True)
            .fillna(0)
            .to_numpy()
        )

        out[f'CUSTOMER_AVG_AMOUNT_{w}D'] = (
            g.rolling(
                f'{w}D',
                on='TX_DATETIME',
                closed='left',
                min_periods=1,
            )['TX_AMOUNT_CLEAN']
            .mean()
            .reset_index(drop=True)
            .fillna(0)
            .to_numpy()
        )

    return out.drop(columns='_COUNT')

#%%
def terminal_count_risk_rolling(df, windows, delay_days=7):
    out = df.sort_values(['TERMINAL_ID', 'TX_DATETIME']).copy()
    g = out.groupby('TERMINAL_ID', sort=False)

    # first transaction for a terminal have no previous transaction in the rolling window
    # fillna(0) = 0 previous transactions
    delayed = (
        g.rolling(
            f'{delay_days}D',
            on='TX_DATETIME',
            closed='left',
            min_periods=1,
        )['TX_FRAUD']
        .agg(['sum', 'count'])
        .reset_index(drop=True)
        .fillna(0)
    )

    for w in windows:
        hist = (
            g.rolling(
                f'{delay_days + w}D',
                on='TX_DATETIME',
                closed='left',
                min_periods=1,
            )['TX_FRAUD']
            .agg(['sum', 'count'])
            .reset_index(drop=True)
            .fillna(0)
        )

        fraud_n = (
            hist['sum'].to_numpy()
            - delayed['sum'].to_numpy()
        )
        tx_n = (
            hist['count'].to_numpy()
            - delayed['count'].to_numpy()
        )

        risk = np.divide(
            fraud_n,
            tx_n,
            out=np.zeros(len(tx_n), dtype=float),
            where=tx_n > 0,
        )

        out[f'TERMINAL_NB_TX_{w}D'] = tx_n
        out[f'TERMINAL_RISK_{w}D'] = risk

    return out


def add_behavior(df):
    out = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME']).copy()
    g = out.groupby('CUSTOMER_ID', sort=False)

    out['CUSTOMER_TIME_GAP_SECONDS'] = (
        g.TX_DATETIME.diff()
        .dt.total_seconds()
        .fillna(-1)
    )

    avg = out['CUSTOMER_AVG_AMOUNT_30D']

    out['CUSTOMER_AMOUNT_RATIO_30D'] = (
        out['TX_AMOUNT_CLEAN']
        .div(avg)
        .where(avg > 0, 0)
        .replace([np.inf, -np.inf], -99)
        # .fillna(0)
    )

    return out

#%%
def add_combination_features(
        df,
        include_customer: bool = True,
        include_terminal: bool = True,
        include_amount: bool = True,
):
    def get_window_columns(prefix:str):
        pattern = re.compile(
            rf"^{re.escape(prefix)}_(\d+)D$"
        )

        columns = {}

        for column in df.columns:
            match = pattern.match(column)

            if match:
                window = int(match.group(1))
                columns[window] = column

        return columns

    def safe_divide(numerator, denominator):
        result = numerator.div(denominator)

        return result.replace(
            [np.inf, -np.inf],
            np.nan,
        )

    customer_tx = get_window_columns(
        "CUSTOMER_NB_TX"
    )

    customer_avg = get_window_columns(
        "CUSTOMER_AVG_AMOUNT"
    )

    terminal_risk = get_window_columns(
        "TERMINAL_RISK"
    )

    # Customer transaction velocity
    if include_customer:

        windows = sorted(customer_tx)

        for i, short_window in enumerate(windows):

            for long_window in windows[i + 1:]:
                short_col = customer_tx[short_window]
                long_col = customer_tx[long_window]

                feature_name = (
                    f"CUSTOMER_TX_RATE_"
                    f"{short_window}D_{long_window}D"
                )

                df[feature_name] = safe_divide(
                    df[short_col],
                    df[long_col],
                )

    # Amount relative to customer behavior
    if include_amount and "TX_AMOUNT_CLEAN" in df.columns:

        for window, column in sorted(customer_avg.items()):
            feature_name = (
                f"CUSTOMER_AMOUNT_RATIO_{window}D"
            )

            df[feature_name] = safe_divide(
                df["TX_AMOUNT_CLEAN"],
                df[column],
            )

    # Short-term vs long-term customer behavior
    if include_customer:

        windows = sorted(customer_avg)

        for i, short_window in enumerate(windows):

            for long_window in windows[i + 1:]:
                short_col = customer_avg[short_window]
                long_col = customer_avg[long_window]

                feature_name = (
                    f"CUSTOMER_AMOUNT_SHIFT_"
                    f"{short_window}D_{long_window}D"
                )

                df[feature_name] = safe_divide(
                    df[short_col],
                    df[long_col],
                )

    # Customer transaction velocity * amount
    if (
            include_customer
            and include_amount
            and "TX_AMOUNT_CLEAN" in df.columns
    ):

        for window, column in sorted(customer_tx.items()):
            feature_name = (
                f"CUSTOMER_ACTIVITY_AMOUNT_{window}D"
            )

            df[feature_name] = (
                    df[column]
                    * df["TX_AMOUNT_CLEAN"]
            )

    # Terminal behavior
    if include_terminal:

        windows = sorted(terminal_risk)

        for i, short_window in enumerate(windows):

            for long_window in windows[i + 1:]:
                short_col = terminal_risk[short_window]
                long_col = terminal_risk[long_window]

                feature_name = (
                    f"TERMINAL_RISK_CHANGE_"
                    f"{short_window}D_{long_window}D"
                )

                df[feature_name] = (
                        df[short_col]
                        - df[long_col]
                )

    # Amount per customer time gap
    if (
            include_customer
            and include_amount
            and "CUSTOMER_TIME_GAP_SECONDS" in df.columns
            and "TX_AMOUNT_CLEAN" in df.columns
    ):
        gap = df["CUSTOMER_TIME_GAP_SECONDS"].where(
            df["CUSTOMER_TIME_GAP_SECONDS"] >= 0
        )

        df["AMOUNT_PER_TIME_GAP"] = safe_divide(
            df["TX_AMOUNT_CLEAN"],
            gap,
        )

    return df

#%%
def add_deviation_features(df):
    if {
        'TX_AMOUNT_CLEAN',
        'CUSTOMER_AVG_AMOUNT_30D'
    }.issubset(df.columns):

        df['CUSTOMER_AMOUNT_DEVIATION_30D'] = (
            df['TX_AMOUNT_CLEAN']
            - df['CUSTOMER_AVG_AMOUNT_30D']
        )

    if {
        'TX_AMOUNT_CLEAN',
        'CUSTOMER_AVG_AMOUNT_30D'
    }.issubset(df.columns):

        df['CUSTOMER_AMOUNT_MULTIPLIER_30D'] = (
            df['TX_AMOUNT_CLEAN']
            / df['CUSTOMER_AVG_AMOUNT_30D']
        ).replace([np.inf, -np.inf], np.nan)

    return df