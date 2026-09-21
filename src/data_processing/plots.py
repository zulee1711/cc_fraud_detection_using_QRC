#%%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

#%%
def save_mpl(fig,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=300,bbox_inches='tight')
    plt.close(fig)


#%%
def plot_amount_time_distributions(
    df:pd.DataFrame,
    max_days:int=10,
    sample_size:int=10000,
    random_state:int=99,
    output_path=None
):
    required_columns = {
        "TX_AMOUNT",
        "TX_TIME_DAYS",
        "TX_TIME_SECONDS",
    }

    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "DataFrame is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    filtered_df = df[
        df["TX_TIME_DAYS"] < max_days
        ]

    if filtered_df.empty:
        raise ValueError(
            f"No transactions found with TX_TIME_DAYS < {max_days}."
        )

    sampled_df = filtered_df.sample(
        n=min(sample_size, len(filtered_df)),
        random_state=random_state,
    )

    amount_values = (
        sampled_df["TX_AMOUNT"]
        # .dropna()
        .to_numpy()
    )

    time_values = (
            sampled_df["TX_TIME_SECONDS"]
            # .dropna()
            .to_numpy()
            / 86400
    )

    if len(amount_values) == 0 or len(time_values) == 0:
        raise ValueError(
            "The selected data contains no valid amount or time values."
        )

    fig, ax = plt.subplots(1, 2, figsize=(18, 4))

    ax[0].hist(
        amount_values,
        bins=100,
        color="red",
        edgecolor="black",
    )
    ax[0].set_title(
        "Distribution of transaction amounts",
        fontsize=14,
    )
    ax[0].set_xlim(
        amount_values.min(),
        amount_values.max(),
    )
    ax[0].set(
        xlabel="Amount",
        ylabel="Number of transactions",
    )

    ax[1].hist(
        time_values,
        bins=100,
        color="blue",
        edgecolor="black",
    )
    ax[1].set_title(
        "Distribution of transaction times",
        fontsize=14,
    )
    ax[1].set_xlim(
        time_values.min(),
        time_values.max(),
    )
    ax[1].set_xticks(range(max_days))
    ax[1].set(
        xlabel="Time (days)",
        ylabel="Number of transactions",
    )

    fig.tight_layout()

    if output_path is not None:
        save_mpl(fig, output_path)

    return fig, ax

#%%
def plot_fraud_and_transactions_stats(daily_stats, output_path=None):
    """
    Plot total transactions, fraudulent transactions, and fraudulent customers/cards per day
    """
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(
        daily_stats.index,
        daily_stats["nb_tx_per_day"] / 50,
        label="# transactions per day (/50)",
        linewidth=2,
    )

    ax.plot(
        daily_stats.index,
        daily_stats["nb_fraud_per_day"],
        label="# fraudulent txs per day",
        linewidth=2,
    )

    ax.plot(
        daily_stats.index,
        daily_stats["nb_fraud_customer_per_day"],
        label="# fraudulent cards per day",
        linewidth=2,
    )

    ax.set_title(
        "Total transactions, and number of fraudulent transactions\n"
        "and number of compromised cards per day",
        fontsize=20,
    )

    ax.set_xlabel(
        "Number of days since beginning of data generation",
        fontsize=14,
    )

    ax.set_ylabel(
        "Number",
        fontsize=14,
    )

    ax.set_ylim(0, 300)

    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize=15,
    )

    fig.tight_layout()

    if output_path is not None:
        save_mpl(fig, output_path)

    return fig, ax

#%%
def plot_fraud_rate_over_time(daily_stats, output_path=None):
    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    ax.plot(
        daily_stats.index,
        daily_stats["fraud_rate_per_day"],
        linewidth=2,
    )

    ax.set_title(
        "Fraud rate per day",
        fontsize=16,
    )

    ax.set_xlabel(
        "Number of days since beginning of data generation"
    )

    ax.set_ylabel(
        "Fraud rate"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    if output_path is not None:
        save_mpl(fig, output_path)

    return fig, ax

#%%
def plot_correlation_matrix(corr_matrix, output_path=None):
    fig, ax = plt.subplots(
        figsize=(16, 14)
    )

    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        ax=ax,
    )

    ax.set_title(
        "Feature correlation matrix",
        fontsize=16,
    )

    fig.tight_layout()

    if output_path is not None:
        save_mpl(fig, output_path)

    return fig, ax

#%%
def plot_pearson_correlation(
    pearson,
    top_n=20,
    output_path=None
):
    values = (
        pearson
        .drop(labels=["TX_FRAUD"], errors="ignore")
        .sort_values(
            key=np.abs,
            ascending=False,
        )
        .head(top_n)
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.barh(
        values.index,
        values.values,
    )

    ax.set_title(
        f"Top {top_n} Pearson correlations with fraud",
        fontsize=16,
    )

    ax.set_xlabel(
        "Pearson correlation"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.axvline(
        0,
        linewidth=0.8,
    )

    fig.tight_layout()

    if output_path is not None:
        save_mpl(fig, output_path)

    return fig, ax

#%%
def plot_information_gain(
    information_gain,
    top_n=20,
    output_path=None
):
    values = (
        information_gain
        .sort_values(
            "information_gain",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            "information_gain"
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.barh(
        values["feature"],
        values["information_gain"],
    )

    ax.set_title(
        f"Top {top_n} features by information gain",
        fontsize=16,
    )

    ax.set_xlabel(
        "Information gain"
    )

    ax.set_ylabel(
        "Feature"
    )

    fig.tight_layout()

    if output_path is not None:
        save_mpl(fig, output_path)

    return fig, ax


# %%
def plot_random_forest_importance(
    importance,
    top_n=20,
    output_path=None
):
    values = (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            "importance"
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.barh(
        values["feature"],
        values["importance"],
    )

    ax.set_title(
        f"Top {top_n} features by random forest importance",
        fontsize=16,
    )

    ax.set_xlabel(
        "Random forest importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    fig.tight_layout()

    if output_path is not None:
        save_mpl(fig, output_path)

    return fig, ax

#%%
def plot_feature_selection_report(
    feature_report,
    top_n=20,
    output_path=None
):
    values = (
        feature_report
        .sort_values(
            "information_gain",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    values = values.sort_values(
        "information_gain"
    )

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    y = np.arange(len(values))
    height = 0.25

    ax.barh(
        y - height,
        values["abs_pearson"],
        height=height,
        label="Absolute Pearson",
    )

    ax.barh(
        y,
        values["information_gain"],
        height=height,
        label="Information gain",
    )

    ax.barh(
        y + height,
        values["importance"],
        height=height,
        label="Random forest",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(values["feature"])

    ax.set_xlabel(
        "Feature importance / association"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        f"Feature selection comparison - top {top_n}",
        fontsize=16,
    )

    ax.legend()

    fig.tight_layout()

    if output_path is not None:
        save_mpl(fig, output_path)

    return fig, ax

#%%
def plot_pca_explained_variance(
    pca,
    output_path=None,
):
    """
    Plot explained variance ratio of each
    principal component.
    """

    components = np.arange(
        1,
        len(pca.explained_variance_ratio_) + 1,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        components,
        pca.explained_variance_ratio_,
        marker="o",
    )

    ax.set_title(
        "PCA explained variance by component"
    )

    ax.set_xlabel(
        "Principal component"
    )

    ax.set_ylabel(
        "Explained variance ratio"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    save_mpl(
        fig,
        output_path,
    )

#%%
def plot_pca_cumulative_variance(
    pca,
    output_path=None,
):
    """
    Plot cumulative explained variance.
    """

    cumulative = np.cumsum(
        pca.explained_variance_ratio_
    )

    components = np.arange(
        1,
        len(cumulative) + 1,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        components,
        cumulative,
        marker="o",
    )

    ax.axhline(
        0.90,
        linestyle="--",
        label="90%",
    )

    ax.axhline(
        0.95,
        linestyle="--",
        label="95%",
    )

    ax.axhline(
        0.99,
        linestyle="--",
        label="99%",
    )

    ax.set_title(
        "PCA cumulative explained variance"
    )

    ax.set_xlabel(
        "Number of principal components"
    )

    ax.set_ylabel(
        "Cumulative explained variance"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    save_mpl(
        fig,
        output_path,
    )

#%%
def plot_pca_loadings(
    pca,
    feature_names,
    component=1,
    top_n=20,
    output_path=None,
):
    """
    Plot the top absolute loadings
    for a selected principal component.

    Parameters
    ----------
    component : int
        Principal component number, starting at 1.
    """

    if component < 1:
        raise ValueError(
            "component must be >= 1"
        )

    if component > pca.n_components_:
        raise ValueError(
            f"component must be <= "
            f"{pca.n_components_}"
        )

    loadings = pca.components_[
        component - 1
    ]

    result = pd.DataFrame({
        "feature": feature_names,
        "loading": loadings,
        "abs_loading": np.abs(loadings),
    })

    result = (
        result
        .sort_values(
            "abs_loading",
            ascending=False,
        )
        .head(top_n)
        .sort_values("loading")
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        result["feature"],
        result["loading"],
    )

    ax.set_title(
        f"Top {top_n} loadings for PC{component}"
    )

    ax.set_xlabel(
        "PCA loading"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    save_mpl(
        fig,
        output_path,
    )

    return result