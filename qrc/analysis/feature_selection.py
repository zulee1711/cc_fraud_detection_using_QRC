#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .core import get_numeric_features, pearson_correlation

#%%
EXCLUDED_FEATURES = [
    "TRANSACTION_ID",
    "TX_FRAUD_SCENARIO",
]

#%%
def prepare_feature_data(
    df,
    target="TX_FRAUD",
    exclude=None,
):
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

    X = (
        df[columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    y = df[target]

    # Temporary imputation for algorithms
    X = X.fillna(X.median())

    return X, y

#%%
def information_gain(
    df,
    target='TX_FRAUD',
    random_state=99,
    exclude=None,
):
    X, y = prepare_feature_data(
        df,
        target=target,
        exclude=exclude,
    )

    scores = mutual_info_classif(
        X,
        y,
        random_state=random_state,
        n_jobs=-1,
    )

    result = pd.DataFrame({
        "feature": X.columns,
        "information_gain": scores,
    })

    return result.sort_values(
        "information_gain",
        ascending=False,
    ).reset_index(drop=True)

#%%
def random_forest_importance(
    df,
    target='TX_FRAUD',
    random_state=99,
    n_estimators=100,
    exclude=None,
):
    X, y = prepare_feature_data(
        df,
        target=target,
        exclude=exclude,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X, y)

    result = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    })

    return result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

#%%
def pca_analysis(
    df,
    target="TX_FRAUD",
    exclude=None,
    n_components=None,
):
    X, _ = prepare_feature_data(
        df,
        target=target,
        exclude=exclude,
    )

    # Standardize features
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(
        n_components=n_components,
    )

    pca.fit(X_scaled)

    return pca, scaler, X.columns.tolist()

#%%
def feature_selection_report(df, target, random_state=99, n_estimators=100, exclude=None,):
    pearson = pearson_correlation(
        df,
        target=target,
        exclude=exclude,
    )

    pearson_df = (
        pearson
        .rename("pearson")
        .reset_index()
        .rename(
            columns={
                "index": "feature"
            }
        )
    )

    ig = information_gain(
        df,
        target=target,
        random_state=random_state,
        exclude=exclude,
    )

    rf = random_forest_importance(
        df,
        target=target,
        random_state=random_state,
        n_estimators=n_estimators,
        exclude=exclude,
    )

    result = pearson_df.merge(
        ig,
        on="feature",
        how="outer",
    )

    result = result.merge(
        rf,
        on="feature",
        how="outer",
    )

    result["abs_pearson"] = (
        result["pearson"].abs()
    )

    return result.sort_values(
        [
            "information_gain",
            "importance",
        ],
        ascending=False,
    ).reset_index(drop=True)

#%%
def pca_summary(pca, feature_names):
    summary = pd.DataFrame({
        "component": [
            f"PC{i + 1}"
            for i in range(
                len(pca.explained_variance_ratio_)
            )
        ],
        "explained_variance_ratio":
            pca.explained_variance_ratio_,
        "explained_variance":
            pca.explained_variance_,
    })

    summary[
        "cumulative_explained_variance"
    ] = (
        summary["explained_variance_ratio"]
        .cumsum()
    )

    return summary

#%%
def pca_loadings(
    pca,
    feature_names,
):
    """
    Return PCA feature loadings.
    """

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[
            f"PC{i + 1}"
            for i in range(
                pca.n_components_
            )
        ],
    )

    return loadings