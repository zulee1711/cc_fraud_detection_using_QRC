#%%
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#%%
class DataProcessor:
    def __init__(
        self,
        feature_sets,
    ):
        self.feature_sets = feature_sets
        self.scalers = {}

    def select_features(
        self,
        df,
        feature_set,
    ):
        features = self.feature_sets[feature_set]

        X = (
            df[features]
            .replace([np.inf, -np.inf], np.nan)
        )

        return X

    def process(
        self,
        train,
        validation,
        test,
        feature_set,
    ):

        X_train = self.select_features(
            train,
            feature_set,
        )

        X_validation = self.select_features(
            validation,
            feature_set,
        )

        X_test = self.select_features(
            test,
            feature_set,
        )

        # Fill missing values using TRAIN statistics
        medians = X_train.median()

        X_train = X_train.fillna(medians)
        X_validation = X_validation.fillna(medians)
        X_test = X_test.fillna(medians)

        # Scale using TRAIN only
        scaler = MinMaxScaler()

        X_train = scaler.fit_transform(X_train)
        X_validation = scaler.transform(X_validation)
        X_test = scaler.transform(X_test)

        self.scalers[feature_set] = scaler

        return {
            "X_train": X_train,
            "X_validation": X_validation,
            "X_test": X_test,

            "y_train": train["TX_FRAUD"].to_numpy(),
            "y_validation": validation["TX_FRAUD"].to_numpy(),
            "y_test": test["TX_FRAUD"].to_numpy(),

            "groups_train": train["CUSTOMER_ID"].to_numpy(),
            "groups_validation": validation["CUSTOMER_ID"].to_numpy(),
            "groups_test": test["CUSTOMER_ID"].to_numpy(),

            "features": self.feature_sets[feature_set],
        }