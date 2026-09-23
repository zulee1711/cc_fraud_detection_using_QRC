#%%
from pathlib import Path

from utils import (
    clean_amount,
    add_time_features,
    customer_spending_behaviour_rolling,
    terminal_count_risk_rolling,
    add_behavior,
    add_combination_features,
    add_deviation_features,
    dumping,
)

from datasets import split_features_by_dates

#%%
class FeatureEngineer:

    def __init__(
        self,
        windows=(1, 7, 30, 90, 180),
        use_behavior=True,
        use_combinations=True,
        use_deviations=True,
        output_dir=None,
        save=True,
    ):
        self.windows = windows
        self.output_dir = Path(output_dir) if output_dir else None
        self.use_behavior = use_behavior
        self.use_combinations = use_combinations
        self.use_deviations = use_deviations
        self.save = save

    def transform(self, df):
        """Create all features for a dataframe."""

        df = clean_amount(df)

        df = add_time_features(df)

        df = customer_spending_behaviour_rolling(
            df,
            self.windows,
        )

        df = terminal_count_risk_rolling(
            df,
            self.windows,
        )

        if self.use_behavior:
            df = add_behavior(df)

        if self.use_combinations:
            df = add_combination_features(df)

        if self.use_deviations:
            df = add_deviation_features(df)

        return df

    def run(
        self,
        full_data,
        train,
        validation,
        test,
    ):
        """Create features and split them using existing date boundaries."""

        full_features = self.transform(full_data)

        if self.save and self.output_dir:
            dumping(
                full_features,
                out_path=self.output_dir / "full_features.pkl",
            )

        (
            train_features,
            validation_features,
            test_features,
        ) = split_features_by_dates(
            full_features,
            train,
            validation,
            test,
        )

        return (
            train_features,
            validation_features,
            test_features,
        )