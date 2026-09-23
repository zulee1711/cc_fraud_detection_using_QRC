#%%
from pathlib import Path

import numpy as np
import pandas as pd

from ..logger import get_logger
from ..datasets import dumping
from .core import (
    sample_for_feature_analysis,
    correlation_matrix,
)
from .plots import (
    plot_correlation_matrix,
    plot_pearson_correlation,
    plot_information_gain,
    plot_random_forest_importance,
    plot_feature_selection_report,
    plot_pca_explained_variance,
    plot_pca_cumulative_variance,
    plot_pca_loadings,
)
from .feature_selection import (
    feature_selection_report,
    pca_analysis,
    pca_summary,
    pca_loadings,
)

logger = get_logger(__name__)

from dataclasses import dataclass, field

#%%
@dataclass
class FeatureAnalysisConfig:
    """
    Configuration for one feature-analysis experiment.

    sample_seed:
        Controls which rows are selected for train_analysis.

    selection_seeds:
        Controls the randomness used by Information Gain and
        Random Forest inside feature_selection_report().
    """
    sample_seed: int = 99
    sample_size: int = 100000

    selection_seeds: list[int] = field(
        default_factory=lambda: [42, 99, 712, 2026]
    )

    n_estimators: int = 100
    top_n: int = 20
    stability_top_k: tuple[int, ...] = (5, 10, 15, 20)
    candidate_sizes: tuple[int, ...] = (5, 10, 15)

    target: str = "TX_FRAUD"

    # Features excluded from correlation analysis.
    correlation_exclude: tuple[str, ...] = (
        "TRANSACTION_ID",
        "TX_FRAUD",
        "TX_FRAUD_SCENARIO",
    )

    # Features excluded from PCA.
    pca_exclude: tuple[str, ...] = (
        "CUSTOMER_ID",
        "TERMINAL_ID",
        "TRANSACTION_ID",
        "TX_FRAUD_SCENARIO",
    )


class FeatureAnalysisExperiment:
    """
    Feature Analysis experiment.

    One experiment is defined by one sample_seed. Inside that
    experiment, compare multiple selection_seeds.

    Directory structure:
        results/
        └── feature_analysis/
            └── sample_seed_99/
                ├── train_analysis.csv
                ├── correlation_matrix.csv
                ├── pca_report.csv
                ├── pca_loadings.csv
                ├── feature_selection/
                │   ├── seed_42/
                │   ├── seed_99/
                │   └── ...
                ├── feature_stability.csv
                ├── pca_feature_contribution.csv
                ├── combined_feature_selection.csv
                ├── candidate_top_5_features.csv
                ├── candidate_top_10_features.csv
                └── candidate_top_15_features.csv
    """
    def __init__(
        self,
        train_features,
        result_dir,
        plot_dir,
        config=None,
        save_results=True,
        make_plots=True,
    ):
        self.train_features = train_features
        self.result_dir = Path(result_dir)
        self.plot_dir = Path(plot_dir)

        self.config = (
            config
            if config is not None
            else FeatureAnalysisConfig()
        )

        self.save_results = save_results
        self.make_plots = make_plots

        self.sample_name = (
            f"sample_seed_{self.config.sample_seed}"
        )

        self.experiment_result_dir = (self.result_dir / "feature_analysis" / self.sample_name)

        self.experiment_plot_dir = (self.plot_dir / "feature_analysis" / self.sample_name)

        self.experiment_result_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_plot_dir.mkdir(parents=True, exist_ok=True)

        self.train_analysis = None
        self.correlation = None
        self.feature_reports = {}
        self.feature_stability = None
        self.pca = None
        self.scaler = None
        self.feature_names = None
        self.pca_report = None
        self.pca_loadings_report = None
        self.pca_feature_contribution = None
        self.combined_feature_selection = None
        self.candidate_feature_sets = {}

    def create_analysis_sample(self):
        """
        Create the fixed train_analysis sample for this experiment.
        """

        self.train_analysis = sample_for_feature_analysis(
            self.train_features,
            sample_size=self.config.sample_size,
            random_state=self.config.sample_seed,
        )

        logger.info(
            "Created train_analysis with sample seed %s (n=%s)",
            self.config.sample_seed,
            len(self.train_analysis),
        )

        if self.save_results:
            dumping(
                self.train_analysis,
                out_path=(self.experiment_result_dir / "train_analysis.csv"),
            )

        return self.train_analysis

    def run_correlation_analysis(self):
        """
        Run correlation analysis once for this sample.
        """

        if self.train_analysis is None:
            self.create_analysis_sample()

        self.correlation = correlation_matrix(
            self.train_analysis,
            exclude=list(self.config.correlation_exclude),
        )

        if self.make_plots:
            plot_correlation_matrix(
                self.correlation,
                output_path=(self.experiment_plot_dir / "correlation_matrix.png"),
            )

        if self.save_results:
            dumping(
                self.correlation,
                out_path=(self.experiment_result_dir / "correlation_matrix.csv"),
            )

        return self.correlation

    def run_feature_selection(
        self,
        random_state,
    ):
        """
        Run Pearson, Information Gain and Random Forest
        for one selection seed.
        """

        if self.train_analysis is None:
            self.create_analysis_sample()

        seed_name = f"seed_{random_state}"

        seed_result_dir = self.experiment_result_dir / "feature_selection" / seed_name

        seed_plot_dir = self.experiment_plot_dir / "feature_selection" / seed_name

        seed_result_dir.mkdir(parents=True, exist_ok=True)

        seed_plot_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Running feature selection: sample_seed=%s, selection_seed=%s",
            self.config.sample_seed,
            random_state,
        )

        feature_report = feature_selection_report(
            self.train_analysis,
            target=self.config.target,
            random_state=random_state,
            n_estimators=self.config.n_estimators,
        )

        pearson = feature_report[["feature", "pearson"]].copy().set_index("feature")["pearson"]

        ig = feature_report[["feature", "information_gain"]].copy()

        rf = feature_report[["feature", "importance"]].copy()

        if self.make_plots:
            plot_pearson_correlation(
                pearson,
                top_n=self.config.top_n,
                output_path=(
                    seed_plot_dir
                    / "pearson_correlation.png"
                ),
            )

            plot_information_gain(
                ig,
                top_n=self.config.top_n,
                output_path=(
                    seed_plot_dir
                    / "information_gain.png"
                ),
            )

            plot_random_forest_importance(
                rf,
                top_n=self.config.top_n,
                output_path=(
                    seed_plot_dir
                    / "random_forest_importance.png"
                ),
            )

            plot_feature_selection_report(
                feature_report,
                top_n=self.config.top_n,
                output_path=(
                    seed_plot_dir
                    / "feature_selection_report.png"
                ),
            )

        if self.save_results:
            dumping(
                pearson,
                out_path=(
                    seed_result_dir
                    / "pearson_correlation.csv"
                ),
            )

            dumping(
                ig,
                out_path=(
                    seed_result_dir
                    / "info_gain.csv"
                ),
            )

            dumping(
                rf,
                out_path=(
                    seed_result_dir
                    / "random_forest_importance.csv"
                ),
            )

            dumping(
                feature_report,
                out_path=(
                    seed_result_dir
                    / "feature_selection_report.csv"
                ),
            )

        return feature_report

    def run_all_feature_selection(self):
        """
        Run feature selection for every configured selection seed.
        """

        self.feature_reports = {}

        for seed in self.config.selection_seeds:
            self.feature_reports[seed] = (
                self.run_feature_selection(
                    random_state=seed
                )
            )

        return self.feature_reports

    def create_feature_stability_report(self):
        """
        Compare Random Forest rankings across selection seeds.
        """

        if not self.feature_reports:
            self.run_all_feature_selection()

        rank_tables = []

        for seed, report in self.feature_reports.items():

            rf = report[["feature", "importance"]].copy()

            rf["rank"] = rf["importance"].rank(ascending=False, method="min")

            rf["seed"] = seed

            rank_tables.append(
                rf[
                    [
                        "feature",
                        "importance",
                        "rank",
                        "seed",
                    ]
                ]
            )

        all_ranks = pd.concat(
            rank_tables,
            ignore_index=True,
        )

        stability = (
            all_ranks
            .groupby("feature")
            .agg(
                mean_importance=(
                    "importance",
                    "mean",
                ),
                std_importance=(
                    "importance",
                    "std",
                ),
                mean_rank=(
                    "rank",
                    "mean",
                ),
                median_rank=(
                    "rank",
                    "median",
                ),
                std_rank=(
                    "rank",
                    "std",
                ),
                seeds_evaluated=(
                    "seed",
                    "nunique",
                ),
            )
            .reset_index()
        )

        total_seeds = len(
            self.feature_reports
        )

        for top_k in (
            self.config.stability_top_k
        ):

            counts = (
                all_ranks
                .assign(
                    in_top_k=lambda x:
                        x["rank"] <= top_k
                )
                .groupby("feature")[
                    "in_top_k"
                ]
                .sum()
            )

            stability[f"top_{top_k}_count"] = stability["feature"].map(counts).fillna(0).astype(int)

            stability[f"top_{top_k}_percentage"] = stability[f"top_{top_k}_count"] / total_seeds * 100

        stability = stability.sort_values(
            by=[
                "top_15_count",
                "mean_rank",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(drop=True)

        self.feature_stability = stability

        if self.save_results:
            dumping(
                stability,
                out_path=(
                    self.experiment_result_dir
                    / "feature_stability.csv"
                ),
            )

        return stability

    def run_pca(self):
        """
        Run PCA for this sample.
        """

        if self.train_analysis is None:
            self.create_analysis_sample()

        (
            self.pca,
            self.scaler,
            self.feature_names,
        ) = pca_analysis(
            self.train_analysis,
            target=self.config.target,
            exclude=list(
                self.config.pca_exclude
            ),
        )

        self.pca_report = pca_summary(
            self.pca,
            self.feature_names,
        )

        self.pca_loadings_report = pca_loadings(
            self.pca,
            self.feature_names,
        )

        if self.make_plots:
            plot_pca_explained_variance(
                self.pca,
                output_path=(
                    self.experiment_plot_dir
                    / "pca_explained_variance.png"
                ),
            )

            plot_pca_cumulative_variance(
                self.pca,
                output_path=(
                    self.experiment_plot_dir
                    / "pca_cumulative_variance.png"
                ),
            )

            plot_pca_loadings(
                self.pca,
                self.feature_names,
                component=1,
                top_n=self.config.top_n,
                output_path=(
                    self.experiment_plot_dir
                    / "pca_pc1_loadings.png"
                ),
            )

        if self.save_results:
            dumping(
                self.pca_report,
                out_path=(
                    self.experiment_result_dir
                    / "pca_report.csv"
                ),
            )

            dumping(
                self.pca_loadings_report,
                out_path=(
                    self.experiment_result_dir
                    / "pca_loadings.csv"
                ),
            )

        return (
            self.pca,
            self.scaler,
            self.feature_names,
        )

    def create_pca_feature_contribution(self):
        """
        Calculate explained-variance-weighted squared PCA loading
        for every original feature.
        """

        if self.pca is None:
            self.run_pca()

        loadings = pd.DataFrame(
            self.pca.components_.T,
            index=self.feature_names,
            columns=[
                f"PC{i + 1}"
                for i in range(
                    self.pca.n_components_
                )
            ],
        )

        explained_variance = pd.Series(
            self.pca.explained_variance_ratio_,
            index=loadings.columns,
        )

        squared_loadings = loadings.pow(2)

        weighted_contributions = (
            squared_loadings
            .mul(
                explained_variance,
                axis=1,
            )
        )

        result = pd.DataFrame(
            {
                "feature": loadings.index,
                "pca_contribution":
                    weighted_contributions.sum(
                        axis=1
                    ),
            }
        )

        result["pca_rank"] = (
            result["pca_contribution"]
            .rank(
                ascending=False,
                method="min",
            )
            .astype(int)
        )

        result = (
            result
            .sort_values(
                "pca_contribution",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        self.pca_feature_contribution = result

        if self.save_results:
            dumping(
                result,
                out_path=(
                    self.experiment_result_dir
                    / "pca_feature_contribution.csv"
                ),
            )

        return result

    def create_combined_feature_selection_report(
        self,
        top_k=15,
    ):
        """
        Combine:
            - Pearson
            - Information Gain
            - Random Forest importance
            - Random Forest stability
            - PCA contribution

        PCA remains a variance-based measure, not predictive
        importance.
        """

        if not self.feature_reports:
            self.run_all_feature_selection()

        if self.feature_stability is None:
            self.create_feature_stability_report()

        if self.pca_feature_contribution is None:
            self.create_pca_feature_contribution()

        reports = []

        for seed, report in (
            self.feature_reports.items()
        ):

            temp = report[
                [
                    "feature",
                    "pearson",
                    "information_gain",
                    "importance",
                ]
            ].copy()

            temp["seed"] = seed
            reports.append(temp)

        all_reports = pd.concat(
            reports,
            ignore_index=True,
        )

        average_metrics = (
            all_reports
            .groupby("feature")
            .agg(
                mean_abs_pearson=(
                    "pearson",
                    lambda x: x.abs().mean(),
                ),
                mean_information_gain=(
                    "information_gain",
                    "mean",
                ),
                mean_rf_importance=(
                    "importance",
                    "mean",
                ),
            )
            .reset_index()
        )

        average_metrics["pearson_rank"] = average_metrics["mean_abs_pearson"].rank(
                ascending=False,
                method="min",
                na_option="bottom"
            ).astype(int)

        average_metrics["information_gain_rank"] = average_metrics["mean_information_gain"].rank(
                ascending=False,
                method="min",
                na_option="bottom"
            ).astype(int)

        average_metrics["rf_importance_rank"] = average_metrics["mean_rf_importance"].rank(
                ascending=False,
                method="min",
                na_option="bottom"
            ).astype(int)

        stability_columns = [
            "feature",
            "mean_rank",
            "median_rank",
            "std_rank",
            "top_5_count",
            "top_5_percentage",
            "top_10_count",
            "top_10_percentage",
            "top_15_count",
            "top_15_percentage",
            "top_20_count",
            "top_20_percentage",
        ]

        combined = average_metrics.merge(
            self.feature_stability[stability_columns],
            on="feature",
            how="left",
        )

        combined = combined.merge(
            self.pca_feature_contribution[
                [
                    "feature",
                    "pca_contribution",
                    "pca_rank",
                ]
            ],
            on="feature",
            how="left",
        )

        combined["rf_top_15"] = combined["top_15_count"] > 0

        combined["ig_top_15"] = combined["information_gain_rank"] <= top_k

        combined["pearson_top_15"] = combined["pearson_rank"] <= top_k

        combined["pca_top_15"] = combined["pca_rank"] <= top_k

        combined["methods_supporting_top_15"] = combined[
                [
                    "rf_top_15",
                    "ig_top_15",
                    "pearson_top_15",
                    "pca_top_15",
                ]
            ].sum(axis=1)

        combined = combined.sort_values(
            by=[
                "methods_supporting_top_15",
                "top_15_count",
                "mean_rank",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        ).reset_index(drop=True)

        combined["consensus_rank"] = np.arange(len(combined)) + 1

        self.combined_feature_selection = (
            combined
        )

        if self.save_results:
            dumping(
                combined,
                out_path=(
                    self.experiment_result_dir
                    / "combined_feature_selection.csv"
                ),
            )

        return combined

    def create_candidate_feature_sets(self):
        """
        Create candidate 5-, 10-, and 15-feature sets.
        """

        if self.combined_feature_selection is None:
            self.create_combined_feature_selection_report()

        ranking = (
            self.combined_feature_selection
            .sort_values(
                by=[
                    "methods_supporting_top_15",
                    "top_15_count",
                    "mean_rank",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        self.candidate_feature_sets = {}

        for size in self.config.candidate_sizes:

            features = (
                ranking
                .head(size)["feature"]
                .tolist()
            )

            self.candidate_feature_sets[size] = features

            if self.save_results:
                feature_set_df = pd.DataFrame(
                    {
                        "rank": range(
                            1,
                            len(features) + 1,
                        ),
                        "feature": features,
                    }
                )

                dumping(
                    feature_set_df,
                    out_path=(
                        self.experiment_result_dir
                        / (
                            f"candidate_top_"
                            f"{size}_features.csv"
                        )
                    ),
                )

            logger.info(
                "Sample seed %s - candidate top %s "
                "features: %s",
                self.config.sample_seed,
                size,
                features,
            )

        return self.candidate_feature_sets

    def run(self):
        """
        Run the complete analysis for this sample seed.
        """
        self.create_analysis_sample()

        self.run_correlation_analysis()

        self.run_all_feature_selection()

        self.create_feature_stability_report()

        self.run_pca()

        self.create_pca_feature_contribution()

        self.create_combined_feature_selection_report()

        self.create_candidate_feature_sets()

        return self

#%%
class FeatureAnalyzer:
    def __init__(
        self,
        result_dir,
        plot_dir,
        sample_seeds=(99, 812, 2026, 287, 123456),
        selection_seeds=(42, 99, 712, 2026),
        sample_size=100000,
        n_estimators=100,
        top_n=20,
        stability_top_k=(5, 10, 15, 20),
        candidate_sizes=(5, 10, 15),
        save=True,
        plots=True,
    ):
        self.result_dir = Path(result_dir)
        self.plot_dir = Path(plot_dir)

        self.sample_seeds = sample_seeds
        self.selection_seeds = selection_seeds
        self.sample_size = sample_size
        self.candidate_sizes = candidate_sizes
        self.n_estimators = n_estimators
        self.top_n = top_n
        self.stability_top_k = stability_top_k

        self.save = save
        self.plots = plots

        self.experiments = {}

    def run(self, train_features):

        for sample_seed in self.sample_seeds:

            config = FeatureAnalysisConfig(
                sample_seed=sample_seed,
                sample_size=self.sample_size,
                selection_seeds=self.selection_seeds,
                n_estimators=self.n_estimators,
                top_n=self.top_n,
                stability_top_k=self.stability_top_k,
                candidate_sizes=self.candidate_sizes,
            )

            experiment = FeatureAnalysisExperiment(
                train_features=train_features,
                result_dir=self.result_dir,
                plot_dir=self.plot_dir,
                config=config,
                save_results=self.save,
                make_plots=self.plots,
            )

            experiment.run()

            self.experiments[sample_seed] = experiment

        return self.experiments