"""Fraud feature engineering.

Builds the customer- and terminal-level features from raw transactions.
Scoring and selection of those features lives in :mod:`qrc.analysis`.
"""

from .feature_engineer import FeatureEngineer
