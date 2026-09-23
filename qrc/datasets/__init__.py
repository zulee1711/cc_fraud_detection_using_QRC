"""Transaction datasets: simulation, train/validation/test splitting and file I/O.

This is code. The dataset *files* live outside the package, at the repository
root (``raw_data/`` and ``data/``, which are git-ignored).
"""

from .dataIO import loading, load_daily_files, load_splits, dumping
from .split_dataset import split_dataset, split_features_by_dates
