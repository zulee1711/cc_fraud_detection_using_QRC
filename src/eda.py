#%%
import os
import sys
import pandas as pd

from pathlib import Path
from utils import *

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#%% Load training data
# train_path = Path.joinpath(PROJECT_ROOT, "data", "transactions_train.pkl")
# transactions_df = loading(train_path)

#%% Load daily training data
train_daily_path = Path.joinpath(PROJECT_ROOT, "data", "train")
daily_df = load_daily_files(
    input_dir=train_daily_path,
    start_date="2025-01-01",
    end_date="2025-09-12"
)
