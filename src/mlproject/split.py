"""One reproducible train/test split. The test set is reserved for the final evaluation."""

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import TARGET

SEED = 42
TEST_SIZE = 0.2


def split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified on the target so train and test keep the same churn rate (the classes are imbalanced)."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])
    return train.reset_index(drop=True), test.reset_index(drop=True)
