"""Splits by patient (subjectid), so no patient's cases sit on both sides."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def train_test(rows: pd.DataFrame, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks over `rows` for the training and test patients."""
    split = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_idx, _ = next(split.split(rows, groups=rows.subjectid))
    train = np.zeros(len(rows), bool)
    train[train_idx] = True
    return train, ~train


def folds(rows: pd.DataFrame, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """k (train, validation) index pairs, grouped by patient."""
    return list(GroupKFold(k, shuffle=True, random_state=seed).split(rows, groups=rows.subjectid))
