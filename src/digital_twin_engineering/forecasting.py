"""Forecasters: one LightGBM model per (target, horizon).

Inputs are every `x.` column of the dataset rows (channels, staleness, expert
signals, covariates, phase); LightGBM handles the NaNs that missing channels and
inapplicable experts leave. Another model family can replace `_model` as long as
it keeps fit/predict(_proba).
"""

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import splits
from .targets import TARGETS


def feature_columns(rows: pd.DataFrame, prefixes: tuple[str, ...] = ("x.",)) -> list[str]:
    return [c for c in rows.columns if c.startswith(prefixes)]


def _model(classification: bool, params: dict[str, Any]):
    params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "n_jobs": 16,
        "verbose": -1,
    } | params
    return lgb.LGBMClassifier(**params) if classification else lgb.LGBMRegressor(**params)


@dataclass
class Forecaster:
    target: str
    horizon_s: int
    features: list[str]
    model: Any

    @property
    def column(self) -> str:
        """The label column this forecaster predicts."""
        return TARGETS[self.target].column(self.horizon_s)

    @property
    def output(self) -> str:
        """The column its predictions go in when fed to later models."""
        return "f." + self.column.removeprefix("y.")

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """Event probability, or the predicted value."""
        X = rows[self.features]
        if TARGETS[self.target].kind == "event":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)


def fit(rows: pd.DataFrame, target: str, horizon_s: int, params: dict[str, Any]) -> Forecaster:
    """Train on the rows whose label is known."""
    column = TARGETS[target].column(horizon_s)
    known = rows[column].notna()
    features = feature_columns(rows)
    model = _model(TARGETS[target].kind == "event", params)
    model.fit(rows.loc[known, features], rows.loc[known, column])
    return Forecaster(target, horizon_s, features, model)


def out_of_fold(
    rows: pd.DataFrame, target: str, horizon_s: int, params: dict[str, Any], k: int, seed: int
) -> np.ndarray:
    """Predictions for every row from a model that never saw that row's patient.

    Later models (the suggesters) train on these, so they learn how much to trust
    forecasts that are as wrong as they will be on new patients.
    """
    pred = np.full(len(rows), np.nan)
    for train_idx, val_idx in splits.folds(rows, k, seed):
        f = fit(rows.iloc[train_idx], target, horizon_s, params)
        pred[val_idx] = f.predict(rows.iloc[val_idx])
    return pred
