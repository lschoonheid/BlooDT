"""Forecasters: one LightGBM model per (target, horizon).

Inputs are every `x.` column of the dataset rows (channels, expert
signals, covariates, phase); LightGBM handles the NaNs that missing channels and
inapplicable experts leave. Another model family can replace `_model` as long as
it keeps fit/predict(_proba).

A value forecaster can also carry two quantile models, the bounds of a central
prediction interval (`interval`): its width follows the inputs, so it widens where
the forecasts tend to be wrong. Runs trained before intervals have none.
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
    # value targets: the interval's lower and upper quantile models, and its coverage
    bounds: tuple[Any, Any] | None = None
    interval_level: float | None = None

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
        return self._predict(rows[self.features].to_numpy(np.float32))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if TARGETS[self.target].kind == "event":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def interval(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
        """Lower and upper bounds of the prediction interval, or None without
        quantile models. Sorted, and widened to contain the forecast itself, as
        separately fitted quantiles can cross."""
        if self.bounds is None:
            return None
        X = rows[self.features].to_numpy(np.float32)
        point = self._predict(X)
        lo, hi = (m.predict(X) for m in self.bounds)
        return np.minimum.reduce([lo, hi, point]), np.maximum.reduce([lo, hi, point])


def _fit(X: np.ndarray, y: np.ndarray, use: np.ndarray, target: str, params: dict[str, Any]):
    """A model trained on rows `use` (indices) of X whose label is known."""
    use = use[np.isfinite(y[use])]
    model = _model(TARGETS[target].kind == "event", params)
    model.fit(X[use], y[use])
    return model


def _fit_bounds(X: np.ndarray, y: np.ndarray, use: np.ndarray, level: float, params: dict[str, Any]):
    """Quantile models of the central `level` interval, trained on rows `use`."""
    use = use[np.isfinite(y[use])]
    tail = (1 - level) / 2
    bounds = []
    for alpha in (tail, 1 - tail):
        model = _model(False, params | {"objective": "quantile", "alpha": alpha})
        model.fit(X[use], y[use])
        bounds.append(model)
    return tuple(bounds)


def _final(features, X, y, target, horizon_s, params, interval) -> Forecaster:
    """A forecaster on every row with a known label, with interval models if asked
    for and the target is a value."""
    everything = np.arange(len(y))
    model = _fit(X, y, everything, target, params)
    if interval is None or TARGETS[target].kind == "event":
        return Forecaster(target, horizon_s, features, model)
    bounds = _fit_bounds(X, y, everything, interval, params)
    return Forecaster(target, horizon_s, features, model, bounds, interval)


def _arrays(
    rows: pd.DataFrame, target: str, horizon_s: int, features: list[str] | None = None
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Feature names, a float32 feature matrix (one copy, reused across folds) and labels."""
    features = features if features is not None else feature_columns(rows)
    X = rows[features].to_numpy(np.float32)
    y = rows[TARGETS[target].column(horizon_s)].to_numpy(np.float64)
    return features, X, y


def fit(
    rows: pd.DataFrame,
    target: str,
    horizon_s: int,
    params: dict[str, Any],
    features: list[str] | None = None,
    interval: float | None = None,
) -> Forecaster:
    """Train on the rows whose label is known, on `features` (default: every `x.` column);
    with `interval`, also the quantile models of a central interval of that coverage."""
    features, X, y = _arrays(rows, target, horizon_s, features)
    return _final(features, X, y, target, horizon_s, params, interval)


def fit_with_out_of_fold(
    rows: pd.DataFrame,
    target: str,
    horizon_s: int,
    params: dict[str, Any],
    k: int,
    seed: int,
    interval: float | None = None,
) -> tuple[Forecaster, np.ndarray]:
    """A forecaster on all rows, plus predictions for every row from a model that
    never saw that row's patient.

    Later models (the suggesters) train on the out-of-fold predictions, so they
    learn how much to trust forecasts that are as wrong as they will be on new
    patients. The interval models (`interval`) are trained on all rows only.
    """
    features, X, y = _arrays(rows, target, horizon_s)
    pred = np.full(len(rows), np.nan)
    for train_idx, val_idx in splits.folds(rows, k, seed):
        model = _fit(X, y, train_idx, target, params)
        f = Forecaster(target, horizon_s, features, model)
        pred[val_idx] = f._predict(X[val_idx])
    return _final(features, X, y, target, horizon_s, params, interval), pred
