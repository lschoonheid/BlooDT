"""Forecast intervals: quantile models, their evaluation, and the live service's fallback."""

import numpy as np
import pandas as pd

from digital_twin_engineering import evaluation, forecasting
from digital_twin_engineering.live import service

PARAMS = {"n_estimators": 100, "num_leaves": 15, "n_jobs": 2}


def rows(n: int, seed: int) -> pd.DataFrame:
    """BIS 5 min ahead: the current value plus noise whose spread grows with x.noisy."""
    rng = np.random.default_rng(seed)
    current = rng.uniform(30, 70, n)
    noisy = rng.uniform(0, 1, n)
    future = current + rng.normal(0, 1 + 9 * noisy)
    return pd.DataFrame(
        {"x.bis": current, "x.noisy": noisy, "y.bis@300": future, "subjectid": rng.integers(0, 50, n)}
    )


def test_interval_covers_and_follows_noise():
    f = forecasting.fit(rows(8000, 0), "bis", 300, PARAMS, interval=0.9)
    test = rows(4000, 1)
    lo, hi = f.interval(test)
    y = test["y.bis@300"].to_numpy()
    assert 0.85 <= np.mean((lo <= y) & (y <= hi)) <= 0.95
    point = f.predict(test)
    assert np.all((lo <= point) & (point <= hi))
    width = hi - lo
    quiet, loud = test["x.noisy"] < 0.2, test["x.noisy"] > 0.8
    assert width[loud].mean() > 3 * width[quiet].mean()

    m = evaluation.forecaster(f, test)["interval"]
    assert m["level"] == 0.9 and 0.85 <= m["coverage"] <= 0.95
    assert m["mean_width"] > 0 and 0 < m["rmse_band_coverage"] <= 1


def test_no_interval_models():
    f = forecasting.fit(rows(500, 0), "bis", 300, PARAMS)
    assert f.bounds is None and f.interval(rows(10, 1)) is None
    assert "interval" not in evaluation.forecaster(f, rows(500, 1))
    # events get no interval even when asked for
    events = rows(500, 0).assign(**{"y.hypotension@300": lambda r: (r["y.bis@300"] > 50).astype(float)})
    assert forecasting.fit(events, "hypotension", 300, PARAMS, interval=0.9).bounds is None


def test_service_interval_prefers_model_then_rmse():
    svc = service.TwinService.__new__(service.TwinService)
    svc.rmse = {("bis", 300): 5.0}
    model = {"bis@300": 40.0, "bis_lo@300": 30.0, "bis_hi@300": 55.0}
    assert svc._interval(model, "bis", 300) == (30.0, 55.0)
    half = service.INTERVAL_Z * 5.0
    assert svc._interval({"bis@300": 40.0}, "bis", 300) == (40.0 - half, 40.0 + half)
    assert svc._interval({"bis@60": 40.0}, "bis", 60) is None
