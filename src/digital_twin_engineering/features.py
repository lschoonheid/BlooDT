"""Forecaster inputs at decision points, from what was observed up to each point.

Every feature column starts with `x.`. Windows look back only, so a row never sees
the future. The same functions serve the live twin (on its recent buffer).
"""

import numpy as np
import pandas as pd

from .channels import CHANNELS, STALENESS
from .patients import Covariates


def window_features(
    frame: pd.DataFrame, windows_s: tuple[int, ...], step_s: float, at: pd.Index
) -> pd.DataFrame:
    """Rolling summaries of every frame column, taken at the decision points `at`.

    Level and setting channels get their last value plus mean, min, max, std and
    slope per window; expert signals (smooth by construction) only last value,
    mean and slope; expert lookaheads (`<expert>.<signal>@<h>`) their value and the
    change from the signal's current value, and what-ifs (`...@<h>.<drug>_x<f>`)
    their effect: the difference from keeping that drug's current target (`_x1`),
    whose own lookahead counts as a plain one; dose
    channels the amount given per window; staleness columns only their current value.
    """
    out = {}
    for col in frame.columns:
        x = frame[col]
        if col.endswith(STALENESS):
            out[f"x.{col}"] = x
            continue
        if "@" in col:
            signal, ahead = col.split("@")
            horizon, _, scenario = ahead.partition(".")
            if scenario and not scenario.endswith("_x1"):
                drug = scenario.rsplit("_x", 1)[0]
                out[f"x.{signal}.effect@{ahead}"] = x - frame[f"{signal}@{horizon}.{drug}_x1"]
            else:
                out[f"x.{col}"] = x
                out[f"x.{signal}.change@{horizon}"] = x - frame[signal]
            continue
        channel = CHANNELS.get(col)
        if channel is not None and channel.kind == "dose":
            for w in windows_s:
                out[f"x.{col}.sum{w}"] = x.rolling(round(w / step_s), min_periods=1).sum()
            continue
        out[f"x.{col}"] = x
        for w in windows_s:
            n = round(w / step_s)
            r = x.rolling(n, min_periods=1)
            out[f"x.{col}.mean{w}"] = r.mean()
            if channel is not None:
                out[f"x.{col}.min{w}"] = r.min()
                out[f"x.{col}.max{w}"] = r.max()
                out[f"x.{col}.std{w}"] = r.std()
            out[f"x.{col}.slope{w}"] = (x - x.shift(n)) / w
    return pd.DataFrame(out, index=frame.index).loc[at]


def covariate_features(c: Covariates) -> dict[str, float]:
    return {
        "x.age": c.age,
        "x.male": float(c.is_male),
        "x.height": c.height,
        "x.weight": c.weight,
        "x.bmi": c.bmi,
        "x.asa": c.asa,
    }


def phase_features(t: np.ndarray, anestart: float, opstart: float, opend: float) -> dict:
    """Time into anaesthesia and whether surgery is under way (VitalDB's EVENT
    track announces surgery start and end live, so these are causal)."""
    return {
        "x.anaesthesia_s": t - anestart,
        "x.surgery": ((t >= opstart) & (t < opend)).astype(float),
    }
