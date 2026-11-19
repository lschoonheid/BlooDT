"""Labelling what happened in a case: hypotension events and interventions.

Labels look at the whole case, including the future, so they are for training and
evaluation only; nothing here may be used as a forecaster input.
"""

import numpy as np
import pandas as pd


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs of the True runs in a boolean array."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def hypotension_mask(
    map_: pd.Series,
    threshold: float = 65,
    min_duration_s: float = 60,
    smooth_s: float = 5,
    step_s: float = 1,
) -> pd.Series:
    """True while a hypotension event is ongoing.

    MAP is median-smoothed over `smooth_s` to ignore single-sample spikes (as in
    STEP-OP), then runs below `threshold` lasting at least `min_duration_s` count.
    Missing MAP never counts as low.
    """
    smoothed = map_.rolling(max(1, round(smooth_s / step_s)), center=True, min_periods=1).median()
    below = (smoothed < threshold).to_numpy()
    mask = np.zeros(len(below), bool)
    for start, stop in _runs(below):
        if (stop - start) * step_s >= min_duration_s:
            mask[start:stop] = True
    return pd.Series(mask, index=map_.index)


def detect_interventions(
    setting: pd.Series,
    min_hold_s: float = 30,
    min_relative_change: float = 0.05,
    step_s: float = 1,
) -> pd.DataFrame:
    """Sustained changes of a pump setting (a TCI target or a rate-mode pump's rate).

    The setting must hold its new value for `min_hold_s` (or until the case ends);
    shorter values, such as the knob passing through 3.5 on the way from 3 to 4,
    are skipped. Changes smaller than `min_relative_change` of the old value are
    ignored. A missing setting means the pump is off (0).

    Returns one row per intervention: t (start of the new value), old, new, direction.
    """
    v = setting.fillna(0).to_numpy()
    t = setting.index.to_numpy()
    starts = np.flatnonzero(np.diff(v, prepend=np.nan) != 0)
    stops = np.append(starts[1:], len(v))
    rows = []
    current = v[0]
    for start, stop in zip(starts[1:], stops[1:]):
        held = (stop - start) * step_s >= min_hold_s or stop == len(v)
        new = v[start]
        if held and abs(new - current) > min_relative_change * max(abs(current), 1e-9):
            rows.append((t[start], current, new, "increase" if new > current else "decrease"))
            current = new
    return pd.DataFrame(rows, columns=["t", "old", "new", "direction"])
