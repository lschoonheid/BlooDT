"""Targets: what forecasters predict. Add a vital by adding an entry to TARGETS.

A "value" target is a channel's observed value one horizon ahead; an "event" target
is whether an event (a boolean mask over the case) is ongoing at any time within the
next horizon. Each (target, horizon) pair gets its own forecaster.

An event target may say when the event is already under way judging by the current
value alone (`ongoing`); evaluation then also scores the forecaster on onsets, the
decision points where it isn't, since "MAP is low now" already predicts most of
"MAP will be low soon".
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .channels import STALENESS
from .labels import hypotension_mask


@dataclass(frozen=True)
class Target:
    name: str
    channel: str
    kind: Literal["value", "event"]
    event: Callable[[pd.Series], pd.Series] | None = None  # for "event": channel -> mask
    max_staleness_s: float = 10  # "value": the future observation must be this fresh
    ongoing: Callable[[np.ndarray], np.ndarray] | None = None  # "event": current value -> under way
    smooth_s: float | None = None  # "value": the median of fresh values over this window around t + h
    persistence: str | None = None  # the baseline's feature column (default: x.<channel>)

    def label(self, frame: pd.DataFrame, horizon_s: float, step_s: float = 1) -> pd.Series:
        """The label at every grid point of a case frame (NaN where unknown)."""
        n = round(horizon_s / step_s)
        x = frame[self.channel]
        if self.kind == "value":
            fresh = frame[self.channel + STALENESS] <= self.max_staleness_s
            if self.smooth_s:
                m = round(self.smooth_s / step_s)
                x = x.where(fresh).rolling(m, center=True, min_periods=m // 2).median()
                fresh = x.notna()
            return x.shift(-n).where(fresh.shift(-n, fill_value=False))
        mask = self.event(x).astype(float)
        # any event in (t, t + horizon]: rolling max over the reversed, shifted series
        ahead = mask[::-1].rolling(n, min_periods=1).max()[::-1].shift(-1)
        observed = x.notna().astype(float)[::-1].rolling(n, min_periods=1).max()[::-1].shift(-1)
        return ahead.where(observed > 0)

    @property
    def persistence_column(self) -> str:
        return self.persistence or f"x.{self.channel}"

    def column(self, horizon_s: float) -> str:
        return f"y.{self.name}@{int(horizon_s)}"


def _below_65(map_: np.ndarray) -> np.ndarray:
    return map_ < 65


TARGETS = {
    t.name: t
    for t in [
        Target("map", "map", "value"),
        # MAP over the minute around the horizon: beat-to-beat swings and short
        # artefacts left out, as in the hypotension event definition
        Target("map_1min", "map", "value", smooth_s=60, persistence="x.map.mean60"),
        Target("bis", "bis", "value"),
        Target("hr", "hr", "value"),
        Target("hypotension", "map", "event", event=hypotension_mask, ongoing=_below_65),
    ]
}


def is_classification(name: str) -> bool:
    return TARGETS[name].kind == "event"

