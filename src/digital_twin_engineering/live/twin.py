"""The twin: follows one case live, from raw records to forecasts and suggestions.

Every grid step the twin
1. carries each channel's last received value forward and updates its staleness
   (channels.to_grid semantics, one step at a time; pump volumes go through the
   same DoseCounter as offline),
2. steps every expert with this step's doses (experts.Runner, as offline),
3. at each decision point, computes features over its rolling buffer
   (features.window_features), runs the forecasters and then the suggesters.

So at a decision point it sees exactly the features the offline rows hold for that
point, except shortly after a pump volume jump, which the twin can only judge a few
readings later (see channels.DoseCounter). `tick` returns the decision; the twin
service (service.py) writes it to InfluxDB.
"""

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import experts
from ..channels import CHANNELS, STALENESS, DoseCounter, resolve
from ..config import Config
from ..experts.anesthesim import TCI_DRUGS
from ..features import covariate_features, phase_features, window_features
from ..patients import Covariates


@dataclass(frozen=True)
class Phases:
    """Case times (s from case start), announced live by VitalDB's EVENT track; one
    not announced yet is NaN (anestart) or inf (opstart, opend)."""

    anestart: float
    opstart: float
    opend: float


class Twin:
    def __init__(
        self,
        config: Config,
        models: Mapping,
        covariates: Covariates,
        tracks: Iterable[str],
        phases: Phases,
    ):
        """`models` as saved by a training run ({"forecasters": [...], "suggesters":
        [...]}); `tracks` are the case's available tracks (a channel uses the first
        of its tracks the case has, as offline)."""
        self.config = config
        self.forecasters = models["forecasters"]
        self.suggesters = models["suggesters"]
        self.covariates = covariates
        self.phases = phases
        names = list(config.all_channels)
        self.channels = names
        self.track_of = resolve(names, set(tracks))
        self.channel_of = {track: name for name, track in self.track_of.items() if track}
        self.last: dict[str, tuple[float, float]] = {}  # level/setting -> (t, value)
        self.counters = {n: DoseCounter() for n in names if CHANNELS[n].kind == "dose"}
        self.dose_seen: dict[str, float] = {}  # dose -> time of the last resolved reading
        self.dose_total = dict.fromkeys(self.counters, np.nan)
        self.runners = [
            experts.Runner(experts.build(spec), covariates, config.step_s, config.lookahead())
            for spec in config.experts
        ]
        longest = round(max(config.features.windows_s) / config.step_s)
        self.buffer: deque[dict[str, float]] = deque(maxlen=longest + 1)
        self.t: float | None = None

    def observe(self, track: str, t: float, value: float) -> None:
        """A raw record arrived (time in s from case start)."""
        name = self.channel_of.get(track)
        if name is None:
            return
        channel = CHANNELS[name]
        factor = dict(channel.tracks)[track]
        if channel.kind == "dose":
            for t_resolved, _, _ in self.counters[name].add(t, value):
                self.dose_seen[name] = t_resolved
            return
        v = value * factor
        if np.isfinite(v) and channel.valid[0] <= v <= channel.valid[1]:
            self.last[name] = (t, v)

    def tick(self, t: float) -> dict | None:
        """Advance to grid time `t` (every record up to `t` observed). Returns the
        decision at decision points, else None."""
        self.t = t
        row = self._grid_row(t)
        for runner in self.runners:
            for k, v in runner.step(t, row).items():
                row[f"{runner.expert.name}.{k}"] = v
        self.buffer.append(row)
        if round(t) % self.config.decision_every_s != 0:
            return None
        return self.decide()

    def _grid_row(self, t: float) -> dict[str, float]:
        row = {}
        for name in self.channels:
            channel = CHANNELS[name]
            if name in self.counters:
                track = self.track_of[name]
                if track is None:  # no such pump in this case
                    row[name] = row[name + STALENESS] = np.nan
                elif name not in self.dose_seen:  # pump present, nothing counted yet
                    row[name], row[name + STALENESS] = 0.0, np.nan
                else:
                    total = self.counters[name].total * dict(channel.tracks)[track]
                    previous = self.dose_total[name]
                    row[name] = total - previous if np.isfinite(previous) else 0.0
                    row[name + STALENESS] = t - self.dose_seen[name]
                    self.dose_total[name] = total
                continue
            if name not in self.last:
                row[name] = row[name + STALENESS] = np.nan
                continue
            t_obs, v = self.last[name]
            staleness = t - t_obs
            row[name] = v if staleness <= channel.max_staleness_s else np.nan
            row[name + STALENESS] = staleness
        return row

    def frame(self) -> pd.DataFrame:
        """The rolling buffer as a grid frame, like the offline case frame's tail."""
        index = pd.Index(
            self.t - self.config.step_s * np.arange(len(self.buffer))[::-1], name="t"
        )
        return pd.DataFrame(list(self.buffer), index=index)

    def features(self) -> pd.DataFrame:
        """One row of forecaster inputs for the current time."""
        frame = self.frame()
        at = frame.index[-1:]
        x = window_features(frame, self.config.features.windows_s, self.config.step_s, at)
        extra = covariate_features(self.covariates)
        p = self.phases
        extra |= phase_features(at.to_numpy(), p.anestart, p.opstart, p.opend)
        return x.astype("float32").assign(**extra)

    def decide(self) -> dict:
        """Forecasts and suggestions for the current time."""
        x = self.features()
        out: dict = {"t": self.t}
        for f in self.forecasters:
            value = float(f.predict(x)[0])
            x[f.output] = value
            out[f.column.removeprefix("y.")] = value
        for s in self.suggesters:
            suggestion = s.suggest(x).iloc[0]
            for k, v in suggestion.items():
                out[f"suggest.{s.drug}.{k}"] = float(v)
            if s.drug not in TCI_DRUGS:
                continue
            # what the first expert able to simulate it expects each suggested change to do
            current = float(x[f"x.{s.setting}"].iloc[0])
            for runner in self.runners:
                effects = {
                    d: runner.effect(s.drug, suggestion[f"setting_{d}"], current)
                    for d in ("decrease", "increase")
                    if np.isfinite(suggestion[f"setting_{d}"])
                }
                if any(e is not None for e in effects.values()):
                    for d, effect in effects.items():
                        for k, v in (effect or {}).items():
                            out[f"suggest.{s.drug}.effect_{d}.{k}"] = v
                    break
        self.last_features = x
        return out
