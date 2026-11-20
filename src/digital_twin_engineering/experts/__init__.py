"""Expert models: mechanistic models stepped alongside a case.

Every expert advances one grid step at a time through `Expert.step`, so the offline
pipeline (`run`) and the live twin drive it the same way. Expert signals are
inputs to the forecasters, not competing predictions; see
docs/adr/0001-expert-models-feed-the-forecaster.md.

An expert may also implement `lookahead(held, horizons_s)` and
`lookahead_columns(horizons_s)`: its signals after holding one step's inputs for
each horizon (and possibly what-if scenarios), leaving its state untouched. `run`
then adds those columns (`<expert>.<signal>@<horizon>...`) at every decision point:
the expert's own forecast of what happens if the current infusion rates are kept.

Experts are built from config specs like `{"kind": "pas", "name": "pas_eleveld",
"model_propo": "Eleveld", ...}`; `kind` picks the class from `KINDS` and every other
key except `name` goes to its constructor.
"""

from collections import deque
from collections.abc import Mapping
from typing import Any, Protocol

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..patients import Covariates


class Expert(Protocol):
    name: str
    inputs: tuple[str, ...]  # channels read each step; "dose" channels give the amount per step
    signals: tuple[str, ...]  # what `step` returns

    def reset(self, covariates: Covariates, step_s: float) -> None:
        """Start a new case for this patient."""
        ...

    def step(self, inputs: Mapping[str, float]) -> Mapping[str, float]:
        """Advance one grid step with this step's inputs; returns the expert signals."""
        ...


@dataclass(frozen=True)
class Lookahead:
    """When and how far experts forecast: every `every_s` seconds of case time (the
    decision points), for each horizon, holding each dose input at its mean over the
    last `rate_window_s` (single steps are too jittery) and other inputs as they are."""

    horizons_s: tuple[int, ...]
    every_s: int
    rate_window_s: float = 60


def kinds() -> dict[str, type]:
    from .anesthesim import AnestheSimExpert
    from .pas import PasExpert

    return {"pas": PasExpert, "anesthesim": AnestheSimExpert}


def build(spec: Mapping[str, Any]) -> Expert:
    params = dict(spec)
    kind = params.pop("kind")
    return kinds()[kind](**params)


class Runner:
    """Drives one expert through a case a grid step at a time, as `run` does offline
    and the live twin does online: missing doses count as 0 (no pump means no
    drug), and at decision points experts that can look ahead do."""

    def __init__(
        self,
        expert: Expert,
        covariates: Covariates,
        step_s: float,
        lookahead: Lookahead | None = None,
    ):
        self.expert = expert
        self.step_s = step_s
        expert.reset(covariates, step_s)
        self.doses = [n for n in expert.inputs if n.endswith("_dose")]
        self.lookahead = lookahead if lookahead and hasattr(expert, "lookahead") else None
        self.ahead: dict[str, float] = {}
        if self.lookahead:
            self.ahead_names = expert.lookahead_columns(self.lookahead.horizons_s)
            self.ahead = dict.fromkeys(self.ahead_names, np.nan)
            window = max(1, round(self.lookahead.rate_window_s / step_s))
            self.recent = {n: deque(maxlen=window) for n in self.doses}

    @property
    def columns(self) -> list[str]:
        """The output columns, without the `<expert>.` prefix."""
        return list(self.expert.signals) + (self.ahead_names if self.lookahead else [])

    def step(self, t: float, inputs: Mapping[str, float]) -> dict[str, float]:
        """This step's signals (and lookaheads, carried forward between decision points)."""
        now = {n: inputs[n] for n in self.expert.inputs}
        for n in self.doses:
            if not now[n] == now[n]:  # NaN
                now[n] = 0.0
        y = self.expert.step(now)
        out = {s: y[s] for s in self.expert.signals}
        if self.lookahead:
            for n in self.doses:
                self.recent[n].append(now[n])
            self.held = now | {n: sum(r) / len(r) for n, r in self.recent.items()}
            if round(t) % self.lookahead.every_s == 0:
                self.ahead = self.expert.lookahead(self.held, self.lookahead.horizons_s)
            out |= {k: self.ahead[k] for k in self.ahead_names}
        return out


    def effect(self, drug: str, target: float, current: float) -> dict[str, float] | None:
        """What the expert expects a TCI target change from `current` to `target` to
        do: each lookahead signal with the drug steered to `target` minus steered to
        `current` (None if the expert can't simulate it). Both go through the TCI,
        so the pump's pausing or boluses after a recent change don't count."""
        if not self.lookahead or not hasattr(self.expert, "scenario") or not current > 0:
            return None
        horizons = self.lookahead.horizons_s
        changed = self.expert.scenario(self.held, horizons, drug, target)
        kept = self.expert.scenario(self.held, horizons, drug, current)
        return {k: v - kept[k] for k, v in changed.items()}


def run(
    experts: list[Expert],
    frame: pd.DataFrame,
    covariates: Covariates,
    step_s: float,
    lookahead: Lookahead | None = None,
) -> pd.DataFrame:
    """Step every expert through a case's channel frame.

    Returns one column per expert signal, named `<expert>.<signal>`, on the frame's
    index, plus lookahead columns for experts that support it (see `Runner`).
    """
    out = {}
    t = frame.index.to_numpy()
    for expert in experts:
        runner = Runner(expert, covariates, step_s, lookahead)
        columns = {n: frame[n].to_numpy() for n in expert.inputs}
        signals = np.full((len(frame), len(runner.columns)), np.nan)
        for i in range(len(frame)):
            y = runner.step(t[i], {n: v[i] for n, v in columns.items()})
            signals[i] = [y[c] for c in runner.columns]
        for j, c in enumerate(runner.columns):
            out[f"{expert.name}.{c}"] = signals[:, j]
    return pd.DataFrame(out, index=frame.index)
