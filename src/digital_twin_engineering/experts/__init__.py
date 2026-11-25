"""Expert models: mechanistic models stepped alongside a case.

Every expert advances one grid step at a time through `Expert.step`, so the offline
pipeline (`run`) and the live twin drive it the same way. Expert signals are
inputs to the forecasters, not competing predictions; see
docs/adr/0001-expert-models-feed-the-forecaster.md.

An expert may also implement `scenario(held, horizons_s, drug, target)`: its
signals after holding one step's inputs for each horizon, with one drug steered
to a TCI target, leaving its state untouched. `Runner.effect` uses it for the
expected effect of a target change (a what-if), asked for only when needed.

Experts are built from config specs like `{"kind": "pas", "name": "pas_marsh",
"model_propo": "Marsh_modified", ...}`; `kind` picks the class from `KINDS` and every other
key except `name` goes to its constructor.
"""

from collections import deque
from collections.abc import Mapping
from typing import Any, Protocol

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


RATE_WINDOW_S = 60  # a what-if holds each dose at its mean over this long (single steps jitter)


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
    drug). For experts that can simulate a target change it keeps the inputs a
    what-if holds: this step's, with each dose at its mean over RATE_WINDOW_S."""

    def __init__(self, expert: Expert, covariates: Covariates, step_s: float):
        self.expert = expert
        self.step_s = step_s
        expert.reset(covariates, step_s)
        self.doses = [n for n in expert.inputs if n.endswith("_dose")]
        self.simulates = hasattr(expert, "scenario")
        if self.simulates:
            window = max(1, round(RATE_WINDOW_S / step_s))
            self.recent = {n: deque(maxlen=window) for n in self.doses}
            self.held: dict[str, float] = {}

    @property
    def columns(self) -> list[str]:
        """The output columns, without the `<expert>.` prefix."""
        return list(self.expert.signals)

    def step(self, inputs: Mapping[str, float]) -> dict[str, float]:
        """This step's signals."""
        now = {n: inputs[n] for n in self.expert.inputs}
        for n in self.doses:
            if not now[n] == now[n]:  # NaN
                now[n] = 0.0
        y = self.expert.step(now)
        if self.simulates:
            for n in self.doses:
                self.recent[n].append(now[n])
            self.held = now | {n: sum(r) / len(r) for n, r in self.recent.items()}
        return {s: y[s] for s in self.expert.signals}

    def effect(
        self, drug: str, target: float, current: float, horizons_s: tuple[int, ...]
    ) -> dict[str, float] | None:
        """What the expert expects a TCI target change from `current` to `target` to
        do: its signals `<signal>@<h>` with the drug steered to `target` minus
        steered to `current` (None if the expert can't simulate it). Both go through
        the TCI, so the pump's pausing or boluses after a recent change don't count."""
        if not self.simulates or not self.held or not current > 0:
            return None
        changed = self.expert.scenario(self.held, horizons_s, drug, target)
        kept = self.expert.scenario(self.held, horizons_s, drug, current)
        return {k: v - kept[k] for k, v in changed.items()}


def run(experts: list[Expert], frame: pd.DataFrame, covariates: Covariates, step_s: float) -> pd.DataFrame:
    """Step every expert through a case's channel frame: one column per expert
    signal, named `<expert>.<signal>`, on the frame's index."""
    out = {}
    for expert in experts:
        runner = Runner(expert, covariates, step_s)
        columns = {n: frame[n].to_numpy() for n in expert.inputs}
        signals = np.full((len(frame), len(runner.columns)), np.nan)
        for i in range(len(frame)):
            y = runner.step({n: v[i] for n, v in columns.items()})
            signals[i] = [y[c] for c in runner.columns]
        for j, c in enumerate(runner.columns):
            out[f"{expert.name}.{c}"] = signals[:, j]
    return pd.DataFrame(out, index=frame.index)
