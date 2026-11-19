"""Expert models: mechanistic models stepped alongside a case.

Every expert advances one grid step at a time through `Expert.step`, so the offline
pipeline (`run`) and the live twin drive it the same way. Expert signals are
inputs to the forecasters, not competing predictions; see
docs/adr/0001-expert-models-feed-the-forecaster.md.

Experts are built from config specs like `{"kind": "pas", "name": "pas_eleveld",
"model_propo": "Eleveld", ...}`; `kind` picks the class from `KINDS` and every other
key except `name` goes to its constructor.
"""

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


def _kinds() -> dict[str, type]:
    from .anesthesim import AnestheSimExpert
    from .pas import PasExpert

    return {"pas": PasExpert, "anesthesim": AnestheSimExpert}


def build(spec: Mapping[str, Any]) -> Expert:
    params = dict(spec)
    kind = params.pop("kind")
    return _kinds()[kind](**params)


def run(
    experts: list[Expert], frame: pd.DataFrame, covariates: Covariates, step_s: float
) -> pd.DataFrame:
    """Step every expert through a case's channel frame.

    Returns one column per expert signal, named `<expert>.<signal>`, on the frame's
    index. Missing inputs are passed as 0 for doses (no pump means no drug) and NaN
    otherwise; experts decide what NaN means for them.
    """
    out = {}
    for expert in experts:
        expert.reset(covariates, step_s)
        inputs = {
            name: frame[name].fillna(0).to_numpy() if name.endswith("_dose") else frame[name].to_numpy()
            for name in expert.inputs
        }
        signals = np.full((len(frame), len(expert.signals)), np.nan)
        for i in range(len(frame)):
            y = expert.step({name: values[i] for name, values in inputs.items()})
            signals[i] = [y[s] for s in expert.signals]
        for j, s in enumerate(expert.signals):
            out[f"{expert.name}.{s}"] = signals[:, j]
    return pd.DataFrame(out, index=frame.index)
