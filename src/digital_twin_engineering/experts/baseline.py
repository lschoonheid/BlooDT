"""Pre-dose baselines: a patient's MAP and HR before any drug, as known live.

Mechanistic models describe drug effects relative to a patient's baseline MAP and
HR. AnestheSIM estimates them from the whole pre-induction recording, which the
twin can't see until induction is over. Here the baseline is the median of what was
observed so far, frozen at the first drug dose, so it is causal.

Before the first dose the arterial line is rarely in place (about 1 case in 10 in
the TIVA cohort, and then usually not yet zeroed), so MAP comes from the cuff
(NIBP), which about 3 cases in 4 have. HR comes from the ECG or pleth.
"""

import math
from collections.abc import Mapping

INPUTS = ("nibp_map", "hr")


class PreDoseBaseline:
    """Feed it every grid step's inputs; `frozen` turns True at the first dose."""

    def __init__(self) -> None:
        self.maps: list[float] = []
        self.hrs: list[float] = []
        self.frozen = False

    def update(self, inputs: Mapping[str, float]) -> bool:
        """Record this step; True on the step the baseline freezes."""
        if self.frozen:
            return False
        nibp, hr = inputs["nibp_map"], inputs["hr"]
        # NIBP is carried forward between cuff readings: count each reading once
        if math.isfinite(nibp) and (not self.maps or nibp != self.maps[-1]):
            self.maps.append(nibp)
        if math.isfinite(hr):
            self.hrs.append(hr)
        if any(v > 0 for k, v in inputs.items() if k.endswith("_dose")):
            self.frozen = True
            return True
        return False

    @property
    def map0(self) -> float | None:
        return _median(self.maps)

    @property
    def hr0(self) -> float | None:
        return _median(self.hrs)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
