"""Online calibration: fitting one model parameter to a patient as the case runs.

A population model's drug sensitivity is off for most patients (AnestheSIM's BIS
is 17% off on the TIVA cohort). `LogScaleFilter` tracks a multiplier s = exp(theta)
on such a parameter with a scalar extended Kalman filter: theta drifts slowly (a
random walk), and every fresh measurement pulls it towards the value that makes
the model match. Measurement errors of BIS are correlated over minutes, so the
noise is set well above the monitor's sample noise; otherwise the filter would
chase every disturbance (an incision) instead of the patient's sensitivity.
"""

import math
from collections.abc import Callable


class LogScaleFilter:
    def __init__(self, prior_sd: float, drift_sd_per_hour: float, noise_sd: float, step_s: float, bound: float = 1.5):
        self.theta = 0.0
        self.var = prior_sd**2
        self.q = drift_sd_per_hour**2 / 3600 * step_s
        self.r = noise_sd**2
        self.bound = bound

    @property
    def scale(self) -> float:
        return math.exp(self.theta)

    def step(self, measured: float | None, predict: Callable[[float], float]) -> None:
        """Advance one grid step; `predict(theta)` is the model's value of the
        measurement for a parameter theta, `measured` None if there is no fresh one."""
        self.var += self.q
        if measured is None:
            return
        y = predict(self.theta)
        gain = (predict(self.theta + 1e-3) - y) / 1e-3
        if gain == 0:
            return
        k = self.var * gain / (gain * gain * self.var + self.r)
        self.theta = min(max(self.theta + k * (measured - y), -self.bound), self.bound)
        self.var *= 1 - k * gain
