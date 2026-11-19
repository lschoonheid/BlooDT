"""The twin service: subscribes to a replayed case, publishes its state (outline).

Every grid step the twin
1. carries each channel's last received value forward and updates its staleness
   (channels.to_grid semantics, one step at a time),
2. steps every expert with this step's doses,
3. at each decision point, computes features over its rolling buffer
   (features.window_features), runs the forecasters and suggesters,
4. writes channels, expert signals, forecasts and suggestions to InfluxDB.
"""

from collections.abc import Mapping

from ..config import Config
from ..patients import Covariates


class Twin:
    def __init__(self, config: Config, models_path: str, covariates: Covariates):
        raise NotImplementedError

    def observe(self, track: str, t: float, value: float) -> None:
        """A raw record arrived from the replayer."""
        raise NotImplementedError

    def tick(self, t: float) -> Mapping[str, float]:
        """Advance to grid time `t`; returns everything to write to InfluxDB."""
        raise NotImplementedError
