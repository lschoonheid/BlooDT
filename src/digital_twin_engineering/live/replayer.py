"""Replays a VitalDB case's tracks onto MQTT as if live (outline).

Each raw record is published at its own time (not on the grid), scaled by `speed`,
with timestamps shifted to a configurable start date, on topic
`twin/<caseid>/track/<device>/<track>` with payload {"t": epoch_s, "v": value}.
Publishing raw records keeps the twin's view identical to the offline one:
it builds its own causal grid and staleness from what arrives.
"""

from datetime import datetime


def replay(caseid: int, broker: str, speed: float = 1.0, start: datetime | None = None) -> None:
    raise NotImplementedError
