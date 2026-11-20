"""What the replayer publishes and the twin service reads, over MQTT.

All topics are under `twin/<caseid>/`; payloads are JSON and times are epoch
seconds (the replayer's start date plus the time from case start):

- `meta` (retained while the case runs): {"start", "duration", "tracks",
  "covariates"}, everything the twin must know before the first record
- `track/<device>/<name>`: one raw record, {"t", "v"}
- `event`: a case phase announced live (VitalDB's EVENT track), {"t", "name"}
  with name one of anestart, opstart, opend
- `clock`: every record up to "t" has been published, {"t"}; the twin ticks
- `end`: the case is over

Raw records rather than grid values keep the twin's view identical to the offline
one: it builds its own causal grid and staleness.
"""

import json
from collections.abc import Iterator
from dataclasses import asdict

import numpy as np

from ..patients import Covariates

PREFIX = "twin"
PHASES = ("anestart", "opstart", "opend")


def topic(caseid: int, kind: str, track: str | None = None) -> str:
    return f"{PREFIX}/{caseid}/{kind}" + (f"/{track}" if track else "")


def parse(topic: str) -> tuple[int, str, str | None]:
    """(caseid, kind, track) of a topic."""
    _, caseid, kind, *track = topic.split("/", 3)
    return int(caseid), kind, track[0] if track else None


def encode(payload: dict) -> bytes:
    return json.dumps(payload, allow_nan=True).encode()


def decode(payload: bytes) -> dict:
    return json.loads(payload) if payload else {}


def stream(
    caseid: int,
    tracks: list[str],
    recs: list[tuple[float, str, float]],
    duration: float,
    covariates: Covariates,
    phases: dict[str, float],
    start: float,
    step_s: float = 1,
) -> Iterator[tuple[float, str, dict]]:
    """(time from case start, topic, payload) in publishing order: meta, then each
    second's events and records followed by its clock, then end."""
    meta = {"start": start, "duration": duration, "tracks": tracks, "covariates": asdict(covariates)}
    yield 0.0, topic(caseid, "meta"), meta
    events = sorted((t, name) for name, t in phases.items() if np.isfinite(t))
    i = j = 0
    for t in np.arange(0, duration, step_s):
        t = float(t)
        while j < len(events) and events[j][0] <= t:
            yield t, topic(caseid, "event"), {"t": start + events[j][0], "name": events[j][1]}
            j += 1
        while i < len(recs) and recs[i][0] <= t:
            record_t, track, value = recs[i]
            yield t, topic(caseid, "track", track), {"t": start + record_t, "v": value}
            i += 1
        yield t, topic(caseid, "clock"), {"t": start + t}
    yield duration, topic(caseid, "end"), {}
