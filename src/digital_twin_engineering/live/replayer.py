"""Replays a VitalDB case's raw records into a twin as if live.

`replay` drives a Twin in-process, as fast as it can: each second of case time it
delivers every record up to that second, then ticks. `replay_mqtt` publishes the
same stream to an MQTT broker (see messages.py) for the twin service, each second
of case time at its own time divided by a speed factor.
"""

import heapq
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .. import vitaldb_cache as vdb
from ..channels import CHANNELS
from ..config import Config
from ..patients import clinical, covariates
from ..pipeline import MODELS_DIR
from . import messages
from .twin import Phases, Twin


def load_run(run: str) -> tuple[Config, dict]:
    """A training run's config and models."""
    out = MODELS_DIR / run
    config = Config.from_dict(json.loads((out / "config.json").read_text())["config"])
    return config, joblib.load(out / "models.joblib")


def records(caseid: int, channels: list[str]) -> tuple[list[str], list[tuple[float, str, float]], float]:
    """The case's available tracks, every record of the channels' tracks as
    (t from case start, track, value) in time order, and the case's duration."""
    tracks = sorted({t for n in channels for t, _ in CHANNELS[n].tracks})
    vf = vdb.load_vital(caseid, tracks)
    available = list(vf.get_track_names())
    streams = []
    for track in available:
        recs = vf.find_track(track).recs
        t = np.array([r["dt"] for r in recs], float) - vf.dtstart
        v = np.array([r["val"] for r in recs], float)
        order = np.argsort(t, kind="stable")
        streams.append([(float(t[i]), track, float(v[i])) for i in order])
    return available, list(heapq.merge(*streams, key=lambda r: r[0])), vf.dtend - vf.dtstart


def phases(caseid: int) -> dict[str, float]:
    row = clinical().loc[caseid]
    return {name: float(row[name]) for name in messages.PHASES}


def make_twin(caseid: int, config: Config, models: dict, tracks: list[str]) -> Twin:
    return Twin(config, models, covariates(caseid), tracks, Phases(**phases(caseid)))


def replay(caseid: int, config: Config, models: dict, until: float | None = None) -> pd.DataFrame:
    """The twin's decisions over a case, one row per decision point."""
    tracks, recs, duration = records(caseid, list(config.all_channels))
    twin = make_twin(caseid, config, models, tracks)
    decisions, i = [], 0
    for t in np.arange(0, duration if until is None else until, config.step_s):
        while i < len(recs) and recs[i][0] <= t:
            twin.observe(recs[i][1], recs[i][0], recs[i][2])
            i += 1
        decision = twin.tick(t)
        if decision is not None:
            decisions.append(decision)
    return pd.DataFrame(decisions).set_index("t")


def stream(
    caseid: int, config: Config, start: float, until: float | None = None
) -> list[tuple[float, str, dict]]:
    """The case's messages for the twin service (see messages.stream), up to case
    time `until` if given."""
    tracks, recs, duration = records(caseid, list(config.all_channels))
    duration = duration if until is None else min(duration, until)
    return list(
        messages.stream(caseid, tracks, recs, duration, covariates(caseid), phases(caseid), start, config.step_s)
    )


def replay_mqtt(
    caseid: int,
    config: Config,
    host: str,
    port: int = 1883,
    speed: float = 30.0,
    start: float | None = None,
    until: float | None = None,
) -> float:
    """Publish a case to an MQTT broker, `speed` times faster than real time (inf:
    as fast as possible). `start` is the epoch time of case time 0 (default now,
    whole seconds). Returns `start`."""
    import paho.mqtt.client as mqtt

    start = math.floor(time.time()) if start is None else start
    msgs = stream(caseid, config, start, until)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"replayer-{caseid}")
    client.connect(host, port)
    client.loop_start()
    began = time.monotonic()
    for t, topic, payload in msgs:
        wait = began + t / speed - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        client.publish(topic, messages.encode(payload), qos=1, retain=topic.endswith("/meta"))
    # clear the retained meta; QoS 1 messages go out in order, so this one last
    client.publish(messages.topic(caseid, "meta"), b"", qos=1, retain=True).wait_for_publish()
    client.loop_stop()
    client.disconnect()
    return start
