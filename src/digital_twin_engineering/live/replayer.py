"""Replays a VitalDB case's raw records into a twin as if live.

`replay` drives a Twin in-process, as fast as it can: each second of case time it
delivers every record up to that second, then ticks. The live layer's replayer
does the same over MQTT (outline): each record published at its own time scaled
by a speed factor, on topic `twin/<caseid>/track/<device>/<track>` with payload
{"t": epoch_s, "v": value}. Raw records rather than grid values keep the twin's
view identical to the offline one: it builds its own causal grid and staleness.
"""

import heapq
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .. import vitaldb_cache as vdb
from ..channels import CHANNELS
from ..config import Config
from ..patients import clinical, covariates
from ..pipeline import MODELS_DIR
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


def make_twin(caseid: int, config: Config, models: dict, tracks: list[str]) -> Twin:
    row = clinical().loc[caseid]
    phases = Phases(row.anestart, row.opstart, row.opend)
    return Twin(config, models, covariates(caseid), tracks, phases)


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


def replay_mqtt(caseid: int, broker: str, speed: float = 1.0) -> None:
    raise NotImplementedError("the MQTT replayer is part of the live layer outline")
