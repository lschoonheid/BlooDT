"""Channels: logical signals in canonical units, put on a causal grid per case.

A channel lists candidate tracks in priority order; a case uses the first one it has.
On the grid, each cell holds the last value observed at or before that second (never
interpolated) and a `<channel>__staleness` column says how old that observation is.
Values outside a channel's valid range are dropped first, so artefacts show up as
growing staleness. See docs/adr/0002-causal-grid-with-staleness.md.

Kinds:
- "level": a measured vital, e.g. MAP. Subject to `Perturbation` during training.
- "setting": a pump setting the clinician controls, e.g. a TCI target concentration.
- "dose": a cumulative infused volume, turned into the amount delivered per grid step.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from . import vitaldb_cache as vdb

STALENESS = "__staleness"
_KPA_TO_VOL_PERCENT = 100 / 101.325


@dataclass(frozen=True)
class Channel:
    name: str
    kind: Literal["level", "setting", "dose"]
    unit: str
    tracks: tuple[tuple[str, float], ...]  # (track, factor to `unit`), highest priority first
    valid: tuple[float, float] = (-np.inf, np.inf)
    max_staleness_s: float = np.inf  # older observations become NaN (staleness is kept)


def _level(name, unit, tracks, valid, max_staleness_s=300.0):
    return Channel(name, "level", unit, tuple((t, 1.0) for t in tracks), valid, max_staleness_s)


CHANNELS = {
    ch.name: ch
    for ch in [
        _level("map", "mmHg", ["Solar8000/ART_MBP", "EV1000/ART_MBP", "Solar8000/FEM_MBP"], (20, 200)),
        _level("sbp", "mmHg", ["Solar8000/ART_SBP", "EV1000/ART_SBP"], (30, 300)),
        _level("dbp", "mmHg", ["Solar8000/ART_DBP", "EV1000/ART_DBP"], (10, 200)),
        _level("nibp_map", "mmHg", ["Solar8000/NIBP_MBP"], (20, 200), max_staleness_s=900),
        _level("hr", "/min", ["Solar8000/HR", "Solar8000/PLETH_HR"], (20, 250)),
        _level("bis", "", ["BIS/BIS"], (1, 100)),
        _level("sqi", "%", ["BIS/SQI"], (0, 100)),
        _level("spo2", "%", ["Solar8000/PLETH_SPO2"], (50, 100)),
        _level("etco2", "mmHg", ["Solar8000/ETCO2", "Primus/ETCO2"], (5, 100)),
        _level("mac", "", ["Primus/MAC"], (0, 5)),
        Channel(
            "sevoflurane_et", "level", "vol%",
            (("Primus/EXP_SEVO", _KPA_TO_VOL_PERCENT),), (0, 10), 300,
        ),
        Channel(
            "desflurane_et", "level", "vol%",
            (("Primus/EXP_DES", _KPA_TO_VOL_PERCENT),), (0, 20), 300,
        ),
        # Pump doses: syringe concentrations are fixed per VitalDB track name.
        Channel("propofol_dose", "dose", "mg", (("Orchestra/PPF20_VOL", 20.0),)),
        Channel(
            "remifentanil_dose", "dose", "ug",
            (("Orchestra/RFTN20_VOL", 20.0), ("Orchestra/RFTN50_VOL", 50.0)),
        ),
        Channel("norepinephrine_dose", "dose", "ug", (("Orchestra/NEPI_VOL", 20.0),)),
        Channel("epinephrine_dose", "dose", "ug", (("Orchestra/EPI_VOL", 20.0),)),
        Channel("phenylephrine_dose", "dose", "ug", (("Orchestra/PHEN_VOL", 100.0),)),
        # Settings the clinician controls (see docs/adr/0004-...).
        Channel("propofol_target", "setting", "ug/mL", (("Orchestra/PPF20_CT", 1.0),), (0, 20)),
        Channel(
            "remifentanil_target", "setting", "ng/mL",
            (("Orchestra/RFTN20_CT", 1.0), ("Orchestra/RFTN50_CT", 1.0)), (0, 50),
        ),
        # mL/h of 20 mcg/mL (or 100 mcg/mL) -> mcg/min
        Channel("norepinephrine_rate", "setting", "ug/min", (("Orchestra/NEPI_RATE", 20 / 60),), (0, 1000)),
        Channel("phenylephrine_rate", "setting", "ug/min", (("Orchestra/PHEN_RATE", 100 / 60),), (0, 5000)),
        # The pump's own TCI model, for verifying the experts.
        Channel("propofol_ce_pump", "level", "ug/mL", (("Orchestra/PPF20_CE", 1.0),), (0, 20), np.inf),
        Channel(
            "remifentanil_ce_pump", "level", "ng/mL",
            (("Orchestra/RFTN20_CE", 1.0), ("Orchestra/RFTN50_CE", 1.0)), (0, 50), np.inf,
        ),
    ]
}


@dataclass(frozen=True)
class Perturbation:
    """Random sensor faults applied to level channels' raw records, for training.

    Each level channel in a case gets a fixed extra delay drawn from [0, max_delay_s]
    and loses Poisson(dropouts_per_hour) spans of exponentially distributed length.
    """

    max_delay_s: float = 0.0
    dropouts_per_hour: float = 0.0
    mean_dropout_s: float = 30.0

    def apply(self, t: np.ndarray, v: np.ndarray, rng: np.random.Generator):
        t = t + rng.uniform(0, self.max_delay_s)
        if self.dropouts_per_hour > 0 and len(t):
            hours = (t[-1] - t[0]) / 3600
            keep = np.ones(len(t), bool)
            for _ in range(rng.poisson(self.dropouts_per_hour * hours)):
                start = rng.uniform(t[0], t[-1])
                keep &= ~((t >= start) & (t < start + rng.exponential(self.mean_dropout_s)))
            t, v = t[keep], v[keep]
        return t, v


def cumulative_dose(v: np.ndarray) -> np.ndarray:
    """Total volume infused so far, from a pump's VOL readings.

    The readings jitter downwards by a fraction of a mL and reset to zero when a
    syringe or the pump is restarted. A drop of more than half is taken as a reset;
    within a segment the volume can only grow.
    """
    if len(v) == 0:
        return v
    reset = np.flatnonzero(v[1:] < 0.5 * v[:-1]) + 1
    total = np.empty_like(v)
    offset = 0.0
    for seg in np.split(np.arange(len(v)), reset):
        total[seg] = offset + np.maximum.accumulate(v[seg])
        offset = total[seg[-1]]
    return total


def resolve(names: list[str], available: set[str]) -> dict[str, str | None]:
    """The track each channel uses in a case with `available` tracks."""
    return {
        n: next((t for t, _ in CHANNELS[n].tracks if t in available), None) for n in names
    }


def to_grid(
    channel: Channel, t: np.ndarray, v: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Values and staleness of one channel's records (times `t`, sorted) on `grid`."""
    if channel.kind == "dose":
        v = cumulative_dose(v)
    else:
        ok = np.isfinite(v) & (v >= channel.valid[0]) & (v <= channel.valid[1])
        t, v = t[ok], v[ok]
    idx = np.searchsorted(t, grid, side="right") - 1
    seen = idx >= 0
    values = np.full(len(grid), np.nan)
    staleness = np.full(len(grid), np.nan)
    values[seen] = v[idx[seen]]
    staleness[seen] = grid[seen] - t[idx[seen]]
    values[staleness > channel.max_staleness_s] = np.nan
    if channel.kind == "dose":
        # amount delivered during each step; nothing known before the first reading
        total = np.where(seen, values, v[0] if len(v) else np.nan)
        values = np.diff(total, prepend=total[0])
    return values, staleness


def load_channels(
    caseid: int,
    names: list[str],
    step_s: float = 1.0,
    perturbation: Perturbation | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """A case's channels on a `step_s` grid from case start (index `t`, seconds).

    Columns: each channel in its canonical unit, plus `<channel>__staleness`.
    Channels the case lacks are all-NaN.
    """
    rng = np.random.default_rng(seed)
    tracks = sorted({t for n in names for t, _ in CHANNELS[n].tracks})
    vf = vdb.load_vital(caseid, tracks)
    chosen = resolve(names, set(vf.get_track_names()))
    grid = np.arange(0, vf.dtend - vf.dtstart, step_s)
    cols = {}
    for name in names:
        channel, track = CHANNELS[name], chosen[name]
        if track is None:
            cols[name] = cols[name + STALENESS] = np.full(len(grid), np.nan)
            continue
        recs = vf.find_track(track).recs
        t = np.array([r["dt"] for r in recs], float) - vf.dtstart
        v = np.array([r["val"] for r in recs], float) * dict(channel.tracks)[track]
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
        if perturbation is not None and channel.kind == "level":
            t, v = perturbation.apply(t, v, rng)
        cols[name], cols[name + STALENESS] = to_grid(channel, t, v, grid)
    return pd.DataFrame(cols, index=pd.Index(grid, name="t"))
