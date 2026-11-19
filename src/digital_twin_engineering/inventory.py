"""Which tracks every case has, read from the .vital file headers once."""

from concurrent.futures import ProcessPoolExecutor
from functools import cache

import pandas as pd
import vitaldb

from . import vitaldb_cache as vdb
from .channels import CHANNELS
from .stages import DERIVED_DIR

PATH = DERIVED_DIR / "track_inventory.parquet"


def _tracks(caseid: int) -> list[tuple[int, str]]:
    vf = vitaldb.VitalFile(str(vdb.case_path(caseid)), header_only=True)
    return [(caseid, name) for name in vf.get_track_names()]


def build(jobs: int = 16) -> pd.DataFrame:
    """Scan every case and save the (caseid, track) table."""
    with ProcessPoolExecutor(jobs) as pool:
        rows = [row for rows in pool.map(_tracks, vdb.case_ids(), chunksize=32) for row in rows]
    df = pd.DataFrame(rows, columns=["caseid", "track"])
    PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PATH)
    load.cache_clear()
    return df


@cache
def load() -> pd.DataFrame:
    """The (caseid, track) table, building it first if needed (cached; don't mutate)."""
    return pd.read_parquet(PATH) if PATH.exists() else build()


def coverage(channels: list[str]) -> pd.DataFrame:
    """Boolean table: does case (row) have any track of channel (column)?"""
    inv = load()
    has = {}
    for name in channels:
        tracks = [t for t, _ in CHANNELS[name].tracks]
        has[name] = inv[inv.track.isin(tracks)].caseid.unique()
    index = pd.Index(sorted(inv.caseid.unique()), name="caseid")
    return pd.DataFrame({name: index.isin(ids) for name, ids in has.items()}, index=index)
