"""What every case has: its tracks (from file headers) and the anaesthetics it
actually received (from the data). Both are scanned once and saved.

Having a track doesn't mean the drug was given: the anaesthesia machine logs
end-tidal sevoflurane as 0 in TIVA cases, and pumps can be connected unused.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import cache

import pandas as pd
import vitaldb

from . import vitaldb_cache as vdb
from .channels import CHANNELS, load_channels
from .stages import DERIVED_DIR

PATH = DERIVED_DIR / "track_inventory.parquet"
USAGE_PATH = DERIVED_DIR / "drug_usage.parquet"

# anaesthetic -> (channel, how much counts as given). Doses count in total units
# over the case; volatiles in minutes with end-tidal above VOLATILE_ON_VOL_PERCENT.
AGENTS = {
    "propofol": ("propofol_dose", 50.0),  # mg
    "remifentanil": ("remifentanil_dose", 50.0),  # ug
    "sevoflurane": ("sevoflurane_et", 10.0),  # min
    "desflurane": ("desflurane_et", 10.0),  # min
}
VOLATILE_ON_VOL_PERCENT = 0.5


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


def _usage(caseid: int) -> dict[str, float]:
    f = load_channels(caseid, [channel for channel, _ in AGENTS.values()])
    out = {"caseid": caseid}
    for agent, (channel, _) in AGENTS.items():
        x = f[channel]
        if CHANNELS[channel].kind == "dose":
            out[agent] = float(x.sum())
        else:
            out[agent] = float((x > VOLATILE_ON_VOL_PERCENT).sum()) / 60
    return out


def build_usage(jobs: int = 16) -> pd.DataFrame:
    """Measure how much of each anaesthetic every case received, and save it.

    Only cases with at least one anaesthetic track are read (~10 min for all).
    """
    tracks = [t for channel, _ in AGENTS.values() for t, _ in CHANNELS[channel].tracks]
    inv = load()
    ids = sorted(inv[inv.track.isin(tracks)].caseid.unique())
    with ProcessPoolExecutor(jobs) as pool:
        rows = list(pool.map(_usage, ids, chunksize=8))
    df = (
        pd.DataFrame(rows, columns=["caseid", *AGENTS])
        .set_index("caseid")
        .reindex(sorted(inv.caseid.unique()), fill_value=0.0)
    )
    df.to_parquet(USAGE_PATH)
    usage.cache_clear()
    return df


@cache
def usage() -> pd.DataFrame:
    """Amount of each anaesthetic per case (see AGENTS for units), building it first
    if needed (cached; don't mutate)."""
    return pd.read_parquet(USAGE_PATH) if USAGE_PATH.exists() else build_usage()


def agents(caseids: pd.Index) -> pd.Series:
    """Which anaesthetics each case actually received, e.g. "propofol+remifentanil"."""
    u = usage().reindex(caseids, fill_value=0.0)
    given = pd.DataFrame({a: u[a] >= threshold for a, (_, threshold) in AGENTS.items()})
    return given.apply(lambda row: "+".join(a for a in AGENTS if row[a]) or "none", axis=1)
