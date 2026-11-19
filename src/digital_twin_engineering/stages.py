"""Per-case cache of pipeline stage outputs, keyed by the stage's config.

Each stage writes `DERIVED_DIR/<stage>/<config hash>/<caseid>.parquet` plus a
`config.json` describing the hash, so changing a stage's config recomputes it while
other configs' results stay on disk.
"""

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

DERIVED_DIR = Path(
    os.environ.get("DTE_DERIVED_DIR")
    or Path(__file__).resolve().parents[2] / "data" / "derived"
)


def config_hash(config: Mapping[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def stage_dir(stage: str, config: Mapping[str, Any]) -> Path:
    d = DERIVED_DIR / stage / config_hash(config)
    if not (d / "config.json").exists():
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(config, sort_keys=True, indent=2, default=str))
    return d


def cached(
    stage: str,
    config: Mapping[str, Any],
    caseid: int,
    compute: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """`compute()`'s result for this case, read from the cache if already there."""
    path = stage_dir(stage, config) / f"{caseid:04d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = compute()
    part = path.with_name(f"{path.name}.{os.getpid()}.part")
    df.to_parquet(part)
    part.replace(path)
    return df
