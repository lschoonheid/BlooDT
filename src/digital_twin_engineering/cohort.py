"""Cohorts: named, reproducible case sets, and an explorer for choosing them.

A cohort is a YAML file under `cohorts/`:

    name: tiva-artline
    channels: [map, bis, hr, propofol_dose, remifentanil_dose]   # required
    where:
      ane_type: General                  # equals
      agents: propofol+remifentanil      # anaesthetics actually given (see inventory.AGENTS)
      department: [General surgery, Thoracic surgery]   # one of
      age: {min: 18}                     # range, inclusive
    query: "weight > 40"                 # optional pandas query on clinical_data
    min_duration_s: 3600                 # anaesthesia time (aneend - anestart)

Cases must also have complete patient covariates (age, sex, height, weight).
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import inventory
from .patients import clinical
from .stages import DERIVED_DIR

REQUIRED_COVARIATES = ["age", "sex", "height", "weight"]

@dataclass(frozen=True)
class Cohort:
    name: str
    channels: tuple[str, ...] = ()
    where: Mapping[str, Any] = field(default_factory=dict)
    query: str | None = None
    min_duration_s: float = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Cohort":
        spec = yaml.safe_load(Path(path).read_text())
        spec["channels"] = tuple(spec.get("channels", ()))
        return cls(**spec)

    def cases(self) -> pd.DataFrame:
        """clinical_data rows of the cases in this cohort."""
        df = clinical()
        df = df[df[REQUIRED_COVARIATES].notna().all(axis=1)]
        df = df[(df.aneend - df.anestart) >= self.min_duration_s]
        if "agents" in self.where:
            df = df.assign(agents=inventory.agents(df.index))
        for column, condition in self.where.items():
            df = df[_matches(df[column], condition)]
        if self.query:
            df = df.query(self.query)
        if self.channels:
            cov = inventory.coverage(list(self.channels))
            df = df[df.index.isin(cov.index[cov.all(axis=1)])]
        return df

    def caseids(self) -> list[int]:
        return sorted(self.cases().index)

    def save_manifest(self) -> Path:
        """Freeze the resolved case list next to the spec, for reproducible runs."""
        path = DERIVED_DIR / "cohorts" / f"{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"spec": asdict(self), "caseids": self.caseids()}, indent=1))
        return path


def _matches(column: pd.Series, condition: Any) -> pd.Series:
    if isinstance(condition, Mapping):
        ok = pd.Series(True, index=column.index)
        if "min" in condition:
            ok &= column >= condition["min"]
        if "max" in condition:
            ok &= column <= condition["max"]
        return ok
    if isinstance(condition, list | tuple):
        return column.isin(condition)
    return column == condition


def explore(
    channels: list[str],
    by: list[str] = ["department", "optype", "agents"],
    base: Cohort | None = None,
) -> pd.DataFrame:
    """How many cases and patients have all `channels`, per group, largest first.

    `by` takes clinical_data columns plus `agents`, the anaesthetics actually
    given. `base` restricts the search to an existing cohort's criteria.
    """
    base = base or Cohort("explore")
    df = Cohort(base.name, tuple(channels), base.where, base.query, base.min_duration_s).cases()
    if "agents" in by and "agents" not in df:
        df = df.assign(agents=inventory.agents(df.index))
    counts = df.groupby(by, dropna=False).agg(
        cases=("subjectid", "size"), patients=("subjectid", "nunique")
    )
    return counts.sort_values("cases", ascending=False)


def channel_coverage(base: Cohort | None = None) -> pd.Series:
    """Fraction of a cohort's cases (default: all) that have each known channel."""
    from .channels import CHANNELS

    ids = (base or Cohort("all")).caseids()
    return inventory.coverage(list(CHANNELS)).loc[ids].mean().sort_values(ascending=False)
