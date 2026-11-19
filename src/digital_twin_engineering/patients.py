"""Patient covariates and identities from VitalDB's clinical_data table."""

from dataclasses import dataclass
from functools import cache

import pandas as pd

from . import vitaldb_cache as vdb

COVARIATES = ("age", "sex", "height", "weight", "bmi", "asa")


@dataclass(frozen=True)
class Covariates:
    """The static facts about a patient at the time of one case."""

    age: float
    sex: str  # "M" or "F"
    height: float  # cm
    weight: float  # kg
    bmi: float
    asa: float  # NaN when not recorded

    @property
    def is_male(self) -> bool:
        return self.sex == "M"


@cache
def clinical() -> pd.DataFrame:
    """clinical_data indexed by caseid (cached; don't mutate)."""
    df = vdb.table("clinical_data").set_index("caseid")
    # ages over 89 are de-identified as ">89"
    df["age"] = pd.to_numeric(df.age.replace(">89", "90"))
    return df


def covariates(caseid: int) -> Covariates:
    row = clinical().loc[caseid]
    return Covariates(**{name: row[name] for name in COVARIATES})


def subject_of(caseid: int) -> int:
    """The patient (subjectid) a case belongs to."""
    return int(clinical().at[caseid, "subjectid"])
