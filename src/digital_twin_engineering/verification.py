"""Expert fidelity: how closely each expert tracks the real case.

- verification: expert effect-site concentrations vs the pump's own TCI model
  (`propofol_ce_pump`, `remifentanil_ce_pump`); a mismatch means our dose
  reconstruction or the model choice differs from the pump's. The Orchestra
  pumps run Minto for remifentanil and (modified) Marsh for propofol, not
  Schnider: replaying our doses through PAS's Marsh_modified matches the pump's
  Ce to MDAPE 0.1-1% in most cases, so a `pas` expert with
  `model_propo: Marsh_modified` is the reference for checking dose reconstruction.
- validation: expert BIS and MAP vs the measured values.

Varvel metrics per case: performance error PE = (measured - predicted) / predicted,
MDPE = median PE, MDAPE = median |PE| (both in %), and RMSE. Only freshly observed
reference values (staleness <= 5 s) during anaesthesia are compared.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

from . import dataset
from .channels import STALENESS
from .config import Config
from .patients import clinical

# (expert signal, reference channel)
COMPARISONS = [
    ("propofol_ce", "propofol_ce_pump"),
    ("remifentanil_ce", "remifentanil_ce_pump"),
    ("bis", "bis"),
    ("map", "map"),
]


def varvel(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(reference) & np.isfinite(predicted) & (predicted > 0)
    if not ok.any():
        return {"mdpe": np.nan, "mdape": np.nan, "rmse": np.nan, "n": 0}
    r, p = reference[ok], predicted[ok]
    pe = 100 * (r - p) / p
    return {
        "mdpe": float(np.median(pe)),
        "mdape": float(np.median(np.abs(pe))),
        "rmse": float(np.sqrt(np.mean((r - p) ** 2))),
        "n": int(ok.sum()),
    }


def case_fidelity(caseid: int, config: Config) -> pd.DataFrame:
    """One row per (expert, signal) for one case."""
    frame = dataset.case_frame(caseid, config)
    row = clinical().loc[caseid]
    during = (frame.index >= row.anestart) & (frame.index < row.aneend)
    rows = []
    for spec in config.experts:
        for signal, reference in COMPARISONS:
            column = f"{spec['name']}.{signal}"
            if column not in frame or reference not in frame:
                continue
            fresh = during & (frame[reference + STALENESS] <= 5).to_numpy()
            metrics = varvel(frame[reference].to_numpy()[fresh], frame[column].to_numpy()[fresh])
            rows.append({"caseid": caseid, "expert": spec["name"], "signal": signal} | metrics)
    return pd.DataFrame(rows)


def fidelity(caseids: list[int], config: Config, jobs: int = 16) -> pd.DataFrame:
    """Per-case metrics for every expert and signal."""
    with ProcessPoolExecutor(jobs) as pool:
        parts = list(pool.map(partial(case_fidelity, config=config), caseids))
    return pd.concat(parts, ignore_index=True)


def summary(per_case: pd.DataFrame) -> pd.DataFrame:
    """Median over cases of each per-case metric, per expert and signal."""
    return (
        per_case.groupby(["signal", "expert"])[["mdpe", "mdape", "rmse"]]
        .median()
        .join(per_case.groupby(["signal", "expert"]).size().rename("cases"))
    )
