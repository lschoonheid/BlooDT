"""Training data: one row per decision point, per case, cached per case.

A case *frame* is its channels plus expert signals on the 1 s grid. A case's
*rows* are its decision points with features (`x.`), forecaster labels
(`y.<target>@<horizon>`) and intervention labels (`y.suggest.<drug>.*`), plus
`caseid`, `subjectid` and `t`.

With `perturbed=True` the features come from a frame with simulated sensor faults
while the labels still come from the clean frame.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

from . import experts, stages
from .channels import CHANNELS, load_channels
from .config import Config
from .features import covariate_features, phase_features, window_features
from .labels import detect_interventions
from .patients import clinical, covariates
from .targets import TARGETS


def case_frame(caseid: int, config: Config, perturbed: bool = False) -> pd.DataFrame:
    def compute():
        frame = load_channels(
            caseid,
            list(config.all_channels),
            config.step_s,
            config.perturbation if perturbed else None,
            seed=config.seed * 100_000 + caseid,
        )
        built = [experts.build(spec) for spec in config.experts]
        if perturbed:
            # perturbations only touch level channels; experts reading none of them
            # would compute exactly what they did on the clean frame
            clean = case_frame(caseid, config)
            reuse = [e for e in built if not any(CHANNELS[n].kind == "level" for n in e.inputs)]
            built = [e for e in built if e not in reuse]
            frame = frame.join(clean[[c for e in reuse for c in clean if c.startswith(e.name + ".")]])
        signals = experts.run(built, frame, covariates(caseid), config.step_s)
        return frame.join(signals)

    return stages.cached("frames", config.frame_key(perturbed), caseid, compute)


def decision_points(caseid: int, frame: pd.DataFrame, every_s: int) -> pd.Index:
    """Grid times from anaesthesia start to end, every `every_s` seconds."""
    row = clinical().loc[caseid]
    t = frame.index.to_numpy()
    keep = (t >= row.anestart) & (t < row.aneend) & (np.round(t) % every_s == 0)
    return frame.index[keep]


def suggestion_labels(
    frame: pd.DataFrame, at: pd.Index, config: Config
) -> dict[str, np.ndarray]:
    """Per drug: the first intervention within the window after each decision point
    ("hold" if none) and the setting it changed to (the current setting on hold)."""
    s = config.suggestion
    out = {}
    for drug, channel in s.drugs.items():
        setting = frame[channel]
        events = detect_interventions(setting, s.min_hold_s, s.min_relative_change, config.step_s)
        t, event_t = at.to_numpy(), events.t.to_numpy()
        i = np.searchsorted(event_t, t, side="right")  # first event strictly after t
        within = i < len(event_t)
        within[within] = event_t[i[within]] - t[within] <= s.window_s
        direction = np.full(len(t), "hold", dtype=object)
        new = setting.fillna(0).loc[at].to_numpy().copy()
        direction[within] = events.direction.to_numpy()[i[within]]
        new[within] = events.new.to_numpy()[i[within]]
        out[f"y.suggest.{drug}.direction"] = direction.astype(str)
        out[f"y.suggest.{drug}.new"] = new
    return out


def case_rows(caseid: int, config: Config, perturbed: bool = False) -> pd.DataFrame:
    def compute():
        clean = case_frame(caseid, config)
        seen = case_frame(caseid, config, perturbed=True) if perturbed else clean
        at = decision_points(caseid, clean, config.decision_every_s)
        # float32 halves memory; plenty for vitals and drug levels
        rows = window_features(seen, config.features.windows_s, config.step_s, at).astype("float32")
        row = clinical().loc[caseid]
        extra = covariate_features(covariates(caseid))
        extra |= phase_features(at.to_numpy(), row.anestart, row.opstart, row.opend)
        for name in config.targets:
            target = TARGETS[name]
            for h in config.horizons_s:
                extra[target.column(h)] = target.label(clean, h, config.step_s).loc[at].to_numpy()
        extra |= suggestion_labels(clean, at, config)
        rows = rows.assign(**extra)
        rows.insert(0, "subjectid", int(row.subjectid))
        rows.insert(0, "caseid", caseid)
        return rows.reset_index()

    return stages.cached("rows", config.rows_key(perturbed), caseid, compute)


def _safe_rows(caseid: int, config: Config, perturbed: bool) -> pd.DataFrame | None:
    try:
        return case_rows(caseid, config, perturbed)
    except Exception as e:  # one broken case shouldn't sink a cohort build
        print(f"case {caseid}: skipped ({type(e).__name__}: {e})", flush=True)
        return None


def build(
    caseids: list[int], config: Config, perturbed: bool = False, jobs: int = 16
) -> pd.DataFrame:
    """Rows of every case, built in parallel (cached cases are just read)."""
    with ProcessPoolExecutor(jobs) as pool:
        parts = list(pool.map(partial(_safe_rows, config=config, perturbed=perturbed), caseids))
    return pd.concat([p for p in parts if p is not None], ignore_index=True)
