"""Transfer: do forecasters trained on one patient group work on another?

Each split divides the cohort in two groups by a clinical fact (sex, age, ...).
Per split, a forecaster is trained on each group's training patients and tested on
both groups' test patients (the run's usual patient split, so no test patient was
seen). The larger group's training patients are subsampled to the smaller group's
number of rows, so both models learn from as much data. Testing a model on the
other group shows the cost of transferring it; on its own group, the reference.
"""

import json

import numpy as np
import pandas as pd

from . import evaluation, forecasting
from .cohort import Cohort
from .config import Config
from .patients import clinical
from .pipeline import MODELS_DIR, _json, split

SPLITS = {
    "sex": lambda c: np.where(c.sex == "F", "female", "male"),
    "age": lambda c: np.where(c.age >= 65, "65 and over", "under 65"),
    "department": lambda c: np.where(c.department == "Thoracic surgery", "thoracic", "general and other"),
    "approach": lambda c: np.where(c.approach == "Open", "open", "videoscopic or robotic"),
    "asa": lambda c: np.where(c.asa >= 3, "ASA 3+", np.where(c.asa >= 1, "ASA 1-2", None)),
}
TARGETS = ("map", "bis", "hypotension")  # at 5 min, or the config's longest horizon


def run(config: Config, cohort: Cohort, name: str, only: list[str] | None = None, jobs: int = 16) -> dict:
    """Train and test both groups' forecasters of every split; metrics go to
    models/<name>/transfer.json (added to the splits already there)."""
    out = MODELS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    train_rows, test_sets, _ = split(config, cohort, jobs)
    test = test_sets["clean"]
    del test_sets
    path = out / "transfer.json"
    results: dict = json.loads(path.read_text()) if path.exists() else {}  # splits run before
    rng = np.random.default_rng(config.seed)
    h = 300 if 300 in config.horizons_s else max(config.horizons_s)
    targets = [t for t in TARGETS if t in config.targets]
    for split_name, group_of in SPLITS.items():
        if only and split_name not in only:
            continue
        train_group = _groups(train_rows, group_of)
        test_group = _groups(test, group_of)
        groups = sorted({g for g in train_group if isinstance(g, str)})  # else not recorded
        n = min((train_group == g).sum() for g in groups)
        result = {"groups": {}}
        for g in groups:
            rows = _subsample(train_rows[train_group == g], n, rng)
            result["groups"][g] = {
                "train_rows": len(rows), "train_patients": int(rows.subjectid.nunique()),
                "test_patients": int(test.subjectid[test_group == g].nunique()),
            }
            for target in targets:
                f = forecasting.fit(rows, target, h, config.model)
                for tested in groups:
                    m = evaluation.forecaster(f, test[test_group == tested].reset_index(drop=True))
                    result.setdefault(f.column, {}).setdefault(g, {})[tested] = m
                print(f"{split_name}: {g} {f.column} -> {_headline(result[f.column][g])}", flush=True)
        results[split_name] = result
        path.write_text(json.dumps(results, indent=2, default=_json))
    return results


def _groups(rows: pd.DataFrame, group_of) -> np.ndarray:
    c = clinical().loc[rows.caseid.unique()]
    return rows.caseid.map(pd.Series(group_of(c), index=c.index)).to_numpy()


def _subsample(rows: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """About `n` rows, taking whole patients in random order."""
    if len(rows) <= n:
        return rows
    sizes = rows.groupby("subjectid").size()
    order = rng.permutation(sizes.index.to_numpy())
    keep = order[: np.searchsorted(np.cumsum(sizes.loc[order].to_numpy()), n) + 1]
    return rows[rows.subjectid.isin(set(keep))]


def _headline(by_tested: dict) -> dict:
    return {t: _metric(m) for t, m in by_tested.items()}


def _metric(m: dict) -> float:
    if "rmse" in m["model"]:
        return round(m["model"]["rmse"], 3)
    return round(m.get("onset", m)["model"]["auroc"], 3)


def table(results: dict) -> pd.DataFrame:
    """One row per (split, target, test group): the metric of the group's own model
    and of the other group's model (RMSE, or onset AUROC for hypotension)."""
    rows = []
    for split_name, r in results.items():
        for column, by_trained in r.items():
            if column == "groups":
                continue
            for tested in by_trained:
                own = _metric(by_trained[tested][tested])
                other = next(g for g in by_trained if g != tested)
                rows.append({
                    "split": split_name, "target": column, "tested on": tested,
                    "own group's model": own, "other group's model": _metric(by_trained[other][tested]),
                })
    return pd.DataFrame(rows)
