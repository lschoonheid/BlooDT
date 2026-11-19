"""End-to-end training run: cohort -> rows -> forecasters -> suggesters -> evaluation.

1. Resolve the cohort and freeze its case list.
2. Build decision-point rows for every case (cached per case).
3. Split patients into train and test.
4. Per (target, horizon): out-of-fold forecasts on train (inputs for the
   suggesters), a final forecaster on all of train, forecasts on test.
5. Per drug: a suggester on train rows plus forecasts.
6. Evaluate on clean test rows, and on perturbed ones when a perturbation is
   configured; save models, metrics and config under models/<run>/.
"""

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import dataset, evaluation, forecasting, splits, suggestion
from .cohort import Cohort
from .config import Config

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _rows(caseids: list[int], config: Config, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    clean = dataset.build(caseids, config, jobs=jobs)
    perturbed = dataset.build(caseids, config, perturbed=True, jobs=jobs) if config.perturbation else None
    return clean, perturbed


def train(config: Config, cohort: Cohort, run: str, jobs: int = 16) -> Path:
    out = MODELS_DIR / run
    out.mkdir(parents=True, exist_ok=True)
    cohort.save_manifest()
    clean, perturbed = _rows(cohort.caseids(), config, jobs)

    is_train, _ = splits.train_test(clean, config.test_fraction, config.seed)
    train_patients = set(clean.subjectid[is_train])
    train_rows = clean[is_train]
    test_sets = {"clean": clean[~is_train]}
    if perturbed is not None:
        in_train = perturbed.subjectid.isin(train_patients)
        # train on clean and faulty views of the same patients
        train_rows = pd.concat([train_rows, perturbed[in_train]], ignore_index=True)
        test_sets["perturbed"] = perturbed[~in_train]
    train_rows = train_rows.reset_index(drop=True)

    metrics: dict = {"cases": int(clean.caseid.nunique()), "rows_train": len(train_rows)}
    forecasters = []
    for target in config.targets:
        for h in config.horizons_s:
            oof = forecasting.out_of_fold(train_rows, target, h, config.model, config.folds, config.seed)
            f = forecasting.fit(train_rows, target, h, config.model)
            train_rows[f.output] = oof
            for name, rows in test_sets.items():
                test_sets[name] = rows.assign(**{f.output: f.predict(rows)})
                metrics.setdefault(name, {})[f.column] = evaluation.forecaster(f, rows)
            forecasters.append(f)
            print(f"forecaster {f.column}: {metrics['clean'][f.column]['model']}", flush=True)

    suggesters = []
    for drug, setting in config.suggestion.drugs.items():
        s = suggestion.fit(train_rows, drug, setting, config.model, config.folds, config.seed)
        for name, rows in test_sets.items():
            metrics[name][f"suggest.{drug}"] = evaluation.suggester(s, rows)
        suggesters.append(s)
        print(f"suggester {drug}: log loss {metrics['clean'][f'suggest.{drug}']['log_loss']:.3f}", flush=True)

    joblib.dump({"forecasters": forecasters, "suggesters": suggesters}, out / "models.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json))
    (out / "config.json").write_text(
        json.dumps({"config": asdict(config), "cohort": asdict(cohort)}, indent=2, default=_json)
    )
    return out


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    return str(o)
