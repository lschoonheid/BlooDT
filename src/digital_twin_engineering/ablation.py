"""Ablation: what each group of forecaster inputs is worth.

Each variant drops one group of `x.` columns (or keeps a single expert) and retrains
the forecasters on the same training rows; comparing their test metrics with the
"all" variant shows what the group adds. Suggesters are left out: they inherit any
difference through the forecasts.
"""

import json

import pandas as pd

from . import evaluation, forecasting
from .channels import STALENESS
from .cohort import Cohort
from .config import Config
from .pipeline import MODELS_DIR, _json, split

COVARIATES = ("x.age", "x.male", "x.height", "x.weight", "x.bmi", "x.asa")
PHASE = ("x.anaesthesia_s", "x.surgery")


def variants(config: Config, features: list[str]) -> dict[str, list[str]]:
    """Variant name -> the feature columns it trains on."""
    experts = [spec["name"] for spec in config.experts]

    def of(expert: str, c: str) -> bool:
        return c.startswith(f"x.{expert}.")

    def without(drop) -> list[str]:
        return [c for c in features if not drop(c)]

    out = {"all": features}
    if experts:
        out["no_experts"] = without(lambda c: any(of(e, c) for e in experts))
        for e in experts:
            out[f"only_{e}"] = without(lambda c, e=e: any(of(o, c) for o in experts if o != e))
    out["no_staleness"] = without(lambda c: c.endswith(STALENESS))
    out["no_covariates"] = without(lambda c: c in COVARIATES)
    out["no_phase"] = without(lambda c: c in PHASE)
    return out


def run(config: Config, cohort: Cohort, name: str, only: list[str] | None = None, jobs: int = 16):
    """Train and evaluate every variant; metrics go to models/<name>/ablation.json."""
    out = MODELS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    train_rows, test_sets, _ = split(config, cohort, jobs)
    results: dict = {}
    for variant, features in variants(config, forecasting.feature_columns(train_rows)).items():
        if only and variant not in only:
            continue
        results[variant] = {"features": len(features)}
        for target in config.targets:
            for h in config.horizons_s:
                f = forecasting.fit(train_rows, target, h, config.model, features)
                for test, rows in test_sets.items():
                    results[variant].setdefault(test, {})[f.column] = evaluation.forecaster(f, rows)
                print(f"{variant} {f.column}: {results[variant]['clean'][f.column]['model']}", flush=True)
        (out / "ablation.json").write_text(json.dumps(results, indent=2, default=_json))
    return results


def table(results: dict, test: str = "clean") -> pd.DataFrame:
    """One row per variant: the headline metric of every forecaster (and onset AUROC)."""
    rows = {}
    for variant, r in results.items():
        row = {"features": r["features"]}
        for column, m in r[test].items():
            if "rmse" in m["model"]:
                row[f"{column} rmse"] = m["model"]["rmse"]
            else:
                row[f"{column} auroc"] = m["model"]["auroc"]
                if "onset" in m:
                    row[f"{column} onset auroc"] = m["onset"]["model"]["auroc"]
        rows[variant] = row
    return pd.DataFrame(rows).T
