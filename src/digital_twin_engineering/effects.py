"""Validating the what-ifs: does a target change do what the expert expects?

At every test decision point after which the clinician changed a TCI target (an
intervention within the suggestion window), the expert simulates the change it
was about to see: its signals with the drug steered to the new target minus
steered to the current one (`Runner.effect`, as the twin reports for its
suggestions). That expected effect is compared with what followed:

- residual: the measured value at the horizon minus the forecaster's prediction
  made at the same decision point. The forecaster doesn't know the change is
  coming, so its residual holds the change's effect plus noise. It may partly
  anticipate changes from the situation that prompts them, which shrinks the
  residual towards zero; residuals at decision points without a change show the
  noise level.
- change: the measured value at the horizon minus the current one (the raw
  course, confounded by whatever prompted the change).

If the expected effects are right, the residuals after changes should follow them:
the same sign, a correlation across changes, and a slope near 1.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

from . import dataset, experts
from .config import Config
from .experts.anesthesim import TCI_DRUGS
from .patients import covariates
from .pipeline import MODELS_DIR
from .suggestion import label_columns
from .targets import TARGETS

MIN_HORIZON_S = 180  # a change comes up to a minute after its decision point: shorter is too early


def changes(rows: pd.DataFrame, drugs: dict[str, str]) -> pd.DataFrame:
    """One row per intervention: the drug, its direction, the setting before and
    after, at the last decision point before it."""
    out = []
    for drug, setting in drugs.items():
        direction, new = label_columns(drug)
        changed = rows[rows[direction] != "hold"]
        out.append(
            pd.DataFrame(
                {
                    "caseid": changed.caseid, "t": changed.t, "drug": drug,
                    "direction": changed[direction], "current": changed[f"x.{setting}"], "new": changed[new],
                }
            )
        )
    out = pd.concat(out, ignore_index=True).sort_values(["caseid", "drug", "t"], ignore_index=True)
    # a change within the window shows at several decision points: keep the last
    # one before it, whose simulated timing is closest
    later = out.shift(-1)
    same = (later.caseid == out.caseid) & (later.drug == out.drug) & (later.new == out.new) & (later.direction == out.direction)
    return out[~same].reset_index(drop=True)


def case_effects(
    caseid: int, config: Config, spec: dict, todo: pd.DataFrame, horizons_s: tuple[int, ...]
) -> pd.DataFrame:
    """The expert's expected effect of each change in `todo` (one case's rows)."""
    frame = dataset.case_channels(caseid, config)
    runner = experts.Runner(experts.build(spec), covariates(caseid), config.step_s)
    inputs = {n: frame[n].to_numpy() for n in runner.expert.inputs}
    at = {t: g for t, g in todo.groupby("t")}
    out = []
    for i, t in enumerate(frame.index.to_numpy()):
        runner.step({n: v[i] for n, v in inputs.items()})
        for _, c in at.get(t, pd.DataFrame()).iterrows():
            effect = runner.effect(c.drug, c.new, c.current, horizons_s)
            if effect:
                out.append({"caseid": caseid, "t": t, "drug": c.drug} | {f"effect.{k}": v for k, v in effect.items()})
        if t >= todo.t.max():
            break
    return pd.DataFrame(out)


def run(run_name: str, jobs: int = 16) -> pd.DataFrame:
    """Expected effects, residuals and changes after every intervention in a training
    run's test cases (written to models/<run>/effects.parquet), plus the decision
    points without one (drug "none")."""
    from .live.replayer import load_run

    config, models = load_run(run_name)
    spec = next(s for s in config.experts if s["kind"] == "anesthesim")  # PAS can't simulate TCI
    test_cases = json.loads((MODELS_DIR / run_name / "test_cases.json").read_text())
    rows = dataset.build(test_cases, config, jobs=jobs)
    for f in models["forecasters"]:
        rows[f.output] = f.predict(rows)
    todo = changes(rows, config.suggestion.drugs)
    todo = todo[todo.drug.isin(list(TCI_DRUGS)) & (todo.current > 0) & (todo.new > 0)]
    horizons = tuple(h for h in config.horizons_s if h >= MIN_HORIZON_S)
    values = [name for name in config.targets if TARGETS[name].kind == "value"]
    work = partial(_safe_case_effects, config=config, spec=spec, horizons_s=horizons)
    with ProcessPoolExecutor(jobs) as pool:
        parts = list(pool.map(work, [(c, g) for c, g in todo.groupby("caseid")]))
    effects = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True)

    observed = rows[["caseid", "t"]].copy()
    for name in values:
        for h in horizons:
            y = rows[f"y.{name}@{h}"]
            observed[f"residual.{name}@{h}"] = y - rows[f"f.{name}@{h}"]
            observed[f"change.{name}@{h}"] = y - rows[TARGETS[name].persistence_column]
    after = todo.merge(effects, on=["caseid", "t", "drug"]).merge(observed, on=["caseid", "t"])
    for name in values:  # the expert's signal for each target (map_1min: its MAP)
        for h in horizons:
            after[f"effect.{name}@{h}"] = after[f"effect.{TARGETS[name].channel}@{h}"]
    holds = np.logical_and.reduce([rows[label_columns(d)[0]] == "hold" for d in config.suggestion.drugs])
    quiet = observed[holds].assign(drug="none", direction="hold")
    result = pd.concat([after, quiet], ignore_index=True)
    result.to_parquet(MODELS_DIR / run_name / "effects.parquet")
    return result


def _safe_case_effects(item, config, spec, horizons_s) -> pd.DataFrame | None:
    caseid, todo = item
    try:
        return case_effects(caseid, config, spec, todo, horizons_s)
    except Exception as e:
        print(f"case {caseid}: skipped ({type(e).__name__}: {e})", flush=True)
        return None


def summary(result: pd.DataFrame) -> pd.DataFrame:
    """Per drug, direction, signal and horizon: the mean expected effect against the
    mean residual and change, and how well they agree across changes."""
    pairs = [c.removeprefix("residual.").split("@") for c in result if c.startswith("residual.")]
    rows = []
    for (drug, direction), g in result.groupby(["drug", "direction"]):
        for s, h in pairs:
            r = g[f"residual.{s}@{h}"].to_numpy()
            row = {
                "drug": drug, "direction": direction, "signal": s, "horizon_s": h,
                "n": int(np.isfinite(r).sum()),
                "residual": np.nanmean(r), "change": np.nanmean(g[f"change.{s}@{h}"]),
            }
            if drug != "none":
                e = g[f"effect.{s}@{h}"].to_numpy()
                row |= {"expected": np.nanmean(e)} | _agreement(e, r)
            rows.append(row)
    # both directions together: does the size of the effect track?
    for drug, g in result[result.drug != "none"].groupby("drug"):
        for s, h in pairs:
            e, r = g[f"effect.{s}@{h}"].to_numpy(), g[f"residual.{s}@{h}"].to_numpy()
            rows.append(
                {"drug": drug, "direction": "both", "signal": s, "horizon_s": h, "n": int(np.isfinite(e + r).sum()),
                 "residual": np.nanmean(r), "change": np.nanmean(g[f"change.{s}@{h}"]), "expected": np.nanmean(e)}
                | _agreement(e, r)
            )
    return pd.DataFrame(rows)


def _agreement(expected: np.ndarray, residual: np.ndarray) -> dict:
    ok = np.isfinite(expected) & np.isfinite(residual)
    e, r = expected[ok], residual[ok]
    if len(e) < 3 or e.std() == 0:
        return {}
    slope = float(np.cov(e, r)[0, 1] / e.var(ddof=1))
    return {
        "corr": float(np.corrcoef(e, r)[0, 1]),
        "slope": slope,
        "same_sign": float((np.sign(e) == np.sign(r))[e != 0].mean()),
    }
