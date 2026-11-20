"""Honest evaluation: every model is compared with a do-nothing-clever baseline.

- value targets: MAE/RMSE vs persistence (the current value stays the same)
- event targets: AUROC/AUPRC and sensitivity at a fixed alert rate vs "MAP already
  low" (score = -current MAP), as in Chaari et al.; also on onsets only (decision
  points where the event isn't already under way), where that baseline is weakest
- suggestions: per-direction AUROC and log loss vs the class prior; MAE of the
  suggested setting for the direction actually taken vs keeping the current one

Probabilities are also checked for calibration (a stated 30% should come true 30%
of the time): Brier score, expected calibration error over equal-count bins, and
the reliability table behind it.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .forecasting import Forecaster
from .suggestion import DIRECTIONS, Suggester, label_columns
from .targets import TARGETS


def sensitivity_at_alert_rate(y: np.ndarray, score: np.ndarray, alert_rate: float) -> float:
    """Share of positives caught when alerting on the top `alert_rate` of scores."""
    threshold = np.quantile(score, 1 - alert_rate)
    return float((score[y == 1] >= threshold).mean()) if (y == 1).any() else np.nan


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    """Brier score, and expected calibration error over `bins` equal-count bins of
    the predicted probability with each bin's mean prediction and observed rate."""
    order = np.argsort(p, kind="stable")
    groups = [g for g in np.array_split(order, bins) if len(g)]
    predicted = np.array([p[g].mean() for g in groups])
    observed = np.array([y[g].mean() for g in groups])
    sizes = np.array([len(g) for g in groups])
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "ece": float(np.sum(sizes * np.abs(predicted - observed)) / sizes.sum()),
        "reliability": {"predicted": predicted.round(4).tolist(), "observed": observed.round(4).tolist()},
    }


def _event_metrics(y, score, alert_rate):
    return {
        "auroc": roc_auc_score(y, score),
        "auprc": average_precision_score(y, score),
        f"sensitivity@{alert_rate:.0%}_alerts": sensitivity_at_alert_rate(y, score, alert_rate),
    }


def forecaster(f: Forecaster, rows: pd.DataFrame, alert_rate: float = 0.1) -> dict:
    target = TARGETS[f.target]
    known = rows[f.column].notna()
    rows = rows[known]
    y = rows[f.column].to_numpy()
    pred = f.predict(rows)
    current = rows[f"x.{target.channel}"].to_numpy()
    if target.kind == "event":
        y = y.astype(int)
        has_current = np.isfinite(current)
        baseline = _event_metrics(y[has_current], -current[has_current], alert_rate)
        result = {
            "n": len(y),
            "prevalence": float(y.mean()),
            "model": _event_metrics(y, pred, alert_rate),
            "baseline_map_already_low": baseline,
            "calibration": calibration(y, pred),
        }
        if target.ongoing is not None:
            onset = has_current & ~target.ongoing(current)
            if 0 < y[onset].sum() < onset.sum():
                result["onset"] = {
                    "n": int(onset.sum()),
                    "prevalence": float(y[onset].mean()),
                    "model": _event_metrics(y[onset], pred[onset], alert_rate),
                    "baseline_map_already_low": _event_metrics(
                        y[onset], -current[onset], alert_rate
                    ),
                }
        return result

    def errors(p):
        ok = np.isfinite(p)
        e = p[ok] - y[ok]
        return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e**2).mean()))}

    return {"n": len(y), "model": errors(pred), "baseline_persistence": errors(current)}


def suggester(s: Suggester, rows: pd.DataFrame) -> dict:
    direction, new = label_columns(s.drug)
    y = rows[direction].to_numpy()
    out = s.suggest(rows)
    proba = out[[f"p_{d}" for d in DIRECTIONS]].to_numpy()
    prior = np.array([(y == d).mean() for d in DIRECTIONS])
    result = {
        "n": len(y),
        "share": dict(zip(DIRECTIONS, prior.round(4).tolist())),
        "log_loss": log_loss(y, proba, labels=list(DIRECTIONS)),
        "log_loss_prior": log_loss(y, np.tile(prior, (len(y), 1)), labels=list(DIRECTIONS)),
        "auroc": {
            d: roc_auc_score(y == d, proba[:, j])
            for j, d in enumerate(DIRECTIONS)
            if 0 < (y == d).sum() < len(y)
        },
        "calibration": {
            d: calibration((y == d).astype(float), proba[:, j]) for j, d in enumerate(DIRECTIONS)
        },
    }
    changed = y != "hold"
    if changed.any() and s.regressors:
        target = rows.loc[changed, new].to_numpy()
        current = rows.loc[changed, f"x.{s.setting}"].fillna(0).to_numpy()
        taken = np.where(y[changed] == "increase", out.loc[changed, "setting_increase"], out.loc[changed, "setting_decrease"])
        result["setting_mae"] = float(np.nanmean(np.abs(taken - target)))
        result["setting_mae_keep_current"] = float(np.nanmean(np.abs(current - target)))
    return result
