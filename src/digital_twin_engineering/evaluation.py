"""Honest evaluation: every model is compared with a do-nothing-clever baseline.

- value targets: MAE/RMSE vs persistence (the current value stays the same)
- event targets: AUROC/AUPRC and sensitivity at a fixed alert rate vs "MAP already
  low" (score = -current MAP), as in Chaari et al.
- suggestions: per-direction AUROC and log loss vs the class prior; MAE of the
  suggested setting vs keeping the current one
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
        return {
            "n": len(y),
            "prevalence": float(y.mean()),
            "model": _event_metrics(y, pred, alert_rate),
            "baseline_map_already_low": baseline,
        }

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
    }
    changed = y != "hold"
    if changed.any() and s.regressor is not None:
        target = rows.loc[changed, new].to_numpy()
        current = rows.loc[changed, f"x.{s.setting}"].fillna(0).to_numpy()
        result["setting_mae"] = float(np.abs(out.loc[changed, "setting"] - target).mean())
        result["setting_mae_keep_current"] = float(np.nanmean(np.abs(current - target)))
    return result
