"""Intervention suggestions: what a clinician would typically change now, per drug.

Per drug, a classifier gives calibrated probabilities of decrease / hold / increase
within the suggestion window, and a regressor per direction gives the setting a change would go
to. Both see the forecasters' outputs (`f.` columns) next to the `x.` features, so a
predicted hypotension can drive a suggested decrease. They imitate observed
practice; a high probability means "typical here", not "correct".
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import splits
from .forecasting import _model, feature_columns

DIRECTIONS = ("decrease", "hold", "increase")


def label_columns(drug: str) -> tuple[str, str]:
    return f"y.suggest.{drug}.direction", f"y.suggest.{drug}.new"


@dataclass
class Suggester:
    drug: str
    setting: str  # the channel it suggests a value for, e.g. propofol_target
    features: list[str]
    classifier: Any
    regressors: dict[str, Any]  # "decrease"/"increase" -> model (missing: no such changes)

    def suggest(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Columns p_decrease, p_hold, p_increase; setting_decrease and setting_increase
        (where a change in that direction would go, at most / at least the current
        setting); and `setting`, that of the more probable of the two."""
        X = rows[self.features].to_numpy(np.float32)
        proba = self.classifier.predict_proba(X)
        out = pd.DataFrame(0.0, index=rows.index, columns=[f"p_{d}" for d in DIRECTIONS])
        for j, cls in enumerate(self.classifier.classes_):
            out[f"p_{cls}"] = proba[:, j]
        current = rows[f"x.{self.setting}"].fillna(0).to_numpy()
        for d, clip in (("decrease", np.minimum), ("increase", np.maximum)):
            model = self.regressors.get(d)
            out[f"setting_{d}"] = clip(model.predict(X), current) if model is not None else np.nan
        up = out.p_increase >= out.p_decrease
        out["setting"] = np.where(up, out.setting_increase, out.setting_decrease)
        return out


def fit(
    rows: pd.DataFrame, drug: str, setting: str, params: dict[str, Any], k: int, seed: int
) -> Suggester:
    direction, new = label_columns(drug)
    features = feature_columns(rows, ("x.", "f."))
    # calibrate on patient-grouped folds so calibration never sees a training patient
    classifier = CalibratedClassifierCV(
        _model(True, params), method="isotonic", cv=splits.folds(rows, k, seed)
    )
    X = rows[features].to_numpy(np.float32)
    classifier.fit(X, rows[direction].to_numpy())
    regressors = {}
    for d in ("decrease", "increase"):
        changed = (rows[direction] == d).to_numpy()
        if changed.any():
            regressors[d] = _model(False, params)
            regressors[d].fit(X[changed], rows[new].to_numpy()[changed])
    return Suggester(drug, setting, features, classifier, regressors)
