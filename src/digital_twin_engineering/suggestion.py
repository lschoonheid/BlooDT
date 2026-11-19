"""Intervention suggestions: what a clinician would typically change now, per drug.

Per drug, a classifier gives calibrated probabilities of decrease / hold / increase
within the suggestion window, and a regressor gives the setting a change would go
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
    regressor: Any | None  # None when the training data had no changes

    def suggest(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Columns p_decrease, p_hold, p_increase and `setting` (the setting a change
        would go to) per row."""
        X = rows[self.features]
        proba = self.classifier.predict_proba(X)
        out = pd.DataFrame(0.0, index=rows.index, columns=[f"p_{d}" for d in DIRECTIONS])
        for j, cls in enumerate(self.classifier.classes_):
            out[f"p_{cls}"] = proba[:, j]
        out["setting"] = self.regressor.predict(X) if self.regressor is not None else np.nan
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
    classifier.fit(rows[features], rows[direction])
    changed = rows[direction] != "hold"
    regressor = None
    if changed.any():
        regressor = _model(False, params)
        regressor.fit(rows.loc[changed, features], rows.loc[changed, new])
    return Suggester(drug, setting, features, classifier, regressor)
