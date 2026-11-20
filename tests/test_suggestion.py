import numpy as np
import pandas as pd

from digital_twin_engineering.suggestion import Suggester


class Constant:
    def __init__(self, value, classes=None):
        self.value, self.classes_ = value, classes

    def predict(self, X):
        return np.full(len(X), self.value)

    def predict_proba(self, X):
        return np.tile(self.value, (len(X), 1))


def test_settings_stay_on_their_side_of_the_current_one_and_follow_the_likelier_change():
    rows = pd.DataFrame({"x.propofol_target": [3.0, 3.0, np.nan], "x.a": [0.0, 0.0, 0.0]})
    s = Suggester(
        "propofol", "propofol_target", ["x.a"],
        Constant([0.1, 0.6, 0.3], np.array(["decrease", "hold", "increase"])),
        {"decrease": Constant(3.5), "increase": Constant(2.5)},  # both on the wrong side
    )
    out = s.suggest(rows)
    assert out.setting_decrease.tolist() == [3.0, 3.0, 0.0]
    assert out.setting_increase.tolist() == [3.0, 3.0, 2.5]
    assert (out.setting == out.setting_increase).all()  # p_increase > p_decrease
    assert np.allclose(out[["p_decrease", "p_hold", "p_increase"]].sum(axis=1), 1)
