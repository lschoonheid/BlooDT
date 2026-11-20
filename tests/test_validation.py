import numpy as np
import pandas as pd

from digital_twin_engineering import effects, transfer


def test_a_change_counts_once_at_its_last_decision_point():
    rows = pd.DataFrame(
        {
            "caseid": [1, 1, 1, 1],
            "t": [0.0, 30.0, 60.0, 90.0],
            "x.propofol_target": [3.0, 3.0, 3.0, 2.5],
            "y.suggest.propofol.direction": ["decrease", "decrease", "hold", "increase"],
            "y.suggest.propofol.new": [2.5, 2.5, 3.0, 3.5],
        }
    )
    c = effects.changes(rows, {"propofol": "propofol_target"})
    assert c[["t", "direction", "current", "new"]].values.tolist() == [[30.0, "decrease", 3.0, 2.5], [90.0, "increase", 2.5, 3.5]]


def test_agreement_of_a_proportional_effect():
    rng = np.random.default_rng(0)
    expected = rng.normal(size=500)
    a = effects._agreement(expected, 0.5 * expected + rng.normal(scale=0.1, size=500))
    assert abs(a["slope"] - 0.5) < 0.05
    assert a["corr"] > 0.9 and a["same_sign"] > 0.9


def test_subsample_takes_whole_patients():
    rows = pd.DataFrame({"subjectid": np.repeat(np.arange(10), 100)})
    sub = transfer._subsample(rows, 450, np.random.default_rng(0))
    assert 450 <= len(sub) < 550 and sub.groupby("subjectid").size().eq(100).all()
