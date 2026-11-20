import numpy as np
import pandas as pd
import pytest

from digital_twin_engineering import experts
from digital_twin_engineering.experts.anesthesim import AnestheSimExpert
from digital_twin_engineering.experts.baseline import PreDoseBaseline
from digital_twin_engineering.experts.pas import PasExpert
from digital_twin_engineering.patients import Covariates

PATIENT = Covariates(age=55, sex="M", height=175, weight=78, bmi=25.5, asa=2)
REST = dict(
    propofol_dose=0.0, remifentanil_dose=0.0, norepinephrine_dose=0.0,
    sevoflurane_et=np.nan, desflurane_et=np.nan, nibp_map=110.0, hr=80.0,
)


def test_baseline_counts_each_cuff_reading_once_and_freezes_at_first_dose():
    b = PreDoseBaseline()
    for nibp in [100.0, 100.0, 100.0, 120.0, np.nan, 130.0]:
        assert not b.update(REST | dict(nibp_map=nibp))
    assert b.map0 == 120.0  # readings 100, 120, 130
    assert b.update(REST | dict(nibp_map=140.0, propofol_dose=1.0))
    assert not b.update(REST | dict(nibp_map=50.0))
    assert b.map0 == 125.0 and b.frozen


def test_baseline_without_observations_is_unknown():
    b = PreDoseBaseline()
    b.update(REST | dict(nibp_map=np.nan, hr=np.nan, propofol_dose=1.0))
    assert b.map0 is None and b.hr0 is None


@pytest.mark.parametrize("make", [lambda: AnestheSimExpert("a", baseline="pre_dose"),
                                  lambda: PasExpert("p", baseline="pre_dose")])
def test_pre_dose_baseline_sets_resting_map_and_hr(make):
    e = make()
    e.reset(PATIENT, 1.0)
    for _ in range(10):
        e.step(REST)
    y = e.step(REST | dict(propofol_dose=1e-6))
    assert y["map"] == pytest.approx(110, abs=0.5)
    assert y["hr"] == pytest.approx(80, abs=0.5)


def test_lookahead_matches_stepping_and_leaves_state_alone():
    e = AnestheSimExpert("a")
    e.reset(PATIENT, 1.0)
    drip = REST | dict(propofol_dose=0.1, remifentanil_dose=0.2, norepinephrine_dose=0.05)
    for _ in range(100):
        e.step(drip)
    ahead = e.lookahead(drip, [300, 60])
    following = [e.step(drip) for _ in range(300)]
    for key in e.lookahead_signals:
        assert ahead[0][key] == pytest.approx(following[59][key])
        assert ahead[1][key] == pytest.approx(following[299][key])


def test_run_adds_lookahead_at_decision_points_only():
    n = 200
    frame = pd.DataFrame(
        {k: np.full(n, v) for k, v in REST.items()}, index=pd.Index(np.arange(n, dtype=float), name="t")
    )
    frame["propofol_dose"] = 0.1
    out = experts.run([AnestheSimExpert("a")], frame, PATIENT, 1.0, experts.Lookahead((60,), 30))
    ahead = out["a.bis@60"]
    assert ahead.notna().all()
    # carried forward between decision points
    assert (ahead.iloc[30:60] == ahead.iloc[30]).all()
    assert ahead.iloc[60] != ahead.iloc[30]
    # what the model says 60 s later, holding the same drip
    assert ahead.iloc[90] == pytest.approx(out["a.bis"].iloc[150], abs=1e-6)
