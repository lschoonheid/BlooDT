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


def test_scenario_matches_stepping_and_leaves_state_alone():
    e = AnestheSimExpert("a")
    e.reset(PATIENT, 1.0)
    drip = REST | dict(propofol_dose=0.1, remifentanil_dose=0.2, norepinephrine_dose=0.05)
    for _ in range(100):
        e.step(drip)
    ahead = e.scenario(drip, (300, 60))
    following = [e.step(drip) for _ in range(300)]
    for key in e.scenario_signals:
        assert ahead[f"{key}@60"] == pytest.approx(following[59][key])
        assert ahead[f"{key}@300"] == pytest.approx(following[299][key])


def test_tci_reaches_and_holds_its_target():
    e = AnestheSimExpert("a")
    e.reset(PATIENT, 1.0)
    rates, ce = [], []
    for _ in range(600):
        rate = e._tci_rate("propofol", 3.0)
        rates.append(rate)
        ce.append(e.step(REST | dict(propofol_dose=rate))["propofol_ce"])
    assert rates[0] > 5 * rates[-1]  # a bolus first, then maintenance
    assert ce[120] == pytest.approx(3.0, rel=0.03)
    assert ce[-1] == pytest.approx(3.0, rel=0.01)
    assert max(ce) < 3.1
    assert max(e.step(REST | dict(propofol_dose=e._tci_rate("propofol", 3.0)))["propofol_ce"]
               for _ in range(60)) < 3.1
    assert e._tci_rate("propofol", 1.5) == 0  # pauses to come down


def test_effect_of_raising_propofol_deepens_and_lowers_pressure():
    runner = experts.Runner(AnestheSimExpert("a"), PATIENT, 1.0)
    e = runner.expert
    inputs = dict(REST)
    for _ in range(900):
        inputs["propofol_dose"] = e._tci_rate("propofol", 3.0)
        inputs["remifentanil_dose"] = e._tci_rate("remifentanil", 2.0)
        runner.step(inputs)
    up = runner.effect("propofol", 3.6, 3.0, (300,))
    down = runner.effect("propofol", 2.4, 3.0, (300,))
    assert up["bis@300"] < 0 < down["bis@300"]
    assert up["map@300"] < 0 < down["map@300"]
    assert up["propofol_ce@300"] == pytest.approx(0.6, rel=0.05)
    assert runner.effect("remifentanil", 2.4, 2.0, (300,))["remifentanil_ce@300"] == pytest.approx(0.4, rel=0.05)
    assert runner.effect("propofol", 3.6, 0.0, (300,)) is None  # no pump running
    assert experts.Runner(PasExpert("p"), PATIENT, 1.0).effect("propofol", 3.6, 3.0, (300,)) is None
