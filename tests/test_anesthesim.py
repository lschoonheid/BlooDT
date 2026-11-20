import itertools
import math
import time

import numpy as np
import pandas as pd
import pytest

from digital_twin_engineering import experts
from digital_twin_engineering.experts.anesthesim import (
    NE_MOLAR_MASS,
    AnestheSimExpert,
    _lbm,
    covariate_map0,
)
from digital_twin_engineering.patients import Covariates

PATIENT = Covariates(age=55, sex="M", height=175, weight=78, bmi=25.5, asa=2)
NO_DRUG = dict(
    propofol_dose=0.0, remifentanil_dose=0.0, norepinephrine_dose=0.0,
    sevoflurane_et=np.nan, desflurane_et=np.nan,
)
MODELS = list(itertools.product(("greco", "bouillon", "minto"), ("separable", "unified")))


def simulate(expert, n, step_s=1.0, covariates=PATIENT, **inputs):
    expert.reset(covariates, step_s)
    return [expert.step(NO_DRUG | inputs) for _ in range(n)]


def test_infusions_converge_to_rate_over_clearance():
    c, step_s = PATIENT, 600.0  # ZOH is exact, so long steps reach steady state cheaply
    y = simulate(
        AnestheSimExpert("a"), 2000, step_s,
        propofol_dose=0.1 * step_s, remifentanil_dose=0.2 * step_s, norepinephrine_dose=0.1 * step_s,
    )[-1]
    lbm = _lbm(c)
    cl_prop = (1.89 + 0.0456 * (c.weight - 77) - 0.0681 * (lbm - 59) + 0.0264 * (c.height - 177)) / 60
    cl_remi = (2.6 - 0.0162 * (c.age - 40) + 0.0191 * (lbm - 55)) / 60
    cl_ne = math.exp(0.64 * math.log(lbm) - 5.52) * 0.49
    assert y["propofol_cp"] == pytest.approx(0.1 / cl_prop, rel=1e-3)
    assert y["propofol_ce"] == pytest.approx(y["propofol_cp"], rel=1e-3)
    assert y["remifentanil_cp"] == pytest.approx(0.2 / cl_remi, rel=1e-3)
    assert y["norepinephrine_cp"] == pytest.approx(0.1 * 1e3 / NE_MOLAR_MASS / cl_ne, rel=1e-3)


def test_effect_site_lags_plasma():
    e = AnestheSimExpert("a")
    e.reset(PATIENT, 1.0)
    on = [e.step(NO_DRUG | dict(propofol_dose=1.0, remifentanil_dose=1.0)) for _ in range(60)]
    off = [e.step(NO_DRUG) for _ in range(600)]
    assert all(y["propofol_ce"] < y["propofol_cp"] for y in on)
    assert all(y["remifentanil_ce"] < y["remifentanil_cp"] for y in on)
    assert off[-1]["propofol_ce"] > off[-1]["propofol_cp"]
    ce = [y["propofol_ce"] for y in on]
    assert np.all(np.diff(ce) > 0)


def test_norepinephrine_transport_lag():
    ys = simulate(AnestheSimExpert("a"), 40, norepinephrine_dose=1.0)
    tlag = math.exp(0.26 + 3.21)  # male, ~32 s
    assert all(y["norepinephrine_cp"] == 0 for y in ys[: int(tlag) - 1])
    assert ys[-1]["norepinephrine_cp"] > 0


@pytest.mark.parametrize("bis_model", ["greco", "bouillon", "minto"])
def test_bis_falls_monotonically_with_propofol(bis_model):
    e = AnestheSimExpert("a", bis_model=bis_model)
    e.reset(PATIENT, 1.0)
    for ce_r in (0.0, 4.0):
        bis = [e._bis(ce_p, ce_r, 0.0, 0.0) for ce_p in np.linspace(0, 20, 200)]
        assert all(0 <= b <= 100 for b in bis)
        assert np.all(np.diff(bis) < 0)
    # Remifentanil deepens the hypnotic effect at any propofol level.
    assert e._bis(3.0, 4.0, 0, 0) < e._bis(3.0, 0.0, 0, 0)


@pytest.mark.parametrize("bis_model,map_model", MODELS)
def test_no_drug_gives_baselines(bis_model, map_model):
    e = AnestheSimExpert("a", bis_model=bis_model, map_model=map_model)
    ys = simulate(e, 600)
    e0 = {"greco": 95.0, "bouillon": 97.4, "minto": 93.0}[bis_model]
    map0 = covariate_map0(PATIENT, PATIENT.weight / 1.75**2)
    for y in ys:
        assert y["bis"] == pytest.approx(e0)
        assert y["map"] == pytest.approx(map0)
        assert y["hr"] == pytest.approx(70.8)
        assert y["volatile_mac"] == 0


def test_baseline_overrides():
    y = simulate(AnestheSimExpert("a", map0=92.0, hr0=64.0), 10)[-1]
    assert (y["map"], y["hr"]) == (pytest.approx(92.0), pytest.approx(64.0))


def test_volatile_nan_is_zero():
    dose = dict(propofol_dose=0.5, remifentanil_dose=0.5)
    nan = simulate(AnestheSimExpert("a"), 300, **dose)
    zero = simulate(AnestheSimExpert("a"), 300, sevoflurane_et=0.0, desflurane_et=0.0, **dose)
    assert nan == zero


def test_sevoflurane_effect_site_and_age_adjusted_bis():
    young = Covariates(age=30, sex="F", height=165, weight=60, bmi=22, asa=1)
    old = Covariates(age=80, sex="F", height=165, weight=60, bmi=22, asa=1)
    ys = simulate(AnestheSimExpert("a"), 3600, covariates=young, sevoflurane_et=2.0)
    assert ys[60]["sevoflurane_ce"] < 2.0
    assert ys[-1]["sevoflurane_ce"] == pytest.approx(2.0, rel=1e-3)
    assert ys[-1]["bis"] < 95
    y_old = simulate(AnestheSimExpert("a"), 3600, covariates=old, sevoflurane_et=2.0)[-1]
    assert y_old["bis"] < ys[-1]["bis"]  # Mapleson: lower C50 with age
    assert y_old["volatile_mac"] > ys[-1]["volatile_mac"]
    # Only reduced Greco carries the volatile interaction.
    y_minto = simulate(AnestheSimExpert("a", bis_model="minto"), 600, sevoflurane_et=2.0)[-1]
    assert y_minto["bis"] == pytest.approx(93.0)


@pytest.mark.parametrize("map_model", ["separable", "unified"])
def test_haemodynamic_directions(map_model):
    base = simulate(AnestheSimExpert("a", map_model=map_model), 1)[-1]
    prop = simulate(AnestheSimExpert("a", map_model=map_model), 1800, propofol_dose=0.1)[-1]
    ne = simulate(AnestheSimExpert("a", map_model=map_model), 1800, norepinephrine_dose=0.1)[-1]
    remi = simulate(AnestheSimExpert("a", map_model=map_model), 1800, remifentanil_dose=0.3)[-1]
    assert prop["map"] < base["map"]
    assert ne["map"] > base["map"]
    assert remi["hr"] < base["hr"]


def test_stimulation_raises_map():
    plain = simulate(AnestheSimExpert("a"), 600)[-1]
    stim = simulate(AnestheSimExpert("a", stim_onset_s=60.0), 600)[-1]
    assert stim["map"] > plain["map"]


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        AnestheSimExpert("a", bis_model="schnider")


def test_run_over_frame():
    n = 120
    frame = pd.DataFrame(
        {
            "propofol_dose": np.r_[np.full(30, 2.0), np.zeros(n - 30)],
            "remifentanil_dose": np.full(n, 0.5),
            "norepinephrine_dose": np.full(n, np.nan),
            "sevoflurane_et": np.r_[np.full(60, np.nan), np.full(n - 60, 1.5)],
            "desflurane_et": np.full(n, np.nan),
        },
        index=pd.Index(np.arange(n) * 2.0, name="t"),
    )
    out = experts.run([experts.build({"kind": "anesthesim", "name": "asim"})], frame, PATIENT, 2.0)
    assert list(out.columns) == [f"asim.{s}" for s in AnestheSimExpert.signals]
    assert len(out) == n and not out.isna().any().any()
    assert out["asim.bis"].iloc[-1] < 95


def test_speed():
    e = AnestheSimExpert("a", map_model="unified")
    e.reset(PATIENT, 1.0)
    inputs = NO_DRUG | dict(propofol_dose=0.2, remifentanil_dose=0.2, norepinephrine_dose=0.1, sevoflurane_et=1.0)
    n = 20_000
    start = time.perf_counter()
    for _ in range(n):
        e.step(inputs)
    per_step_ms = (time.perf_counter() - start) / n * 1e3
    print(f"\nAnestheSIM expert: {per_step_ms * 1e3:.1f} us per step")
    assert per_step_ms < 5


def test_bis_calibration_finds_a_more_sensitive_patient():
    dose = NO_DRUG | dict(propofol_dose=0.15, remifentanil_dose=0.002)
    patient = AnestheSimExpert("p")  # the "real" patient: 1.5x as sensitive as the model
    patient.reset(PATIENT, 1.0)
    patient.bis_scale = 1.5
    calibrated = AnestheSimExpert("c", calibrate_bis=True)
    calibrated.reset(PATIENT, 1.0)
    plain = AnestheSimExpert("n")
    plain.reset(PATIENT, 1.0)
    for i in range(3600):
        measured = patient.step(dose)["bis"]
        y = calibrated.step(dose | {"bis": measured, "bis__staleness": 0.0})
        y_plain = plain.step(dose)
    assert y["bis_sensitivity"] == pytest.approx(1.5, rel=0.05)
    assert abs(y["bis"] - measured) < 0.2 * abs(y_plain["bis"] - measured)


def test_bis_calibration_ignores_stale_and_implausible_readings():
    dose = NO_DRUG | dict(propofol_dose=0.15)
    e = AnestheSimExpert("c", calibrate_bis=True)
    e.reset(PATIENT, 1.0)
    for _ in range(600):
        e.step(dose | {"bis": 20.0, "bis__staleness": 5.0})
        y = e.step(dose | {"bis": 0.0, "bis__staleness": 0.0})
    assert y["bis_sensitivity"] == 1.0
