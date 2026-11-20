import numpy as np
import pandas as pd

from digital_twin_engineering.channels import STALENESS
from digital_twin_engineering.labels import detect_interventions, hypotension_mask
from digital_twin_engineering.targets import TARGETS


def series(values):
    return pd.Series(np.asarray(values, float), index=pd.Index(np.arange(len(values)), name="t"))


def test_hypotension_needs_a_full_minute():
    short = series([80] * 10 + [60] * 59 + [80] * 10)
    long = series([80] * 10 + [60] * 61 + [80] * 10)
    assert not hypotension_mask(short).any()
    assert hypotension_mask(long).sum() >= 60


def test_single_spikes_do_not_break_an_event():
    v = [80] * 10 + [60] * 40 + [90] + [60] * 40 + [80] * 10
    assert hypotension_mask(series(v)).sum() >= 80


def test_missing_map_is_not_hypotension():
    assert not hypotension_mask(series([np.nan] * 120)).any()


def test_interventions_skip_knob_transients():
    # 3 -> (3.5 for 2 s) -> 4, then 4 -> 2
    v = [3.0] * 60 + [3.5] * 2 + [4.0] * 60 + [2.0] * 60
    events = detect_interventions(series(v))
    assert events[["old", "new", "direction"]].values.tolist() == [
        [3.0, 4.0, "increase"],
        [4.0, 2.0, "decrease"],
    ]
    assert events.t.tolist() == [62, 122]


def test_pump_start_is_an_intervention_and_tiny_changes_are_not():
    v = [np.nan] * 30 + [4.0] * 60 + [4.1] * 60
    events = detect_interventions(series(v))
    assert events[["old", "new"]].values.tolist() == [[0.0, 4.0]]


def frame(map_values, staleness=0.0):
    m = series(map_values)
    return pd.DataFrame({"map": m, "map" + STALENESS: staleness}, index=m.index)


def test_value_target_looks_ahead_and_needs_a_fresh_observation():
    f = frame(np.arange(100.0))
    y = TARGETS["map"].label(f, horizon_s=10)
    assert y.iloc[0] == 10 and np.isnan(y.iloc[95])
    stale = frame(np.arange(100.0), staleness=60.0)
    assert TARGETS["map"].label(stale, horizon_s=10).isna().all()


def test_event_target_is_any_event_within_the_horizon():
    f = frame([80] * 200 + [55] * 100 + [80] * 100)
    y = TARGETS["hypotension"].label(f, horizon_s=60)
    assert y.iloc[100] == 0  # event starts 100 s later, beyond the horizon
    assert y.iloc[150] == 1  # within 60 s
    assert y.iloc[199] == 1


def test_smoothed_target_is_the_median_around_the_horizon():
    values = np.full(300, 80.0)
    values[150] = 20.0  # a flush artefact
    y = TARGETS["map_1min"].label(frame(values), horizon_s=150)
    assert y.iloc[0] == 80.0
    assert TARGETS["map"].label(frame(values), horizon_s=150).iloc[0] == 20.0
