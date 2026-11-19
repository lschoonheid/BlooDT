import numpy as np

from digital_twin_engineering.channels import CHANNELS, Perturbation, cumulative_dose, to_grid


def test_cumulative_dose_ignores_jitter_and_survives_resets():
    # 0.18 mL downward jitter (seen in VitalDB) is not a reset; a drop to 0 is
    v = np.array([0, 5, 8.69, 8.52, 9.71, 12.0, 0.0, 1.0, 3.0])
    total = cumulative_dose(v)
    assert np.all(np.diff(total) >= 0)
    assert total[5] == 12.0
    assert total[-1] == 12.0 + 3.0


def test_level_grid_is_causal_and_reports_staleness():
    t = np.array([0.0, 2.0, 4.0])
    v = np.array([80.0, 70.0, 60.0])
    values, staleness = to_grid(CHANNELS["map"], t, v, np.arange(0, 6.0))
    # never interpolated: t=1 still holds the t=0 reading
    assert values.tolist() == [80, 80, 70, 70, 60, 60]
    assert staleness.tolist() == [0, 1, 0, 1, 0, 1]


def test_artefacts_are_dropped_and_show_as_staleness():
    t = np.arange(4.0)
    v = np.array([80.0, -9.0, -9.0, 75.0])  # Solar8000 writes -9 for no signal
    values, staleness = to_grid(CHANNELS["map"], t, v, t)
    assert values.tolist() == [80, 80, 80, 75]
    assert staleness.tolist() == [0, 1, 2, 0]


def test_values_beyond_max_staleness_become_missing():
    values, staleness = to_grid(CHANNELS["map"], np.array([0.0]), np.array([80.0]), np.array([0.0, 299, 301]))
    assert np.isnan(values[2]) and staleness[2] == 301
    assert values[1] == 80


def test_nothing_known_before_the_first_record():
    values, staleness = to_grid(CHANNELS["map"], np.array([5.0]), np.array([80.0]), np.arange(0, 7.0))
    assert np.isnan(values[:5]).all() and np.isnan(staleness[:5]).all()


def test_dose_grid_conserves_total_across_gaps():
    # a 100 s gap in VOL readings: the dose arrives when the next reading does
    t = np.array([0.0, 1, 2, 102, 103])
    v = np.array([0.0, 1, 2, 12, 13])
    values, _ = to_grid(CHANNELS["propofol_dose"], t, v * 20, np.arange(0, 110.0))
    assert np.isclose(values.sum(), 13 * 20)
    assert values[50] == 0


def test_perturbation_delays_and_drops_records():
    rng = np.random.default_rng(0)
    t = np.arange(0, 7200.0, 2)
    v = np.ones_like(t)
    pt, pv = Perturbation(max_delay_s=10, dropouts_per_hour=10, mean_dropout_s=60).apply(t, v, rng)
    assert len(pt) < len(t) and len(pt) == len(pv)
    assert pt[0] >= t[0]
