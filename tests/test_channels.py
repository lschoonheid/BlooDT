import numpy as np
import pandas as pd

from digital_twin_engineering.channels import CHANNELS, Perturbation, cumulative_dose, to_grid


def test_cumulative_dose_ignores_jitter_and_survives_resets():
    # 0.18 mL downward jitter (seen in VitalDB) is not a reset; a lasting drop to 0 is
    v = np.array([0, 0.3, 0.69, 0.52, 0.91, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3])
    total = cumulative_dose(np.arange(len(v)), v)
    assert np.all(np.diff(total) >= 0)
    assert total[5] == 1.2
    assert np.isclose(total[-1], 1.2 + 0.3)


def test_reset_to_a_nonzero_volume_counts_only_new_growth():
    # case 7: RFTN20_VOL drops 65.09 -> 17.92 and carries on from there
    v = np.array([65.0, 65.09, 17.92, 17.92, 17.93, 17.93, 17.93, 18.1])
    total = cumulative_dose(np.arange(len(v)), v)
    assert np.isclose(total[-1], 65.09 + (18.1 - 17.92))


def test_a_one_off_glitch_to_zero_is_not_a_reset():
    v = np.array([10.0, 10.1, 0.0, 10.2, 10.3, 10.4, 10.5, 10.6])
    assert np.isclose(cumulative_dose(np.arange(len(v)), v)[-1], 10.6)


def test_impossible_spikes_are_ignored():
    # case 926: one reading of 249037.83 mL between ~13 mL readings
    t = np.array([0.0, 187, 188, 189, 190, 191, 192, 193])
    v = np.array([12.48, 249037.83, 13.88, 13.88, 13.89, 13.89, 13.9, 13.9])
    assert np.isclose(cumulative_dose(t, v)[-1], 13.9)


def test_persistent_impossible_jump_is_rebased_not_counted():
    t = np.arange(10.0)
    v = np.array([5.0, 5.1, 900.0, 900.0, 900.1, 900.1, 900.2, 900.2, 900.3, 900.3])
    assert np.isclose(cumulative_dose(t, v)[-1], 5.1 + 0.3)


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
    v = np.array([0.0, 0.3, 0.6, 20.0, 20.3])
    values, _ = to_grid(CHANNELS["propofol_dose"], t, v, np.arange(0, 110.0), factor=20)
    assert np.isclose(values.sum(), 20.3 * 20)
    assert values[50] == 0


def test_perturbation_delays_and_drops_records():
    rng = np.random.default_rng(0)
    t = np.arange(0, 7200.0, 2)
    v = np.ones_like(t)
    pt, pv = Perturbation(max_delay_s=10, dropouts_per_hour=10, mean_dropout_s=60).apply(t, v, rng)
    assert len(pt) < len(t) and len(pt) == len(pv)
    assert pt[0] >= t[0]


def test_perturbation_of_a_channel_ignores_other_channels(monkeypatch):
    from digital_twin_engineering import channels

    t = np.arange(0, 3600, 2.0)

    class Track:
        recs = [{"dt": x, "val": 80.0 + (x % 7)} for x in t]

    class Vital:
        dtstart, dtend = 0.0, 3600.0

        def get_track_names(self):
            return ["Solar8000/ART_MBP", "Solar8000/HR"]

        def find_track(self, name):
            return Track()

    monkeypatch.setattr(channels.vdb, "load_vital", lambda caseid, tracks: Vital())
    p = channels.Perturbation(max_delay_s=10, dropouts_per_hour=4)
    alone = channels.load_channels(1, ["map"], perturbation=p, seed=3)
    together = channels.load_channels(1, ["hr", "map"], perturbation=p, seed=3)
    pd.testing.assert_series_equal(alone["map"], together["map"])


def test_a_persistent_jump_counts_only_once_it_is_known():
    # a refill re-bases the counter, but that is only clear 5 readings later, so the
    # dose delivered meanwhile shows up when it becomes known, as it would live
    t = np.arange(12.0)
    v = np.array([1.0, 1.1, 1.2, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8])
    grid = np.arange(12.0)
    dose, staleness = to_grid(CHANNELS["propofol_dose"], t, v, grid)
    assert np.allclose(dose[3:8], 0)
    assert np.isclose(dose[8], 0.5)  # readings 4-8, 0.1 each
    assert np.isclose(dose.sum(), 1.2 - 1.0 + 9.8 - 9.0)
    assert np.isclose(staleness[8], 0)
