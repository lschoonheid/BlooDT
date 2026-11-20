"""The twin, fed raw records one at a time, sees exactly what the offline pipeline
computes for a case: this is what makes the offline features causal."""

from collections import deque

import numpy as np
import pytest

from digital_twin_engineering import dataset, stages, vitaldb_cache
from digital_twin_engineering.config import Config
from digital_twin_engineering.live.replayer import make_twin, records

CASE = 45  # its induction bolus makes the pump volume jump (see channels.DoseCounter)

pytestmark = pytest.mark.skipif(
    not (vitaldb_cache.CACHE_DIR / f"vital_files/{CASE:04d}.vital").exists(),
    reason="VitalDB case not downloaded",
)


def test_twin_grid_matches_offline_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "DERIVED_DIR", tmp_path)
    config = Config.from_dict(
        {
            "channels": ["map", "hr", "bis"],
            "experts": [{"kind": "anesthesim", "name": "asim", "baseline": "pre_dose", "calibrate_bis": True}],
            "expert_lookahead_s": [300],
            "decision_every_s": 30,
            "features": {"windows_s": [60, 300]},
        }
    )
    offline = dataset.case_frame(CASE, config)
    tracks, recs, duration = records(CASE, list(config.all_channels))
    twin = make_twin(CASE, config, {"forecasters": [], "suggesters": []}, tracks)
    twin.buffer = deque()  # keep the whole case
    i = 0
    for t in offline.index:
        while i < len(recs) and recs[i][0] <= t:
            record_t, track, value = recs[i]
            twin.observe(track, record_t, value)
            i += 1
        twin.tick(t)
    live = twin.frame()[offline.columns]
    np.testing.assert_allclose(live.to_numpy(), offline.to_numpy(), rtol=1e-9, atol=1e-9)
