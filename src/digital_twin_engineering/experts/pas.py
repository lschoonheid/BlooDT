"""Expert backed by the Python Anesthesia Simulator (PAS).

See docs/adr/0003-pas-from-git-on-python-3-14.md for why PAS comes from a git commit.
"""

from collections.abc import Mapping

import python_anesthesia_simulator as pas

from ..patients import Covariates
from . import baseline as baselines


class PasExpert:
    """One PAS patient configuration driven by the case's pump doses.

    With `baseline="pre_dose"` the patient's MAP and HR at rest are set to those
    observed before the first dose (see baseline.py) instead of PAS's population
    values; stroke volume keeps its population value, so the MAP baseline sets the
    vascular resistance.
    """

    inputs = ("propofol_dose", "remifentanil_dose", "norepinephrine_dose")
    signals = ("propofol_ce", "remifentanil_ce", "bis", "map", "hr", "co")

    def __init__(
        self,
        name: str,
        model_propo: str = "Schnider",
        model_remi: str = "Minto",
        model_bis: str = "Bouillon",
        model_hemo: str = "VitalDB",
        model_nore: str = "Beloeil",
        baseline: str = "covariates",  # or "pre_dose"
    ):
        self.name = name
        self.models = dict(
            model_propo=model_propo,
            model_remi=model_remi,
            model_bis=model_bis,
            model_hemo=model_hemo,
            model_nore=model_nore,
        )
        if baseline not in ("covariates", "pre_dose"):
            raise ValueError(f"unknown baseline {baseline!r}")
        self.baseline = baseline
        if baseline == "pre_dose":
            self.inputs = type(self).inputs + baselines.INPUTS

    def reset(self, covariates: Covariates, step_s: float) -> None:
        c = covariates
        self.step_s = step_s
        self.characteristics = [c.age, c.height, c.weight, int(c.is_male)]
        self.patient = pas.Patient(self.characteristics, ts=step_s, **self.models)
        self.pre_dose = baselines.PreDoseBaseline() if self.baseline == "pre_dose" else None

    def _set_baseline(self, map0: float | None, hr0: float | None) -> None:
        """Rebuild the patient around the observed baseline (no drug given yet)."""
        p = self.patient
        hr = min(max(hr0, 35.0), 140.0) if hr0 is not None else p.hr_base
        map_ = min(max(map0, 50.0), 150.0) if map0 is not None else p.map_base
        self.patient = pas.Patient(
            self.characteristics, co_base=p.co_base, hr_base=hr, map_base=map_,
            ts=self.step_s, **self.models,
        )

    def step(self, inputs: Mapping[str, float]) -> Mapping[str, float]:
        if self.pre_dose is not None and self.pre_dose.update(inputs):
            self._set_baseline(self.pre_dose.map0, self.pre_dose.hr0)
        p, dt = self.patient, self.step_s
        bis, _loc, co, map_, _tol, _tof = p.one_step(
            u_propo=inputs["propofol_dose"] / dt,  # mg/s
            u_remi=inputs["remifentanil_dose"] / dt,  # ug/s
            u_nore=inputs["norepinephrine_dose"] / dt,  # ug/s
        )
        return {
            "propofol_ce": float(p.c_es_propo),
            "remifentanil_ce": float(p.c_es_remi),
            "bis": float(bis),
            "map": float(map_),
            "hr": float(p.hr),
            "co": float(co),
        }
