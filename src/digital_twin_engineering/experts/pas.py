"""Expert backed by the Python Anesthesia Simulator (PAS).

See docs/adr/0003-pas-from-git-on-python-3-14.md for why PAS comes from a git commit.
"""

from collections.abc import Mapping

import python_anesthesia_simulator as pas

from ..patients import Covariates


class PasExpert:
    """One PAS patient configuration driven by the case's pump doses."""

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
    ):
        self.name = name
        self.models = dict(
            model_propo=model_propo,
            model_remi=model_remi,
            model_bis=model_bis,
            model_hemo=model_hemo,
            model_nore=model_nore,
        )

    def reset(self, covariates: Covariates, step_s: float) -> None:
        c = covariates
        self.step_s = step_s
        self.patient = pas.Patient(
            [c.age, c.height, c.weight, int(c.is_male)], ts=step_s, **self.models
        )

    def step(self, inputs: Mapping[str, float]) -> Mapping[str, float]:
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
