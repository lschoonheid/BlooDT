"""Expert ported from AnestheSIM 1.0.0 (MATLAB/Simulink; github.com/izabirs/AnestheSIM).

Copyright (c) 2026 Isabela Birs. Ported under the MIT License; the full licence text
is the LICENSE file of the AnestheSIM repository, and this notice must be kept.

The port follows simulator/build_anesthesim_model.m: covariate PK models
(Schnider, Minto, Joachim norepinephrine, volatile effect sites) feed a BIS
interaction surface and a MAP/HR model. Deviations from the Simulink model:

- Linear compartments advance with an exact zero-order-hold step (as simulateLTI.m)
  instead of ode15s; the turnover MAP/HR model uses an exponential step with
  concentrations and HR held at their step midpoints.
- AnestheSIM takes MAP0/HR0 from the pre-induction recording (estimateBaselines.m),
  which is not causal. Here MAP0 defaults to that file's covariate fallback and HR0
  to the published 70.8 bpm; both can be overridden.
- AnestheSIM starts the noxious-stimulation term of the turnover model at t = 0.
  The incision time is not known causally, so stimulation is off unless
  `stim_onset_s` is given.
"""

import math
from collections import deque
from collections.abc import Mapping

import numpy as np
from scipy.linalg import block_diag, expm

from ..patients import Covariates

NE_MOLAR_MASS = 169.18  # g/mol, buildPatientModels.m
MAC40 = {"sevoflurane": 1.8, "desflurane": 6.6}  # vol%, Mapleson 1996 (VolatileModel.m)
MAPLESON_AGE_DECAY = -0.00269  # 1/year, VolatileModel.m

# BIS surfaces, bisParameters.m: (C50, gamma) per drug and the interaction E0, Emax, gamma.
BIS_PARAMS = {
    "greco": dict(prop=(3.34, 1.2), remi=(2.41, 1.2), e0=95.0, emax=70.0, gamma=1.2),
    "bouillon": dict(prop=(4.47, 1.43), remi=(19.3, 1.43), e0=97.4, emax=97.4, gamma=1.43),
    "minto": dict(prop=(3.35, 1.58), remi=(7.96, 7.85), e0=93.0, emax=83.0, gamma=1.0),
}


def _lbm(c: Covariates) -> float:
    """James lean body mass [kg] (Patient.m)."""
    if c.is_male:
        return 1.10 * c.weight - 128 * (c.weight / c.height) ** 2
    return 1.07 * c.weight - 148 * (c.weight / c.height) ** 2


def _three_compartment(v1, v2, v3, cl1, cl2, cl3, ke0):
    """States [C1, C2, C3, Ce], input mass rate; clearances in L/min, ke0 in 1/s.

    threeCompartmentModel.m
    """
    k10, k12, k13 = cl1 / v1 / 60, cl2 / v1 / 60, cl3 / v1 / 60
    k21, k31 = cl2 / v2 / 60, cl3 / v3 / 60
    a = np.array(
        [
            [-(k10 + k12 + k13), k21, k31, 0],
            [k12, -k21, 0, 0],
            [k13, 0, -k31, 0],
            [ke0, 0, 0, -ke0],
        ]
    )
    return a, np.array([1 / v1, 0, 0, 0])


def _schnider(c: Covariates, lbm: float, ke0_per_min: float):
    # SchniderModel.m; ke0 1.21/min (Orchestra/Marsh) instead of Schnider's 0.456.
    age, w, h = c.age, c.weight, c.height
    return _three_compartment(
        4.27,
        18.9 - 0.391 * (age - 53),
        238,
        1.89 + 0.0456 * (w - 77) - 0.0681 * (lbm - 59) + 0.0264 * (h - 177),
        1.29 - 0.024 * (age - 53),
        0.836,
        ke0_per_min / 60,
    )


def _minto(c: Covariates, lbm: float):
    # MintoModel.m
    age = c.age
    return _three_compartment(
        5.1 - 0.0201 * (age - 40) + 0.072 * (lbm - 55),
        9.82 - 0.0811 * (age - 40) + 0.108 * (lbm - 55),
        5.42,
        2.6 - 0.0162 * (age - 40) + 0.0191 * (lbm - 55),
        2.05 - 0.0301 * (age - 40),
        0.076 - 0.00113 * (age - 40),
        (0.595 - 0.007 * (age - 40)) / 60,
    )


def _joachim(c: Covariates, lbm: float, bmi: float) -> dict:
    """Joachim 2024 norepinephrine PK/PD; states [depot, central, peripheral] in nmol.

    JoachimModel.m: smoking and hypertension set to 0, depot non-linearity linearised.
    """
    male = float(c.is_male)
    kel = math.exp(0.64 * math.log(lbm) - 5.52)  # 1/s
    ka = math.exp(-3.31)  # 1/s
    v1, k12, k21 = 0.49, 0.06, 0.04  # L, 1/s, 1/s
    a = np.array([[-ka, 0, 0], [ka, -(k12 + kel), k21], [0, k12, -k21]])
    return dict(
        a=a,
        b=np.array([1.0, 0, 0]),
        v1=v1,
        kel=kel,
        tlag=math.exp(0.26 * male + 3.21),  # s
        cendo=math.exp(0.57 * math.log(c.age) + 0.67 * math.log(bmi) - 4.56),  # nmol/L
        emax=math.exp(-0.46 * male + 4.09),  # mmHg
        ec50=15.7,  # nmol/L
        gamma=1.46,
    )


def covariate_map0(c: Covariates, bmi: float) -> float:
    """Heuristic awake MAP [mmHg] (estimateBaselines.m, covariate fallback)."""
    map0 = 75 + 0.4 * max(0.0, c.age - 30) + 5 * c.is_male + 0.5 * max(0.0, bmi - 25)
    return min(max(map0, 65.0), 115.0)


def _hill(u: float, gamma: float) -> float:
    ug = max(u, 0.0) ** gamma
    return ug / (1 + ug)


class AnestheSimExpert:
    """AnestheSIM's patient model driven by pump doses and end-tidal volatile levels.

    `bis_model`: "greco" (reduced Greco, the only surface with the volatile agents),
    "bouillon" or "minto" (intravenous only). `map_model`: "separable" (turnover model
    plus Joachim pressor effect) or "unified" (three-drug effect-site model).
    HR always comes from the turnover model.
    """

    inputs = (
        "propofol_dose",
        "remifentanil_dose",
        "norepinephrine_dose",
        "sevoflurane_et",
        "desflurane_et",
    )
    signals = (
        "propofol_cp",  # ug/mL
        "propofol_ce",  # ug/mL
        "remifentanil_cp",  # ng/mL
        "remifentanil_ce",  # ng/mL
        "norepinephrine_cp",  # nmol/L, exogenous only
        "sevoflurane_ce",  # vol%
        "desflurane_ce",  # vol%
        "volatile_mac",  # summed effect-site MAC fraction, age-adjusted
        "bis",
        "map",  # mmHg
        "hr",  # /min
    )

    def __init__(
        self,
        name: str,
        bis_model: str = "greco",
        map_model: str = "separable",
        propofol_ke0: float = 1.21,  # 1/min
        map0: float | None = None,  # mmHg; default: covariate estimate
        hr0: float | None = None,  # /min; default: 70.8
        stim_onset_s: float | None = None,  # s from case start; None: no stimulation
    ):
        if bis_model not in BIS_PARAMS:
            raise ValueError(f"unknown bis_model {bis_model!r}")
        if map_model not in ("separable", "unified"):
            raise ValueError(f"unknown map_model {map_model!r}")
        self.name = name
        self.bis_model = bis_model
        self.map_model = map_model
        self.propofol_ke0 = propofol_ke0
        self.map0_override = map0
        self.hr0_override = hr0
        self.stim_onset_s = stim_onset_s

    def reset(self, covariates: Covariates, step_s: float) -> None:
        c = covariates
        self.step_s = step_s
        lbm = _lbm(c)
        bmi = c.weight / (c.height / 100) ** 2

        # Linear part, states [propofol C1-3 Ce | remi C1-3 Ce | NE depot, central,
        # peripheral | Ce sevo | Ce des], inputs [mg/s, ug/s, nmol/s, vol%, vol%].
        prop_a, prop_b = _schnider(c, lbm, self.propofol_ke0)
        remi_a, remi_b = _minto(c, lbm)
        self.ne = ne = _joachim(c, lbm, bmi)
        vol_ke0 = 0.47 / 60  # 1/s, VolatileModel.m (both agents)
        a = block_diag(prop_a, remi_a, ne["a"], [[-vol_ke0]], [[-vol_ke0]])
        b = np.zeros((13, 5))
        b[0:4, 0], b[4:8, 1], b[8:11, 2] = prop_b, remi_b, ne["b"]
        b[11, 3] = b[12, 4] = vol_ke0
        m = expm(np.block([[a, b], [np.zeros((5, 18))]]) * step_s)
        self.ad, self.bd = m[:13, :13], m[:13, 13:]
        self.x = np.zeros(13)
        self.u = np.zeros(5)
        # Transport lag of the NE input, rounded to the grid (simulateLTI.m).
        self.ne_queue = deque([0.0] * round(ne["tlag"] / step_s))

        mac_age = 10 ** (MAPLESON_AGE_DECAY * (c.age - 40))
        self.mac = {k: v * mac_age for k, v in MAC40.items()}
        # bisParameters.m / VolatileModel.m: sevo C50 1.2 vol% at 40 y, des scaled by MAC ratio.
        self.c50_sevo = 1.2 * mac_age
        self.c50_des = 1.2 * MAC40["desflurane"] / MAC40["sevoflurane"] * mac_age
        self.bis_params = BIS_PARAMS[self.bis_model]

        self.map0 = self.map0_override if self.map0_override is not None else covariate_map0(c, bmi)
        self.hr0 = self.hr0_override if self.hr0_override is not None else 70.8
        # hemodynamicParameters.m (Garraza-Obaldia 2024, plasma concentrations)
        self.kout = 0.126 / 60  # 1/s
        self.decay = math.exp(-self.kout * step_s)
        self.map_turnover, self.hr = self.map0, self.hr0
        self.t = 0.0

    def _bis(self, ce_p: float, ce_r: float, ce_sevo: float, ce_des: float) -> float:
        # bisMinto.m, bisBouillon.m, bisGrecoVolatile.m
        p = self.bis_params
        (c50_p, g_p), (c50_r, g_r) = p["prop"], p["remi"]
        ce_p, ce_r = max(ce_p, 0.0), max(ce_r, 0.0)
        if self.bis_model == "minto":
            s = (ce_p / c50_p) ** g_p + (ce_r / c50_r) ** g_r
            effect = s / (1 + s)
        elif self.bis_model == "bouillon":
            effect = _hill(ce_p / c50_p + ce_r / c50_r, p["gamma"])
        else:
            hypnotic = ce_p / c50_p + max(ce_sevo, 0.0) / self.c50_sevo + max(ce_des, 0.0) / self.c50_des
            effect = _hill(hypnotic * (1 + ce_r / c50_r), p["gamma"])
        return min(max(p["e0"] - p["emax"] * effect, 0.0), 100.0)

    def _ne_emax(self, cc: float) -> float:
        ne = self.ne
        return ne["emax"] * _hill(cc / ne["ec50"], ne["gamma"])

    def _turnover(self, cp_p: float, cp_r: float) -> None:
        """Advance MAP/HR one step (hemodynamicTurnover.m), inputs held at mid-step."""
        cp_p, cp_r = max(cp_p, 0.0), max(cp_r, 0.0)
        f_p = 0.77 * cp_p / (cp_p + 7.38)  # EmaxP, C50P [ug/mL]
        g_r = 0.65 * _hill(cp_r / 3.63, 1.37)  # EmaxR, C50R [ng/mL], gammaR
        hr_ss = self.hr0 * (1 - g_r)
        hr_new = hr_ss + (self.hr - hr_ss) * self.decay
        h = 1.0
        t_mid = self.t + self.step_s / 2
        if self.stim_onset_s is not None and t_mid >= self.stim_onset_s:
            h += 0.277 * math.exp(-0.018 / 60 * (t_mid - self.stim_onset_s))  # Stim, kStim
        hr_mid = max((self.hr + hr_new) / 2, 1e-6)
        production = self.map0 * (1 - f_p) * (hr_mid / self.hr0) ** 0.842 * h  # FBhr
        self.map_turnover = production + (self.map_turnover - production) * self.decay
        self.hr = hr_new

    def step(self, inputs: Mapping[str, float]) -> Mapping[str, float]:
        dt = self.step_s
        u = self.u
        u[0] = inputs["propofol_dose"] / dt  # mg/s
        u[1] = inputs["remifentanil_dose"] / dt  # ug/s
        ne_rate = inputs["norepinephrine_dose"] / dt * 1e3 / NE_MOLAR_MASS  # nmol/s
        if self.ne_queue:
            self.ne_queue.append(ne_rate)
            ne_rate = self.ne_queue.popleft()
        u[2] = ne_rate
        sevo, des = inputs["sevoflurane_et"], inputs["desflurane_et"]
        u[3] = sevo if sevo == sevo else 0.0  # NaN: no volatile
        u[4] = des if des == des else 0.0

        x0 = self.x
        x = self.x = self.ad @ x0 + self.bd @ u
        cp_p, ce_p, cp_r, ce_r = float(x[0]), float(x[3]), float(x[4]), float(x[7])
        cc_ne = float(x[9]) / self.ne["v1"]
        ce_sevo, ce_des = float(x[11]), float(x[12])

        self._turnover((float(x0[0]) + cp_p) / 2, (float(x0[4]) + cp_r) / 2)
        self.t += dt

        if self.map_model == "separable":
            # norepinephrineEffect.m: pressor effect above the endogenous level
            cendo = self.ne["cendo"]
            map_ = self.map_turnover + self._ne_emax(max(cc_ne, 0.0) + cendo) - self._ne_emax(cendo)
        else:
            # mapUnified.m / mapUnifiedParameters.m: the propofol/remifentanil values
            # are AnestheSIM's illustrative ones (no published set); NE on exogenous Cc.
            ce_pp, ce_rr = max(ce_p, 0.0), max(ce_r, 0.0)
            c50 = max(2.0 * (1 - 0.5 * ce_rr / (ce_rr + 2.0)), 1e-6)  # C50P, kR, C50R
            map_ = self.map0 - 30.0 * ce_pp / (ce_pp + c50) + self._ne_emax(max(cc_ne, 0.0))  # EmaxP

        return {
            "propofol_cp": cp_p,
            "propofol_ce": ce_p,
            "remifentanil_cp": cp_r,
            "remifentanil_ce": ce_r,
            "norepinephrine_cp": cc_ne,
            "sevoflurane_ce": ce_sevo,
            "desflurane_ce": ce_des,
            "volatile_mac": max(ce_sevo, 0.0) / self.mac["sevoflurane"]
            + max(ce_des, 0.0) / self.mac["desflurane"],
            "bis": self._bis(ce_p, ce_r, ce_sevo, ce_des),
            "map": map_,
            "hr": self.hr,
        }
